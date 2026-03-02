import fs from "node:fs";
import path from "node:path";
import { test, expect } from "@playwright/test";

const FIXTURE_MP3 = path.resolve(import.meta.dirname, "../fixtures/test-speech.mp3");
const FIXTURE_PCM = path.resolve(import.meta.dirname, "../fixtures/test-speech.pcm");

test.describe("MP3 file upload session", () => {
  test("streams MP3 through whisper-tts and shows transcript", async ({ page }) => {
    test.setTimeout(120_000);

    // Read pre-decoded PCM fixture (Int16LE 48kHz mono) as base64.
    // This is ~370KB base64 vs ~1.6MB JSON array of floats.
    const pcmBase64 = fs.readFileSync(FIXTURE_PCM).toString("base64");
    const sampleCount = fs.statSync(FIXTURE_PCM).size / 2;

    // Monkey-patch AudioContext.decodeAudioData and OfflineAudioContext
    // so they work in headless Chrome (which cannot decode MP3 audio).
    // The pre-decoded PCM data is injected directly.
    await page.addInitScript(
      ({ b64, count }: { b64: string; count: number }) => {
        // Decode base64 → Int16 → Float32
        const raw = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
        const i16 = new Int16Array(raw.buffer);
        const pcmData = new Float32Array(count);
        for (let i = 0; i < count; i++) {
          pcmData[i] = i16[i] < 0 ? i16[i] / 0x8000 : i16[i] / 0x7fff;
        }

        AudioContext.prototype.decodeAudioData = function (
          _arrayBuffer: ArrayBuffer,
          successCb?: DecodeSuccessCallback | null,
          _errorCb?: DecodeErrorCallback | null,
        ): Promise<AudioBuffer> {
          const buf = this.createBuffer(1, count, 48000);
          buf.copyToChannel(pcmData, 0);
          if (successCb) successCb(buf);
          return Promise.resolve(buf);
        };

        // OfflineAudioContext.startRendering may also fail in headless mode.
        // Patch it to pass the source buffer through without real rendering.
        const OrigOffline = globalThis.OfflineAudioContext;
        class MockOfflineAudioContext extends OrigOffline {
          private _srcBuf: AudioBuffer | null = null;

          createBufferSource(): AudioBufferSourceNode {
            const node = super.createBufferSource();
            const desc = Object.getOwnPropertyDescriptor(
              AudioBufferSourceNode.prototype,
              "buffer",
            );
            const ctx = this;
            Object.defineProperty(node, "buffer", {
              get() {
                return desc?.get?.call(this) ?? null;
              },
              set(val: AudioBuffer | null) {
                ctx._srcBuf = val;
                desc?.set?.call(this, val);
              },
            });
            return node;
          }

          startRendering(): Promise<AudioBuffer> {
            if (this._srcBuf) return Promise.resolve(this._srcBuf);
            return super.startRendering();
          }
        }
        globalThis.OfflineAudioContext = MockOfflineAudioContext as typeof OfflineAudioContext;
      },
      { b64: pcmBase64, count: sampleCount },
    );

    await page.goto("/");
    await expect(page.getByText("Sessions", { exact: true })).toBeVisible();

    // Open the New Session modal
    await page.locator('button[aria-label="New session"]').click();
    await expect(page.getByText("New Session")).toBeVisible();

    // Select Whisper TTS pipeline
    const pipelineSelect = page.locator("select").first();
    await pipelineSelect.selectOption({ label: "Whisper TTS" });

    // Switch to MP3 File source
    await page.getByText("MP3 File", { exact: true }).click();
    await expect(page.getByText("No file selected")).toBeVisible();

    // Upload the MP3 fixture
    const fileChooserPromise = page.waitForEvent("filechooser");
    await page.getByText("Choose MP3 file").click();
    const fileChooser = await fileChooserPromise;
    await fileChooser.setFiles(FIXTURE_MP3);
    await expect(page.getByText("test-speech.mp3")).toBeVisible();

    // Start the session
    await page.locator('button:has-text("Start Session")').click();

    // Wait for active session and WebSocket connection
    await expect(page.getByText("Active Session")).toBeVisible({ timeout: 10_000 });
    await expect(page.getByText("Streaming")).toBeVisible({ timeout: 10_000 });

    // Wait for transcript — the file is 2.88s streamed at real-time pace,
    // and Whisper buffers 3s before transcribing, so ~6-10s total.
    await expect(page.getByText("Transcript", { exact: true })).toBeVisible({ timeout: 60_000 });

    const transcriptArea = page.locator('[data-testid="transcript-area"]');
    await expect(transcriptArea).toContainText(/grace/i, { timeout: 60_000 });
  });
});
