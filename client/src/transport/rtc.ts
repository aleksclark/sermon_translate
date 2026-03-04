import type { StreamTransport, TransportEvent } from "./base.ts";

const ICE_GATHER_TIMEOUT_MS = 2_000;

export class WebRTCTransport implements StreamTransport {
  private pc: RTCPeerConnection | null = null;
  private dc: RTCDataChannel | null = null;
  private audioEl: HTMLAudioElement | null = null;
  private eventCallbacks: ((event: TransportEvent) => void)[] = [];
  private closeCallbacks: (() => void)[] = [];

  constructor(
    private sessionId: string,
    private inputStream: MediaStream,
    private outputDeviceId?: string,
  ) {}

  async connect(): Promise<void> {
    this.pc = new RTCPeerConnection({ iceServers: [] });

    const audioTrack = this.inputStream.getAudioTracks()[0];
    if (audioTrack) {
      this.pc.addTrack(audioTrack, this.inputStream);
    }

    this.pc.ontrack = (ev) => {
      const audio = document.createElement("audio");
      audio.autoplay = true;
      audio.srcObject = ev.streams[0] ?? new MediaStream([ev.track]);
      if (this.outputDeviceId && "setSinkId" in audio) {
        (audio as unknown as { setSinkId: (id: string) => Promise<void> })
          .setSinkId(this.outputDeviceId)
          .catch(() => {});
      }
      audio.play().catch(() => {});
      this.audioEl = audio;
    };

    const dcOpen = new Promise<void>((resolve) => {
      this.dc = this.pc!.createDataChannel("events");
      this.dc.onopen = () => resolve();
      this.dc.onmessage = (ev) => {
        try {
          const parsed = JSON.parse(ev.data as string) as TransportEvent;
          for (const cb of this.eventCallbacks) cb(parsed);
        } catch {
          // ignore malformed events
        }
      };
      this.dc.onclose = () => {
        for (const cb of this.closeCallbacks) cb();
      };
    });

    this.pc.onconnectionstatechange = () => {
      if (this.pc?.connectionState === "failed" || this.pc?.connectionState === "disconnected") {
        for (const cb of this.closeCallbacks) cb();
      }
    };

    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);
    await this.waitForIceGathering();

    const resp = await fetch(`/api/sessions/${this.sessionId}/offer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: this.pc.localDescription!.sdp,
        type: this.pc.localDescription!.type,
      }),
    });
    if (!resp.ok) {
      throw new Error(`Signaling failed: ${resp.status}`);
    }
    const answer = (await resp.json()) as { sdp: string; type: RTCSdpType };
    await this.pc.setRemoteDescription(new RTCSessionDescription(answer));

    await dcOpen;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  sendAudio(_data: ArrayBuffer): void {
    // no-op: WebRTC handles audio natively via addTrack
  }

  sendEvent(event: TransportEvent): void {
    if (!this.dc || this.dc.readyState !== "open") return;
    this.dc.send(JSON.stringify(event));
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  onAudio(_cb: (data: ArrayBuffer) => void): void {
    // no-op: WebRTC playback via <audio> element
  }

  onEvent(cb: (event: TransportEvent) => void): void {
    this.eventCallbacks.push(cb);
  }

  onClose(cb: () => void): void {
    this.closeCallbacks.push(cb);
  }

  setMuted(muted: boolean): void {
    if (this.audioEl) {
      this.audioEl.muted = muted;
    }
  }

  close(): void {
    this.dc?.close();
    this.dc = null;
    if (this.audioEl) {
      this.audioEl.srcObject = null;
      this.audioEl = null;
    }
    this.pc?.close();
    this.pc = null;
  }

  private waitForIceGathering(): Promise<void> {
    return new Promise((resolve) => {
      if (!this.pc) {
        resolve();
        return;
      }
      if (this.pc.iceGatheringState === "complete") {
        resolve();
        return;
      }
      const timer = setTimeout(() => {
        resolve();
      }, ICE_GATHER_TIMEOUT_MS);
      const check = () => {
        if (this.pc?.iceGatheringState === "complete") {
          clearTimeout(timer);
          this.pc.removeEventListener("icegatheringstatechange", check);
          resolve();
        }
      };
      this.pc.addEventListener("icegatheringstatechange", check);
    });
  }
}
