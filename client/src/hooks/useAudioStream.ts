import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ListenProduct,
  MetadataEnvelope,
  SessionStats,
  StageKind,
  TranslateProduct,
} from "../api/index.ts";
import type { AudioSource } from "../components/NewSessionModal.tsx";
import type { TransportEvent } from "../transport/index.ts";
import { WebRTCTransport } from "../transport/index.ts";

const MAX_STREAM_LINES = 200;

interface AudioStreamOptions {
  sessionId: string;
  sampleRate: number;
  channels: number;
  inputDeviceId: string;
  outputDeviceId: string;
  audioSource: AudioSource;
}

export interface TranscriptLine {
  stream: string;
  text: string;
  timestamp: number;
}

export interface MetadataUpdate {
  stream: string;
  envelope: MetadataEnvelope;
  timestamp: number;
}

export type StageProduct = ListenProduct | TranslateProduct | Record<string, unknown>;

export interface StageProductUpdate {
  stage: StageKind;
  product: StageProduct;
  timestamp: number;
}

const STAGE_KINDS = new Set<StageKind>(["listen", "translate", "speak", "prosody"]);

function appendCapped<T>(items: T[] | undefined, item: T, max = MAX_STREAM_LINES): T[] {
  const next = [...(items || []), item];
  return next.length > max ? next.slice(next.length - max) : next;
}

export function parseMetadataEvent(evt: TransportEvent): MetadataUpdate | null {
  if (evt.type !== "pipeline.event" || evt.payload.kind !== "metadata") return null;
  const stream = (evt.payload.stream as string) || "metadata";
  const envelope = evt.payload.metadata as unknown as MetadataEnvelope;
  if (!envelope) return null;
  return { stream, envelope, timestamp: Date.now() };
}

export function parseStageProductEvent(evt: TransportEvent): StageProductUpdate | null {
  if (evt.type !== "pipeline.event" || evt.payload.kind !== "stage.product") return null;
  const stage = evt.payload.stage;
  if (typeof stage !== "string" || !STAGE_KINDS.has(stage as StageKind)) return null;
  const product = evt.payload.product;
  if (product == null || typeof product !== "object") return null;
  return {
    stage: stage as StageKind,
    product: product as StageProduct,
    timestamp: Date.now(),
  };
}

interface FileMediaStreamResult {
  stream: MediaStream;
  durationMs: number;
}

async function createFileMediaStream(file: File, sampleRate: number): Promise<FileMediaStreamResult> {
  const audioCtx = new AudioContext({ sampleRate });
  await audioCtx.resume();
  const arrayBuffer = await file.arrayBuffer();
  const decoded = await audioCtx.decodeAudioData(arrayBuffer);

  const source = audioCtx.createBufferSource();
  source.buffer = decoded;
  const dest = audioCtx.createMediaStreamDestination();
  source.connect(dest);
  source.start();

  source.onended = () => {
    dest.stream.getTracks().forEach((t) => t.stop());
  };

  return { stream: dest.stream, durationMs: decoded.duration * 1000 };
}

export function useAudioStream(options: AudioStreamOptions | null) {
  const [connected, setConnected] = useState(false);
  const [muted, setMuted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [liveStats, setLiveStats] = useState<SessionStats | null>(null);
  const [transcripts, setTranscripts] = useState<Record<string, TranscriptLine[]>>({});
  const [metadata, setMetadata] = useState<Record<string, MetadataUpdate[]>>({});
  const [stageProducts, setStageProducts] = useState<Record<string, StageProductUpdate[]>>({});
  const transportRef = useRef<WebRTCTransport | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const cancelledRef = useRef(false);

  const stop = useCallback(() => {
    cancelledRef.current = true;
    transportRef.current?.sendEvent({
      type: "session.stop",
      session_id: "",
      payload: {},
    });
    transportRef.current?.close();
    transportRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setConnected(false);
    setMuted(false);
    setLiveStats(null);
    setTranscripts({});
    setMetadata({});
    setStageProducts({});
  }, []);

  const toggleMute = useCallback(() => {
    setMuted((prev) => {
      const next = !prev;
      transportRef.current?.setMuted(next);
      return next;
    });
  }, []);

  useEffect(() => {
    if (!options) return;
    cancelledRef.current = false;

    async function start() {
      const { sessionId, sampleRate, channels, audioSource, inputDeviceId, outputDeviceId } =
        options!;
      setError(null);

      let inputStream: MediaStream;
      let fileDurationMs: number | null = null;

      try {
        if (audioSource.type === "mic") {
          inputStream = await navigator.mediaDevices.getUserMedia({
            audio: {
              deviceId: inputDeviceId ? { exact: inputDeviceId } : undefined,
              sampleRate: { ideal: sampleRate },
              channelCount: { ideal: channels },
              echoCancellation: false,
              noiseSuppression: false,
              autoGainControl: false,
            },
          });
        } else if (audioSource.type === "file" && audioSource.file) {
          const result = await createFileMediaStream(audioSource.file, sampleRate);
          inputStream = result.stream;
          fileDurationMs = result.durationMs;
        } else {
          return;
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : "failed to open audio input");
        return;
      }

      if (cancelledRef.current) {
        inputStream.getTracks().forEach((t) => t.stop());
        return;
      }
      streamRef.current = inputStream;

      const transport = new WebRTCTransport(sessionId, inputStream, outputDeviceId);
      try {
        await transport.connect();
      } catch (err) {
        inputStream.getTracks().forEach((t) => t.stop());
        setError(err instanceof Error ? err.message : "failed to connect transport");
        return;
      }
      if (cancelledRef.current) {
        transport.close();
        return;
      }

      transportRef.current = transport;
      setConnected(true);

      if (audioSource.type === "file" && fileDurationMs != null) {
        const track = inputStream.getAudioTracks()[0];
        let audioEndSent = false;
        const sendAudioEnd = () => {
          if (audioEndSent) return;
          audioEndSent = true;
          transport.sendEvent({
            type: "audio.end",
            session_id: sessionId,
            payload: {},
          });
        };
        if (track) {
          track.addEventListener("ended", sendAudioEnd);
        }
        setTimeout(sendAudioEnd, fileDurationMs + 500);
      }

      transport.onEvent((evt: TransportEvent) => {
        if (evt.type === "session.stats") {
          setLiveStats(evt.payload as unknown as SessionStats);
        } else if (evt.type === "error") {
          const detail = evt.payload.detail;
          setError(typeof detail === "string" && detail ? detail : "stream error");
        } else if (evt.type === "pipeline.event" && evt.payload.kind === "transcript") {
          const streamName = (evt.payload.stream as string) || "transcript";
          const text = evt.payload.text as string;
          const line: TranscriptLine = { stream: streamName, text, timestamp: Date.now() };
          setTranscripts((prev) => ({
            ...prev,
            [streamName]: appendCapped(prev[streamName], line),
          }));
        } else {
          const stageUpdate = parseStageProductEvent(evt);
          if (stageUpdate) {
            setStageProducts((prev) => ({
              ...prev,
              [stageUpdate.stage]: appendCapped(prev[stageUpdate.stage], stageUpdate),
            }));
            return;
          }
          const update = parseMetadataEvent(evt);
          if (update) {
            setMetadata((prev) => ({
              ...prev,
              [update.stream]: appendCapped(prev[update.stream], update),
            }));
          }
        }
      });

      transport.onClose(() => {
        setConnected(false);
      });
    }

    start();
    return () => {
      cancelledRef.current = true;
      stop();
    };
  }, [options?.sessionId]);

  return {
    connected,
    muted,
    error,
    liveStats,
    transcripts,
    metadata,
    stageProducts,
    stop,
    toggleMute,
  };
}
