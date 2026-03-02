import { useCallback, useEffect, useRef, useState } from "react";
import type { SessionStats } from "../api/index.ts";
import type { AudioSource } from "../components/NewSessionModal.tsx";
import type { StreamTransport, TransportEvent } from "../transport/index.ts";
import { WebSocketTransport } from "../transport/index.ts";

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

const CHUNK_SIZE = 4096;

function floatToInt16(pcm: Float32Array): Int16Array {
  const i16 = new Int16Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]));
    i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return i16;
}

async function streamFileAudio(
  file: File,
  sampleRate: number,
  transport: StreamTransport,
  audioCtx: AudioContext,
  cancelledRef: { current: boolean },
): Promise<void> {
  const arrayBuffer = await file.arrayBuffer();
  const decoded = await audioCtx.decodeAudioData(arrayBuffer);

  const offlineCtx = new OfflineAudioContext(1, Math.ceil(decoded.duration * sampleRate), sampleRate);
  const src = offlineCtx.createBufferSource();
  src.buffer = decoded;
  src.connect(offlineCtx.destination);
  src.start();
  const rendered = await offlineCtx.startRendering();

  const pcm = rendered.getChannelData(0);
  const chunkDurationMs = (CHUNK_SIZE / sampleRate) * 1000;

  for (let offset = 0; offset < pcm.length; offset += CHUNK_SIZE) {
    if (cancelledRef.current) return;
    const end = Math.min(offset + CHUNK_SIZE, pcm.length);
    const chunk = pcm.subarray(offset, end);
    const i16 = floatToInt16(chunk);
    transport.sendAudio(i16.buffer as ArrayBuffer);
    await new Promise((r) => setTimeout(r, chunkDurationMs));
  }

  if (!cancelledRef.current) {
    transport.sendAudio(new ArrayBuffer(0));
  }
}

export function useAudioStream(options: AudioStreamOptions | null) {
  const [connected, setConnected] = useState(false);
  const [liveStats, setLiveStats] = useState<SessionStats | null>(null);
  const [transcripts, setTranscripts] = useState<Record<string, TranscriptLine[]>>({});
  const [playbackDelay, setPlaybackDelay] = useState(0);
  const transportRef = useRef<StreamTransport | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const cancelledRef = useRef(false);
  const nextPlayTimeRef = useRef(0);
  const delayIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stop = useCallback(() => {
    cancelledRef.current = true;
    transportRef.current?.close();
    transportRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    contextRef.current?.close();
    contextRef.current = null;
    nextPlayTimeRef.current = 0;
    if (delayIntervalRef.current) {
      clearInterval(delayIntervalRef.current);
      delayIntervalRef.current = null;
    }
    setConnected(false);
    setLiveStats(null);
    setTranscripts({});
    setPlaybackDelay(0);
  }, []);

  useEffect(() => {
    if (!options) return;
    cancelledRef.current = false;

    async function start() {
      const { sessionId, sampleRate, audioSource, inputDeviceId, outputDeviceId } = options!;

      const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${wsProto}//${location.host}/ws/stream/${sessionId}`;
      const transport = new WebSocketTransport(wsUrl);

      try {
        await transport.connect();
      } catch {
        return;
      }
      if (cancelledRef.current) {
        transport.close();
        return;
      }

      transportRef.current = transport;
      setConnected(true);

      const audioCtx = new AudioContext({ sampleRate });
      contextRef.current = audioCtx;

      if (audioSource.type === "mic") {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            deviceId: inputDeviceId ? { exact: inputDeviceId } : undefined,
            sampleRate: { ideal: sampleRate },
            channelCount: { ideal: options!.channels },
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
          },
        });
        if (cancelledRef.current) {
          stream.getTracks().forEach((t) => t.stop());
          transport.close();
          return;
        }
        streamRef.current = stream;

        const source = audioCtx.createMediaStreamSource(stream);
        const processor = audioCtx.createScriptProcessor(CHUNK_SIZE, 1, 1);
        processor.onaudioprocess = (e: AudioProcessingEvent) => {
          const pcm = e.inputBuffer.getChannelData(0);
          transport.sendAudio(floatToInt16(pcm).buffer as ArrayBuffer);
        };
        source.connect(processor);
        processor.connect(audioCtx.destination);
      } else if (audioSource.type === "file" && audioSource.file) {
        streamFileAudio(audioSource.file, sampleRate, transport, audioCtx, cancelledRef);
      }

      transport.onAudio((buf: ArrayBuffer) => {
        const i16 = new Int16Array(buf);
        const f32 = new Float32Array(i16.length);
        for (let i = 0; i < i16.length; i++) {
          f32[i] = i16[i] / (i16[i] < 0 ? 0x8000 : 0x7fff);
        }
        const abuf = audioCtx.createBuffer(1, f32.length, sampleRate);
        abuf.copyToChannel(f32, 0);
        const src = audioCtx.createBufferSource();
        src.buffer = abuf;

        if (outputDeviceId && "setSinkId" in audioCtx) {
          (audioCtx as unknown as { setSinkId: (id: string) => Promise<void> })
            .setSinkId(outputDeviceId)
            .catch(() => {});
        }

        src.connect(audioCtx.destination);

        const now = audioCtx.currentTime;
        const startAt = Math.max(now, nextPlayTimeRef.current);
        src.start(startAt);
        nextPlayTimeRef.current = startAt + abuf.duration;
      });

      delayIntervalRef.current = setInterval(() => {
        if (!audioCtx || audioCtx.state === "closed") return;
        const behind = Math.max(0, nextPlayTimeRef.current - audioCtx.currentTime);
        setPlaybackDelay(behind);
      }, 250);

      transport.onEvent((evt: TransportEvent) => {
        if (evt.type === "session.stats") {
          setLiveStats(evt.payload as unknown as SessionStats);
        } else if (evt.type === "pipeline.event" && evt.payload.kind === "transcript") {
          const streamName = (evt.payload.stream as string) || "transcript";
          const text = evt.payload.text as string;
          const line: TranscriptLine = { stream: streamName, text, timestamp: Date.now() };
          setTranscripts((prev) => ({
            ...prev,
            [streamName]: [...(prev[streamName] || []), line],
          }));
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

  return { connected, liveStats, transcripts, playbackDelay, stop };
}
