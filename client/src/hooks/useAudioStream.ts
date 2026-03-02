import { useCallback, useEffect, useRef, useState } from "react";
import type { SessionStats } from "../api/index.ts";
import type { StreamTransport, TransportEvent } from "../transport/index.ts";
import { WebSocketTransport } from "../transport/index.ts";

interface AudioStreamOptions {
  sessionId: string;
  sampleRate: number;
  channels: number;
  inputDeviceId: string;
  outputDeviceId: string;
}

export function useAudioStream(options: AudioStreamOptions | null) {
  const [connected, setConnected] = useState(false);
  const [liveStats, setLiveStats] = useState<SessionStats | null>(null);
  const transportRef = useRef<StreamTransport | null>(null);
  const contextRef = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const stop = useCallback(() => {
    transportRef.current?.close();
    transportRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    contextRef.current?.close();
    contextRef.current = null;
    setConnected(false);
    setLiveStats(null);
  }, []);

  useEffect(() => {
    if (!options) return;
    let cancelled = false;

    async function start() {
      const { sessionId, sampleRate, inputDeviceId, outputDeviceId } =
        options!;

      const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${wsProto}//${location.host}/ws/stream/${sessionId}`;
      const transport = new WebSocketTransport(wsUrl);

      try {
        await transport.connect();
      } catch {
        return;
      }
      if (cancelled) {
        transport.close();
        return;
      }

      transportRef.current = transport;
      setConnected(true);

      const audioCtx = new AudioContext({ sampleRate });
      contextRef.current = audioCtx;

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
      if (cancelled) {
        stream.getTracks().forEach((t) => t.stop());
        transport.close();
        return;
      }
      streamRef.current = stream;

      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processor.onaudioprocess = (e: AudioProcessingEvent) => {
        const pcm = e.inputBuffer.getChannelData(0);
        const i16 = new Int16Array(pcm.length);
        for (let i = 0; i < pcm.length; i++) {
          const s = Math.max(-1, Math.min(1, pcm[i]));
          i16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }
        transport.sendAudio(i16.buffer);
      };
      source.connect(processor);
      processor.connect(audioCtx.destination);

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
        src.start();
      });

      transport.onEvent((evt: TransportEvent) => {
        if (evt.type === "session.stats") {
          setLiveStats(evt.payload as unknown as SessionStats);
        }
      });

      transport.onClose(() => {
        setConnected(false);
      });
    }

    start();
    return () => {
      cancelled = true;
      stop();
    };
  }, [options?.sessionId]);

  return { connected, liveStats, stop };
}
