import { useCallback, useEffect, useRef, useState } from "react";
import type { SessionStats } from "../api/index.ts";
import type { AudioSource } from "../components/NewSessionModal.tsx";
import type { TransportEvent, WebRTCMetrics } from "../transport/index.ts";
import { WebRTCTransport } from "../transport/index.ts";

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

export interface PipelineStatus {
  phase: "connecting" | "loading" | "ready" | "streaming" | "error" | "stopped";
  detail: string;
}

interface SampleMediaStreamResult {
  stream: MediaStream;
  durationMs: number;
  audioBuffer: AudioBuffer;
  audioContext: AudioContext;
  sourceNode: AudioBufferSourceNode;
}

async function createSampleMediaStream(
  url: string,
  sampleRate: number,
): Promise<SampleMediaStreamResult> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch sample: ${resp.status}`);
  const arrayBuffer = await resp.arrayBuffer();

  const audioCtx = new AudioContext({ sampleRate });
  await audioCtx.resume();
  const decoded = await audioCtx.decodeAudioData(arrayBuffer);

  const source = audioCtx.createBufferSource();
  source.buffer = decoded;
  const dest = audioCtx.createMediaStreamDestination();
  source.connect(dest);
  source.start();

  source.onended = () => {
    dest.stream.getTracks().forEach((t) => t.stop());
  };

  return {
    stream: dest.stream,
    durationMs: decoded.duration * 1000,
    audioBuffer: decoded,
    audioContext: audioCtx,
    sourceNode: source,
  };
}

export interface AudioNodes {
  sourceAnalyser: AnalyserNode | null;
  outputAnalyser: AnalyserNode | null;
  sourceGain: GainNode | null;
  outputGain: GainNode | null;
}

export function useAudioStream(options: AudioStreamOptions | null) {
  const [connected, setConnected] = useState(false);
  const [muted, setMuted] = useState(false);
  const [liveStats, setLiveStats] = useState<SessionStats | null>(null);
  const [transcripts, setTranscripts] = useState<Record<string, TranscriptLine[]>>({});
  const [audioNodes, setAudioNodes] = useState<AudioNodes>({
    sourceAnalyser: null,
    outputAnalyser: null,
    sourceGain: null,
    outputGain: null,
  });
  const [rtcMetrics, setRtcMetrics] = useState<WebRTCMetrics | null>(null);
  const [pipelineStatus, setPipelineStatus] = useState<PipelineStatus>({
    phase: "stopped",
    detail: "",
  });
  const transportRef = useRef<WebRTCTransport | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const sourceCtxRef = useRef<AudioContext | null>(null);
  const outputCtxRef = useRef<AudioContext | null>(null);
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
    sourceCtxRef.current?.close().catch(() => {});
    sourceCtxRef.current = null;
    outputCtxRef.current?.close().catch(() => {});
    outputCtxRef.current = null;
    setConnected(false);
    setMuted(false);
    setLiveStats(null);
    setTranscripts({});
    setRtcMetrics(null);
    setPipelineStatus({ phase: "stopped", detail: "" });
    setAudioNodes({
      sourceAnalyser: null,
      outputAnalyser: null,
      sourceGain: null,
      outputGain: null,
    });
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

      setPipelineStatus({ phase: "connecting", detail: "Acquiring audio…" });

      let inputStream: MediaStream;
      let fileDurationMs: number | null = null;
      let sourceAnalyser: AnalyserNode | null = null;
      let sourceGain: GainNode | null = null;
      let srcCtx: AudioContext | null = null;

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
      } else if (audioSource.type === "sample" && audioSource.sampleUrl) {
        const result = await createSampleMediaStream(audioSource.sampleUrl, sampleRate);
        inputStream = result.stream;
        fileDurationMs = result.durationMs;
        srcCtx = result.audioContext;

        sourceAnalyser = srcCtx.createAnalyser();
        sourceAnalyser.fftSize = 256;
        sourceGain = srcCtx.createGain();

        result.sourceNode.connect(sourceAnalyser);
        result.sourceNode.connect(sourceGain);
        sourceGain.connect(srcCtx.destination);
      } else {
        return;
      }

      if (cancelledRef.current) {
        inputStream.getTracks().forEach((t) => t.stop());
        srcCtx?.close().catch(() => {});
        return;
      }
      streamRef.current = inputStream;
      sourceCtxRef.current = srcCtx;

      setPipelineStatus({ phase: "connecting", detail: "WebRTC signaling…" });

      const transport = new WebRTCTransport(sessionId, inputStream, outputDeviceId);
      try {
        await transport.connect();
      } catch (err) {
        inputStream.getTracks().forEach((t) => t.stop());
        srcCtx?.close().catch(() => {});
        setPipelineStatus({
          phase: "error",
          detail: `Connection failed: ${err instanceof Error ? err.message : String(err)}`,
        });
        return;
      }
      if (cancelledRef.current) {
        transport.close();
        srcCtx?.close().catch(() => {});
        return;
      }

      transportRef.current = transport;

      setPipelineStatus({ phase: "connecting", detail: "DataChannel open, setting up audio…" });

      const outCtx = new AudioContext();
      await outCtx.resume();
      outputCtxRef.current = outCtx;

      const outputAnalyser = outCtx.createAnalyser();
      outputAnalyser.fftSize = 256;
      const outputGain = outCtx.createGain();

      transport.setupAudioOutput(outCtx, outputAnalyser, outputGain);

      setAudioNodes({ sourceAnalyser, outputAnalyser, sourceGain, outputGain });
      setConnected(true);

      if (audioSource.type === "sample" && fileDurationMs != null) {
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
        } else if (evt.type === "pipeline.status") {
          const phase = evt.payload.phase as PipelineStatus["phase"];
          const detail = (evt.payload.detail as string) || "";
          setPipelineStatus({ phase, detail });
          if (phase === "ready") {
            setPipelineStatus({ phase: "streaming", detail });
          }
        } else if (evt.type === "error") {
          setPipelineStatus({
            phase: "error",
            detail: (evt.payload.detail as string) || "Unknown error",
          });
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
        setPipelineStatus((prev) =>
          prev.phase === "error" ? prev : { phase: "stopped", detail: "Connection closed" },
        );
      });

      transport.onMetrics((m) => setRtcMetrics(m));
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
    liveStats,
    transcripts,
    audioNodes,
    rtcMetrics,
    pipelineStatus,
    stop,
    toggleMute,
  };
}
