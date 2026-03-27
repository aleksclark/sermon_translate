import type { StreamTransport, TransportEvent } from "./base.ts";

const ICE_GATHER_TIMEOUT_MS = 2_000;
const STATS_INTERVAL_MS = 2_000;

export interface WebRTCMetrics {
  connectionState: string;
  iceState: string;
  roundTripMs: number | null;
  jitterMs: number | null;
  packetsLost: number;
  bytesReceived: number;
  bytesSent: number;
}

export class WebRTCTransport implements StreamTransport {
  private pc: RTCPeerConnection | null = null;
  private dc: RTCDataChannel | null = null;
  private audioEl: HTMLAudioElement | null = null;
  private eventCallbacks: ((event: TransportEvent) => void)[] = [];
  private closeCallbacks: (() => void)[] = [];
  private outputSourceNode: MediaStreamAudioSourceNode | null = null;
  private metricsCallbacks: ((m: WebRTCMetrics) => void)[] = [];
  private metricsTimer: ReturnType<typeof setInterval> | null = null;

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
      audio.muted = true;
      audio.srcObject = ev.streams[0] ?? new MediaStream([ev.track]);
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
      this.pollMetrics();
      if (this.pc?.connectionState === "failed" || this.pc?.connectionState === "disconnected") {
        for (const cb of this.closeCallbacks) cb();
      }
    };

    this.pc.oniceconnectionstatechange = () => this.pollMetrics();

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

    this.metricsTimer = setInterval(() => this.pollMetrics(), STATS_INTERVAL_MS);
  }

  setupAudioOutput(
    ctx: AudioContext,
    analyser: AnalyserNode,
    gain: GainNode,
  ): void {
    if (!this.audioEl) return;

    const source = ctx.createMediaStreamSource(
      this.audioEl.srcObject as MediaStream,
    );
    this.outputSourceNode = source;

    source.connect(analyser);
    source.connect(gain);
    gain.connect(ctx.destination);
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
    // no-op: WebRTC playback via audio output chain
  }

  onEvent(cb: (event: TransportEvent) => void): void {
    this.eventCallbacks.push(cb);
  }

  onClose(cb: () => void): void {
    this.closeCallbacks.push(cb);
  }

  onMetrics(cb: (m: WebRTCMetrics) => void): void {
    this.metricsCallbacks.push(cb);
  }

  setMuted(muted: boolean): void {
    if (this.audioEl) {
      this.audioEl.muted = muted;
    }
  }

  close(): void {
    if (this.metricsTimer) {
      clearInterval(this.metricsTimer);
      this.metricsTimer = null;
    }
    this.dc?.close();
    this.dc = null;
    this.outputSourceNode?.disconnect();
    this.outputSourceNode = null;
    if (this.audioEl) {
      this.audioEl.srcObject = null;
      this.audioEl = null;
    }
    this.pc?.close();
    this.pc = null;
  }

  private pollMetrics(): void {
    if (!this.pc) return;
    const m: WebRTCMetrics = {
      connectionState: this.pc.connectionState,
      iceState: this.pc.iceConnectionState,
      roundTripMs: null,
      jitterMs: null,
      packetsLost: 0,
      bytesReceived: 0,
      bytesSent: 0,
    };
    this.pc.getStats().then((stats) => {
      stats.forEach((report) => {
        if (report.type === "candidate-pair" && report.state === "succeeded") {
          m.roundTripMs = report.currentRoundTripTime != null
            ? Math.round(report.currentRoundTripTime * 1000)
            : null;
        }
        if (report.type === "inbound-rtp" && report.kind === "audio") {
          m.jitterMs = report.jitter != null ? Math.round(report.jitter * 1000) : null;
          m.packetsLost = report.packetsLost ?? 0;
          m.bytesReceived = report.bytesReceived ?? 0;
        }
        if (report.type === "outbound-rtp" && report.kind === "audio") {
          m.bytesSent = report.bytesSent ?? 0;
        }
      });
      for (const cb of this.metricsCallbacks) cb(m);
    }).catch(() => {});
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
