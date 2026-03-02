import type { StreamTransport, TransportEvent } from "./base.ts";

const AUDIO_TAG = 0x01;
const EVENT_TAG = 0x02;

export class WebSocketTransport implements StreamTransport {
  private ws: WebSocket | null = null;
  private audioCallbacks: ((data: ArrayBuffer) => void)[] = [];
  private eventCallbacks: ((event: TransportEvent) => void)[] = [];
  private closeCallbacks: (() => void)[] = [];

  constructor(private url: string) {}

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url);
      this.ws.binaryType = "arraybuffer";

      this.ws.onopen = () => resolve();
      this.ws.onerror = () => reject(new Error("WebSocket connection failed"));

      this.ws.onmessage = (ev: MessageEvent) => {
        if (!(ev.data instanceof ArrayBuffer)) return;
        const view = new Uint8Array(ev.data);
        if (view.length < 1) return;

        const tag = view[0];
        const body = ev.data.slice(1);

        if (tag === AUDIO_TAG) {
          for (const cb of this.audioCallbacks) cb(body);
        } else if (tag === EVENT_TAG) {
          try {
            const text = new TextDecoder().decode(body);
            const parsed = JSON.parse(text) as TransportEvent;
            for (const cb of this.eventCallbacks) cb(parsed);
          } catch {
            // ignore malformed events
          }
        }
      };

      this.ws.onclose = () => {
        for (const cb of this.closeCallbacks) cb();
      };
    });
  }

  sendAudio(data: ArrayBuffer): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    const tagged = new Uint8Array(1 + data.byteLength);
    tagged[0] = AUDIO_TAG;
    tagged.set(new Uint8Array(data), 1);
    this.ws.send(tagged.buffer);
  }

  sendEvent(event: TransportEvent): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    const json = new TextEncoder().encode(JSON.stringify(event));
    const tagged = new Uint8Array(1 + json.byteLength);
    tagged[0] = EVENT_TAG;
    tagged.set(json, 1);
    this.ws.send(tagged.buffer);
  }

  onAudio(cb: (data: ArrayBuffer) => void): void {
    this.audioCallbacks.push(cb);
  }

  onEvent(cb: (event: TransportEvent) => void): void {
    this.eventCallbacks.push(cb);
  }

  onClose(cb: () => void): void {
    this.closeCallbacks.push(cb);
  }

  close(): void {
    this.ws?.close();
    this.ws = null;
  }
}
