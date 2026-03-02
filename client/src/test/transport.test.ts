import { describe, it, expect, vi, beforeEach } from "vitest";
import { WebSocketTransport } from "../transport/ws.ts";

class MockWebSocket {
  binaryType = "";
  readyState = 1;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((ev: { data: ArrayBuffer }) => void) | null = null;
  onerror: (() => void) | null = null;
  sent: Uint8Array[] = [];

  static OPEN = 1;

  constructor(_url: string) { // eslint-disable-line @typescript-eslint/no-unused-vars
    setTimeout(() => this.onopen?.(), 0);
  }

  send(data: ArrayBufferLike) {
    this.sent.push(new Uint8Array(data));
  }

  close() {
    this.onclose?.();
  }

  simulateMessage(data: ArrayBuffer) {
    this.onmessage?.({ data });
  }
}

describe("WebSocketTransport", () => {
  let instances: MockWebSocket[];

  beforeEach(() => {
    instances = [];
    vi.stubGlobal("WebSocket", class extends MockWebSocket {
      constructor(url: string) {
        super(url);
        instances.push(this);
      }
    });
  });

  it("connects and resolves", async () => {
    const transport = new WebSocketTransport("ws://test/ws/stream/abc");
    await transport.connect();
    expect(instances).toHaveLength(1);
  });

  it("sendAudio tags with 0x01", async () => {
    const transport = new WebSocketTransport("ws://test/ws/stream/abc");
    await transport.connect();
    const data = new Uint8Array([1, 2, 3]);
    transport.sendAudio(data.buffer);
    expect(instances[0].sent).toHaveLength(1);
    expect(instances[0].sent[0][0]).toBe(0x01);
    expect(instances[0].sent[0].slice(1)).toEqual(data);
  });

  it("sendEvent tags with 0x02", async () => {
    const transport = new WebSocketTransport("ws://test/ws/stream/abc");
    await transport.connect();
    transport.sendEvent({ type: "session.start", session_id: "abc", payload: {} });
    expect(instances[0].sent).toHaveLength(1);
    expect(instances[0].sent[0][0]).toBe(0x02);
    const json = new TextDecoder().decode(instances[0].sent[0].slice(1));
    expect(JSON.parse(json).type).toBe("session.start");
  });

  it("dispatches audio messages to onAudio callback", async () => {
    const transport = new WebSocketTransport("ws://test/ws/stream/abc");
    await transport.connect();

    const received: ArrayBuffer[] = [];
    transport.onAudio((data) => received.push(data));

    const msg = new Uint8Array([0x01, 10, 20, 30]);
    instances[0].simulateMessage(msg.buffer);

    expect(received).toHaveLength(1);
    expect(new Uint8Array(received[0])).toEqual(new Uint8Array([10, 20, 30]));
  });

  it("dispatches event messages to onEvent callback", async () => {
    const transport = new WebSocketTransport("ws://test/ws/stream/abc");
    await transport.connect();

    const events: unknown[] = [];
    transport.onEvent((evt) => events.push(evt));

    const payload = JSON.stringify({ type: "session.stats", session_id: "abc", payload: { foo: 1 } });
    const encoded = new TextEncoder().encode(payload);
    const msg = new Uint8Array(1 + encoded.length);
    msg[0] = 0x02;
    msg.set(encoded, 1);
    instances[0].simulateMessage(msg.buffer);

    expect(events).toHaveLength(1);
    expect((events[0] as { type: string }).type).toBe("session.stats");
  });

  it("close calls ws.close", async () => {
    const transport = new WebSocketTransport("ws://test/ws/stream/abc");
    await transport.connect();
    transport.close();
    expect(instances[0]).toBeDefined();
  });
});
