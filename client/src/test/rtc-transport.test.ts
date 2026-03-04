import { describe, it, expect, vi, beforeEach } from "vitest";
import { WebRTCTransport } from "../transport/rtc.ts";

class MockRTCDataChannel {
  readyState = "open";
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  sent: string[] = [];

  send(data: string) {
    this.sent.push(data);
  }

  close() {
    this.onclose?.();
  }
}

class MockMediaStreamTrack {
  kind = "audio";
  stop = vi.fn();
}

class MockMediaStream {
  private tracks: MockMediaStreamTrack[];
  constructor(tracks?: MockMediaStreamTrack[]) {
    this.tracks = tracks ?? [new MockMediaStreamTrack()];
  }
  getAudioTracks() {
    return this.tracks;
  }
  getTracks() {
    return this.tracks;
  }
}

class MockRTCPeerConnection {
  iceGatheringState = "complete";
  connectionState = "connected";
  localDescription: { sdp: string; type: string } | null = null;
  ontrack: ((ev: { streams: MediaStream[]; track: MediaStreamTrack }) => void) | null = null;
  onconnectionstatechange: (() => void) | null = null;
  addedTracks: { track: MockMediaStreamTrack; stream: MockMediaStream }[] = [];
  createdChannels: { label: string; channel: MockRTCDataChannel }[] = [];
  closed = false;

  addTrack(track: MockMediaStreamTrack, stream: MockMediaStream) {
    this.addedTracks.push({ track, stream });
  }

  createDataChannel(label: string): MockRTCDataChannel {
    const ch = new MockRTCDataChannel();
    this.createdChannels.push({ label, channel: ch });
    setTimeout(() => ch.onopen?.(), 0);
    return ch;
  }

  async createOffer() {
    return { sdp: "v=0\r\noffer", type: "offer" };
  }

  async setLocalDescription(desc: { sdp: string; type: string }) {
    this.localDescription = desc;
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async setRemoteDescription(_desc: { sdp: string; type: string }) {}

  addEventListener(_event: string, handler: () => void) {
    handler();
  }

  removeEventListener() {}

  close() {
    this.closed = true;
  }
}

describe("WebRTCTransport", () => {
  let instances: MockRTCPeerConnection[];

  beforeEach(() => {
    instances = [];
    vi.stubGlobal(
      "RTCPeerConnection",
      class extends MockRTCPeerConnection {
        constructor() {
          super();
          instances.push(this);
        }
      },
    );
    vi.stubGlobal("RTCSessionDescription", class {
      sdp: string;
      type: string;
      constructor(init: { sdp: string; type: string }) {
        this.sdp = init.sdp;
        this.type = init.type;
      }
    });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ sdp: "v=0\r\nanswer", type: "answer" }),
      }),
    );
  });

  it("connect creates peer connection and exchanges SDP", async () => {
    const stream = new MockMediaStream() as unknown as MediaStream;
    const transport = new WebRTCTransport("sess-1", stream);
    await transport.connect();

    expect(instances).toHaveLength(1);
    expect(instances[0].addedTracks).toHaveLength(1);
    expect(instances[0].createdChannels).toHaveLength(1);
    expect(instances[0].createdChannels[0].label).toBe("events");
    expect(fetch).toHaveBeenCalledWith(
      "/api/sessions/sess-1/offer",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("sendEvent sends JSON through data channel", async () => {
    const stream = new MockMediaStream() as unknown as MediaStream;
    const transport = new WebRTCTransport("sess-1", stream);
    await transport.connect();

    transport.sendEvent({ type: "session.stop", session_id: "sess-1", payload: {} });
    const dc = instances[0].createdChannels[0].channel;
    expect(dc.sent).toHaveLength(1);
    const parsed = JSON.parse(dc.sent[0]);
    expect(parsed.type).toBe("session.stop");
  });

  it("onEvent dispatches DataChannel messages", async () => {
    const stream = new MockMediaStream() as unknown as MediaStream;
    const transport = new WebRTCTransport("sess-1", stream);
    await transport.connect();

    const events: unknown[] = [];
    transport.onEvent((evt) => events.push(evt));

    const dc = instances[0].createdChannels[0].channel;
    dc.onmessage?.({
      data: JSON.stringify({ type: "session.stats", session_id: "sess-1", payload: { foo: 1 } }),
    });

    expect(events).toHaveLength(1);
    expect((events[0] as { type: string }).type).toBe("session.stats");
  });

  it("sendAudio is a no-op", async () => {
    const stream = new MockMediaStream() as unknown as MediaStream;
    const transport = new WebRTCTransport("sess-1", stream);
    await transport.connect();
    transport.sendAudio(new ArrayBuffer(100));
  });

  it("close tears down everything", async () => {
    const stream = new MockMediaStream() as unknown as MediaStream;
    const transport = new WebRTCTransport("sess-1", stream);
    await transport.connect();
    transport.close();
    expect(instances[0].closed).toBe(true);
  });

  it("connect throws on signaling failure", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({ ok: false, status: 500, json: async () => ({}) }),
    );

    const stream = new MockMediaStream() as unknown as MediaStream;
    const transport = new WebRTCTransport("sess-1", stream);
    await expect(transport.connect()).rejects.toThrow("Signaling failed");
  });
});
