# WebRTC Audio Transport Refactor

## Motivation

The current transport uses WebSocket with a custom binary wire protocol (`0x01` + raw PCM for audio, `0x02` + JSON for events). This works but has drawbacks:

- **No codec compression** — raw s16le PCM at 48 kHz mono = 96 KB/s each direction. WebRTC Opus compresses to ~12 KB/s with better quality.
- **Manual pacing** — the server sleeps to pace audio output chunks because WebSocket has no concept of media timing. WebRTC handles jitter buffering and playout timing natively.
- **No NAT traversal** — WebSocket requires a direct connection to the server. WebRTC's ICE framework handles NAT traversal, making future peer-to-peer or edge deployments possible.
- **No echo cancellation** — the client manually disables echoCancellation/noiseSuppression on getUserMedia. WebRTC's media pipeline provides these for free.
- **Deprecated APIs** — the client uses `createScriptProcessor` (deprecated) to capture PCM chunks. WebRTC replaces this with `RTCPeerConnection.addTrack()`.

## Current Architecture

```
┌──────────────────────────────┐            ┌──────────────────────────────┐
│          Client              │            │          Server              │
│                              │            │                              │
│  getUserMedia ──► ScriptProc │            │  WebSocketTransport          │
│      ↓ (PCM s16le chunks)    │   WS binary│     ↓                       │
│  transport.sendAudio(buf) ──────────────────► recv_audio() → pipeline    │
│                              │   frames   │     ↓                       │
│  onAudio(buf) ◄─────────────────────────────── send_audio() ← pipeline  │
│      ↓                       │            │                              │
│  AudioBufferSource.start()   │            │  send_event(stats/text)      │
│                              │   0x02+JSON│     ↓                       │
│  onEvent(evt) ◄─────────────────────────────── JSON events               │
└──────────────────────────────┘            └──────────────────────────────┘
```

Key files:
- `server/src/transport/base.py` — `TransportConnection` ABC (recv_audio, send_audio, send_event, recv_event, close)
- `server/src/transport/ws.py` — `WebSocketTransport` implementing the ABC
- `server/src/transport/handler.py` — session lifecycle, pipeline wiring, audio pacing, stop event
- `client/src/transport/base.ts` — `StreamTransport` interface
- `client/src/transport/ws.ts` — `WebSocketTransport` implementing the interface
- `client/src/hooks/useAudioStream.ts` — React hook: getUserMedia, PCM conversion, playback, event dispatch

## Target Architecture

```
┌──────────────────────────────┐            ┌──────────────────────────────┐
│          Client              │            │          Server              │
│                              │  ICE/DTLS  │                              │
│  getUserMedia ──► addTrack() │◄──────────►│  RTCPeerConnection (aiortc)  │
│     (browser handles Opus    │  Opus RTP  │     ↓                       │
│      encode automatically)   │────────────►  on("track") → AudioFrame   │
│                              │            │     → PCM s16 → pipeline     │
│  ontrack → remoteStream      │  Opus RTP  │     ↓                       │
│     (browser handles Opus    │◄────────────  addTrack(OutputAudioTrack)  │
│      decode + jitter buffer) │            │     ← pipeline PCM → frame  │
│                              │            │                              │
│  dc.onmessage(evt)           │ DataChannel│  dc.send(JSON)              │
│                              │◄──────────►│                              │
│  dc.send(JSON)               │  (SCTP)   │  dc.onmessage(evt)          │
└──────────────────────────────┘            └──────────────────────────────┘
```

### What Changes

| Concern | Current (WebSocket) | Target (WebRTC) |
|---|---|---|
| **Audio input** | `getUserMedia` → `ScriptProcessor` → manual s16le encode → WS binary | `getUserMedia` → `addTrack(stream)` — browser encodes Opus automatically |
| **Audio output** | WS binary → manual s16le decode → `AudioBufferSource.start()` | `ontrack` → attach to `<audio>` element or `AudioContext` — browser decodes Opus automatically |
| **Audio pacing** | Server `asyncio.sleep()` in `forward_audio()` | Eliminated — RTP + jitter buffer handle timing |
| **Audio codec** | None (raw PCM) | Opus (48 kHz, negotiated via SDP) |
| **Events** | `0x02` + JSON over WS binary frames | WebRTC DataChannel (ordered, reliable) |
| **Signaling** | WS URL is the connection | HTTP POST `/api/sessions/{id}/offer` exchanges SDP; ICE candidates via DataChannel or HTTP |
| **NAT traversal** | None | ICE with STUN (optional TURN) |
| **Server library** | Starlette WebSocket | `aiortc` (already in deps: `aioquic` present) |
| **Client library** | Native `WebSocket` API | Native `RTCPeerConnection` API (no extra deps) |
| **Server audio format** | s16le bytes throughout | `av.AudioFrame` at boundary, s16le bytes internally for pipelines |

### What Stays the Same

- **`TransportConnection` ABC** — same interface, new implementation
- **`StreamTransport` interface** — same interface, new implementation
- **Pipeline layer** — completely unchanged. Pipelines still consume `AsyncIterator[bytes]` of s16le PCM and produce `AsyncIterator[bytes]` of s16le PCM.
- **`handler.py` orchestration** — same task structure (audio_input, forward_audio, forward_text, stats_loop, listen_for_stop), just uses new transport
- **REST API** — sessions still created via `POST /api/sessions`, listed, deleted
- **React hooks structure** — `useAudioStream` still returns `{ connected, liveStats, transcripts, stop }`
- **UI components** — completely unchanged
- **File upload path** — still decoded client-side and fed through the same transport interface

## Detailed Plan

### Phase 1: Server-Side WebRTC Transport

#### 1.1 Add `aiortc` dependency

```toml
# server/pyproject.toml
"aiortc>=1.12.0",
```

`aiortc` depends on `av` (PyAV) which is already transitively available. It also depends on `pyee` for event emitting.

#### 1.2 Create `server/src/transport/rtc.py`

New `WebRTCTransport` class implementing `TransportConnection`:

```python
class WebRTCTransport(TransportConnection):
    def __init__(self, pc: RTCPeerConnection):
        self._pc = pc
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._event_queue: asyncio.Queue[TransportEvent] = asyncio.Queue()
        self._dc: RTCDataChannel | None = None
        self._output_track: OutputAudioTrack | None = None
        self._closed = False
```

**Receiving audio** — When the client's audio track arrives via `pc.on("track")`:
1. Read `av.AudioFrame` objects from the track via `await track.recv()`
2. Resample to session sample rate if needed (Opus always delivers 48 kHz)
3. Convert to s16le bytes via `frame.to_ndarray().tobytes()`
4. Put into `_audio_queue` for `recv_audio()` to yield

**Sending audio** — Custom `MediaStreamTrack` subclass (`OutputAudioTrack`):
1. `send_audio(data: bytes)` puts PCM chunks into an internal queue
2. `recv()` (called by aiortc's media loop) reads from the queue
3. Converts s16le bytes to `av.AudioFrame(format='s16', layout='mono', samples=N)`
4. Sets `pts` and `time_base` for proper RTP timestamping
5. aiortc encodes to Opus and sends via RTP automatically

**Events** — WebRTC DataChannel:
1. Server creates DataChannel on the peer connection: `pc.createDataChannel("events")`
2. `send_event()` serializes to JSON and calls `dc.send(json_str)`
3. `dc.on("message")` parses incoming JSON and puts into `_event_queue`
4. `recv_event()` yields from `_event_queue`

**Close** — `await self._pc.close()` tears down all tracks and data channels.

#### 1.3 Create `OutputAudioTrack` (in `rtc.py`)

```python
class OutputAudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._sample_rate = sample_rate
        self._pts = 0

    async def recv(self) -> AudioFrame:
        data = await self._queue.get()
        if data is None:
            self.stop()
            raise MediaStreamError
        samples = len(data) // 2  # s16le = 2 bytes per sample
        arr = np.frombuffer(data, dtype=np.int16).reshape(1, -1)
        frame = AudioFrame.from_ndarray(arr, format='s16', layout='mono')
        frame.sample_rate = self._sample_rate
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, self._sample_rate)
        self._pts += samples
        return frame

    def push(self, data: bytes) -> None:
        self._queue.put_nowait(data)

    def finish(self) -> None:
        self._queue.put_nowait(None)
```

#### 1.4 Add signaling endpoint

```python
# server/src/api/routes.py

@router.post("/sessions/{session_id}/offer")
async def webrtc_offer(session_id: str, body: dict) -> dict:
    """Exchange SDP offer/answer to establish WebRTC connection."""
```

This endpoint:
1. Looks up the session
2. Creates an `RTCPeerConnection`
3. Creates the `OutputAudioTrack` and adds it to the PC
4. Creates a DataChannel "events"
5. Sets the remote description (client's offer)
6. Creates and returns the answer
7. Spawns `handle_stream_rtc(pc, transport, session_id)` as a background task

The handler function is nearly identical to the current `handle_stream()` but:
- Receives a `WebRTCTransport` instead of `WebSocketTransport`
- No manual audio pacing (remove the `next_play` / `asyncio.sleep` logic from `forward_audio`)
- Waits for the PC connection state to reach "connected" before starting the pipeline

#### 1.5 Refactor `handler.py`

Extract the orchestration logic so it's transport-agnostic:

```python
async def run_session(transport: TransportConnection, session_id: str) -> None:
    """Transport-agnostic session lifecycle."""
    # ... same as current handle_stream minus the WS-specific setup
```

The WebSocket route calls `run_session(ws_transport, session_id)`.
The WebRTC signaling endpoint calls `run_session(rtc_transport, session_id)`.

**Remove audio pacing from `forward_audio()`** when using WebRTC — the RTP layer handles it. Either:
- (a) Check transport type and skip pacing for WebRTC, or
- (b) Move pacing into `WebSocketTransport.send_audio()` so each transport handles its own timing

Option (b) is cleaner — the `TransportConnection` contract becomes "send_audio delivers at the right time" and each implementation decides how.

### Phase 2: Client-Side WebRTC Transport

#### 2.1 Create `client/src/transport/rtc.ts`

New `WebRTCTransport` class implementing `StreamTransport`:

```typescript
export class WebRTCTransport implements StreamTransport {
  private pc: RTCPeerConnection | null = null;
  private dc: RTCDataChannel | null = null;
  private audioCallbacks: ((data: ArrayBuffer) => void)[] = [];
  private eventCallbacks: ((event: TransportEvent) => void)[] = [];
  private closeCallbacks: (() => void)[] = [];
  private localStream: MediaStream | null = null;

  constructor(
    private sessionId: string,
    private inputStream: MediaStream,
  ) {}

  async connect(): Promise<void> {
    this.pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });

    // Add local audio track
    const audioTrack = this.inputStream.getAudioTracks()[0];
    this.pc.addTrack(audioTrack, this.inputStream);

    // Receive remote audio (translated output)
    this.pc.ontrack = (ev) => {
      // Attach to an <audio> element or AudioContext for playback
    };

    // DataChannel for events
    this.dc = this.pc.createDataChannel("events");
    this.dc.onmessage = (ev) => { /* parse JSON, dispatch */ };

    // Create offer
    const offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);

    // Wait for ICE gathering to complete
    await this.waitForIceGathering();

    // Send offer to server, get answer
    const resp = await fetch(`/api/sessions/${this.sessionId}/offer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: this.pc.localDescription!.sdp,
        type: this.pc.localDescription!.type,
      }),
    });
    const answer = await resp.json();
    await this.pc.setRemoteDescription(answer);
  }
}
```

**Audio input** — The browser's `getUserMedia` stream is added directly to the peer connection. No `ScriptProcessor`, no manual PCM encoding.

**Audio output** — The `ontrack` event fires when the server adds its output track. Attach the remote stream to an `<audio>` element for playback:

```typescript
this.pc.ontrack = (ev) => {
  const audio = new Audio();
  audio.srcObject = ev.streams[0];
  audio.play();
};
```

Or for output device selection (`setSinkId`):

```typescript
this.pc.ontrack = (ev) => {
  const audio = document.createElement("audio");
  audio.srcObject = ev.streams[0];
  if (outputDeviceId) {
    (audio as any).setSinkId(outputDeviceId);
  }
  audio.play();
};
```

**Events** — DataChannel replaces the `0x02` + JSON wire protocol. `sendEvent()` calls `dc.send(JSON.stringify(event))`. `dc.onmessage` parses and dispatches.

**`sendAudio()` / `onAudio()`** — These become no-ops or can be removed from the interface, since WebRTC handles the audio track natively. But for interface compatibility (and for file upload, which still needs to push PCM), they could use the DataChannel as a fallback or a `MediaStreamTrack` generated from the file.

#### 2.2 File Upload Path

The current file upload flow decodes the file client-side, resamples to the session rate, and pushes PCM chunks via `transport.sendAudio()`. With WebRTC, this needs adaptation:

**Option A: Client-side MediaStreamTrack from file** (preferred)
- Decode the file to an AudioBuffer (already done)
- Create a `MediaStreamAudioSourceNode` → `MediaStreamAudioDestinationNode` to get a `MediaStream`
- Add that stream's track to the peer connection

**Option B: Keep a WebSocket sidecar for file uploads**
- Only use this if Option A proves unreliable

Option A is cleanest and keeps everything on the WebRTC path.

#### 2.3 Simplify `useAudioStream.ts`

The hook becomes much simpler:

```typescript
export function useAudioStream(options: AudioStreamOptions | null) {
  // State: connected, liveStats, transcripts (same as before)
  // No AudioContext for manual PCM conversion
  // No ScriptProcessor
  // No manual s16le encode/decode

  async function start() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { ... } });
    // OR: generate stream from file
    const transport = new WebRTCTransport(sessionId, stream);
    await transport.connect();

    transport.onEvent((evt) => { /* same as before */ });
    transport.onClose(() => setConnected(false));
  }

  // Return same shape: { connected, liveStats, transcripts, stop }
}
```

Removed:
- `floatToInt16()` / int16-to-float conversion
- `createScriptProcessor()` / `onaudioprocess`
- `AudioBufferSource` for output playback
- `AudioContext` management (for mic capture)
- `setSinkId` workaround (moved to `<audio>` element in transport)

### Phase 3: Remove Server-Side Audio Pacing

With WebRTC, the server no longer needs to pace audio output. The `forward_audio()` function simplifies to:

```python
async def forward_audio(stream: AsyncIterator[bytes]) -> None:
    async for chunk in pipeline.process(stream):
        if stop_event.is_set():
            break
        session.stats.bytes_sent += len(chunk)
        session.stats.chunks_sent += 1
        stats_tracker.total_bytes_processed += len(chunk)
        await transport.send_audio(chunk)
```

The `audio_delay_seconds` metric changes meaning — it's now the amount of audio queued in the `OutputAudioTrack` waiting for aiortc to read it, rather than a sleep-based estimate. Still useful for monitoring pipeline throughput vs. real-time.

### Phase 4: Clean Up and Deprecate WebSocket Transport

Keep the WebSocket transport for now as a fallback (some environments block WebRTC). The `StreamTransport` / `TransportConnection` interfaces make this easy — the client picks which transport to use based on browser support or configuration.

Eventually:
- Remove `server/src/transport/ws.py`
- Remove `client/src/transport/ws.ts`
- Remove the `/ws/stream/{session_id}` WebSocket route
- Remove the `0x01`/`0x02` wire protocol

## File Change Summary

### New Files

| File | Description |
|---|---|
| `server/src/transport/rtc.py` | `WebRTCTransport` + `OutputAudioTrack` |
| `client/src/transport/rtc.ts` | `WebRTCTransport` implementing `StreamTransport` |
| `server/tests/test_rtc_transport.py` | Unit tests for WebRTC transport |
| `client/src/test/rtc-transport.test.ts` | Unit tests for WebRTC transport |

### Modified Files

| File | Changes |
|---|---|
| `server/pyproject.toml` | Add `aiortc>=1.12.0` |
| `server/src/api/routes.py` | Add `POST /sessions/{id}/offer` signaling endpoint |
| `server/src/transport/handler.py` | Extract `run_session()`, make pacing transport-aware |
| `server/src/main.py` | Wire new signaling route (already via router) |
| `client/src/transport/index.ts` | Export `WebRTCTransport` |
| `client/src/hooks/useAudioStream.ts` | Use `WebRTCTransport`, remove manual PCM handling |
| `server/src/models/session.py` | (optional) Add `transport_type` field if we support both |

### Unchanged Files

| File | Reason |
|---|---|
| `server/src/pipelines/*` | Pipeline interface unchanged — still s16le PCM `AsyncIterator[bytes]` |
| `server/src/transport/base.py` | ABC stays the same |
| `client/src/transport/base.ts` | Interface stays the same |
| `client/src/components/*` | UI components unchanged |
| `client/src/api/*` | REST API client unchanged |
| `server/src/api/store.py` | Session storage unchanged |

## Implementation Order

1. **Server `rtc.py`** — implement `WebRTCTransport` + `OutputAudioTrack` + unit tests
2. **Server signaling endpoint** — `POST /sessions/{id}/offer`
3. **Refactor `handler.py`** — extract `run_session()`, make pacing transport-aware
4. **Client `rtc.ts`** — implement `WebRTCTransport` + unit tests
5. **Client `useAudioStream.ts`** — switch to WebRTC transport, simplify
6. **File upload** — adapt to MediaStreamTrack-based approach
7. **E2E tests** — update to verify WebRTC flow
8. **Remove WebSocket transport** (deferred — keep as fallback)

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **aiortc maturity** | Server crashes or audio glitches | Extensive unit tests; keep WS fallback |
| **Opus ↔ pipeline sample rate mismatch** | Opus is 48 kHz; pipelines downsample to 16 kHz internally. But Opus output from server is also 48 kHz, matching the session default. | No change needed — Opus at 48 kHz matches the existing session `sample_rate: 48000`. Pipelines already resample internally. |
| **ICE/STUN failures in restrictive networks** | Connection fails behind strict corporate firewalls | TURN server config (optional); WebSocket fallback |
| **Browser compatibility** | `RTCPeerConnection` not available | Universal in modern browsers; WS fallback for edge cases |
| **Headless testing (Playwright)** | Playwright may not support WebRTC audio tracks in headless mode | Use `--use-fake-device-for-media-stream` and `--use-fake-ui-for-media-stream` Chromium flags; or test signaling only and mock audio |
| **File upload over WebRTC** | Creating a MediaStreamTrack from a decoded AudioBuffer is not straightforward in all browsers | Use `AudioContext` → `MediaStreamDestination` pattern; fallback to DataChannel chunking if needed |
| **Increased server CPU** | Opus encode/decode on server adds CPU load | Opus is very efficient; aiortc uses C bindings. Measure and profile. |

## Audio Format Flow (Before → After)

### Before (WebSocket)
```
Client mic → getUserMedia(48kHz) → ScriptProcessor → float32→int16 → WS binary
  → Server recv_audio() → s16le bytes → pipeline (downsamples to 16kHz internally)
  → pipeline output s16le → asyncio.sleep(pacing) → WS binary
  → Client onAudio → int16→float32 → AudioBufferSource.start()
```

### After (WebRTC)
```
Client mic → getUserMedia(48kHz) → addTrack() → [browser Opus encode] → RTP
  → Server aiortc Opus decode → AudioFrame → .to_ndarray().tobytes() → s16le → pipeline
  → pipeline output s16le → OutputAudioTrack.push() → AudioFrame → [aiortc Opus encode] → RTP
  → Client ontrack → [browser Opus decode + jitter buffer] → <audio>.play()
```

The pipeline boundary is identical: s16le `bytes` in, s16le `bytes` out. All Opus encode/decode happens at the transport edges.
