# WebRTC Audio Transport — Implementation Notes

This document records the design rationale and implementation of the WebRTC transport that replaced the original WebSocket-based audio streaming.

## Motivation

The original transport used WebSocket with a custom binary wire protocol (`0x01` + raw PCM for audio, `0x02` + JSON for events). Drawbacks:

- **No codec compression** — raw s16le PCM at 48 kHz mono = 96 KB/s each direction. WebRTC Opus compresses to ~12 KB/s with better quality.
- **Manual pacing** — the server had to sleep to pace audio output. WebRTC handles jitter buffering and playout timing natively.
- **No NAT traversal** — WebSocket requires a direct connection. WebRTC's ICE framework handles NAT traversal.
- **Deprecated APIs** — the client used `createScriptProcessor` (deprecated) to capture PCM chunks.

## Architecture

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

## Key Files

### Server

| File | Role |
|---|---|
| `server/src/transport/base.py` | `TransportConnection` ABC (`recv_audio`, `send_audio`, `send_event`, `recv_event`, `close`) |
| `server/src/transport/rtc.py` | `WebRTCTransport` + `OutputAudioTrack` implementing the ABC |
| `server/src/transport/handler.py` | Transport-agnostic session lifecycle (`run_session`) |
| `server/src/api/routes.py` | `POST /sessions/{id}/offer` SDP signaling endpoint |

### Client

| File | Role |
|---|---|
| `client/src/transport/base.ts` | `StreamTransport` interface |
| `client/src/transport/rtc.ts` | `WebRTCTransport` implementing the interface |
| `client/src/hooks/useAudioStream.ts` | React hook: mic/file source → WebRTC → stats/transcripts |

## Audio Format Flow

```
Client mic → getUserMedia(48kHz) → addTrack() → [browser Opus encode] → RTP
  → Server aiortc Opus decode → AudioFrame → .to_ndarray().tobytes() → s16le → pipeline
  → pipeline output s16le → OutputAudioTrack.push() → AudioFrame → [aiortc Opus encode] → RTP
  → Client ontrack → [browser Opus decode + jitter buffer] → <audio>.play()
```

The pipeline boundary is s16le `bytes` in, s16le `bytes` out. All Opus encode/decode happens at the transport edges.

## Design Decisions

### Server-side audio pacing

`OutputAudioTrack` generates silence frames on a 20 ms wall-clock timer so the RTP stream never goes quiet. When real PCM data is available it replaces silence. This eliminated the manual `asyncio.sleep()` pacing the WebSocket transport required.

### DataChannel for events

Events (session lifecycle, stats, transcripts) flow over a reliable ordered DataChannel rather than multiplexed binary WebSocket frames. The server creates the DataChannel during signaling; the client listens via `pc.ondatachannel`.

### Stereo downmix

aiortc may decode Opus to stereo. `WebRTCTransport._read_track` downmixes to mono so pipelines always receive single-channel audio.

### File upload

File uploads are decoded client-side into a `MediaStream` via `AudioContext` → `MediaStreamDestination`, then sent over the same WebRTC audio track as live mic input. The client sends an `audio.end` event on the DataChannel when playback completes.

## What Was Removed

- `server/src/transport/ws.py` — WebSocket transport implementation
- `client/src/transport/ws.ts` — WebSocket transport implementation
- `/ws/stream/{session_id}` WebSocket route
- `0x01`/`0x02` tagged binary wire protocol
- `ScriptProcessor` PCM capture on the client
- Manual `AudioBufferSource` playback on the client
- Server-side `asyncio.sleep()` audio pacing
