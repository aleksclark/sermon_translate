# Sermon Translate

Real-time sermon translation platform with a React frontend and FastAPI backend.

## Architecture

```
client/          React 19 + Mantine 8 UI (pnpm, Vite)
server/          FastAPI + uvicorn (uv, Python 3.12)
e2e/             Playwright end-to-end tests (Docker Compose)
```

### Transport

The client connects to the server two ways:

1. **REST API** (`/api/*`) — CRUD for sessions, pipelines, server stats.
2. **WebRTC** (`POST /api/sessions/{session_id}/offer`) — SDP offer/answer exchange establishes a peer connection with Opus-encoded audio tracks and a reliable DataChannel for JSON events.

Audio flows over Opus/RTP tracks (browser ↔ aiortc). Events (stats, transcripts, session lifecycle) flow over a DataChannel. The transport is abstracted behind `StreamTransport` (client) / `TransportConnection` (server) interfaces.

### Pipelines

Translation pipelines implement `BasePipeline` and are registered in the `PipelineRegistry`. Five pipelines ship out of the box:

| Pipeline | Description |
|---|---|
| **Echo** | Returns audio after a 5-second delay (testing) |
| **Whisper TTS** | English speech-to-text via faster-whisper |
| **Spanish Translation** | Whisper ASR → Opus-MT translation → Edge TTS synthesis |
| **Spanish Direct** | SeamlessM4T direct speech-to-speech translation |
| **Seamless Streaming** | Simultaneous English→Spanish via SeamlessStreaming with monotonic attention |

Pipelines consume and produce s16le PCM `AsyncIterator[bytes]`. Shared audio utilities (downsample, TTS synthesis, MP3 decode) live in `pipelines/_audio.py`. Composable stage protocols (`ASRStage`, `TranslationStage`, `TTSStage`) are defined in `pipelines/stages.py` for building new pipeline combinations.

## Quick Start

### Server

```bash
cd server
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### Client

```bash
cd client
pnpm install
pnpm dev
```

Then open http://localhost:5173. The Vite dev server proxies `/api` to the backend on port 8000.

### Both (tmux)

```bash
./dev.sh
```

## Testing

```bash
# Server (108 tests)
cd server && uv run pytest

# Client (21 tests)
cd client && pnpm test

# E2E (10 tests)
cd e2e && bash run.sh
```

## Type Generation

Pydantic models are the single source of truth for the API contract. Run `pnpm typegen` from `client/` after changing any model in `server/src/models/`. Never edit `client/src/api/types.gen.ts` by hand.
