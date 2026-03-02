# Sermon Translate

Real-time sermon translation platform with a React frontend and FastAPI backend.

## Architecture

```
client/          React + Mantine UI (pnpm, Vite)
server/          FastAPI + uvicorn (uv)
```

### Transport

The client connects to the server two ways:

1. **REST API** (`/api/*`) — CRUD for sessions, pipelines, server stats
2. **WebSocket** (`/ws/stream/{session_id}`) — bidirectional audio streaming + events

The WebSocket transport uses a tagged binary protocol and is abstracted behind
`StreamTransport` (client) / `TransportConnection` (server) interfaces so it can
be swapped for WebTransport or any other bidirectional channel.

### Pipelines

Translation pipelines implement `BasePipeline` and are registered in the
`PipelineRegistry`. The included **Echo** pipeline returns audio after a 5-second
delay for testing.

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

Then open http://localhost:5173. The Vite dev server proxies `/api` and `/ws` to
the backend on port 8000.
