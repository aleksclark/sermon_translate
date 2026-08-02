# Phase 5 — Out-of-Process Stage Workers

## Objective

Allow each stage to run in an **independent worker process** (local subprocess
first, network client second) behind the `StageRuntime` interface from Phase 4,
so GPU isolation and Nomad placement become possible without changing
`ComposedPipeline` logic.

## Motivation

Heavy stages (Kyutai STT, Qwen3-TTS, translation LLM) need separate address
spaces, restart domains, and eventually separate Nomad allocations. The
orchestrator should only stream messages.

## Design

### 1. Worker protocol

Length-prefixed JSON control + binary media frames over stdio **or** TCP/WebSocket.
Prefer **WebSocket** for parity with future Nomad service networking and with
vLLM-Omni style endpoints.

#### Control messages (JSON text frames)

```json
{"type": "hello", "stage_id": "…", "session_id": "…", "config": {}}
{"type": "start"}
{"type": "stop"}
{"type": "error", "message": "…"}
{"type": "ready"}
```

#### Data messages

 dual frames: small JSON header + binary payload when needed.

```json
{"type": "audio_in", "seq": 1, "sample_rate": 48000, "channels": 1, "pcm_bytes": 1920}
// followed by binary PCM

{"type": "audio_out", "seq": 1, ...}
{"type": "listen_product", "product": { ...ListenProduct }}
{"type": "translate_product", "product": { ... }}
{"type": "metadata", "envelope": { ...MetadataEnvelope }}
{"type": "eos"}
```

Keep schema in Pydantic (`server/src/runtime/protocol.py`) and share with worker
entrypoints.

### 2. Worker entrypoint

```
python -m src.runtime.worker --stage-id passthrough-listen --port 0
```

- Loads settings + `ModelCache`
- Builds stage from registry
- Serves one session at a time (v1); multi-session later if needed
- Health: HTTP `/healthz` on the same port or adjacent port

### 3. Runtimes

| Runtime | `STAGE_RUNTIME` | Behavior |
|---------|-----------------|----------|
| Local | `local` | In-process (Phase 4) |
| Subprocess | `subprocess` | Spawn worker module; connect WS to localhost port |
| Remote | `remote` | Connect to URL from config/service discovery |

```python
class SubprocessStageRuntime:
    async def spawn(self, stage_id, session) -> StageHandle:
        # start process, wait ready, return handle wrapping WS streams
```

Handle maps:
- Listen: audio_in → listen_product (+ optional metadata)
- Translate: listen_product in → translate_product out
- Speak: translate_product in → audio_out
- Prosody: audio_in → metadata

### 4. Config

| Env | Meaning |
|-----|---------|
| `STAGE_RUNTIME` | `local` \| `subprocess` \| `remote` |
| `STAGE_WORKER_PYTHON` | interpreter for subprocess (default: sys.executable) |
| `STAGE_REMOTE_URLS` | JSON map `stage_id → ws://host:port` for remote mode |
| `STAGE_WORKER_START_TIMEOUT` | seconds (default 60) |

### 5. Failure semantics

- Worker crash → surface `error` pipeline event to admin UI; end session cleanly.
- Backpressure: bounded queues; drop policy **must not** drop audio silently —
  block with timeout then error.
- Always `stop()` workers on session end; subprocess runtime kills process group.

### 6. Testing strategy

- Use **passthrough** stages over subprocess runtime in pytest (skip if
  environment forbids process spawn).
- Protocol unit tests with mocked WS.
- Do not pull GPU models in CI.

## Files Likely Touched

```
server/src/runtime/protocol.py
server/src/runtime/worker.py
server/src/runtime/subprocess_runtime.py
server/src/runtime/remote_runtime.py
server/src/runtime/ws_handle.py
server/src/config.py
server/src/pipelines/composed.py      # unchanged if handle API is solid
server/src/api/deps.py
server/tests/test_worker_protocol.py
server/tests/test_subprocess_runtime.py
pyproject.toml                        # scripts entry if needed
```

## Tests

1. Protocol encode/decode round-trip.
2. Subprocess runtime: spawn passthrough-listen, push audio, receive product, stop.
3. Worker crash produces error path.
4. Local runtime still default and green.

## Exit Criteria

- [ ] `STAGE_RUNTIME=subprocess` runs composed passthrough graph
- [ ] Worker loads cache env from `MODEL_CACHE_DIR`
- [ ] Clean shutdown; no zombie processes in tests
- [ ] Remote runtime can point at a manually started worker
- [ ] Docs: how to run a worker locally

## Out of Scope

- Nomad jobspecs and service discovery automation (Phase 6)
- GPU model packaging (Phase 7)
- Autoscaling / pool of warm workers (future)
