# Phase 1 — Stage Contracts and Composition

## Objective

Make listen / translate / speak (plus prosody) **first-class selectable stages**
and introduce a `ComposedPipeline` that wires them into a single
`BasePipeline`, so sessions can choose models per stage instead of one
monolithic pipeline id.

## Motivation

Today `PipelineRegistry` only knows whole pipelines. Stage protocols exist in
`pipelines/stages.py` but are not registered, discoverable, or selectable via
the session API. Composition happens ad-hoc inside each pipeline class.

## Design

### 1. Stage kinds and info models

Add Pydantic models (codegen to TS):

```python
class StageKind(StrEnum):
    LISTEN = "listen"       # ASR
    TRANSLATE = "translate"
    SPEAK = "speak"         # TTS
    PROSODY = "prosody"     # parallel analyzer

class StageInfo(BaseModel):
    id: str                 # e.g. "baseline-prosody", "echo-listen"
    kind: StageKind
    name: str
    description: str
    # optional resource hints for later phases
    requires_gpu: bool = False
    default_for_kind: bool = False
```

Extend `PipelineInfo` (or add a sibling) so the composed pipeline advertises
that it is stage-driven:

```python
class StageSelection(BaseModel):
    listen: str
    translate: str
    speak: str
    prosody: str | None = None  # None = disabled; default = baseline
```

### 2. Session create API

Extend `SessionCreate` / `Session`:

```python
class SessionCreate(BaseModel):
    pipeline_id: str                    # keep; use "composed" for stage graph
    stages: StageSelection | None = None
    # ... existing fields ...
```

Rules:
- If `pipeline_id` is a legacy monolithic pipeline → ignore `stages` (or 400 if set).
- If `pipeline_id == "composed"` → `stages` required; each id must exist in the
  stage registry and match the expected kind.
- Persist resolved `stages` on `Session` so the handler can rebuild the graph.

API additions:
- `GET /api/stages` → `list[StageInfo]`
- `GET /api/stages?kind=listen` optional filter

### 3. Stage registry

```python
class StageRegistry:
    def register(self, stage_factory: StageFactory) -> None: ...
    def get(self, stage_id: str) -> StageFactory | None: ...
    def list_all(self, kind: StageKind | None = None) -> list[StageInfo]: ...
```

`StageFactory` builds a fresh stage instance (stages are not necessarily
singleton; model-heavy stages may still share weights via runtime in phase 4).

Register **stub / passthrough** stages in this phase so composition is testable
without heavy models:

| id | kind | behavior |
|----|------|----------|
| `passthrough-listen` | listen | yields placeholder transcript chunks from silence/energy gates or fixed markers |
| `passthrough-translate` | translate | identity or trivial EN→ES stub |
| `passthrough-speak` | speak | silence or tone for each text chunk (reuse `_audio` helpers if useful) |
| `baseline-prosody` | prosody | wrap existing `BaselineProsodyStage` |

Keep real model integrations for Phase 7. Existing monolithic pipelines stay
registered and working.

### 4. Widen stage protocols (minimal)

Current protocols only stream bare `str` / `bytes`. For composition we need a
session-aware lifecycle but **do not** yet introduce full message envelopes
(that is Phase 2). Minimal change:

```python
class ASRStage(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def transcribe(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]: ...

# same shapes as today for translate/speak/prosody
```

Optional: add `info: StageInfo` property on concrete stages for self-description.

### 5. `ComposedPipeline(BasePipeline)`

```python
class ComposedPipeline(BasePipeline):
    def __init__(self, stage_registry: StageRegistry): ...

    # process():
    #   resolve Session.stages → construct listen, translate, speak, prosody
    #   start all; tee audio to listen + prosody
    #   listen → translate → speak → yield audio
    #   publish text streams: "listen", "translate"
    #   publish metadata stream: "prosody"
    #   stop all on exit
```

Output streams (declared dynamically or fixed for composed):

| name | kind | label |
|------|------|-------|
| `audio` | audio | Translated speech |
| `listen` | text | Source transcript |
| `translate` | text | Target transcript |
| `prosody` | metadata | Prosody |

Implementation notes:
- Use asyncio queues between stages (bounded, capacity 8 to match existing
  backpressure constants).
- `process()` owns the speak→audio path; `iter_stream` drains listen/translate
  text queues; `iter_metadata_stream` drains prosody.
- Per-session stage instances: do **not** share mutable stage state across
  sessions in this phase.
- `discard_session_outputs` must clean stage queues.

### 6. Handler / deps wiring

- `create_default_registry()` also builds a `StageRegistry` and registers
  `ComposedPipeline` with id `composed`.
- Expose stage registry via `deps` (same pattern as pipeline registry).
- `create_session` validates stage selection when `pipeline_id == "composed"`.

### 7. Client (minimal for this phase)

- Regenerate types (`pnpm typegen`).
- Session create types accept optional `stages`.
- New session modal: when pipeline is `composed`, show four selects populated
  from `GET /api/stages`. (Full debug UI is Phase 3; this phase only needs
  selection + create.)

If modal work threatens scope, server+tests alone are acceptable **only if**
types are generated and a thin API client method `fetchStages()` exists; UI
pickers can complete in Phase 3. Prefer finishing pickers here if small.

## Files Likely Touched

```
server/src/models/session.py          # StageKind, StageInfo, StageSelection, Session fields
server/src/models/__init__.py
server/src/codegen.py                 # ensure new models exported
server/src/pipelines/stages.py        # info property if needed
server/src/pipelines/stage_registry.py  # NEW
server/src/pipelines/composed.py        # NEW
server/src/pipelines/stub_stages.py     # NEW passthrough stages
server/src/pipelines/registry.py
server/src/pipelines/__init__.py
server/src/api/deps.py
server/src/api/routes.py              # GET /api/stages, create validation
server/tests/test_stage_registry.py   # NEW
server/tests/test_composed_pipeline.py # NEW
server/tests/test_api.py              # stages endpoint + session create
client/src/api/types.gen.ts           # via typegen
client/src/api/client.ts              # fetchStages
client/src/components/NewSessionModal.tsx  # optional stage pickers
```

## Tests

1. Stage registry list/filter/get.
2. Composed pipeline with passthrough stages: audio in → audio out; text
   streams emit; prosody metadata emits.
3. Session create rejects unknown stage ids / wrong kinds.
4. Session create with legacy `pipeline_id` still works without `stages`.
5. Existing pipeline tests green.

## Exit Criteria

- [ ] `GET /api/stages` returns registered stages
- [ ] `POST /api/sessions` with `pipeline_id=composed` + valid `stages` works
- [ ] `ComposedPipeline` runs passthrough graph end-to-end in unit tests
- [ ] Prosody still flows as metadata on composed sessions
- [ ] Legacy pipelines unchanged and tested
- [ ] Types regenerated; server lint/typecheck/tests pass

## Out of Scope

- Real model backends (Phase 7)
- Word-aligned prosody tokens / instruction markup (Phase 2)
- Admin debug panels beyond basic selection (Phase 3)
- Out-of-process workers (Phases 5–6)
- Model cache (Phase 4)
