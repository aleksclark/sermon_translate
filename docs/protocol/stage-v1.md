# stage.v1 protocol (normative)

> Frozen contract copy of the Wave 0 normative specification.
> Source plan: `stage-v1-critical-path.md` §2.
> Semantic changes are forbidden without a new major/minor protocol revision.

## 2. Normative `stage.v1` protocol

The words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative. JSON examples are illustrative; the field tables are authoritative.

### 2.1 Connection and version negotiation

- Endpoint: `GET /stage/v1/stream` upgraded to WebSocket.
- Required WebSocket subprotocol: `stage.v1`.
- Production transport: `wss://`; bearer/workload credentials are sent in the HTTP upgrade headers, never in a protocol event or URL. A worker MUST authenticate/authorize before model/session allocation.
- The orchestrator sends `hello` first. The worker replies `accepted` or closes with an `error`.
- `schema_version` is the literal `stage.v1`. An unknown major version MUST fail closed with `VERSION_UNSUPPORTED` before allocating a stage session. Additive fields under the same major MUST be ignored by readers that do not need them.
- The worker advertises supported stage kind, capabilities, audio formats, limits, boot ID, and immutable provenance in `accepted`.
- One connection carries one stage kind and one active stage attempt in v1. A worker process may accept multiple connections up to declared capacity.

Example handshake:

```json
{
  "schema_version": "stage.v1",
  "event_type": "hello",
  "message_id": "0198...",
  "event_sequence": 0,
  "created_at": "2026-08-08T16:00:00.000Z",
  "correlation_id": "run-0198...",
  "session_id": "product-session-id",
  "owner_generation": 7,
  "stage_kind": "listen",
  "stage_id": "whisper-listen",
  "attempt_id": "0198...",
  "cancel_id": "0198...",
  "deadline_at": "2026-08-08T18:00:00.000Z",
  "traceparent": null,
  "payload": {
    "audio_formats": [{"codec":"pcm_s16le","sample_rate_hz":16000,"channels":1}],
    "limits_requested": {"max_frame_bytes":65536,"max_inflight_events":32}
  }
}
```

### 2.2 Event envelope

Every JSON event and every binary-frame JSON header MUST contain this envelope. Fields marked conditional are omitted only where stated.

| Field | Type | Rule |
|---|---|---|
| `schema_version` | string | Exactly `stage.v1`. |
| `event_type` | enum string | One of the event types in §2.3. |
| `message_id` | UUID string | Globally unique; a duplicate with identical canonical bytes is idempotent, a duplicate with different bytes is `DUPLICATE_CONFLICT`. UUIDv7 preferred. |
| `event_sequence` | non-negative integer | Strictly increases per connection direction, starts at 0, never reused; maximum is JavaScript safe integer. |
| `created_at` | RFC3339 UTC string | Sender wall-clock diagnostic time; never used to order media. |
| `correlation_id` | opaque string | Stable across product → stage → adapter hops; safe for logs. |
| `session_id` | opaque 1–128 character string | Product session identity; never synthesized by a worker. |
| `owner_generation` | non-negative integer | Product/media ownership fence. Test/dev uses 0; production MUST supply current generation. |
| `stage_kind` | `listen` \| `translate` \| `speak` \| `prosody` | Fixed for the connection. |
| `stage_id` | opaque string | Selected implementation, e.g. `whisper-listen`. |
| `stage_version` | semantic-version string | Exact implementation contract/runtime version; absent only in initial `hello`, then echoed on every event. |
| `model_revision` | opaque immutable string | Exact local checkpoint commit/revision or provider model revision; absent only in initial `hello`, then echoed on worker products/health and referenced by requests. |
| `model_artifact_digest` | string | `sha256:<hex>` for local artifacts or an honest `provider_managed:<provider>:<revision>` identity; absent only in initial `hello`. `unavailable` cannot be promoted. |
| `stage_instance_id` | UUID string | Worker allocation/process identity; absent only in initial `hello`, required in worker output and echoed by the orchestrator after `accepted`. |
| `attempt_id` | UUID string | New for every open/retry; never reused across reconnect. |
| `cancel_id` | UUID string | Fence token for this attempt. All attempt data and terminal events carry it. |
| `utterance_id` | UUID/opaque string | Required for utterance-scoped data; absent for connection/session control. Assigned once by orchestrator and never reused. |
| `utterance_sequence` | non-negative integer | Required with `utterance_id`; strict product-session order assigned by orchestrator. |
| `deadline_at` | RFC3339 UTC string | Required for work/data; receiver checks before enqueue and before emit. May be absent on health/handshake. |
| `traceparent`, `tracestate` | nullable strings | Passed through unchanged when present. OTel export is deferred. |
| `provenance_id` | `sha256:<hex>` | Required on worker products/health; identifies the accepted provenance block. |
| `payload` | object | Event-specific schema. No untyped catch-all for core semantics. |

Fence acceptance is exact: a consumer MUST reject output unless `(session_id, owner_generation, stage_kind, stage_id, attempt_id, cancel_id, stage_instance_id)` equals the active attempt, the deadline has not expired, and sequence/revision rules pass. `message_id` dedupe alone is not a fence.

### 2.3 Event types and terminal behavior

| Event | Direction | Purpose | Terminal scope |
|---|---|---|---|
| `hello`, `accepted` | O→W, W→O | Negotiate version, identity, limits, capabilities, provenance. | connection on rejection |
| `open`, `opened` | O→W, W→O | Allocate isolated per-attempt state after model is already warm. | attempt on rejection |
| `listen.audio` | O→W binary | Source PCM frame. | no |
| `listen.product` | W→O JSON | Cumulative transcript revision and committed boundary. | final product closes utterance output after input EOS |
| `translate.request` | O→W JSON | Newly committed source span plus context references/snapshot. | no |
| `translate.product` | W→O JSON | Target revision and committed boundary mapped to source span. | final product closes source span |
| `speak.request` | O→W JSON | One committed target span, voice, output format, prosody. | no |
| `speak.audio` | W→O binary | Ordered PCM for one committed target span. | no |
| `speak.complete` | W→O JSON | Duration/chunk count/finality/prosody-consumption report. | target span |
| `window` | either | Absolute available event/byte credits and queue age. | no |
| `ack` | either | Highest contiguous accepted event/media sequence; acceptance, not publication. | no |
| `eos` | either | No more input for a stream/utterance; does not cancel processing already accepted. | input stream |
| `cancel`, `cancelled` | either | Abort declared scope and acknowledge state disposal. | utterance/attempt/session as declared |
| `gap` | either | Required span could not be processed/published. | affected span |
| `dropped` | either | Explicit coalescing/drop at an allowed semantic boundary. | affected revisions/span |
| `error` | either | Typed failure. | event/span/attempt/connection per payload |
| `health`, `draining` | W→O | Warm/readiness/capacity state and planned shutdown notice. | no/new admission |

After `cancel`, the worker MUST stop accepting new data for the scope, cancel queued/in-flight inference where the provider permits, dispose per-attempt state, and emit exactly one `cancelled`. Any later product is stale and MUST be discarded by the orchestrator even if provider cancellation failed.

### 2.4 Binary audio framing

Each audio event is one WebSocket binary message:

```text
0..3   ASCII "STG1"
4..7   unsigned 32-bit big-endian JSON-header byte length N
8..    N bytes UTF-8 JSON envelope/header
...    raw audio payload, exactly payload_bytes bytes
```

Rules:

- Header limit: 16 KiB. Control/product text-frame limit: 64 KiB. Exceeding either is `FRAME_TOO_LARGE` and terminal for the attempt.
- `max_frame_bytes` is negotiated; baseline default is 65,536 payload bytes and may only be lowered by `accepted`.
- Baseline codec is **mono signed 16-bit little-endian PCM** (`pcm_s16le`). Every compliant Listen and Speak implementation MUST support it. Opus and floating PCM are not baseline `stage.v1`; add only as a later capability, never by guessing bytes.
- Header payload MUST include:

```json
{
  "stream_id": "source|translated:<opaque>",
  "media_sequence": 12,
  "start_sample": 3840,
  "sample_count": 320,
  "payload_bytes": 640,
  "format": {"codec":"pcm_s16le","sample_rate_hz":16000,"channels":1},
  "capture_time": "2026-08-08T16:00:00.240Z",
  "discontinuity": false
}
```

- `media_sequence` starts at 0 and is contiguous per `stream_id`; `start_sample` is the authoritative monotonic media clock and equals the prior `start_sample + sample_count` unless a preceding `gap` declares otherwise.
- `capture_time` is optional diagnostic wall time; algorithms MUST use sample offsets, not wall-clock timestamps, for ordering/alignment.
- Required baseline frame profile is 20 ms PCM; workers MUST accept negotiated 10–100 ms frames. This is a transport interoperability profile, not a production latency SLO.
- Channels other than 1, malformed sample counts, non-contiguous sequence without `gap`, payload-length mismatch, and format changes inside a stream are hard errors.
- `eos` is a JSON event carrying the last `media_sequence`, last sample end, and utterance ID. An empty stream is legal and must not hallucinate text.

### 2.5 IDs, sequencing, revisions, and commit barriers

1. The product/orchestrator owns `session_id`, `owner_generation`, `utterance_id`, and `utterance_sequence`.
2. The worker owns `stage_instance_id`; the orchestrator owns `attempt_id` and `cancel_id`.
3. `event_sequence` orders protocol events only. `media_sequence`/`start_sample` order audio. `utterance_sequence` orders publication. Never substitute one for another.
4. Product revisions start at 0 and strictly increase per `(stage_kind, utterance_id or source_span_id)`. A repeated revision with identical bytes is idempotent; changed bytes are `DUPLICATE_CONFLICT`; a skipped revision is `SEQUENCE_GAP`.
5. A cumulative text product carries `text`, `revision`, `committed_prefix_chars`, and `is_final`. `committed_prefix_chars` is monotonic; bytes before the previous committed boundary MUST remain identical. A violation is `COMMIT_RETRACTION` and fails the utterance.
6. `is_final=true` requires `committed_prefix_chars == text.length` and is the last revision for that product scope.
7. The orchestrator creates a `translate.request` only for the delta newly crossing the committed Listen boundary. It creates a `speak.request` only for the delta newly crossing the committed Translate boundary.
8. The ordered publication barrier releases Speak audio only in `utterance_sequence` and target-span order. A later completion waits within its deadline; if an earlier unit cannot complete, publish a declared `gap` before advancing. It MUST NOT silently reorder or wait forever.
9. A published `(utterance_sequence, target_span_id)` is immutable/idempotent. A retry may regenerate only a unit that has not crossed the publication barrier.

### 2.6 Listen contract

`listen.audio` carries source audio. `listen.product.payload` contains:

| Field | Rule |
|---|---|
| `revision` | Strictly increasing from 0. |
| `text` | Cumulative UTF-8 transcript for this utterance. |
| `committed_prefix_chars` | Monotonic immutable prefix; may advance before input EOS. |
| `is_final` | Final after source EOS or explicit endpointing. |
| `language` | BCP-47, expected `en`; never inferred silently when stage only supports English. |
| `source_start_sample`, `source_end_sample` | Covered source span. |
| `words` | Optional real/best-effort `{text,start_sample,end_sample,confidence}`; `timing_kind` says `model`, `chunk`, or `unavailable`. Fabricated even token spacing MUST be labeled `chunk`, not model timing. |
| `confidence` | Optional model confidence with declared scale/name in capability. |

A compliant remote Listen path MUST emit at least one `listen.product` with a non-empty committed prefix while the test source iterator is deliberately withholding EOS. A model that only produces unstable partials before EOS is not sufficient for downstream pre-EOS speech unless an explicit, tested commit policy promotes a stable boundary.

### 2.7 Translate contract

`translate.request.payload` contains:

- `source_span_id`, `source_revision`, `source_char_start`, `source_char_end`, and committed `text`;
- `source_language` (`en`) and `target_language` (`es` for the critical path);
- preceding committed source and target context, bounded by negotiated/configured limits;
- sermon-note snapshot or immutable reference plus `sermon_notes_revision`;
- glossary entries or immutable reference plus `glossary_revision`;
- prompt/policy revision and deadline.

`translate.product.payload` contains:

- `source_span_id`, stable `target_span_id`, `revision`, cumulative target `text`, `committed_prefix_chars`, `is_final`;
- source character-span mapping for the target span, `target_language`, and optional model adequacy metadata;
- terminology decisions/violations when available;
- `prompt_revision`, `glossary_revision`, and provenance reference.

Rules:

- Only committed source spans enter translation.
- The stage MUST fail closed if it cannot return target-language text. Source English, bracketed error text, or mock output MUST NOT be sent to Speak as Spanish.
- Context and glossary inputs are explicit protocol data, not hidden process globals.
- Index-based source-word→Spanish-word mapping is not represented as real alignment. Use span mapping with confidence/kind, or mark alignment unavailable.

### 2.8 Speak and prosody contract

`speak.request.payload` contains:

- `target_span_id`, committed Spanish `text`, `target_language`, and publication order;
- requested output PCM format;
- `voice_id`, `voice_revision`, and immutable voice/config digest;
- `prosody` object and `prosody_required` boolean.

Minimal `prosody.v1` object:

```json
{
  "schema_version": "prosody.v1",
  "overall": {"rate": 1.0, "pitch_semitones": 0.0, "energy": 1.0, "style": "neutral"},
  "markers": [
    {"target_char_start": 0, "target_char_end": 6, "emphasis": 0.7},
    {"after_target_char": 12, "pause_ms": 220}
  ],
  "alignment_kind": "model|human|heuristic|unavailable",
  "confidence": 0.0
}
```

- Overall `rate` is in `[0.5, 2.0]`, `pitch_semitones` in `[-12, 12]`, `energy` in `[0, 2]`, marker `emphasis` and `confidence` in `[0, 1]`, and `pause_ms` in `[0, 5000]`; values outside these protocol safety ranges are `INVALID_ARGUMENT`. These are validation bounds, not quality/SLO targets.
- `style` is a negotiated enum; an unknown value is unsupported rather than silently remapped.
- Marker offsets target committed Spanish characters, not source word indexes.
- A Speak worker advertises supported prosody fields in `accepted`.
- If `prosody_required=true` and any required field is unsupported, reject before synthesis with `UNSUPPORTED_CAPABILITY`.
- Otherwise `speak.complete` reports `prosody_status: applied|partial|unsupported`, `consumed_fields`, and `ignored_fields`. Claiming applied while ignoring fields is a conformance failure.

`speak.audio` binary headers additionally carry `target_span_id` and `audio_chunk_sequence`. `speak.complete` contains `chunk_count`, `sample_count`, `duration_ms`, `is_final=true`, and the prosody report. Audio chunk sequences and sample offsets MUST be contiguous. The stage SHOULD yield generator chunks as they are produced; collecting a complete utterance before the first yield is allowed only when the selected backend cannot stream and is visible as a capability, not described as streaming.

### 2.9 Backpressure, capacity, age, and deadlines

- Every worker has configured hard limits: `max_sessions`, `max_inflight_events`, `max_inflight_bytes`, `max_frame_bytes`, `input_queue_capacity`, and `max_queue_age_ms`.
- `accepted` and `/health/ready` report limits and current capacity. Admission with no capacity fails immediately with `RESOURCE_EXHAUSTED` and optional `retry_after_ms`; the worker does not queue unbounded sessions.
- `window` contains absolute `available_events`, `available_bytes`, `credit_epoch`, and `oldest_queue_age_ms` for a stream. A sender MUST NOT exceed the latest window. WebSocket/TCP buffering is not application backpressure.
- A bounded `asyncio.Queue` exists on both send and receive sides. Queue `put` waits only until the event deadline. Timeout emits `DEADLINE_EXCEEDED`; it never falls back to unbounded buffering.
- Before enqueue and before model emission, receivers compare `deadline_at` and queue age. Expired work is cancelled/fenced and represented by `gap`/`error` as appropriate.
- Permitted coalescing: superseded **uncommitted** `listen.product` or `translate.product` revisions. The coalescer emits `dropped` with revision range and reason `superseded_uncommitted`.
- Forbidden silent drop: source audio, committed text spans, Speak audio, EOS, cancel, errors, or gap records.
- Oldest queued-audio age, queue occupancy, window stalls, dropped/coalesced revisions, and deadline failures are exposed in health/debug snapshots. OTel export comes later.

### 2.10 Cancellation and stale-result fencing

`cancel.payload` contains `scope` (`utterance|attempt|session`), `reason`, and affected IDs. Required behavior:

1. stop admission for the scope;
2. cancel queue producers/consumers and provider futures where supported;
3. dispose decoder/context/output state, but not resident model weights;
4. emit `cancelled` after disposal;
5. reject any further input carrying the cancelled fence;
6. orchestrator discards any late product failing the exact active-fence match;
7. publication barrier never releases audio from a cancelled/stale fence.

A cancel timeout is an orchestrator-side hard stop: close the WebSocket, mark the attempt failed, retain the stale fence, and open a fresh attempt only if policy/deadline permits.

### 2.11 Errors

`error.payload` contains `code`, safe `message`, `retryable`, `scope`, optional `retry_after_ms`, and affected event/span IDs. It MUST NOT contain credentials, prompts with private sermon notes, or raw audio.

Required codes:

`VERSION_UNSUPPORTED`, `AUTHENTICATION_FAILED`, `AUTHORIZATION_FAILED`, `INVALID_ARGUMENT`, `FRAME_TOO_LARGE`, `UNSUPPORTED_FORMAT`, `UNSUPPORTED_CAPABILITY`, `DUPLICATE_CONFLICT`, `SEQUENCE_GAP`, `COMMIT_RETRACTION`, `STALE_FENCE`, `RESOURCE_EXHAUSTED`, `DEADLINE_EXCEEDED`, `CANCELLED`, `MODEL_UNAVAILABLE`, `INFERENCE_FAILED`, `INTERNAL`.

Errors are fail-closed. No error path may synthesize mock audio or substitute English as Spanish.

### 2.12 Health, readiness, warm models, and provenance

HTTP endpoints on the worker:

| Endpoint | 200 condition | 503 condition |
|---|---|---|
| `/health/live` | process/event loop responsive | process unable to serve |
| `/health/startup` | model artifacts verified and model loaded | loading/verifying/failed |
| `/health/ready` | startup complete, warmup canary passed, not draining, capacity available | cold, failed canary, digest mismatch, no admission capacity, draining |

Readiness JSON includes stage kind/id/version, `stage_instance_id`, worker `boot_id`, active/max sessions, queue limits/age, model loaded/warm booleans, last canary time/result, and provenance.

The immutable provenance block is canonical-JSON hashed into `provenance_id` and contains:

- stage implementation ID and semantic version;
- code Git SHA;
- container image digest (nullable only in explicit local-dev mode);
- model/provider ID and exact revision;
- model artifact manifest digest and status (`verified`, `provider_managed`, `unavailable`);
- runtime/library versions;
- prompt, glossary, voice, and stage-config digests where applicable;
- hardware class and worker boot ID.

Promotion requires verified local artifact/model digests or an honest provider-managed identity; `latest`, floating Git branches, or `unavailable` digest status cannot be promoted. Cache presence alone is not provenance. A checksum manifest is verified before readiness.

Worker lifecycle:

- process lifespan constructs and loads a `StageHost` once;
- `StageHost.open_session()` creates isolated state for each accepted attempt;
- `close_session()` never unloads the model;
- model unload occurs only on process drain/exit;
- planned shutdown emits `draining`, rejects new opens, allows bounded completion, then cancels remaining attempts.

### 2.13 Reconnect/resume policy

- v1 has no durable worker replay log and no transparent in-flight Listen resume.
- On disconnect, the orchestrator marks the attempt failed, cancels/fences it, and rejects all output from its IDs.
- A new connection uses a new `attempt_id`, `cancel_id`, and worker `stage_instance_id`/boot identity from handshake.
- Listen recovery starts a new utterance at the next available audio boundary and emits `gap` for source samples not processed. An optional orchestrator-owned bounded audio ring may replay only frames known not accepted by `ack`; this optimization is not required for v1 conformance and must never duplicate accepted frames.
- Translate/Speak may replay an immutable committed request only if its target unit has not crossed the publication barrier. Reuse its source/target span ID for idempotency but use the new attempt/cancel fence.
- Already published Speak audio is never replayed automatically.
- A `hello.payload.resume` request is rejected with `RESUME_UNSUPPORTED` in baseline v1; the orchestrator then follows the boundary restart policy. This explicit rejection is preferable to false resume.
- Reconnect uses capped exponential backoff with jitter bounded by the original deadline. Once the deadline expires, emit a gap/error and stop retrying. Exact retry timings are configuration, not protocol constants.
