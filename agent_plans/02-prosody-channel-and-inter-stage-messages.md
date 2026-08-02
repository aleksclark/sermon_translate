# Phase 2 — Prosody Channel and Inter-Stage Messages

## Objective

Define a **shared message schema** between stages so prosody is not a side-channel
bolted onto strings, but a structured stream that Listen produces, Translate
realigns, and Speak consumes via an instruction channel.

## Motivation

Phase 1 composes stages with bare `str` / `bytes`. That cannot carry:

- word timings from ASR
- 5-dim quantized prosody tokens (ProsodyLM-style)
- target-side instruction markup for TTS (Qwen3-TTS instruction channel,
  CosyVoice style prompts)

`MetadataEnvelope` / `ProsodyFrame` already exist for the **admin/debug wire**.
This phase adds the **internal stage bus** and upgrades prosody to word-aligned
tokens while keeping wire compatibility.

## Design

### 1. Internal stage messages (Pydantic)

```python
class ProsodyToken(BaseModel):
    """5-dim quantized prosody token aligned to a word/span."""
    pitch_median: int   # quantized bin
    pitch_range: int
    pitch_slope: int
    duration: int
    energy: int
    # optional continuous values for debug UI
    f0_hz: float | None = None
    energy_rms: float | None = None
    start_ms: float | None = None
    end_ms: float | None = None

class WordSpan(BaseModel):
    text: str
    start_ms: float | None = None
    end_ms: float | None = None
    conf: float | None = None
    prosody: ProsodyToken | None = None

class ListenProduct(BaseModel):
    sequence: int
    utterance_id: str
    text: str                    # partial or final transcript chunk
    is_final: bool = False
    words: list[WordSpan] = []
    language: str = "en"

class TranslateProduct(BaseModel):
    sequence: int
    source_utterance_id: str
    target_utterance_id: str
    text: str                    # Spanish chunk
    is_final: bool = False
    words: list[WordSpan] = []   # target words with realigned prosody
    # instruction-channel markup for Speak (model-neutral)
    instructions: SynthesisInstructions | None = None

class SpeakProduct(BaseModel):
    sequence: int
    target_utterance_id: str
    pcm: bytes                   # not for JSON wire; internal only
    sample_rate: int
    start_ms: float | None = None
    end_ms: float | None = None
```

Notes:
- `SpeakProduct.pcm` stays internal (binary). Public debug events use metadata
  + text, not raw PCM dumps over DataChannel (audio already has its own track).
- Reuse `SynthesisInstructions` from `models/metadata.py`.
- Quantization helpers live in `pipelines/prosody_tokens.py` with documented
  bin edges (defaults can be tuned later).

### 2. Upgrade stage protocols

```python
class ASRStage(Protocol):
    def transcribe(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[ListenProduct]: ...

class TranslationStage(Protocol):
    def translate(
        self,
        text_stream: AsyncIterator[ListenProduct],
        *,
        prosody: AsyncIterator[MetadataEnvelope] | None = None,
    ) -> AsyncIterator[TranslateProduct]: ...

class TTSStage(Protocol):
    def synthesize(
        self, text_stream: AsyncIterator[TranslateProduct]
    ) -> AsyncIterator[bytes]: ...  # or AsyncIterator[SpeakProduct]

class ProsodyStage(Protocol):
    def analyze(
        self, audio_stream: AsyncIterator[bytes], stream_name: str
    ) -> AsyncIterator[MetadataEnvelope]: ...
```

**Alignment strategy (this phase):**

1. Prosody stage continues to emit frame-level `MetadataEnvelope` (baseline).
2. A new `ProsodyAligner` utility merges frame-level prosody into `ListenProduct.words`
   by time overlap (median F0, energy, duration → quantized `ProsodyToken`).
3. Translate stage (passthrough stub) copies tokens onto target words by
   monotonic index mapping (1:1 stub); real LLM alignment in Phase 7.
4. Translate emits `SynthesisInstructions.markers` describing prosody for Speak.
5. Speak stub may ignore instructions; real models consume them in Phase 7.

### 3. Wire events for admin (prepare Phase 3)

Extend `pipeline.event` payload kinds (handler already forwards opaque payloads):

| kind | payload | purpose |
|------|---------|---------|
| `transcript` | `{stream, text}` | **keep** for back-compat |
| `stage.product` | `{stage, product}` | structured Listen/Translate products |
| `metadata` | existing | prosody frames + instructions |

`ComposedPipeline` should:
- Still publish plain transcript strings on `listen` / `translate` text streams
  (derived from `product.text`) so existing UI keeps working.
- Additionally publish `stage.product` events via a new publish path **or**
  encode products as metadata/instructions envelopes.

Preferred approach: add optional callback / event bus on `ComposedPipeline`:

```python
# BasePipeline or Composed-only:
async def _publish_stage_event(self, stage: StageKind, product: BaseModel, session): ...
```

Handler gains `forward_stage_events` similar to metadata, **or** composed
pipeline folds stage products into metadata streams named `listen.product` /
`translate.product`. Pick one and document it; prefer explicit `stage.product`
events for clarity in Phase 3 UI.

### 4. Prosody token quantization

Implement:

```python
def quantize_prosody(
    *,
    f0_values: Sequence[float],
    energy_values: Sequence[float],
    duration_ms: float,
    n_bins: int = 32,
) -> ProsodyToken: ...
```

Unit-test stability: same inputs → same bins; silence → pause-friendly defaults.

### 5. Update passthrough stages

- Listen stub: emit `ListenProduct` with crude word splits + empty prosody;
  aligner fills tokens when prosody frames available.
- Translate stub: map words, attach `SynthesisInstructions` with markers.
- Speak stub: read `TranslateProduct.text` (ignore instructions OK).

### 6. Migration of existing protocols

Any code still typing `AsyncIterator[str]` for ASR/MT must be updated.
Monolithic pipelines that do not use stage protocols are unaffected.

## Files Likely Touched

```
server/src/models/metadata.py         # ProsodyToken fields if folded into ProsodyFrame
server/src/models/stage_messages.py   # NEW
server/src/models/__init__.py
server/src/pipelines/stages.py        # protocol signatures
server/src/pipelines/prosody_tokens.py # NEW quantize + align
server/src/pipelines/composed.py
server/src/pipelines/stub_stages.py
server/src/transport/handler.py       # stage.product forwarding if needed
server/tests/test_prosody_tokens.py
server/tests/test_composed_pipeline.py
server/tests/test_stage_messages.py
client typegen
```

## Tests

1. Quantization determinism and bin bounds.
2. Aligner attaches tokens to words by time overlap.
3. Composed graph: ListenProduct → TranslateProduct carries prosody markers.
4. Wire: `stage.product` events observed in handler test transport.
5. Back-compat transcript events still emitted.
6. Existing prosody-echo + baseline prosody tests still pass.

## Exit Criteria

- [ ] Inter-stage message models exist and are codegen'd
- [ ] Composed pipeline moves `ListenProduct` / `TranslateProduct` between stages
- [ ] Prosody tokens (5-dim) produced and attached to words
- [ ] Translate emits `SynthesisInstructions` for Speak
- [ ] Admin wire has structured stage products (or clearly staged equivalent)
- [ ] No regression in metadata envelope schema versioning

## Out of Scope

- Real ASR word timings from Kyutai/Voxtral (Phase 7)
- Real LLM prosody re-alignment (Phase 7)
- Admin UI visualization (Phase 3)
- PENN/pyworld backends (Phase 7; keep YIN baseline)
