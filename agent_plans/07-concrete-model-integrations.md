# Phase 7 — Concrete Model Integrations

## Objective

Replace passthrough stubs with production stage backends:

| Kind | Options |
|------|---------|
| Listen | Kyutai STT-1B; Voxtral Mini Realtime |
| Prosody | PENN or pyworld F0 + energy + pause → 5-dim tokens (keep YIN baseline) |
| Translate | Small fp16 incremental EN→ES LLM with prosody carry-through |
| Speak | Qwen3-TTS-12Hz 0.6B (default); CosyVoice3-0.5B; Kyutai Pocket TTS `spanish_24l` fallback |

## Prerequisites

Phases 1–2 (contracts + messages), 4 (cache/runtime). Phases 3/5/6 improve
operability but model code should still run under `LocalStageRuntime` for dev.

## Design

### 1. Packaging / optional deps

Heavy deps must be **optional extras** so CPU CI stays light:

```toml
# pyproject.toml extras (illustrative names)
[project.optional-dependencies]
listen-kyutai = [...]
listen-voxtral = [...]
tts-qwen = [...]
tts-cosyvoice = [...]
tts-pocket = [...]
prosody-penn = [...]
translate-llm = [...]
stages-all = [/* union */]
```

Registry registration pattern (same as Seamless today):

```python
try:
    register_kyutai_listen(registry)
except ImportError:
    logger.info("kyutai not installed; skipping")
```

### 2. Listen — Kyutai STT-1B

- True streaming ASR; target ~0.5s delay.
- Emit `ListenProduct` with partial/final flags and best-effort word timings
  (if model provides; else chunk-level spans).
- Load weights via `ModelCache` / HF hub into `MODEL_CACHE_DIR`.
- Device from `Settings.compute_device`.

### 3. Listen — Voxtral Mini Realtime

- Alternate streaming text path.
- Same `ListenProduct` surface so Translate is unaware of which Listen ran.

### 4. Prosody — enhanced tracker

- Prefer **pyworld** or **PENN** when extra installed; fall back to
  `YinPitchTracker` / `BaselineProsodyStage`.
- Output remains `MetadataEnvelope` + aligner → `ProsodyToken` on words.
- Must stay cheap enough to run alongside ASR on CPU if needed.

### 5. Translate — incremental LLM

- Small fp16 instruct model fine for EN→ES sermon domain (exact checkpoint
  configurable via env `TRANSLATE_MODEL_ID`).
- Input: stream of `ListenProduct` (+ aligned prosody on words).
- Output: `TranslateProduct` with Spanish text, realigned `words[].prosody`,
  and `SynthesisInstructions` markers for Speak.
- Strategy v1: sentence/chunk boundary buffering with overlap; do not wait for
  full utterance if latency budget exceeded (config `translate_max_latency_ms`).

### 6. Speak — Qwen3-TTS-12Hz 0.6B (default)

- Native Spanish; ES WER ~1.0–1.5 per research notes.
- Bidirectional streaming via **vLLM-Omni WebSocket** when available; local
  fallback path if library supports non-server inference.
- Consume `SynthesisInstructions` on the instruction channel.
- Apache 2.0; ~1.2GB fp16 — fits a V100 with headroom if not colocated poorly.

Worker mode: often this stage **is** a remote vLLM-Omni service; implement
`Qwen3TTSStage` as a client conforming to `TTSStage`, reusable under
`RemoteStageRuntime` or direct HTTP/WS inside local runtime.

### 7. Speak — CosyVoice3-0.5B

- NL style instruct + official cross-lingual cloning (EN ref → ES speech).
- Session config may pass `voice_reference_path` or bytes handle (careful with
  size limits); v1 can read path from cache dir.

### 8. Speak — Kyutai Pocket TTS `spanish_24l`

- 100M, MIT, ~6× realtime on CPU.
- Register with `requires_gpu=False`.
- Intended fallback when GPU speak job unhealthy; selection is manual in v1
  (operator picks stage id), not automatic failover (document as future work).

### 9. Default stage selection

```python
DEFAULT_STAGES = StageSelection(
    listen="kyutai-stt-1b",          # or first available listen
    translate="sermon-mt-small",
    speak="qwen3-tts-0.6b",
    prosody="baseline-prosody",     # upgrade id when penn/pyworld ready
)
```

`StageInfo.default_for_kind` marks defaults among **registered** stages so
environments without GPU extras still default to passthrough/pocket.

### 10. Resource & quality acceptance (manual / GPU host)

Not CI gates — document in phase exit as operator checklist:

| Path | Expectation |
|------|-------------|
| Listen partial latency | ~0.5–1.0s first text |
| Speak first audio | ~100–300ms after translate final chunk (model-dependent) |
| Pocket TTS | realtime factor ≥ 1 on CPU |
| Prosody | no more than ~5–10% of one CPU core at 48 kHz baseline |

### 11. Security / license

- Respect model licenses in README.
- No commit of weight binaries; only cache paths.
- Network fetches only at runtime/build with explicit operator action.

## Files Likely Touched

```
server/pyproject.toml
server/src/pipelines/stages_listen/
server/src/pipelines/stages_translate/
server/src/pipelines/stages_speak/
server/src/pipelines/stages_prosody/
server/src/pipelines/stage_registry.py
server/Dockerfile.gpu
server/tests/test_*_stage.py          # mocked model IO
docs/README.md or deploy/README.md    # model ops
```

Prefer package-per-kind folders with barrel exports; keep stub stages.

## Tests

1. Each backend: protocol conformance with **mocked** model client.
2. Registry skips missing extras.
3. Instructions pass from translate mock → speak mock.
4. Integration test marked `@pytest.mark.gpu` optional, skipped in CI.

## Exit Criteria

- [ ] At least one real Listen, Translate, and Speak backend implemented and selectable
- [ ] Pocket TTS CPU fallback registered
- [ ] Prosody enhanced path optional with baseline default
- [ ] Weights resolve under `MODEL_CACHE_DIR`
- [ ] Composed pipeline works with real backends on a GPU host (documented runbook)
- [ ] CI remains green without optional extras

## Out of Scope

- Automatic GPU→CPU speak failover
- Domain fine-tuning training loop (use existing train job separately)
- Mobile/edge deployment
