# Staged Pipeline Refactor — Overview

## Goal

Replace monolithic end-to-end pipelines with a **composable stage graph**:

```
audio ──► Listen (ASR) ──► Translate ──► Speak (TTS) ──► audio
            │                  │              │
            └──── Prosody ─────┴──────────────┘
                  (parallel track, carried through)
```

Operators pick independent models for **listen**, **translate**, and **speak**.
Each stage product streams to the admin UI for live debugging. Stages run
in-process locally or as independent Nomad jobs, loading weights from a shared
MooseFS model cache.

## Current State (baseline)

| Area | Today |
|------|--------|
| Pipelines | Monolithic `BasePipeline` subclasses (Echo, WhisperTTS, Spanish, Seamless, ProsodyEcho) |
| Stage protocols | `ASRStage`, `TranslationStage`, `TTSStage`, `ProsodyStage` exist but are barely used; pipelines still own the full graph |
| Session create | `pipeline_id` only (`SessionCreate`) |
| Debug streams | Transcripts + prosody metadata via `pipeline.event`; no per-stage product view |
| Deployment | Single GPU Nomad job for the whole server |
| Model cache | None; each container downloads/loads independently |
| Prosody | `BaselineProsodyStage` (YIN F0 + energy + pause); `MetadataEnvelope` / `ProsodyFrame` already on the wire |

## Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Session Orchestrator (existing FastAPI server)             │
│  - WebRTC / Crosstalk transport                             │
│  - Stage graph wiring & backpressure                        │
│  - Fan-out of stage products → DataChannel (admin UI)       │
└────────┬──────────────┬──────────────┬──────────────────────┘
         │              │              │
    ┌────▼────┐   ┌─────▼─────┐  ┌────▼────┐
    │ Listen  │   │ Translate │  │  Speak  │   (+ Prosody sidecar)
    │ worker  │   │  worker   │  │ worker  │
    └────┬────┘   └─────┬─────┘  └────┬────┘
         │              │              │
         └──────────────┴──────────────┘
                        │
              MODEL_CACHE_DIR (MooseFS)
```

### Stage options (product targets)

**Listen**
- Kyutai STT-1B — true streaming, ~0.5s delay, DSM, fp16
- Voxtral Mini Realtime — alternate text stream

**Prosody (parallel with Listen)**
- Lightweight tracker: PENN/pyworld F0 + energy + pause
- Emits ProsodyLM-style word-aligned tokens (5-dim quantized: pitch median/range/slope, duration, energy)

**Translate**
- Small fp16 LLM, incremental EN→ES chunk translation
- Carries prosody tokens through to Spanish word positions
- Emits instruction-channel markup for Speak

**Speak**
- Qwen3-TTS-12Hz 0.6B (default) — native Spanish, bidirectional streaming via vLLM-Omni WS, instruction channel for prosody
- CosyVoice3-0.5B — expressive NL style instruct + cross-lingual cloning
- Kyutai Pocket TTS `spanish_24l` — CPU fallback, 6× realtime, zero VRAM

## Phases

| # | Doc | Summary | Depends on |
|---|-----|---------|------------|
| 1 | [01-stage-contracts-and-composition.md](./01-stage-contracts-and-composition.md) | Stage identity, registries, `ComposedPipeline`, session stage selection API | — |
| 2 | [02-prosody-channel-and-inter-stage-messages.md](./02-prosody-channel-and-inter-stage-messages.md) | Structured stage messages, word-aligned prosody tokens, instruction channel | 1 |
| 3 | [03-admin-ui-stage-debug-streams.md](./03-admin-ui-stage-debug-streams.md) | Per-stage product events to admin UI; stage pickers | 1, 2 |
| 4 | [04-stage-runtime-and-model-cache.md](./04-stage-runtime-and-model-cache.md) | Runtime abstraction (local in-process), `MODEL_CACHE_DIR` | 1 |
| 5 | [05-out-of-process-stage-workers.md](./05-out-of-process-stage-workers.md) | Stage worker process protocol; local subprocess + remote client | 4 |
| 6 | [06-nomad-stage-jobs.md](./06-nomad-stage-jobs.md) | Per-stage Nomad jobs, MooseFS cache mount, service discovery | 5 |
| 7 | [07-concrete-model-integrations.md](./07-concrete-model-integrations.md) | Wire real Listen/Translate/Speak model backends behind stage interfaces | 2, 4 |

## Non-Goals (this refactor)

- Replacing WebRTC / Crosstalk transports
- Training / fine-tuning infrastructure (existing train Nomad job stays)
- Multi-tenant auth / billing
- Changing the browser Opus wire format (stages still use s16le PCM internally)

## Invariants to Preserve

1. **Pydantic models are the schema source of truth** — run `pnpm typegen` after model changes.
2. **Transport is an interface** — no stage logic branches on WebRTC vs Crosstalk.
3. **Pipelines stay pluggable** — monolithic pipelines remain registered for back-compat until stages fully replace them; `ComposedPipeline` is one more `BasePipeline`.
4. **No secrets in logs**; no comments that narrate *what*.
5. **Tests after every phase** — server pytest, client vitest, relevant lint/typecheck.
6. **Prosody is first-class** — never drop the metadata channel when composing stages.

## Suggested Implementation Order per Phase

1. Models / contracts
2. Server wiring + unit tests
3. Client (if phase touches UI) + typegen
4. Deploy artifacts (if phase touches Nomad)
5. Docs touch-up in phase file "Exit criteria" checklist

## Review Checklist (every phase)

- [ ] Phase doc exit criteria all met
- [ ] No regressions in existing pipeline tests
- [ ] Prosody / metadata path still works end-to-end where applicable
- [ ] `uv run ruff check` + `uv run pyright` + `uv run pytest` clean for server changes
- [ ] `pnpm lint` + `pnpm typecheck` + `pnpm test` clean for client changes
- [ ] `pnpm typegen:check` clean if models changed
- [ ] AGENTS.md / README updated only when public architecture changed
