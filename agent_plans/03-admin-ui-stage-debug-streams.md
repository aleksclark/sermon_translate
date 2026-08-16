# Phase 3 — Admin UI Stage Debug Streams

## Objective

Surface **every stage product** in the admin session UI for live debugging:
listen transcript + words/prosody tokens, translate output + instructions,
prosody frames, and speak status — while finishing stage selection UX.

## Motivation

Operators need to see *where* latency or quality fails (ASR vs MT vs TTS vs
prosody alignment) without attaching a debugger to the server.

## Design

### 1. Client event model

Extend parsing in `useAudioStream` (or a dedicated hook):

```ts
type StageKind = "listen" | "translate" | "speak" | "prosody";

interface StageProductUpdate {
  stage: StageKind;
  product: ListenProduct | TranslateProduct | Record<string, unknown>;
  timestamp: number;
}
```

Handle `pipeline.event` with `kind === "stage.product"`:

```ts
payload: {
  kind: "stage.product",
  stage: "listen" | "translate" | "speak",
  product: { ... }  // matches generated types
}
```

Keep existing transcript + metadata handling.

### 2. Active session debug panel

In `ActiveSessionPanel` (or new `StageDebugPanel`):

| Panel | Content |
|-------|---------|
| Listen | Rolling transcript, last N words with prosody token chips (pitch/energy bins) |
| Translate | Target text, instruction markers summary |
| Prosody | Sparkline or compact table of recent frames (energy, f0, pause) — data already in `metadata` |
| Speak | Last utterance id, chunk count / bytes (from stats or speak products) |

UX guidelines:
- Use Mantine `Card` / `ScrollArea` consistent with `TranscriptBox`.
- Cap retained lines (e.g. 200 per stream) to avoid memory growth.
- Collapsible sections; default expanded for listen + translate.
- `data-testid` attributes for e2e later (`stage-debug-listen`, etc.).

### 3. Stage selection in New Session modal

If not completed in Phase 1:

- When `pipeline_id === "composed"`, show NativeSelects for listen / translate /
  speak / prosody filtered by `StageInfo.kind`.
- Prefill defaults where `default_for_kind`.
- POST `stages: { listen, translate, speak, prosody }`.

Show stage names (not only ids) from `GET /api/stages`.

### 4. Session list / active header

Show resolved stage ids on the active session header (e.g.
`listen=… · translate=… · speak=…`) so recordings/screenshots capture config.

### 5. Server guarantees for UI

Confirm handler emits:
1. `transcript` for listen & translate (human-readable).
2. `stage.product` for structured products.
3. `metadata` for prosody frames and any instruction envelopes.

Optional: throttle high-rate prosody frames on the wire (e.g. every Nth frame
or 10 Hz) behind a session flag `debug_prosody_hz` — only if UI jank appears;
default can remain full rate for short admin sessions.

## Files Likely Touched

```
client/src/api/types.gen.ts           # typegen
client/src/api/client.ts
client/src/hooks/useAudioStream.ts
client/src/hooks/useStageProducts.ts  # NEW optional extract
client/src/components/ActiveSessionPanel.tsx
client/src/components/StageDebugPanel.tsx  # NEW
client/src/components/NewSessionModal.tsx
client/src/App.tsx
client/src/test/metadata.test.ts
client/src/test/stage-products.test.ts  # NEW
server/src/transport/handler.py       # only if event shape needs tweak
```

## Tests

1. Client unit: parse `stage.product` events.
2. Client unit: ignore malformed products.
3. Component/render tests optional; prefer pure parse tests + light RTL if patterns exist.
4. Server: ensure composed pipeline emits products (regression from Phase 2).

## Exit Criteria

- [ ] Admin UI shows live listen + translate structured output during a composed session
- [ ] Prosody frames visible (reuse/enhance existing metadata display)
- [ ] Stage pickers work for composed pipeline create
- [ ] Existing transcript boxes still work for legacy pipelines
- [ ] Client lint/typecheck/tests pass

## Out of Scope

- Multi-user admin auth
- Persisting debug traces to disk
- Audio waveform visualization
- E2E Playwright coverage (nice-to-have; add only if cheap)
