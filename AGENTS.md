# AGENTS.md

## Project

Real-time sermon translation platform. Python FastAPI server, React + Mantine client, Playwright e2e tests.

## Repo Layout

```
server/     Python 3.12, FastAPI, uv
client/     React 19, Mantine 8, Vite, pnpm
e2e/        Playwright in Docker Compose
```

## Commands

### Server (`cd server`)

```
uv run ruff check src/ tests/    # lint
uv run pyright src/               # typecheck
uv run pytest                     # unit tests (36)
```

### Client (`cd client`)

```
pnpm lint                         # eslint
pnpm typecheck                    # tsc --noEmit
pnpm test                         # vitest (20)
pnpm typegen                      # regenerate types.gen.ts from Pydantic models
pnpm typegen:check                # verify types.gen.ts is not stale
```

### E2E (`cd e2e`)

```
bash run.sh                       # builds client dist, runs Docker Compose Playwright suite (9)
```

### Dev Environment

```
./dev.sh                          # tmux session: server left, client right
```

## Type Generation

Pydantic models are the single source of truth. `server/src/codegen.py` generates `client/src/api/types.gen.ts`. Never edit `types.gen.ts` by hand — run `pnpm typegen` from `client/` after changing models.

## DRY & Modularity

- **One source of truth for types.** Pydantic models define the schema once; codegen produces the TypeScript counterpart. Do not duplicate type definitions across server and client.
- **Abstract before duplicating.** If logic appears in two places, extract it. Server: new module under the relevant package. Client: new file in `hooks/`, `api/`, or `components/`.
- **Transport is an interface.** `TransportConnection` (server) and `StreamTransport` (client) are abstract. Add new transports by implementing the interface — never branch on transport type inside business logic.
- **Pipelines are pluggable.** Subclass `BasePipeline`, register in `PipelineRegistry`. Pipeline selection is data-driven, not conditional.
- **Barrel exports.** Each package has an `__init__.py` or `index.ts` that re-exports public API. Import from the package, not from internal modules.
- **Thin API routes.** Routes delegate to stores and pipelines. Keep business logic out of route handlers.
- **Components own their UI, hooks own their data.** Don't fetch data inside a component — use or create a hook. Don't render UI inside a hook.

## Code Style

- **Server:** ruff (E, F, I, UP, B, SIM), pyright standard mode, line length 100, `from __future__ import annotations` in every file.
- **Client:** strict TypeScript, `verbatimModuleSyntax`, eslint + typescript-eslint recommended.
- No comments unless they explain *why*. No commented-out code.

## Testing

- Run the relevant test suite after every change.
- Server tests: `pytest` with `pytest-asyncio` (auto mode), `httpx` ASGITransport for API tests.
- Client tests: `vitest` with jsdom, `@testing-library/react`.
- E2E tests: Playwright against Docker Compose stack (server + nginx + Playwright container). Use `getByText` with `{ exact: true }` or aria-label selectors to avoid ambiguity.

## Wire Protocol

WebSocket at `/ws/stream/{session_id}`. Tagged binary frames:
- `0x01` + bytes = audio (PCM int16)
- `0x02` + UTF-8 JSON = event
