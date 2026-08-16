# Phase 4 — Stage Runtime Abstraction and Model Cache

## Objective

Introduce a **stage runtime** interface so stages load and run the same way
in-process today and out-of-process later, plus a configurable **shared model
cache directory** (MooseFS-backed on the Nomad cluster) for downloaded weights.

## Motivation

Without a runtime boundary, Phase 5/6 will fork stage code. Without a shared
cache, every GPU job re-downloads multi-GB weights and wastes disk/network
across the cluster.

## Design

### 1. Config

Extend `Settings` / env:

| Env | Meaning | Default |
|-----|---------|---------|
| `MODEL_CACHE_DIR` | Root for HF/torch/custom weights | `~/.cache/sermon-translate/models` |
| `STAGE_RUNTIME` | `local` (in-process) \| `subprocess` \| `remote` | `local` |
| `HF_HOME` / `TRANSFORMERS_CACHE` / `TORCH_HOME` | Optional passthrough; if unset, derive from `MODEL_CACHE_DIR` | derived |

```python
@dataclass(frozen=True):
class Settings:
    ...
    model_cache_dir: Path
    stage_runtime: str = "local"  # local | subprocess | remote
```

On startup, ensure `model_cache_dir` exists (mkdir parents) when writable;
log the resolved path once.

Document in `deploy/README.md` that Nomad jobs should mount MooseFS and set:

```
MODEL_CACHE_DIR=/mnt/moosefs/sermon-translate/models
```

(Exact MooseFS subpath is operator-configurable; do not hardcode cluster
internals beyond examples.)

### 2. Model cache helper

```python
# server/src/runtime/model_cache.py
class ModelCache:
    def __init__(self, root: Path): ...
    def path_for(self, *parts: str) -> Path:
        """Return absolute path under root; create parents on demand."""
    def environ(self) -> dict[str, str]:
        """Env vars to inject into workers so libs share the cache."""
```

Conventions under root:

```
{MODEL_CACHE_DIR}/
  huggingface/          # HF_HOME
  torch/                # TORCH_HOME
  custom/{stage_id}/    # stage-specific blobs
```

### 3. Stage runtime interface

```python
# server/src/runtime/base.py
class StageHandle(Protocol):
    info: StageInfo
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    # kind-specific stream methods delegated to underlying stage

class StageRuntime(Protocol):
    async def spawn(self, stage_id: str, session: Session) -> StageHandle: ...
```

`LocalStageRuntime`:
- Uses `StageRegistry` factories.
- Injects `ModelCache` into factories that accept it (optional kw-only).
- In-process; no serialization of audio beyond existing iterators.

Factory signature evolution:

```python
class StageFactory(Protocol):
    info: StageInfo
    def create(self, *, cache: ModelCache, session: Session) -> Any: ...
```

### 4. Wire `ComposedPipeline` through runtime

`ComposedPipeline` takes a `StageRuntime` instead of constructing stages
directly from the registry. Default deps: `LocalStageRuntime(stage_registry, cache)`.

### 5. Weight download policy (local)

- Stages **must** resolve weights via `ModelCache.path_for` / HF env, not ad-hoc
  `~/.cache/huggingface` only.
- No automatic cross-node sync logic — MooseFS provides the shared FS.
- Optional: tiny `cache_probe` utility/test that writes+reads a marker file
  under the cache root.

### 6. Deploy docs only (no new Nomad job yet)

Update `deploy/README.md`:
- Describe `MODEL_CACHE_DIR` and MooseFS host volume `moosefs`.
- Example volume mount snippet for the existing GPU job (commented or
  optional variable) so Phase 6 can promote it.

Optionally extend `sermon-translate-gpu.nomad.hcl` with:

```hcl
variable "model_cache_dir" {
  default = "/mnt/moosefs/sermon-translate/models"
}
# volume mount moosefs → MODEL_CACHE_DIR
```

Keep it backward compatible (cache can be empty local path).

## Files Likely Touched

```
server/src/config.py
server/src/runtime/__init__.py       # NEW
server/src/runtime/model_cache.py    # NEW
server/src/runtime/local.py          # NEW
server/src/runtime/base.py           # NEW
server/src/pipelines/composed.py
server/src/pipelines/stage_registry.py
server/src/pipelines/stub_stages.py  # accept cache kw
server/src/api/deps.py
server/src/main.py                   # log cache path
server/tests/test_model_cache.py
server/tests/test_config_device.py   # or new config test
deploy/README.md
deploy/nomad/sermon-translate-gpu.nomad.hcl  # optional volume
```

## Tests

1. `ModelCache.path_for` creates parents; stays under root (reject `..` escape).
2. Settings load `MODEL_CACHE_DIR` from env.
3. `LocalStageRuntime.spawn` returns working passthrough stages.
4. Composed pipeline still passes integration tests with runtime injection.

## Exit Criteria

- [ ] `MODEL_CACHE_DIR` configurable and documented
- [ ] Stages constructed only via `StageRuntime`
- [ ] Cache path traversal-safe
- [ ] Deploy docs explain MooseFS shared cache
- [ ] Default local dev unchanged (cache under home dir)

## Out of Scope

- Actual subprocess/remote transports (Phase 5)
- Per-stage Nomad job files (Phase 6)
- Downloading production model weights in CI
