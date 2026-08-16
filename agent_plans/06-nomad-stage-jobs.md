# Phase 6 — Nomad Per-Stage Jobs

## Objective

Ship Nomad job specs so Listen / Translate / Speak (and optional Prosody) each
run as **independent services** on the cluster, with a **shared MooseFS model
cache**, while the orchestrator (existing API/WebRTC server) wires them via the
remote stage runtime.

## Motivation

One fat GPU job cannot place ASR and TTS on different GPUs or fall back Speak
to CPU (Pocket TTS) when VRAM is tight. Stage jobs match the product topology.

## Design

### 1. Job layout

```
deploy/nomad/
  sermon-translate-orchestrator.nomad.hcl   # API + WebRTC, no GPU required
  sermon-translate-stage-listen.nomad.hcl
  sermon-translate-stage-translate.nomad.hcl
  sermon-translate-stage-speak.nomad.hcl
  sermon-translate-stage-prosody.nomad.hcl  # optional; can stay in-orchestrator
```

Keep `sermon-translate-gpu.nomad.hcl` as a **monolithic legacy** path until
stages are production-default; document deprecation intent in README.

### 2. Common task shape (stage jobs)

- Driver: docker (`runtime = "nvidia"` when GPU needed)
- Command: `python -m src.runtime.worker --stage-id <id> --host 0.0.0.0 --port ${NOMAD_PORT_ws}`
- Ports: `ws` (worker protocol), optional `health`
- Service name: `sermon-stage-<kind>` or `sermon-stage-<stage_id>` with Nomad
  provider (match existing `provider = "nomad"` pattern)
- Health check: HTTP `/healthz`
- Env:
  - `MODEL_CACHE_DIR=/alloc/data/models` or mount path
  - `COMPUTE_DEVICE=cuda|cpu`
  - `STAGE_ID=...`

### 3. MooseFS model cache

Cluster already exposes host volume `moosefs` → `/mnt/moosefs` (see deploy README).

```hcl
volume "models" {
  type      = "host"
  source    = "moosefs"
  read_only = false
}

# in task:
volume_mount {
  volume      = "models"
  destination = "/models"
  read_only   = false
}

env {
  MODEL_CACHE_DIR = "/models/sermon-translate/models"
}
```

Make the subpath a variable `model_cache_subdir` defaulting to
`sermon-translate/models`.

**Cache contract:**
- All stage jobs mount the same volume+subdir.
- First pull wins; subsequent jobs reuse weights.
- Orchestrator also mounts it if it runs any local stages.

### 4. GPU placement

| Stage job | GPU | Notes |
|-----------|-----|-------|
| listen (Kyutai/Voxtral) | 1× V100 preferred | pin model like existing job |
| translate (small LLM) | 1× V100 or share | fp16 |
| speak Qwen3 / CosyVoice | 1× V100 | |
| speak Pocket TTS | CPU only | fallback job class |
| prosody baseline | CPU | can colocate with orchestrator |

Reuse constraints from `sermon-translate-gpu.nomad.hcl`:
- `meta.gpu=true`, model pin `Tesla V100-SXM2-16GB`, VRAM ≥ 16000 MiB
- Never schedule on M2000

### 5. Orchestrator discovery

Orchestrator settings:

| Env | Meaning |
|-----|---------|
| `STAGE_RUNTIME=remote` | use remote handles |
| `STAGE_SERVICE_DISCOVERY=nomad` | resolve via Nomad API |
| `NOMAD_ADDR` | e.g. `http://192.168.0.99:4646` |
| `STAGE_REMOTE_URLS` | static fallback JSON map |

Discovery implementation (minimal viable):
1. Static `STAGE_REMOTE_URLS` required for v1 (operator fills after deploy).
2. Optional helper script `deploy/scripts/resolve-stage-services.sh` prints
   current allocations/IPs for copy-paste.
3. If time permits: poll Nomad HTTP API for service `sermon-stage-*` and cache
   endpoints (read-only).

Do **not** auto-submit jobs from the app.

### 6. Images

- Reuse / split `Dockerfile.gpu`:
  - `Dockerfile.stage` slim image with worker entrypoint + torch stack
  - Or one GPU image with different commands per job
- Prefer **one image, different args** to reduce build matrix initially.

### 7. Safety

Same rules as existing deploy docs:
- Artifacts only; no automatic `nomad job run` from CI unless already practiced.
- Preflight script extended to check MooseFS mount presence on target nodes
  (read-only).

## Files Likely Touched

```
deploy/nomad/sermon-translate-orchestrator.nomad.hcl
deploy/nomad/sermon-translate-stage-*.nomad.hcl
deploy/scripts/preflight-gpu.sh          # moosefs + multi-job notes
deploy/scripts/resolve-stage-services.sh # NEW optional
deploy/README.md
server/Dockerfile.gpu                    # worker CMD examples
server/src/runtime/remote_runtime.py
server/src/config.py
```

## Tests

- HCL validates with `nomad job validate` if nomad CLI available; otherwise
  syntax-review only.
- Unit tests for URL config parsing / discovery client with mocked HTTP.
- No live cluster mutations in CI.

## Exit Criteria

- [ ] Separate Nomad job files for listen/translate/speak (+ docs)
- [ ] Shared MooseFS `MODEL_CACHE_DIR` mount documented and declared
- [ ] Orchestrator can be configured with remote stage URLs
- [ ] CPU speak fallback job documented
- [ ] Safety/preflight notes updated
- [ ] Legacy monolithic GPU job still present and documented

## Out of Scope

- Implementing full Kyutai/Qwen weights in the image (Phase 7)
- Multi-cluster federation
- Spot/preemptible scheduling policies
