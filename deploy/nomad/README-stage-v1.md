# stage.v1 Nomad packaging (Wave 6 / G6 + immutable worker image)

Declaration-only job specs for independent Listen / Translate / Speak /
Prosody workers plus a **private listen canary**. **No automatic
`nomad job run`.** Operators validate, then submit explicitly after preflight.

## Files

| Job file | Default `stage_id` | GPU | Health admission | Public edge |
|----------|--------------------|-----|------------------|-------------|
| `sermon-translate-stage-listen.nomad.hcl` | `whisper-listen` | preferred | `/health/ready` | no (fleet) |
| `sermon-translate-stage-translate.nomad.hcl` | `opus-mt-en-es` | preferred | `/health/ready` | no (fleet) |
| `sermon-translate-stage-speak.nomad.hcl` | `edge-tts-es` | CPU default | `/health/ready` | no (fleet) |
| `sermon-translate-stage-prosody.nomad.hcl` | `baseline-prosody` | CPU | `/health/ready` | no (fleet) |
| `sermon-translate-stage-canary.nomad.hcl` | `whisper-listen` | preferred (`gpu_mode=device`) | `/health/ready` | **PRIVATE only** |

Also see orchestrator + legacy monolith in this directory.

## Health model

stage.v1 workers expose:

| Path | Meaning |
|------|---------|
| `/health/live` | Process up |
| `/health/startup` | Model load finished (or failed) |
| `/health/ready` | Model warm + canary OK; safe for new sessions |
| `/healthz` | Liveness alias only (compat) |

Service checks use **`/health/ready`** for admission and `/health/live` as a
secondary liveness probe. Do not route traffic on `/healthz` alone.

Live WebSocket path is **`/stage/v1/stream`** (not `/stage/v1/ws`).

## Immutable stage-worker image

Lean CPU-capable worker image (build context = `server/`):

| Item | Value |
|------|-------|
| Dockerfile | `server/Dockerfile.stage-worker` |
| Default registry | `997533895598.dkr.ecr.us-east-2.amazonaws.com/sermon-translate-stage-worker` |
| Entrypoint | `python -m src.runtime.worker` (via `uv run --no-sync`) |
| GPU variant | `server/Dockerfile.gpu` (CUDA; optional seamless extra) |

### Build + pin digest

```sh
# From repo root — tags image with full git SHA, writes iidfile, prints pin values
./deploy/scripts/build-stage-worker-image.sh

# Push and resolve registry RepoDigest
PUSH=1 ./deploy/scripts/build-stage-worker-image.sh

# Manual equivalent:
#   docker build -f server/Dockerfile.stage-worker \
#     --iidfile /tmp/stage-worker.iid \
#     -t "$REGISTRY:$(git rev-parse HEAD)" server/
#   docker push "$REGISTRY:$(git rev-parse HEAD)"
#   docker inspect --format='{{index .RepoDigests 0}}' "$REGISTRY:$(git rev-parse HEAD)"
```

Script env:

| Env | Default / purpose |
|-----|-------------------|
| `REGISTRY` | ECR path above |
| `TAG` | full `git rev-parse HEAD` |
| `PUSH` | `0` — set `1` to push |
| `UV_SYNC_EXTRA_ARGS` | optional extras (e.g. `--extra tts-pocket`) |

Prefer Nomad `image` + `image_digest` (immutable) over floating tags.

## Image + digest + auth variables

Every stage job accepts:

```hcl
variable "image"          { /* registry path without digest */ }
variable "image_digest"   { default = "" }  # e.g. "sha256:abc..."
variable "auth_token"     { /* → STAGE_AUTH_TOKEN; pass via -var only */ }
variable "wss_path"       { default = "/stage/v1/stream" }
variable "stage_v1_mode"  { default = "dev" }   # canary defaults to "production"
variable "trust_proxy"    { default = false }   # canary defaults to true
```

When `image_digest` is non-empty the task image becomes
`${image}@${image_digest}` for immutable deploys.

**Never commit tokens.** Pass secrets only via CLI `-var` / `-var-file` that is
gitignored, or the Nomad UI variable UI.

```sh
nomad job validate \
  -var="image=997533895598.dkr.ecr.us-east-2.amazonaws.com/sermon-translate-stage-worker" \
  -var="image_digest=sha256:deadbeef..." \
  -var="auth_token=${STAGE_AUTH_TOKEN}" \
  -var="stage_v1_mode=production" \
  -var="trust_proxy=true" \
  deploy/nomad/sermon-translate-stage-listen.nomad.hcl
```

## Auth / transport env (enforced by worker)

| Env | Purpose |
|-----|---------|
| `STAGE_WSS_PATH` | Documented private path (default `/stage/v1/stream`) |
| `STAGE_AUTH_TOKEN` | Bearer / `X-Stage-Auth` workload token (job var `auth_token`) |
| `STAGE_V1_MODE` | `production` fail-closed; `dev`/`test` allow loopback |
| `STAGE_TRUST_PROXY` | Honor `X-Forwarded-Proto` from trusted internal proxies |

Production mode refuses to boot without a non-empty `STAGE_AUTH_TOKEN`.

Job `meta.private_wss_path` mirrors the path for operators/discovery.

## Private canary job

`sermon-translate-stage-canary.nomad.hcl`:

- Job: `sermon-translate-stage-canary`, **count=1**
- Service: **`sermon-stage-canary-listen`** (Nomad provider only)
- Stage: `whisper-listen` (one stage_id per worker process)
- `meta.canary=true`, tags `canary` / `private` / `stage.v1`
- **No** Traefik router tags, **no** Cloudflare public hostname
- Default `stage_v1_mode=production`, `trust_proxy=true`, `gpu_mode=device`
- Optional `-var=gpu_mode=cpu` if VRAM is contended; optional `-var=node_name=node-6`

Validate / submit examples (**declaration path only — orchestrator deploys**):

```sh
# Validate (dummy token only for schema; production submit uses real secret from env)
nomad job validate \
  -var="image=997533895598.dkr.ecr.us-east-2.amazonaws.com/sermon-translate-stage-worker" \
  -var="image_digest=sha256:REPLACE_AFTER_PUSH" \
  -var="auth_token=${STAGE_AUTH_TOKEN:?set STAGE_AUTH_TOKEN in the environment}" \
  -var="trust_proxy=true" \
  deploy/nomad/sermon-translate-stage-canary.nomad.hcl

# Operator submit AFTER preflight + image push (not CI):
# NOMAD_ADDR=http://192.168.0.99:4646 deploy/scripts/preflight-gpu.sh node-6
# nomad job run \
#   -var="image=997533895598.dkr.ecr.us-east-2.amazonaws.com/sermon-translate-stage-worker" \
#   -var="image_digest=sha256:..." \
#   -var="auth_token=${STAGE_AUTH_TOKEN}" \
#   -var="trust_proxy=true" \
#   deploy/nomad/sermon-translate-stage-canary.nomad.hcl
```

CPU fallback canary:

```sh
nomad job validate \
  -var="auth_token=${STAGE_AUTH_TOKEN}" \
  -var="gpu_mode=cpu" \
  deploy/nomad/sermon-translate-stage-canary.nomad.hcl
```

## Warm model replacement notes (D6)

| Stage | Adapter factory | Session binder | Load once |
|-------|-----------------|----------------|-----------|
| listen | `build_whisper_listen_host` | `open_whisper_session_stage` | faster-whisper weights |
| translate | `build_opus_mt_host` | `open_opus_mt_session_stage` | Opus-MT CT2 |
| speak (edge) | `build_edge_tts_host` | `open_edge_tts_session_stage` | no resident weights |
| speak (pocket) | `build_pocket_tts_host` | `open_pocket_tts_session_stage` | optional extra |

Session close clears decoder/stream/voice state only. Process exit /
`StageHost.shutdown()` unloads weights.

Defaults moved off passthrough stubs:

- listen: `whisper-listen` (was `passthrough-listen`)
- translate: `opus-mt-en-es` (was `passthrough-translate`)
- speak: `edge-tts-es` + `gpu_mode=cpu` (was `passthrough-speak` + GPU)

## Validation (no deploy)

```sh
# From repo root
for f in deploy/nomad/sermon-translate-stage-*.nomad.hcl; do
  echo "== $f =="
  nomad job validate -var="auth_token=validate-only-not-a-secret" "$f" \
    || echo "nomad CLI missing or validate failed"
done
```

Preflight before any submit:

```sh
NOMAD_ADDR=http://192.168.0.99:4646 deploy/scripts/preflight-gpu.sh node-6
```

## Operator submit order (manual)

1. Preflight GPU + MooseFS mount.
2. Build/push stage-worker image; record **registry** digest (`RepoDigests`).
3. Submit canary first (private), confirm `/health/ready`.
4. `nomad job run -var=image=... -var=image_digest=sha256:... -var=auth_token="$STAGE_AUTH_TOKEN" deploy/nomad/sermon-translate-stage-listen.nomad.hcl` (etc.).
5. `bash deploy/scripts/resolve-stage-services.sh` → fill orchestrator `stage_remote_urls`.
6. Submit orchestrator with `STAGE_RUNTIME=remote`.

## E2E evidence (not in git)

Real EN→ES pre-EOS evidence lives outside the repo:

`/home/aleks/work/reviews/live-translation/implementation-runs/stage-v1-e2e-evidence/`

Integration re-runs write to:

`.../stage-v1-e2e-evidence-integration/`

Fixtures under `server/tests/fixtures/audio/` are committed; large run PCM is not.
