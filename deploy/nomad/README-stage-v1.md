# stage.v1 Nomad packaging (Wave 6 / G6)

Declaration-only job specs for independent Listen / Translate / Speak /
Prosody workers. **No automatic `nomad job run`.** Operators validate, then
submit explicitly after preflight.

## Files

| Job file | Default `stage_id` | GPU | Health admission |
|----------|--------------------|-----|------------------|
| `sermon-translate-stage-listen.nomad.hcl` | `whisper-listen` | preferred | `/health/ready` |
| `sermon-translate-stage-translate.nomad.hcl` | `opus-mt-en-es` | preferred | `/health/ready` |
| `sermon-translate-stage-speak.nomad.hcl` | `edge-tts-es` | CPU default | `/health/ready` |
| `sermon-translate-stage-prosody.nomad.hcl` | `baseline-prosody` | CPU | `/health/ready` |

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

## Image + digest variables

Every stage job accepts:

```hcl
variable "image"        { default = "sermon-translate-server:gpu" }
variable "image_digest" { default = "" }  # e.g. "sha256:abc..."
```

When `image_digest` is non-empty the task image becomes
`${image}@${image_digest}` for immutable deploys.

```sh
nomad job validate \
  -var="image=ghcr.io/example/sermon-translate-server" \
  -var="image_digest=sha256:deadbeef..." \
  deploy/nomad/sermon-translate-stage-listen.nomad.hcl
```

## Private WSS / auth placeholders

Env (not yet enforced by worker code — placeholders for gateway wiring):

| Env | Purpose |
|-----|---------|
| `STAGE_WSS_PATH` | Private path (default `/stage/v1/ws`) |
| `STAGE_AUTH_TOKEN` | Bearer token (job var `auth_token`, sensitive) |

Job `meta.private_wss_path` mirrors the path for operators/discovery.

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
  nomad job validate "$f" || echo "nomad CLI missing or validate failed"
done
```

Preflight before any submit:

```sh
NOMAD_ADDR=http://192.168.0.99:4646 deploy/scripts/preflight-gpu.sh node-6
```

## Operator submit order (manual)

1. Preflight GPU + MooseFS mount.
2. Build/push image; record digest.
3. `nomad job run -var=image_digest=sha256:... deploy/nomad/sermon-translate-stage-listen.nomad.hcl` (etc.).
4. `bash deploy/scripts/resolve-stage-services.sh` → fill orchestrator `stage_remote_urls`.
5. Submit orchestrator with `STAGE_RUNTIME=remote`.

## E2E evidence (not in git)

Real EN→ES pre-EOS evidence lives outside the repo:

`/home/aleks/work/reviews/live-translation/implementation-runs/stage-v1-e2e-evidence/`

Integration re-runs write to:

`.../stage-v1-e2e-evidence-integration/`

Fixtures under `server/tests/fixtures/audio/` are committed; large run PCM is not.
