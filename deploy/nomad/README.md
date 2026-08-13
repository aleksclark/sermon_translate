# Nomad deployment contract (sermon-translate) — **normalized disabled**

Plan 03 schema version 1 contract for `aleksclark/sermon_translate`.

## Status

| Item | Value |
|---|---|
| S-state | **normalized-disabled** (source-only; no live jobs) |
| Fleet source | **must stay `enabled: false` / mode `disabled`** |
| Live Nomad submit | **forbidden** from this repo and from reconciler until separate approval |
| Training batch | **never dispatched by reconciliation** (`type=batch`, manual only) |

This phase (I31 / plan 03 P7) only lands the contract on the default branch.
It does **not** start workloads, enroll fleet sources, or claim runtime migration.

## Layout

```text
deploy/nomad/
├── deployment.yaml              # plan 03 manifest (two release sets)
├── jobs/<job-id>.nomad.hcl      # seven jobspecs
├── env/home.nomadvars.hcl       # non-secret overlay
├── images.lock.hcl              # digest-only (placeholder digests until publish)
├── README.md
└── tests/
    ├── contract.sh              # L1 static contract
    └── expected-services.json
.github/CODEOWNERS
```

## Release sets

1. **`sermon-translate-runtime`** (serial): gpu, orchestrator, stage-listen,
   stage-prosody, stage-speak, stage-translate.
2. **`sermon-translate-train`** (serial, separate): batch train job only —
   register-eligible later, **never** reconciler-dispatched.

## Images

Production images are not yet published as fleet digests. `images.lock.hcl`
uses placeholder `sha256:0000…` digests so the contract rejects `:latest` /
floating tags as authority. Replace with real digests before any enablement.

Jobspecs take `image` + `image_digest` variables and compose
`image@sha256:…` via `local.resolved_image`. Empty image defaults prevent
accidental local-tag authority.

## Secrets (key names only)

Nomad Variable paths (create before any future enablement):

- `nomad/jobs/sermon-translate-gpu`
- `nomad/jobs/sermon-translate-orchestrator`
- `nomad/jobs/sermon-translate-stage-listen`
- `nomad/jobs/sermon-translate-stage-prosody`
- `nomad/jobs/sermon-translate-stage-speak`
- `nomad/jobs/sermon-translate-stage-translate`
- `nomad/jobs/sermon-translate-train`

Typical keys (values never in git): stage auth tokens, TURN username/credential
where used. Pass via Nomad Variables / `-var` at operator submit time — never
commit secrets.

## Workstation / local-driver notes

- Default train host volume is `moosefs` (not workstation `local-data`).
- GPU jobs still document node-6 V100 constraints; preflight remains operator-side.
- No raw host path preferred when a Nomad host volume can express the mount.

## Local validation (no cluster write)

```bash
bash deploy/nomad/tests/contract.sh
# optional:
# nomad fmt -check deploy/nomad/jobs
# nomad job run -output -var=image=… -var=image_digest=sha256:… <spec> >/dev/null
```

## Out of scope here

- Fleet source catalog enrollment
- Live `nomad job run` / reconciler apply
- Real image publish / digest pin PR
- Training dispatch automation
