#!/usr/bin/env bash
# Regenerate deploy/stage-v1/conformance-bundle and integration-manifest.json
# from source fixtures / schema_gen with deterministic sha256 digests.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

BUNDLE_DIR="$ROOT/deploy/stage-v1/conformance-bundle"
MANIFEST_OUT="$ROOT/deploy/stage-v1/integration-manifest.json"
FIXTURE_ROOT="$ROOT/server/tests/fixtures/stage_v1"
AUDIO_FIXTURE_ROOT="$ROOT/server/tests/fixtures/audio"
SCHEMA_GEN="$ROOT/server/src/stage_v1/schema_gen.py"

sha256_file() {
  sha256sum -b "$1" | awk '{print $1}'
}

rel_sha() {
  # print "path sha256" for a file relative to ROOT
  local abs="$1"
  local rel="${abs#"$ROOT"/}"
  printf '%s %s\n' "$rel" "$(sha256_file "$abs")"
}

echo "== stage.v1 conformance bundle builder =="
echo "repo_root=$ROOT"

# --- optional schema refresh via schema_gen (source of truth) ---
SCHEMA_REFRESHED=0
if [[ "${SKIP_SCHEMA_GEN:-0}" != "1" ]]; then
  if command -v uv >/dev/null 2>&1 && [[ -d "$ROOT/server" ]]; then
    echo "Refreshing JSON Schema fixtures via schema_gen..."
    (
      cd "$ROOT/server"
      # schema_gen writes into tests/fixtures/stage_v1/schema
      if uv run python -m src.stage_v1.schema_gen; then
        SCHEMA_REFRESHED=1
      else
        echo "WARN: uv run schema_gen failed; using committed schema fixtures" >&2
      fi
    ) || true
  else
    echo "uv not available; using committed schema fixtures"
  fi
fi

# Rebuild fixture MANIFEST.sha256.json from on-disk files (canonical sorted)
python3 - <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path

root = Path("server/tests/fixtures/stage_v1")
entries = []
for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    if path.name == "MANIFEST.sha256.json":
        continue
    rel = path.relative_to(root).as_posix()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entries.append({"path": rel, "sha256": digest})

manifest = {
    "schema_version": "stage.v1",
    "algorithm": "sha256",
    "files": entries,
}
# RFC8785-ish: sorted keys, compact
text = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
(root / "MANIFEST.sha256.json").write_text(text, encoding="utf-8")
print(f"updated fixture MANIFEST with {len(entries)} files")
PY

# --- pack conformance bundle ---
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/schema" "$BUNDLE_DIR/json" "$BUNDLE_DIR/binary" "$BUNDLE_DIR/audio"

# Copy wire schemas + golden JSON/binary fixtures
cp -a "$FIXTURE_ROOT/schema/." "$BUNDLE_DIR/schema/"
cp -a "$FIXTURE_ROOT/json/." "$BUNDLE_DIR/json/"
cp -a "$FIXTURE_ROOT/binary/." "$BUNDLE_DIR/binary/"

# Include protocol doc + models for consumers
cp -a "$ROOT/docs/protocol/stage-v1.md" "$BUNDLE_DIR/stage-v1.md"
cp -a "$ROOT/server/src/stage_v1/models.py" "$BUNDLE_DIR/models.py"
cp -a "$FIXTURE_ROOT/MANIFEST.sha256.json" "$BUNDLE_DIR/fixtures-MANIFEST.sha256.json"

# Audio golden fixtures (public-domain + synthetic) for E2E consumers
if [[ -d "$AUDIO_FIXTURE_ROOT" ]]; then
  # copy audio files + NOTICE + MANIFEST (skip nothing material)
  for f in "$AUDIO_FIXTURE_ROOT"/*; do
    base="$(basename "$f")"
    cp -a "$f" "$BUNDLE_DIR/audio/$base"
  done
fi

# README for consumers
cat > "$BUNDLE_DIR/README.md" << 'EOF'
# stage.v1 portable conformance bundle

Machine-consumable golden artifacts for **canonical sermon-translate** (and
compatible stage.v1 peers) to validate wire protocol framing, JSON payloads,
JSON Schema, and public-domain audio fixtures.

## Contents

| Path | Purpose |
|------|---------|
| `schema/` | JSON Schema generated from Pydantic models (`schema_gen.py`) |
| `json/` | Golden JSON events (hello, listen.product, error, …) |
| `binary/` | STG1 binary audio frames + meta sidecars |
| `audio/` | 16 kHz mono PCM/WAV fixtures + `MANIFEST.sha256.json` |
| `stage-v1.md` | Protocol document snapshot |
| `models.py` | Pydantic model source snapshot |
| `fixtures-MANIFEST.sha256.json` | Source fixture root digests |
| `bundle-manifest.json` | sha256 of every file in this bundle |

## Consumer usage (canonical sermon-translate)

1. **Pin the bundle** by `bundle-manifest.json` digest (or the git SHA recorded
   in `deploy/stage-v1/integration-manifest.json`).
2. **Schema validate** outbound/inbound JSON events against `schema/*.json`
   (start from `schema/EventEnvelope.json` + `schema/event_type.*.json`).
3. **Golden compare** framing:
   - decode `binary/*.stg1` with STG1 framing;
   - expect meta fields in the matching `*.meta.json`.
4. **JSON goldens** under `json/` must round-trip through the peer codec without
   dropping required fields (additive unknown fields OK per protocol).
5. **Audio**: feed `audio/public-domain-en-01.wav` (or `.pcm`) into Listen at
   16 kHz s16le mono; use Spanish references in `audio/MANIFEST.sha256.json`
   only as soft checks (ASR/MT nondeterminism).

### Minimal Python check

```python
import hashlib, json
from pathlib import Path
bundle = Path("deploy/stage-v1/conformance-bundle")
man = json.loads((bundle / "bundle-manifest.json").read_text())
for entry in man["files"]:
    data = (bundle / entry["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
print("bundle integrity OK", man["file_count"], "files")
```

### Regenerate

From repo root:

```bash
./deploy/stage-v1/build-conformance-bundle.sh
```

Requires `sha256sum` and `python3`. Schema refresh uses `uv run` inside
`server/` when available (`SKIP_SCHEMA_GEN=1` to force committed schemas).

## Auth / stream contract (for integration tests)

- Stream: `GET /stage/v1/stream` with WebSocket subprotocol `stage.v1`
- Production (`STAGE_V1_MODE=production`): require wss/TLS (or trusted
  `X-Forwarded-Proto` when `STAGE_TRUST_PROXY=1`); non-empty `STAGE_AUTH_TOKEN`
  via `Authorization: Bearer` or `X-Stage-Auth` on the **upgrade only**
- No credentials in hello payload or URL query
- Health admission: `/health/ready` (not `/healthz` alone)

See `deploy/stage-v1/integration-manifest.json` for full code digests and model
identity claims.
EOF

# bundle-manifest.json over every file except itself
python3 - <<'PY'
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

bundle = Path("deploy/stage-v1/conformance-bundle")
entries = []
for path in sorted(bundle.rglob("*")):
    if not path.is_file():
        continue
    if path.name == "bundle-manifest.json":
        continue
    rel = path.relative_to(bundle).as_posix()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entries.append({"path": rel, "sha256": digest, "bytes": path.stat().st_size})

# Deterministic timestamp from git HEAD committer date (UTC).
import subprocess
git_ts = subprocess.check_output(
    ["git", "log", "-1", "--format=%cI", "HEAD"],
    text=True,
).strip()
# Normalize to Zulu if offset present
from datetime import datetime as _dt
try:
    _d = _dt.fromisoformat(git_ts.replace("Z", "+00:00"))
    generated_at = _d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
except Exception:
    generated_at = git_ts

manifest = {
    "schema_version": "stage.v1",
    "bundle": "stage.v1-conformance",
    "algorithm": "sha256",
    "generated_at": generated_at,
    "source_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "file_count": len(entries),
    "files": entries,
}
text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
(bundle / "bundle-manifest.json").write_text(text, encoding="utf-8")
print(f"wrote bundle-manifest.json ({len(entries)} files)")
PY

# --- integration-manifest.json ---
GIT_SHA="$(git rev-parse HEAD)"
GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
GENERATED_AT="$(git log -1 --format=%cI HEAD | python3 -c 'import sys; from datetime import datetime, timezone; s=sys.stdin.read().strip().replace("Z","+00:00"); print(datetime.fromisoformat(s).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))')"

python3 - <<PY
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

root = Path(".").resolve()
git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()

def sha(p: str | Path) -> str:
    path = root / p
    return hashlib.sha256(path.read_bytes()).hexdigest()

def file_entry(path: str) -> dict:
    p = root / path
    return {
        "path": path,
        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "bytes": p.stat().st_size,
    }

code_files = [
    "server/src/stage_v1/models.py",
    "server/src/stage_v1/framing.py",
    "server/src/stage_v1/auth.py",
    "server/src/stage_v1/server.py",
    "server/src/stage_v1/health.py",
    "server/src/stage_v1/host.py",
    "server/src/stage_v1/adapters.py",
    "server/src/stage_v1/provenance.py",
    "server/src/stage_v1/schema_gen.py",
]

fixture_manifest = json.loads((root / "server/tests/fixtures/stage_v1/MANIFEST.sha256.json").read_text())
audio_manifest = json.loads((root / "server/tests/fixtures/audio/MANIFEST.sha256.json").read_text())
bundle_manifest = json.loads((root / "deploy/stage-v1/conformance-bundle/bundle-manifest.json").read_text())

# Expand fixture digests keyed by category
fixture_files = {e["path"]: e["sha256"] for e in fixture_manifest["files"]}
schema_files = {k: v for k, v in fixture_files.items() if k.startswith("schema/")}
json_files = {k: v for k, v in fixture_files.items() if k.startswith("json/")}
binary_files = {k: v for k, v in fixture_files.items() if k.startswith("binary/")}

audio_files = {
    name: meta["sha256"] if isinstance(meta, dict) else meta
    for name, meta in audio_manifest.get("files", {}).items()
}

nomad_jobs = [
    "deploy/nomad/sermon-translate-stage-listen.nomad.hcl",
    "deploy/nomad/sermon-translate-stage-translate.nomad.hcl",
    "deploy/nomad/sermon-translate-stage-speak.nomad.hcl",
    "deploy/nomad/sermon-translate-stage-prosody.nomad.hcl",
    "deploy/nomad/sermon-translate-stage-canary.nomad.hcl",
]
nomad_jobs = [j for j in nomad_jobs if (root / j).is_file()]

manifest = {
    "schema_version": "stage.v1",
    "kind": "integration-manifest",
    "generated_at": "${GENERATED_AT}",
    "code": {
        "git_sha": git_sha,
        "git_branch": branch,
        "note": "Recompute with git rev-parse HEAD before release pin; may advance if other workers commit first.",
    },
    "protocol": {
        **file_entry("docs/protocol/stage-v1.md"),
        "title": "stage.v1 wire protocol",
    },
    "pydantic_models": file_entry("server/src/stage_v1/models.py"),
    "implementation": {Path(p).name: file_entry(p) for p in code_files},
    "fixtures": {
        "root": "server/tests/fixtures/stage_v1/",
        "manifest": file_entry("server/tests/fixtures/stage_v1/MANIFEST.sha256.json"),
        "schema": schema_files,
        "json": json_files,
        "binary": binary_files,
        "audio": {
            "root": "server/tests/fixtures/audio/",
            "manifest": file_entry("server/tests/fixtures/audio/MANIFEST.sha256.json"),
            "files": audio_files,
            "sample_format": audio_manifest.get("sample_format"),
        },
    },
    "auth_tls_contract": {
        "production_mode_env": "STAGE_V1_MODE=production",
        "requires_wss_or_tls": True,
        "trust_proxy_env": "STAGE_TRUST_PROXY=1",
        "trust_proxy_header": "X-Forwarded-Proto",
        "trust_proxy_accepted_values": ["https", "wss"],
        "auth_token_env": "STAGE_AUTH_TOKEN",
        "auth_token_transport": [
            "Authorization: Bearer <token>",
            "X-Stage-Auth: <token>",
        ],
        "auth_on_upgrade_only": True,
        "credentials_forbidden_in_hello": True,
        "credentials_forbidden_in_url": True,
        "empty_token_fails_boot": True,
        "empty_token_boot_error": (
            "STAGE_V1_MODE=production requires non-empty STAGE_AUTH_TOKEN "
            "(fail-closed: refuse to start without workload credentials)"
        ),
        "subprotocol": "stage.v1",
        "notes": [
            "authorize_stage_upgrade runs before accept/open_session/model bind",
            "dev/test may allow loopback credential-free upgrades",
            "Auth failure sends AUTHENTICATION_FAILED then close; no open_session",
        ],
    },
    "health": {
        "live": "/health/live",
        "startup": "/health/startup",
        "ready_admission": "/health/ready",
        "liveness_alias_only": "/healthz",
        "admission_rule": "Service checks must use /health/ready; never admit on /healthz alone",
    },
    "stream": {
        "method": "GET",
        "path": "/stage/v1/stream",
        "subprotocol": "stage.v1",
        "binary_framing": "STG1",
        "legacy_compat_path": "/ws",
        "legacy_note": "WorkerMessage JSON+base64 compat only; prefer /stage/v1/stream",
    },
    "model_identities": [
        {
            "stage_kind": "listen",
            "stage_id": "whisper-listen",
            "provider": "faster-whisper",
            "model_revision_env": "WHISPER_MODEL_SIZE",
            "model_revision_default": "base",
            "model_artifact_digest": "unavailable",
            "model_artifact_status": "unavailable",
            "promotion_blocked_if_unavailable": True,
            "adapter": "build_whisper_listen_host",
            "session_binder": "open_whisper_session_stage",
            "availability": "required",
        },
        {
            "stage_kind": "translate",
            "stage_id": "opus-mt-en-es",
            "provider": "helsinki-nlp",
            "model_id_default": "Helsinki-NLP/opus-mt-en-es",
            "model_revision_env": "TRANSLATE_MODEL_ID",
            "model_artifact_digest": "unavailable",
            "model_artifact_status": "unavailable",
            "promotion_blocked_if_unavailable": True,
            "adapter": "build_opus_mt_host",
            "session_binder": "open_opus_mt_session_stage",
            "availability": "required",
        },
        {
            "stage_kind": "speak",
            "stage_id": "edge-tts-es",
            "provider": "edge-tts",
            "voice": "es-ES-AlvaroNeural",
            "model_revision": "edge-tts:es-ES-AlvaroNeural",
            "model_artifact_digest": "provider_managed:edge-tts:es-ES-AlvaroNeural",
            "model_artifact_status": "provider_managed",
            "promotion_blocked_if_unavailable": True,
            "streams_pcm": False,
            "adapter": "build_edge_tts_host",
            "session_binder": "open_edge_tts_session_stage",
            "availability": "required",
            "note": "Network voice; no resident weights. Digest is provider-managed revision tag, not a content hash.",
        },
        {
            "stage_kind": "speak",
            "stage_id": "pocket-tts-spanish-24l",
            "provider": "pocket-tts",
            "model_revision_default": "pocket-tts:spanish_24l:lola",
            "model_artifact_digest": "unavailable",
            "model_artifact_status": "unavailable",
            "promotion_blocked_if_unavailable": True,
            "adapter": "build_pocket_tts_host",
            "session_binder": "open_pocket_tts_session_stage",
            "availability": "conditional",
            "install": "uv sync --extra tts-pocket",
            "detect": "import pocket_tts",
            "unavailable_if_not_installed": True,
            "note": "CONDITIONAL PROCEED with edge-tts when pocket-tts is absent; do not fake pocket.",
        },
    ],
    "honest_artifact_digest_policy": {
        "model_artifact_digest_may_be": [
            "unavailable",
            "provider_managed:<provider>:<revision>",
        ],
        "promotion_blocked_if_unavailable": True,
        "rationale": (
            "Local weights may lack a pinned content digest at boot; network voices "
            "are provider-managed. Promotion/release gates must fail closed when a "
            "required digest is unavailable rather than inventing one."
        ),
    },
    "image": {
        "registry": None,
        "repository": None,
        "tag": None,
        "digest": None,
        "note": "Placeholders for image worker — fill registry/repo/tag/digest after build/push. Nomad jobs accept image + image_digest vars.",
    },
    "nomad": {
        "readme": "deploy/nomad/README-stage-v1.md",
        "jobs": [file_entry(p) for p in nomad_jobs],
        "admission_health_path": "/health/ready",
        "validate_only": True,
        "note": "Declaration packaging only — do not nomad job run from integration automation.",
    },
    "conformance_bundle": {
        "path": "deploy/stage-v1/conformance-bundle/",
        "build_script": "deploy/stage-v1/build-conformance-bundle.sh",
        "bundle_manifest": file_entry("deploy/stage-v1/conformance-bundle/bundle-manifest.json"),
        "file_count": bundle_manifest["file_count"],
        "bundle_manifest_sha256": sha("deploy/stage-v1/conformance-bundle/bundle-manifest.json"),
    },
}

out = root / "deploy/stage-v1/integration-manifest.json"
out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"wrote {out}")
print(f"git_sha={git_sha}")
print(f"protocol_sha256={manifest['protocol']['sha256']}")
print(f"models_sha256={manifest['pydantic_models']['sha256']}")
print(f"bundle_files={bundle_manifest['file_count']}")
print(f"bundle_manifest_sha256={manifest['conformance_bundle']['bundle_manifest_sha256']}")
PY

echo "== done =="
echo "bundle: $BUNDLE_DIR"
echo "manifest: $MANIFEST_OUT"
