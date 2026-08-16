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
