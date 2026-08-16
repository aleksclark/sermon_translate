from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from src.stage_v1.models import (
    PAYLOAD_TYPES,
    EventEnvelope,
    EventType,
    StageModel,
)
from src.stage_v1.provenance import canonical_json_bytes, sha256_hex


def model_json_schema(model_cls: type[StageModel]) -> dict[str, Any]:
    return model_cls.model_json_schema()


def generate_schemas() -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {
        "EventEnvelope": model_json_schema(EventEnvelope),
    }
    for event_type, payload_cls in PAYLOAD_TYPES.items():
        if payload_cls is None:
            continue
        name = payload_cls.__name__
        schemas[name] = model_json_schema(payload_cls)
        schemas[f"event_type.{event_type.value}"] = {
            "event_type": event_type.value,
            "payload_schema": name,
        }
    return schemas


def write_schemas(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    schemas = generate_schemas()
    for name, schema in schemas.items():
        path = out_dir / f"{name}.json"
        path.write_bytes(canonical_json_bytes(schema) + b"\n")
        written.append(path)
    index = {
        "schema_version": "stage.v1",
        "models": sorted(k for k in schemas if not k.startswith("event_type.")),
        "event_types": sorted(et.value for et in EventType),
    }
    index_path = out_dir / "index.json"
    index_path.write_bytes(canonical_json_bytes(index) + b"\n")
    written.append(index_path)
    return written


def write_manifest(root: Path, paths: list[Path]) -> Path:
    entries: list[dict[str, str]] = []
    for path in sorted(paths, key=lambda p: str(p.relative_to(root))):
        rel = path.relative_to(root).as_posix()
        digest = sha256_hex(path.read_bytes())
        entries.append({"path": rel, "sha256": digest})
    manifest = {
        "schema_version": "stage.v1",
        "algorithm": "sha256",
        "files": entries,
    }
    manifest_path = root / "MANIFEST.sha256.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest_path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


if __name__ == "__main__":
    fixtures = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "stage_v1"
    schema_dir = fixtures / "schema"
    written = write_schemas(schema_dir)
    print(f"wrote {len(written)} schema files to {schema_dir}")
