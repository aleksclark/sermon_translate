from __future__ import annotations

import hashlib
import json
from typing import Any

from src.stage_v1.models import ProvenanceBlock


def canonical_json_bytes(value: Any) -> bytes:
    """RFC 8785-style deterministic JSON: sorted keys, no whitespace, UTF-8."""
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def provenance_id_from_block(block: ProvenanceBlock | dict[str, Any]) -> str:
    """Hash an immutable provenance block into ``sha256:<hex>``."""
    if isinstance(block, ProvenanceBlock):
        payload = block.model_dump(mode="json", exclude_none=True)
    else:
        payload = block
    digest = sha256_hex(canonical_json_bytes(payload))
    return f"sha256:{digest}"


def message_canonical_bytes(event: dict[str, Any]) -> bytes:
    return canonical_json_bytes(event)
