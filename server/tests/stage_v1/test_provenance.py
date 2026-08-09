"""Provenance block hashing and host wiring."""

from __future__ import annotations

import pytest

from src.stage_v1.host import StageHost
from src.stage_v1.models import ArtifactDigestStatus, ProvenanceBlock, StageKind
from src.stage_v1.provenance import (
    canonical_json_bytes,
    provenance_id_from_block,
    sha256_hex,
)


def test_provenance_id_stable_for_equivalent_blocks() -> None:
    block = ProvenanceBlock(
        stage_id="whisper-listen",
        stage_version="1.0.0",
        code_git_sha="b67c697a11dcd27568aec7e4e5f946818d718f70",
        container_image_digest=None,
        model_provider_id="faster-whisper",
        model_revision="base",
        model_artifact_digest="sha256:" + "11" * 32,
        model_artifact_status=ArtifactDigestStatus.VERIFIED,
        runtime_versions={"python": "3.12", "torch": "2.0"},
        boot_id="boot-1",
    )
    a = provenance_id_from_block(block)
    b = provenance_id_from_block(block.model_dump(mode="json", exclude_none=True))
    assert a == b
    assert a.startswith("sha256:")
    # Recompute manually
    payload = block.model_dump(mode="json", exclude_none=True)
    expected = f"sha256:{sha256_hex(canonical_json_bytes(payload))}"
    assert a == expected


def test_provenance_id_changes_when_field_changes() -> None:
    base = {
        "stage_id": "opus-mt-en-es",
        "stage_version": "1.0.0",
        "code_git_sha": "aaa",
        "model_provider_id": "huggingface",
        "model_revision": "Helsinki-NLP/opus-mt-en-es",
        "model_artifact_digest": "sha256:" + "22" * 32,
        "model_artifact_status": "verified",
        "boot_id": "boot-x",
    }
    p1 = provenance_id_from_block(ProvenanceBlock.model_validate(base))
    changed = {**base, "model_revision": "other-rev"}
    p2 = provenance_id_from_block(ProvenanceBlock.model_validate(changed))
    assert p1 != p2


def test_key_order_does_not_affect_hash() -> None:
    block = ProvenanceBlock(
        stage_id="s",
        stage_version="1",
        code_git_sha="c",
        model_provider_id="p",
        model_revision="r",
        model_artifact_digest="sha256:" + "33" * 32,
        model_artifact_status=ArtifactDigestStatus.PROVIDER_MANAGED,
        runtime_versions={"b": "2", "a": "1"},
        boot_id="boot",
    )
    d1 = block.model_dump(mode="json", exclude_none=True)
    # Shuffle top-level insertion order
    d2 = {
        "boot_id": d1["boot_id"],
        "stage_id": d1["stage_id"],
        "model_revision": d1["model_revision"],
        "code_git_sha": d1["code_git_sha"],
        "model_artifact_status": d1["model_artifact_status"],
        "model_provider_id": d1["model_provider_id"],
        "stage_version": d1["stage_version"],
        "model_artifact_digest": d1["model_artifact_digest"],
        "runtime_versions": {"a": "1", "b": "2"},
    }
    assert provenance_id_from_block(d1) == provenance_id_from_block(d2)


@pytest.mark.asyncio
async def test_host_builds_provenance_after_load() -> None:
    host = StageHost(
        stage_kind=StageKind.SPEAK,
        stage_id="pocket-tts",
        stage_version="0.2.0",
        model_loader=lambda: object(),
        code_git_sha="gitsha",
        model_provider_id="pocket",
        model_revision="24l",
        model_artifact_digest="sha256:" + "44" * 32,
        model_artifact_status=ArtifactDigestStatus.VERIFIED,
        runtime_versions={"pocket_tts": "0.1.0"},
        voice_digest="sha256:" + "55" * 32,
        boot_id="boot-prov",
        local_dev=True,
    )
    assert host.provenance is None
    await host.load()
    assert host.provenance is not None
    assert host.provenance_id == provenance_id_from_block(host.provenance)
    assert host.provenance.stage_id == "pocket-tts"
    assert host.provenance.boot_id == "boot-prov"
    assert host.provenance.voice_digest == "sha256:" + "55" * 32
    await host.warmup()
    # Warmup rebuild keeps stable id for same fields
    assert host.provenance_id == provenance_id_from_block(host.provenance)


@pytest.mark.asyncio
async def test_unavailable_digest_status_still_hashes_but_is_not_promotable_marker() -> None:
    """Unavailable is allowed for local hashing; promotion policy is external."""
    block = ProvenanceBlock(
        stage_id="x",
        stage_version="0.0.1",
        code_git_sha="dev",
        model_provider_id="dev",
        model_revision="latest",
        model_artifact_digest="unavailable",
        model_artifact_status=ArtifactDigestStatus.UNAVAILABLE,
        boot_id="b",
    )
    pid = provenance_id_from_block(block)
    assert pid.startswith("sha256:")
    assert block.model_artifact_status == ArtifactDigestStatus.UNAVAILABLE
