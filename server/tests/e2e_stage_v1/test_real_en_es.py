"""G4 real EN→ES pre-EOS E2E tests for stage.v1.

Uses real faster-whisper + Opus-MT + edge-tts when available.
Skips honestly when models/network are missing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.stage_v1.pipeline_e2e import (
    RealModelUnavailable,
    chunk_pcm,
    create_warm_bundle,
    looks_like_english_not_spanish,
    pcm_rms,
    probe_real_stack_available,
    read_wav_pcm,
    run_pre_eos_pipeline,
    sha256_file,
    shutdown_bundle,
)

pytestmark = pytest.mark.stage_v1_real

SERVER_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = SERVER_ROOT / "fixtures" / "audio" / "public-domain-en-01.wav"
FIXTURE_MANIFEST = SERVER_ROOT / "fixtures" / "audio" / "MANIFEST.sha256.json"
EVIDENCE_ROOT = SERVER_ROOT.parent / ".evidence" / "stage-v1-real-e2e"


def _skip_reason() -> str | None:
    if os.environ.get("STAGE_V1_REAL_E2E", "1") in {"0", "false", "no"}:
        return "STAGE_V1_REAL_E2E disabled"
    ok, reason = probe_real_stack_available()
    if not ok:
        return reason
    if not FIXTURE.is_file():
        return f"fixture missing: {FIXTURE}"
    return None


@pytest.fixture(scope="module")
def real_stack_or_skip() -> None:
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)


def test_fixture_pack_present_and_digests() -> None:
    assert FIXTURE.is_file(), f"missing {FIXTURE}"
    assert FIXTURE_MANIFEST.is_file()
    notice = SERVER_ROOT / "fixtures" / "audio" / "NOTICE.md"
    assert notice.is_file()
    manifest = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["sample_format"]["codec"] == "pcm_s16le"
    assert manifest["sample_format"]["sample_rate_hz"] == 16000
    for name, meta in manifest["files"].items():
        path = SERVER_ROOT / "fixtures" / "audio" / name
        assert path.is_file(), name
        assert sha256_file(path) == meta["sha256"], name


def test_english_heuristic() -> None:
    assert looks_like_english_not_spanish(
        "In the beginning God created the heaven and the earth"
    )
    assert not looks_like_english_not_spanish(
        "En el principio creó Dios los cielos y la tierra"
    )


@pytest.mark.asyncio
async def test_real_en_es_pre_eos_two_warm_runs(real_stack_or_skip: None, tmp_path: Path) -> None:
    """Hard G4: first speak.audio before source EOS; warm loaders stay at 1."""
    os.environ.setdefault("WHISPER_MODEL_SIZE", "base")
    pcm, rate = read_wav_pcm(FIXTURE)
    assert rate == 16000
    assert len(pcm) > rate * 2  # >2s

    bundle = None
    try:
        bundle = await create_warm_bundle(
            listen_id="whisper-listen",
            translate_id="opus-mt-en-es",
            speak_id="edge-tts-es",
            sample_rate=rate,
        )
    except RealModelUnavailable as exc:
        pytest.skip(str(exc))

    assert bundle is not None
    results = []
    try:
        for i in range(2):
            result = await run_pre_eos_pipeline(
                bundle,
                pcm,
                run_index=i,
                require_first_audio_before_source_eos=True,
                max_wait_s=float(os.environ.get("STAGE_V1_E2E_MAX_WAIT_S", "180")),
            )
            results.append(result)
            if result.error:
                # Fail closed with detail (not skip) once models loaded.
                pytest.fail(result.error)
    finally:
        await shutdown_bundle(bundle)

    assert len(results) == 2
    for r in results:
        assert r.first_speak_before_source_eos, r.timeline[-5:]
        assert r.pcm_sample_count > 0
        assert r.pcm_energy_rms > 1e-4
        assert r.contiguous_frames
        assert r.target_language == "es"
        assert not r.english_spoken_as_spanish
        assert r.listen_texts, "expected non-empty listen products"
        assert r.translate_texts, "expected non-empty translate products"
        assert r.loader_counts_after["whisper"] == 1
        assert r.loader_counts_after["opus_mt"] == 1

    # Persist lightweight evidence for the pytest path too.
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    out = EVIDENCE_ROOT / "pytest-last.json"
    out.write_text(
        json.dumps(
            {
                "fixture_sha256": sha256_file(FIXTURE),
                "runs": [r.to_dict() for r in results],
                "output_rms": [r.pcm_energy_rms for r in results],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_protocol_helpers_no_models() -> None:
    """Protocol-only unit checks that always run (no GPU/network)."""
    silence = b"\x00\x00" * 1600
    assert pcm_rms(silence) == 0.0
    frames = chunk_pcm(silence, sample_rate=16000, frame_ms=20)
    assert len(frames) == 5  # 100ms / 20ms
    assert all(len(f) == 640 for f in frames)
