from __future__ import annotations

from pathlib import Path

import pytest

from src.harness.runner import RunResult, load_audio, run_pipeline
from src.pipelines.echo import EchoPipeline

FIXTURE_MP3 = Path(__file__).resolve().parent.parent.parent / "e2e" / "fixtures" / "test-speech.mp3"


class TestLoadAudio:
    def test_loads_chunks(self) -> None:
        if not FIXTURE_MP3.exists():
            pytest.skip("fixture MP3 not found")
        chunks = load_audio(FIXTURE_MP3)
        assert len(chunks) > 0
        assert all(isinstance(c, bytes) for c in chunks)

    def test_chunk_size(self) -> None:
        if not FIXTURE_MP3.exists():
            pytest.skip("fixture MP3 not found")
        chunks = load_audio(FIXTURE_MP3)
        full_chunks = chunks[:-1]
        for c in full_chunks:
            assert len(c) == 960 * 2  # 20ms at 48kHz, s16le


class TestRunPipeline:
    async def test_echo_pipeline(self) -> None:
        pipeline = EchoPipeline()
        chunks = [b"\x00\x00" * 960] * 5
        result = await run_pipeline(pipeline, chunks)
        assert isinstance(result, RunResult)
        assert result.pipeline_id == "echo"
        assert result.error is None
        assert result.wall_seconds > 0
        assert result.audio_chunks_out == 5
        assert result.audio_bytes_out == 5 * 960 * 2

    async def test_records_first_audio_time(self) -> None:
        pipeline = EchoPipeline()
        chunks = [b"\x00\x00" * 960] * 3
        result = await run_pipeline(pipeline, chunks)
        assert result.first_audio_seconds is not None
        assert result.first_audio_seconds > 0

    async def test_empty_chunks(self) -> None:
        pipeline = EchoPipeline()
        result = await run_pipeline(pipeline, [])
        assert result.audio_chunks_out == 0
        assert result.error is None

    async def test_reports_audio_duration(self) -> None:
        pipeline = EchoPipeline()
        chunks = [b"\x00\x00" * 960] * 50  # 50 * 20ms = 1s
        result = await run_pipeline(pipeline, chunks)
        assert abs(result.audio_duration_seconds - 1.0) < 0.01
