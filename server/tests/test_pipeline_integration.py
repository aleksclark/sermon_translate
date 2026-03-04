"""Generic integration tests for all pipelines.

Feeds real audio (the e2e fixture MP3 decoded to 48 kHz s16le PCM) into
every registered pipeline and asserts that each one produces the output
its ``output_streams`` promise: audio bytes and/or transcript text.

These tests load real models and are slow. Run with::

    uv run pytest tests/test_pipeline_integration.py -v

Skip individual heavy pipelines with standard pytest deselection if needed.
"""
from __future__ import annotations

import asyncio
import importlib
import logging
from collections.abc import AsyncIterator

import pytest

from src.pipelines.base import BasePipeline, OutputStreamKind

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Collect every pipeline that can be instantiated in this environment
# ---------------------------------------------------------------------------

def _discover_pipelines() -> list[BasePipeline]:
    """Import every known pipeline module and instantiate what's available."""
    pipelines: list[BasePipeline] = []

    # (module_path, class_name)
    known = [
        ("src.pipelines.echo", "EchoPipeline"),
        ("src.pipelines.whisper_tts", "WhisperTTSPipeline"),
        ("src.pipelines.spanish", "SpanishTranslationPipeline"),
        ("src.pipelines.spanish_direct", "SpanishDirectPipeline"),
        ("src.pipelines.seamless_streaming", "SeamlessStreamingPipeline"),
    ]

    for mod_path, cls_name in known:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            pipelines.append(cls())
        except Exception:  # noqa: BLE001
            logger.info("skipping %s.%s (import failed)", mod_path, cls_name)

    return pipelines


_ALL_PIPELINES = _discover_pipelines()


@pytest.fixture(params=_ALL_PIPELINES, ids=lambda p: p.info.id)
def pipeline(request: pytest.FixtureRequest) -> BasePipeline:
    return request.param


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _audio_stream(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


async def _collect_audio(
    pipeline: BasePipeline, chunks: list[bytes],
) -> list[bytes]:
    output: list[bytes] = []
    async for data in pipeline.process(_audio_stream(chunks)):
        output.append(data)
    return output


async def _collect_text(
    pipeline: BasePipeline, stream_name: str, chunks: list[bytes],
) -> list[str]:
    """Run process() in the background and drain the named text stream."""
    it = pipeline.iter_stream(stream_name, _audio_stream(chunks))
    if it is None:
        return []

    audio_task = asyncio.create_task(
        _drain_audio(pipeline.process(_audio_stream(chunks)))
    )

    lines: list[str] = []
    async for text in it:
        lines.append(text)

    await audio_task
    return lines


async def _drain_audio(stream: AsyncIterator[bytes]) -> None:
    async for _ in stream:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineProducesOutput:
    """Every pipeline must produce output matching its declared streams."""

    async def test_has_audio_output(
        self, pipeline: BasePipeline, fixture_pcm_chunks: list[bytes],
    ) -> None:
        has_audio_stream = any(
            s.kind == OutputStreamKind.AUDIO for s in pipeline.output_streams
        )
        if not has_audio_stream:
            pytest.skip(f"{pipeline.info.id} declares no audio stream")

        try:
            await pipeline.start()
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"{pipeline.info.id} deps not installed: {exc}")

        try:
            output = await _collect_audio(pipeline, fixture_pcm_chunks)
        finally:
            await pipeline.stop()

        total_bytes = sum(len(c) for c in output)
        logger.info(
            "%s: audio output = %d chunks, %d bytes",
            pipeline.info.id, len(output), total_bytes,
        )
        assert len(output) > 0, (
            f"{pipeline.info.id} declares audio output stream "
            f"but produced 0 audio chunks"
        )
        assert total_bytes > 0

    async def test_has_text_output(
        self, pipeline: BasePipeline, fixture_pcm_chunks: list[bytes],
    ) -> None:
        text_streams = [
            s for s in pipeline.output_streams
            if s.kind == OutputStreamKind.TEXT
        ]
        if not text_streams:
            pytest.skip(f"{pipeline.info.id} declares no text stream")

        try:
            await pipeline.start()
        except (ImportError, ModuleNotFoundError) as exc:
            pytest.skip(f"{pipeline.info.id} deps not installed: {exc}")

        try:
            for desc in text_streams:
                lines = await _collect_text(pipeline, desc.name, fixture_pcm_chunks)
                logger.info(
                    "%s[%s]: text output = %d lines: %r",
                    pipeline.info.id, desc.name, len(lines),
                    [ln[:60] for ln in lines],
                )
                assert len(lines) > 0, (
                    f"{pipeline.info.id} declares text stream '{desc.name}' "
                    f"but produced 0 lines"
                )
                assert all(isinstance(ln, str) and ln.strip() for ln in lines)
        finally:
            await pipeline.stop()
