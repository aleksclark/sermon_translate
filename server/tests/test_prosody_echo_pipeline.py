from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np

from src.models import MetadataKind, Session
from src.pipelines import ProsodyEchoPipeline, create_default_registry


def _pcm(samples: np.ndarray) -> bytes:
    return (samples * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


async def _stream(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


class TestProsodyEchoPipeline:
    def test_registered_by_default(self) -> None:
        registry = create_default_registry()
        assert registry.get("prosody-echo") is not None

    def test_declares_audio_and_metadata_streams(self) -> None:
        pipeline = ProsodyEchoPipeline()
        kinds = {s.kind.value for s in pipeline.output_streams}
        assert kinds == {"audio", "metadata"}
        info = pipeline.info
        assert {s.kind for s in info.output_streams} == {"audio", "metadata"}

    async def test_echoes_audio_unchanged(self) -> None:
        pipeline = ProsodyEchoPipeline()
        session = Session(pipeline_id="prosody-echo", sample_rate=16000)
        chunks = [_pcm(np.zeros(1600, dtype=np.float32))]
        out = [c async for c in pipeline.process(_stream(chunks), session=session)]
        assert out == chunks

    async def test_emits_baseline_prosody_frames(self) -> None:
        pipeline = ProsodyEchoPipeline()
        session = Session(pipeline_id="prosody-echo", sample_rate=16000)
        rate = 16000
        t = np.arange(rate, dtype=np.float32) / rate
        tone = _pcm(0.5 * np.sin(2 * np.pi * 200.0 * t))
        iterator = pipeline.iter_metadata_stream("prosody", _stream([tone]), session=session)
        assert iterator is not None
        frames = [f async for f in iterator]

        assert frames
        assert [f.sequence for f in frames] == list(range(len(frames)))
        assert all(f.kind == MetadataKind.PROSODY for f in frames)
        assert all(f.prosody is not None and f.prosody.is_pause is False for f in frames)

    async def test_unknown_metadata_stream_returns_none(self) -> None:
        pipeline = ProsodyEchoPipeline()
        assert pipeline.iter_metadata_stream("other", _stream([])) is None
