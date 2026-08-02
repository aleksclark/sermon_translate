from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest

from src.models import MetadataKind, Session, StageSelection
from src.pipelines import ComposedPipeline, create_default_registry, create_default_stage_registry


def _pcm(samples: np.ndarray) -> bytes:
    return (samples * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


async def _stream(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def _tone_chunks(sample_rate: int = 16000, seconds: float = 0.2) -> list[bytes]:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    tone = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    raw = _pcm(tone)
    chunk_bytes = int(sample_rate * 0.02) * 2
    return [raw[i : i + chunk_bytes] for i in range(0, len(raw), chunk_bytes)]


class TestComposedPipeline:
    def test_registered_by_default(self) -> None:
        registry = create_default_registry()
        pipeline = registry.get("composed")
        assert pipeline is not None
        assert pipeline.info.id == "composed"

    def test_declares_expected_streams(self) -> None:
        pipeline = ComposedPipeline(create_default_stage_registry())
        names = {s.name: s.kind.value for s in pipeline.output_streams}
        assert names == {
            "audio": "audio",
            "listen": "text",
            "translate": "text",
            "prosody": "metadata",
        }

    async def test_passthrough_graph_emits_audio_text_and_prosody(self) -> None:
        stages = create_default_stage_registry()
        pipeline = ComposedPipeline(stages)
        session = Session(
            pipeline_id="composed",
            sample_rate=16000,
            stages=StageSelection(
                listen="passthrough-listen",
                translate="passthrough-translate",
                speak="passthrough-speak",
                prosody="baseline-prosody",
            ),
        )
        chunks = _tone_chunks()

        async def collect_text(name: str) -> list[str]:
            iterator = pipeline.iter_stream(name, _stream([]), session=session)
            assert iterator is not None
            return [text async for text in iterator]

        async def collect_prosody() -> list:
            iterator = pipeline.iter_metadata_stream("prosody", _stream([]), session=session)
            assert iterator is not None
            return [frame async for frame in iterator]

        text_listen = asyncio.create_task(collect_text("listen"))
        text_translate = asyncio.create_task(collect_text("translate"))
        prosody_task = asyncio.create_task(collect_prosody())

        audio_out = [c async for c in pipeline.process(_stream(chunks), session=session)]

        listen_texts = await text_listen
        translate_texts = await text_translate
        prosody_frames = await prosody_task

        assert audio_out
        assert all(isinstance(c, (bytes, bytearray)) and len(c) > 0 for c in audio_out)
        assert listen_texts
        assert translate_texts == listen_texts
        assert prosody_frames
        assert all(f.kind == MetadataKind.PROSODY for f in prosody_frames)
        assert all(f.prosody is not None for f in prosody_frames)

    async def test_requires_stage_selection(self) -> None:
        pipeline = ComposedPipeline(create_default_stage_registry())
        session = Session(pipeline_id="composed", sample_rate=16000)
        with pytest.raises(ValueError, match="session.stages"):
            _ = [c async for c in pipeline.process(_stream([]), session=session)]
