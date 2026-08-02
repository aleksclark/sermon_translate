from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np

from src.models import MetadataKind
from src.pipelines.stages import BaselineProsodyStage


def _pcm(samples: np.ndarray) -> bytes:
    return (samples * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


async def _one_chunk(chunk: bytes) -> AsyncIterator[bytes]:
    yield chunk


class TestBaselineProsodyStage:
    async def test_silence_is_detected_as_pause(self) -> None:
        stage = BaselineProsodyStage(sample_rate=16000, frame_ms=100.0)
        silence = _pcm(np.zeros(16000, dtype=np.float32))
        frames = [f async for f in stage.analyze(_one_chunk(silence), "prosody")]

        assert len(frames) == 10
        for frame in frames:
            assert frame.kind == MetadataKind.PROSODY
            assert frame.prosody is not None
            assert frame.prosody.is_pause is True
            assert frame.prosody.energy == 0.0

    async def test_tone_has_energy_and_pitch(self) -> None:
        rate = 16000
        stage = BaselineProsodyStage(sample_rate=rate, frame_ms=100.0)
        t = np.arange(rate, dtype=np.float32) / rate
        tone = _pcm(0.5 * np.sin(2 * np.pi * 220.0 * t))
        frames = [f async for f in stage.analyze(_one_chunk(tone), "prosody")]

        assert frames
        prosody = frames[0].prosody
        assert prosody is not None
        assert prosody.is_pause is False
        assert prosody.energy is not None and prosody.energy > 0.1
        assert prosody.f0_hz is not None and 180.0 < prosody.f0_hz < 260.0

    async def test_frames_are_sequential_and_ordered(self) -> None:
        rate = 16000
        stage = BaselineProsodyStage(sample_rate=rate, frame_ms=50.0)
        t = np.arange(rate, dtype=np.float32) / rate
        tone = _pcm(0.5 * np.sin(2 * np.pi * 200.0 * t))
        frames = [f async for f in stage.analyze(_one_chunk(tone), "prosody")]

        assert [f.sequence for f in frames] == list(range(len(frames)))
        for prev, nxt in zip(frames, frames[1:], strict=False):
            assert prev.end_ms is not None and nxt.start_ms is not None
            assert nxt.start_ms >= prev.end_ms - 1e-6

    async def test_deterministic_features(self) -> None:
        stage = BaselineProsodyStage(sample_rate=16000, frame_ms=100.0)
        silence = _pcm(np.zeros(8000, dtype=np.float32))
        first = [f.prosody async for f in stage.analyze(_one_chunk(silence), "prosody")]
        second = [f.prosody async for f in stage.analyze(_one_chunk(silence), "prosody")]
        assert [p.model_dump() for p in first] == [p.model_dump() for p in second]
