from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.models import Session, StageKind, StageSelection
from src.pipelines import ComposedPipeline, create_default_stage_registry
from src.runtime.model_cache import ModelCache
from src.runtime.subprocess_runtime import SubprocessStageRuntime


def _tone(sample_rate: int = 16000, seconds: float = 0.1) -> bytes:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    tone = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    return (tone * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


@pytest.mark.asyncio
async def test_subprocess_runtime_listen_round_trip(tmp_path: Path) -> None:
    stages = create_default_stage_registry()
    cache = ModelCache(tmp_path / "models")
    runtime = SubprocessStageRuntime(stages, cache, start_timeout=30.0)
    session = Session(
        pipeline_id="composed",
        sample_rate=16000,
        stages=StageSelection(
            listen="passthrough-listen",
            translate="passthrough-translate",
            speak="passthrough-speak",
        ),
    )
    handle = await runtime.spawn("passthrough-listen", session, kind=StageKind.LISTEN)
    try:
        await handle.start()

        async def audio():
            yield _tone()

        products = [p async for p in handle.transcribe(audio())]
        assert products
        assert all(p.text for p in products)
    finally:
        await handle.stop()
        await runtime.stop_all()


@pytest.mark.asyncio
async def test_composed_pipeline_with_subprocess_runtime(tmp_path: Path) -> None:
    stages = create_default_stage_registry()
    cache = ModelCache(tmp_path / "models")
    runtime = SubprocessStageRuntime(stages, cache, start_timeout=30.0)
    pipeline = ComposedPipeline(stages, runtime=runtime, cache=cache)
    session = Session(
        pipeline_id="composed",
        sample_rate=16000,
        stages=StageSelection(
            listen="passthrough-listen",
            translate="passthrough-translate",
            speak="passthrough-speak",
            prosody=None,
        ),
    )

    async def audio():
        yield _tone()

    try:
        out = [chunk async for chunk in pipeline.process(audio(), session=session)]
        assert out
        assert all(isinstance(chunk, (bytes, bytearray)) and len(chunk) > 0 for chunk in out)
    finally:
        await runtime.stop_all()
