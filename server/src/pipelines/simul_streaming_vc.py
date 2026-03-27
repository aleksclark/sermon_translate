"""SimulStreaming + F5-TTS voice clone variant.

Same as SimulStreamingPipeline but uses F5-TTS fine-tuned on real
Spanish sermon audio instead of Edge TTS.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.models import PipelineInfo
from src.pipelines._f5_tts import release_f5_model, synthesize_f5
from src.pipelines.simul_streaming import SimulStreamingPipeline

logger = logging.getLogger(__name__)


async def _synthesize_f5_async(
    text: str, sample_rate: int, rate_pct: int,  # noqa: ARG001
) -> bytes:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, synthesize_f5, text, None, None, sample_rate,
    )


class SimulStreamingVoiceClonePipeline(SimulStreamingPipeline):
    """SimulStreaming with F5-TTS voice cloning."""

    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            id="simul-streaming-vc",
            name="SimulStreaming + Voice Clone (F5-TTS)",
            description=(
                "SimulStreaming AlignAtt ASR + Opus-MT + F5-TTS "
                "fine-tuned on Spanish sermon audio."
            ),
            output_streams=self._build_output_stream_info(),
        )

    async def _do_start(self) -> None:
        await super()._do_start()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._warmup_f5)

    @staticmethod
    def _warmup_f5() -> None:
        from src.pipelines._f5_tts import _get_f5_model

        _get_f5_model()
        logger.info("F5-TTS warmed up")

    async def _do_stop(self) -> None:
        await super()._do_stop()
        release_f5_model()

    def _get_synth_fn(self) -> Any:
        return _synthesize_f5_async
