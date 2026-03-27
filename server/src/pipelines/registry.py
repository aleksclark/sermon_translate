from __future__ import annotations

import logging

from src.models import PipelineInfo
from src.pipelines.base import BasePipeline
from src.pipelines.echo import EchoPipeline
from src.pipelines.spanish import SpanishTranslationPipeline
from src.pipelines.spanish_direct import SpanishDirectPipeline
from src.pipelines.spanish_fast import SpanishFastPipeline
from src.pipelines.whisper_tts import WhisperTTSPipeline

logger = logging.getLogger(__name__)


class PipelineRegistry:
    """Central registry of available translation pipelines."""

    def __init__(self) -> None:
        self._pipelines: dict[str, BasePipeline] = {}

    def register(self, pipeline: BasePipeline) -> None:
        self._pipelines[pipeline.info.id] = pipeline

    def get(self, pipeline_id: str) -> BasePipeline | None:
        return self._pipelines.get(pipeline_id)

    def list_all(self) -> list[PipelineInfo]:
        return [p.info for p in self._pipelines.values()]

    def __len__(self) -> int:
        return len(self._pipelines)


def create_default_registry() -> PipelineRegistry:
    registry = PipelineRegistry()
    registry.register(EchoPipeline())
    registry.register(WhisperTTSPipeline())
    registry.register(SpanishTranslationPipeline())
    registry.register(SpanishDirectPipeline())
    registry.register(SpanishFastPipeline())
    try:
        from src.pipelines.spanish_fast_v2 import SpanishFastV2Pipeline

        registry.register(SpanishFastV2Pipeline())
    except ImportError:
        pass
    try:
        from src.pipelines.moonshine_pipeline import MoonshineStreamingPipeline

        registry.register(MoonshineStreamingPipeline())
    except ImportError:
        logger.info("moonshine_voice not installed, skipping MoonshineStreamingPipeline")
    try:
        from src.pipelines.nova_sonic import NovaSonicPipeline

        registry.register(NovaSonicPipeline())
    except ImportError:
        logger.info("aws SDK not installed, skipping NovaSonicPipeline")
    try:
        from src.pipelines.simul_streaming import SimulStreamingPipeline as SimulStreamPipe

        registry.register(SimulStreamPipe())
    except ImportError:
        logger.info("SimulStreaming deps missing, skipping")
    try:
        from src.pipelines.simul_streaming_vc import SimulStreamingVoiceClonePipeline

        registry.register(SimulStreamingVoiceClonePipeline())
    except ImportError:
        logger.info("F5-TTS not installed, skipping voice clone pipeline")
    try:
        from src.pipelines.gpu_pipelines import (
            GPUS2STPipeline,
            GPUWhisperOpusPipeline,
            GPUWhisperT2STPipeline,
        )

        registry.register(GPUWhisperT2STPipeline())
        registry.register(GPUS2STPipeline())
        registry.register(GPUWhisperOpusPipeline())
    except Exception:
        logger.info("GPU pipelines not available")
    try:
        from src.pipelines.seamless_streaming import SeamlessStreamingPipeline

        registry.register(SeamlessStreamingPipeline())
    except ImportError:
        logger.info("seamless_communication not installed, skipping SeamlessStreamingPipeline")
    return registry
