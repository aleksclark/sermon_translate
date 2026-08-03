from __future__ import annotations

import logging

from src.config import default_model_cache_dir
from src.models import PipelineInfo
from src.pipelines.base import BasePipeline
from src.pipelines.composed import ComposedPipeline
from src.pipelines.echo import EchoPipeline
from src.pipelines.prosody_echo import ProsodyEchoPipeline
from src.pipelines.spanish import SpanishTranslationPipeline
from src.pipelines.spanish_direct import SpanishDirectPipeline
from src.pipelines.stage_registry import StageRegistry, create_default_stage_registry
from src.pipelines.whisper_tts import WhisperTTSPipeline
from src.runtime.base import StageRuntime
from src.runtime.local import LocalStageRuntime
from src.runtime.model_cache import ModelCache

logger = logging.getLogger(__name__)


class PipelineRegistry:
    """Central registry of available translation pipelines."""

    def __init__(self, stage_registry: StageRegistry | None = None) -> None:
        self._pipelines: dict[str, BasePipeline] = {}
        self.stage_registry = stage_registry or StageRegistry()

    def register(self, pipeline: BasePipeline) -> None:
        self._pipelines[pipeline.info.id] = pipeline

    def get(self, pipeline_id: str) -> BasePipeline | None:
        return self._pipelines.get(pipeline_id)

    def list_all(self) -> list[PipelineInfo]:
        return [p.info for p in self._pipelines.values()]

    def __len__(self) -> int:
        return len(self._pipelines)


def create_default_registry(
    stage_registry: StageRegistry | None = None,
    *,
    cache: ModelCache | None = None,
    runtime: StageRuntime | None = None,
) -> PipelineRegistry:
    stages = stage_registry or create_default_stage_registry()
    model_cache = cache or ModelCache(default_model_cache_dir())
    stage_runtime = runtime or LocalStageRuntime(stages, model_cache)
    registry = PipelineRegistry(stage_registry=stages)
    registry.register(EchoPipeline())
    registry.register(ProsodyEchoPipeline())
    registry.register(WhisperTTSPipeline())
    registry.register(SpanishTranslationPipeline())
    registry.register(SpanishDirectPipeline())
    registry.register(ComposedPipeline(stages, runtime=stage_runtime, cache=model_cache))
    try:
        from src.pipelines.seamless_streaming import SeamlessStreamingPipeline

        registry.register(SeamlessStreamingPipeline())
    except ImportError:
        logger.info("seamless_communication not installed, skipping SeamlessStreamingPipeline")
    return registry
