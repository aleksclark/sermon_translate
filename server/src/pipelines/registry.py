from __future__ import annotations

from src.models import PipelineInfo
from src.pipelines.base import BasePipeline
from src.pipelines.echo import EchoPipeline
from src.pipelines.spanish import SpanishTranslationPipeline
from src.pipelines.whisper_tts import WhisperTTSPipeline


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
    return registry
