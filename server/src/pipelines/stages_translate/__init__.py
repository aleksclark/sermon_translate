from __future__ import annotations

from src.pipelines.stage_registry import StageRegistry
from src.pipelines.stages_translate.opus_mt import OpusMTTranslateFactory


def register_translate_stages(registry: StageRegistry) -> None:
    registry.register(OpusMTTranslateFactory())
