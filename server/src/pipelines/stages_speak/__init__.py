from __future__ import annotations

import logging

from src.pipelines.stage_registry import StageRegistry
from src.pipelines.stages_speak.edge_tts import EdgeTTSSpeakFactory
from src.pipelines.stages_speak.qwen3_client import Qwen3TTSFactory

logger = logging.getLogger(__name__)


def register_speak_stages(registry: StageRegistry) -> None:
    registry.register(EdgeTTSSpeakFactory())
    registry.register(Qwen3TTSFactory())
    try:
        import pocket_tts  # type: ignore[import-not-found]  # noqa: F401

        from src.pipelines.stages_speak.pocket_tts import PocketTTSFactory

        registry.register(PocketTTSFactory())
    except ImportError:
        logger.info("pocket-tts extra not installed; skipping CPU speak fallback registration")
