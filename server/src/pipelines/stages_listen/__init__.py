from __future__ import annotations

import logging

from src.pipelines.stage_registry import StageRegistry
from src.pipelines.stages_listen.whisper import WhisperListenFactory

logger = logging.getLogger(__name__)


def register_listen_stages(registry: StageRegistry) -> None:
    registry.register(WhisperListenFactory())
    try:
        import kyutai  # type: ignore[import-not-found]  # noqa: F401

        from src.pipelines.stages_listen.kyutai import KyutaiListenFactory

        registry.register(KyutaiListenFactory())
    except ImportError:
        logger.info("kyutai listen extra not installed; skipping")
