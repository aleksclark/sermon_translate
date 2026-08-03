from __future__ import annotations

import logging

from src.pipelines.stage_registry import StageRegistry
from src.pipelines.stages_listen.whisper import WhisperListenFactory

logger = logging.getLogger(__name__)


def register_listen_stages(registry: StageRegistry) -> None:
    registry.register(WhisperListenFactory())
    try:
        from src.runtime.nvidia_libs import ensure_nvidia_library_path

        ensure_nvidia_library_path()
        import moshi  # type: ignore[import-not-found]  # noqa: F401

        from src.pipelines.stages_listen.kyutai import KyutaiListenFactory

        registry.register(KyutaiListenFactory())
    except ImportError:
        logger.info("moshi (kyutai STT) extra not installed; skipping")
