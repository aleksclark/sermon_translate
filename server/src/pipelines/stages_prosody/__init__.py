from __future__ import annotations

import logging

from src.pipelines.stage_registry import StageRegistry

logger = logging.getLogger(__name__)


def register_prosody_stages(registry: StageRegistry) -> None:
    # baseline-prosody is registered with stubs; optional enhanced trackers go here.
    try:
        from src.pipelines.stages_prosody.pyworld_stage import PyworldProsodyFactory

        registry.register(PyworldProsodyFactory())
    except ImportError:
        logger.info("pyworld prosody extra not installed; using baseline only")
