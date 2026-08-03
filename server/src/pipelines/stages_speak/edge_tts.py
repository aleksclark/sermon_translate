from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from src.models import StageInfo, StageKind, TranslateProduct
from src.pipelines._audio import synthesize_spanish

logger = logging.getLogger(__name__)


class EdgeTTSSpeakStage:
    """Spanish TTS via edge-tts (es-ES). Consumes instruction markers best-effort."""

    def __init__(self, *, sample_rate: int = 48000, cache: Any = None, **_: object) -> None:
        self._sample_rate = sample_rate
        self._cache = cache
        self.info = StageInfo(
            id="edge-tts-es",
            kind=StageKind.SPEAK,
            name="Edge TTS Spanish",
            description="edge-tts Spanish neural voice (network).",
            requires_gpu=False,
            default_for_kind=True,
        )

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def synthesize(
        self, text_stream: AsyncIterator[TranslateProduct]
    ) -> AsyncIterator[bytes]:
        async for product in text_stream:
            text = product.text.strip()
            if not text:
                continue
            # Instruction channel is available for future expressive backends.
            _ = product.instructions
            pcm = await synthesize_spanish(text, self._sample_rate)
            if pcm:
                yield pcm


class EdgeTTSSpeakFactory:
    @property
    def info(self) -> StageInfo:
        return StageInfo(
            id="edge-tts-es",
            kind=StageKind.SPEAK,
            name="Edge TTS Spanish",
            description="edge-tts Spanish neural voice (network).",
            requires_gpu=False,
            default_for_kind=True,
        )

    def create(self, **kwargs: Any) -> EdgeTTSSpeakStage:
        return EdgeTTSSpeakStage(**kwargs)
