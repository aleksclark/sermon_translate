from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from src.models import StageInfo, StageKind, TranslateProduct

logger = logging.getLogger(__name__)


class PocketTTSSpeakStage:
    """Kyutai Pocket TTS spanish_24l CPU fallback.

    Registered only when the optional pocket-tts extra/import is available.
    """

    def __init__(self, *, sample_rate: int = 48000, cache: Any = None, **_: object) -> None:
        self._sample_rate = sample_rate
        self._cache = cache
        self._engine: Any = None
        self.info = StageInfo(
            id="pocket-tts-spanish-24l",
            kind=StageKind.SPEAK,
            name="Pocket TTS Spanish",
            description="Kyutai Pocket TTS spanish_24l CPU fallback (optional extra).",
            requires_gpu=False,
            default_for_kind=False,
        )

    async def start(self) -> None:
        try:
            import pocket_tts  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("pocket-tts extra is not installed") from exc
        # Exact API may vary by package version; keep a thin adapter.
        self._engine = pocket_tts
        logger.info("pocket-tts engine ready")

    async def stop(self) -> None:
        self._engine = None

    async def synthesize(
        self, text_stream: AsyncIterator[TranslateProduct]
    ) -> AsyncIterator[bytes]:
        if self._engine is None:
            await self.start()
        async for product in text_stream:
            text = product.text.strip()
            if not text:
                continue
            pcm = await self._synth(text)
            if pcm:
                yield pcm

    async def _synth(self, text: str) -> bytes:
        # Placeholder adapter: if the installed package exposes a different API,
        # operators can swap this method. Prefer silence-free failure over crash.
        synth = getattr(self._engine, "synthesize", None)
        if synth is None:
            logger.warning("pocket-tts has no synthesize(); skipping chunk")
            return b""
        result = synth(text, sample_rate=self._sample_rate, language="es")
        if isinstance(result, (bytes, bytearray)):
            return bytes(result)
        return b""


class PocketTTSFactory:
    @property
    def info(self) -> StageInfo:
        return StageInfo(
            id="pocket-tts-spanish-24l",
            kind=StageKind.SPEAK,
            name="Pocket TTS Spanish",
            description="Kyutai Pocket TTS spanish_24l CPU fallback (optional extra).",
            requires_gpu=False,
            default_for_kind=False,
        )

    def create(self, **kwargs: Any) -> PocketTTSSpeakStage:
        return PocketTTSSpeakStage(**kwargs)
