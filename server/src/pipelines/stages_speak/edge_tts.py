from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from src.models import StageInfo, StageKind, TranslateProduct
from src.pipelines._audio import EDGE_TTS_VOICE, synthesize_spanish

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EdgeTTSLoadedModel:
    """Immutable edge-tts backend handle (network voice; no local weights).

    Loaded once via StageHost so session close never 'unloads' the backend.
    Capability: utterance-buffered PCM (edge-tts does not stream PCM frames).
    """

    voice_id: str
    revision: str = "edge-tts-network"
    streams_pcm: bool = False


def load_edge_tts_model(*, voice_id: str | None = None) -> EdgeTTSLoadedModel:
    """Construct the immutable edge-tts backend descriptor (sync, cheap)."""
    voice = voice_id or EDGE_TTS_VOICE
    return EdgeTTSLoadedModel(voice_id=voice, revision=f"edge-tts:{voice}", streams_pcm=False)


class EdgeTTSSpeakStage:
    """Spanish TTS via edge-tts (es-ES). Consumes instruction markers best-effort.

    Accepts optional ``loaded_model`` so StageHost can warm the backend once.
    ``stop()`` clears only per-session utterance state.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        cache: Any = None,
        loaded_model: EdgeTTSLoadedModel | None = None,
        voice_id: str | None = None,
        **_: object,
    ) -> None:
        self._sample_rate = sample_rate
        self._cache = cache
        self._loaded: EdgeTTSLoadedModel | None = loaded_model
        self._owns_model = loaded_model is None
        self._voice_id = voice_id or (loaded_model.voice_id if loaded_model else EDGE_TTS_VOICE)
        self._session_active = False
        self._utterance_count = 0
        self.info = StageInfo(
            id="edge-tts-es",
            kind=StageKind.SPEAK,
            name="Edge TTS Spanish",
            description="edge-tts Spanish neural voice (network).",
            requires_gpu=False,
            default_for_kind=True,
        )

    @property
    def loaded_model(self) -> EdgeTTSLoadedModel | None:
        return self._loaded

    @property
    def streams_pcm(self) -> bool:
        """edge-tts buffers full utterance MP3 before yielding PCM."""
        return False if self._loaded is None else self._loaded.streams_pcm

    async def start(self) -> None:
        self._session_active = True
        self._utterance_count = 0
        if self._loaded is not None:
            return
        self._loaded = load_edge_tts_model(voice_id=self._voice_id)
        self._owns_model = True

    async def stop(self) -> None:
        """Clear per-session utterance counters. Never drops a preloaded backend."""
        self._session_active = False
        self._utterance_count = 0
        if self._owns_model:
            self._loaded = None

    async def synthesize(
        self, text_stream: AsyncIterator[TranslateProduct]
    ) -> AsyncIterator[bytes]:
        if self._loaded is None:
            await self.start()
        assert self._loaded is not None

        async for product in text_stream:
            text = product.text.strip()
            if not text:
                continue
            # Instruction channel is available for future expressive backends.
            _ = product.instructions
            pcm = await synthesize_spanish(text, self._sample_rate)
            self._utterance_count += 1
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
