from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from src.models import ListenProduct, StageInfo, StageKind, WordSpan
from src.pipelines._audio import downsample
from src.pipelines.whisper_tts import WHISPER_SAMPLE_RATE, _transcribe_sync

logger = logging.getLogger(__name__)

BUFFER_SECONDS = 3.0
MIN_BUFFER_SECONDS = 1.0


@dataclass(frozen=True, slots=True)
class WhisperLoadedModel:
    """Immutable resident Whisper weights. Outlives sessions (D6)."""

    model: Any
    model_size: str
    revision: str = "unknown"


def load_whisper_model(
    *,
    model_size: str | None = None,
    cache: Any = None,
) -> WhisperLoadedModel:
    """Load faster-whisper weights once (sync). Used by StageHost model_loader."""
    from faster_whisper import WhisperModel

    from src.config import get_settings

    size = model_size or os.environ.get("WHISPER_MODEL_SIZE", "base")
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "device": settings.compute_device,
        "compute_type": settings.resolved_compute_type(),
    }
    if cache is not None:
        download_root = str(cache.path_for("custom", "whisper-listen"))
        kwargs["download_root"] = download_root
    model = WhisperModel(size, **kwargs)
    revision = str(size)
    return WhisperLoadedModel(model=model, model_size=size, revision=revision)


class WhisperListenStage:
    """Streaming-ish ASR via faster-whisper buffered chunks.

    Weights may be injected via ``loaded_model`` so ``stop()`` never unloads them.
    Without a preloaded model the stage still loads on ``start()`` (legacy path)
    and only then owns teardown on ``stop()``.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        model_size: str | None = None,
        cache: Any = None,
        loaded_model: WhisperLoadedModel | None = None,
        **_: object,
    ) -> None:
        self._sample_rate = sample_rate
        self._model_size = model_size or os.environ.get("WHISPER_MODEL_SIZE", "base")
        self._cache = cache
        self._loaded: WhisperLoadedModel | None = loaded_model
        self._owns_model = loaded_model is None
        self._session_active = False
        # Per-session decode bookkeeping (cleared on stop/close_session).
        self._buffer = np.zeros(0, dtype=np.float32)
        self._sequence = 0
        self._emitted_ms = 0.0
        self.info = StageInfo(
            id="whisper-listen",
            kind=StageKind.LISTEN,
            name="Whisper Listen",
            description="faster-whisper buffered ASR (EN).",
            requires_gpu=True,
            default_for_kind=True,
        )

    @property
    def loaded_model(self) -> WhisperLoadedModel | None:
        return self._loaded

    @property
    def _model(self) -> Any | None:
        return None if self._loaded is None else self._loaded.model

    @_model.setter
    def _model(self, value: Any | None) -> None:
        """Compatibility setter used by tests that inject a fake model."""
        if value is None:
            if self._owns_model:
                self._loaded = None
            return
        if self._loaded is not None and not self._owns_model:
            # Keep resident weights; ignore overwrite of preloaded model.
            return
        size = self._model_size
        if self._loaded is not None:
            size = self._loaded.model_size
        self._loaded = WhisperLoadedModel(model=value, model_size=size, revision=str(size))

    async def start(self) -> None:
        """Ensure weights are available; reset per-session runtime state."""
        self._reset_session_state()
        self._session_active = True
        if self._loaded is not None:
            return
        loop = asyncio.get_running_loop()
        loaded = await loop.run_in_executor(
            None,
            partial(load_whisper_model, model_size=self._model_size, cache=self._cache),
        )
        self._loaded = loaded
        self._owns_model = True
        logger.info("whisper-listen model loaded: %s", loaded.model_size)

    async def stop(self) -> None:
        """Clear per-session decoder/stream state. Never unloads preloaded weights."""
        self._reset_session_state()
        self._session_active = False
        if self._owns_model:
            self._loaded = None

    def _reset_session_state(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._sequence = 0
        self._emitted_ms = 0.0

    def _load_model(self) -> Any:
        """Legacy sync loader used by older call sites / tests."""
        loaded = load_whisper_model(model_size=self._model_size, cache=self._cache)
        return loaded.model

    async def transcribe(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[ListenProduct]:
        if self._loaded is None:
            await self.start()
        assert self._loaded is not None
        self._session_active = True

        min_samples = int(self._sample_rate * MIN_BUFFER_SECONDS)
        max_samples = int(self._sample_rate * BUFFER_SECONDS)
        loop = asyncio.get_running_loop()

        async for chunk in audio_stream:
            if not chunk:
                continue
            pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            self._buffer = np.concatenate([self._buffer, pcm])
            while self._buffer.size >= max_samples:
                frame, self._buffer = self._buffer[:max_samples], self._buffer[max_samples:]
                product = await self._decode_frame(
                    loop,
                    frame,
                    sequence=self._sequence,
                    start_ms=self._emitted_ms,
                    is_final=False,
                )
                self._emitted_ms += (frame.size / self._sample_rate) * 1000.0
                self._sequence += 1
                if product is not None:
                    yield product

        # Stream EOS: flush remainder and mark final.
        if self._buffer.size >= min_samples:
            product = await self._decode_frame(
                loop,
                self._buffer,
                sequence=self._sequence,
                start_ms=self._emitted_ms,
                is_final=True,
            )
            self._buffer = np.zeros(0, dtype=np.float32)
            self._sequence += 1
            if product is not None:
                yield product
        elif self._sequence > 0:
            # Emit a terminal final marker with empty text only if we already
            # produced mid-stream products; adapters handle empty finals.
            yield ListenProduct(
                sequence=self._sequence,
                utterance_id=f"whisper-{self._sequence}",
                text="",
                is_final=True,
                words=[],
                language="en",
            )

    async def _decode_frame(
        self,
        loop: asyncio.AbstractEventLoop,
        frame: np.ndarray,
        *,
        sequence: int,
        start_ms: float,
        is_final: bool = True,
    ) -> ListenProduct | None:
        assert self._loaded is not None
        audio_16k = downsample(frame, self._sample_rate, WHISPER_SAMPLE_RATE)
        texts: list[str] = await loop.run_in_executor(
            None, partial(_transcribe_sync, self._loaded.model, audio_16k)
        )
        text = " ".join(texts).strip()
        if not text and not is_final:
            return None
        duration_ms = (frame.size / self._sample_rate) * 1000.0
        words = _split_words(text, start_ms=start_ms, end_ms=start_ms + duration_ms)
        return ListenProduct(
            sequence=sequence,
            utterance_id=f"whisper-{sequence}",
            text=text,
            is_final=is_final,
            words=words,
            language="en",
        )


def _split_words(text: str, *, start_ms: float, end_ms: float) -> list[WordSpan]:
    tokens = [part for part in text.split() if part]
    if not tokens:
        return []
    span = max(0.0, end_ms - start_ms)
    step = span / len(tokens) if span > 0 else 0.0
    words: list[WordSpan] = []
    cursor = start_ms
    for token in tokens:
        next_ms = cursor + step
        words.append(WordSpan(text=token, start_ms=cursor, end_ms=next_ms, conf=1.0))
        cursor = next_ms
    return words


class WhisperListenFactory:
    @property
    def info(self) -> StageInfo:
        return StageInfo(
            id="whisper-listen",
            kind=StageKind.LISTEN,
            name="Whisper Listen",
            description="faster-whisper buffered ASR (EN).",
            requires_gpu=True,
            default_for_kind=True,
        )

    def create(self, **kwargs: Any) -> WhisperListenStage:
        return WhisperListenStage(**kwargs)
