from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

import numpy as np

from src.models import ListenProduct, StageInfo, StageKind, WordSpan
from src.pipelines._audio import downsample
from src.pipelines.whisper_tts import WHISPER_SAMPLE_RATE, _transcribe_sync

logger = logging.getLogger(__name__)

BUFFER_SECONDS = 3.0
MIN_BUFFER_SECONDS = 1.0


class WhisperListenStage:
    """Streaming-ish ASR via faster-whisper buffered chunks."""

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        model_size: str | None = None,
        cache: Any = None,
        **_: object,
    ) -> None:
        self._sample_rate = sample_rate
        self._model_size = model_size or os.environ.get("WHISPER_MODEL_SIZE", "base")
        self._cache = cache
        self._model: Any = None
        self.info = StageInfo(
            id="whisper-listen",
            kind=StageKind.LISTEN,
            name="Whisper Listen",
            description="faster-whisper buffered ASR (EN).",
            requires_gpu=True,
            default_for_kind=True,
        )

    async def start(self) -> None:
        if self._model is not None:
            return
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, self._load_model)
        logger.info("whisper-listen model loaded: %s", self._model_size)

    async def stop(self) -> None:
        self._model = None

    def _load_model(self) -> Any:
        from faster_whisper import WhisperModel

        from src.config import get_settings

        settings = get_settings()
        kwargs: dict[str, Any] = {
            "device": settings.compute_device,
            "compute_type": settings.resolved_compute_type(),
        }
        if self._cache is not None:
            download_root = str(self._cache.path_for("custom", "whisper-listen"))
            kwargs["download_root"] = download_root
        return WhisperModel(self._model_size, **kwargs)

    async def transcribe(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[ListenProduct]:
        if self._model is None:
            await self.start()
        assert self._model is not None

        buffer = np.zeros(0, dtype=np.float32)
        min_samples = int(self._sample_rate * MIN_BUFFER_SECONDS)
        max_samples = int(self._sample_rate * BUFFER_SECONDS)
        sequence = 0
        emitted_ms = 0.0
        loop = asyncio.get_running_loop()

        async for chunk in audio_stream:
            if not chunk:
                continue
            pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            buffer = np.concatenate([buffer, pcm])
            while buffer.size >= max_samples:
                frame, buffer = buffer[:max_samples], buffer[max_samples:]
                product = await self._decode_frame(
                    loop, frame, sequence=sequence, start_ms=emitted_ms
                )
                emitted_ms += (frame.size / self._sample_rate) * 1000.0
                sequence += 1
                if product is not None:
                    yield product

        if buffer.size >= min_samples:
            product = await self._decode_frame(
                loop, buffer, sequence=sequence, start_ms=emitted_ms
            )
            if product is not None:
                yield product

    async def _decode_frame(
        self,
        loop: asyncio.AbstractEventLoop,
        frame: np.ndarray,
        *,
        sequence: int,
        start_ms: float,
    ) -> ListenProduct | None:
        audio_16k = downsample(frame, self._sample_rate, WHISPER_SAMPLE_RATE)
        texts: list[str] = await loop.run_in_executor(
            None, partial(_transcribe_sync, self._model, audio_16k)
        )
        text = " ".join(texts).strip()
        if not text:
            return None
        duration_ms = (frame.size / self._sample_rate) * 1000.0
        words = _split_words(text, start_ms=start_ms, end_ms=start_ms + duration_ms)
        return ListenProduct(
            sequence=sequence,
            utterance_id=f"whisper-{sequence}",
            text=text,
            is_final=True,
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
