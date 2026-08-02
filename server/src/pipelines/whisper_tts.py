from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from functools import partial

import numpy as np

from src.models import PipelineInfo, Session
from src.pipelines._audio import downsample
from src.pipelines.base import BasePipeline, OutputStreamDescriptor, OutputStreamKind

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000
BUFFER_SECONDS = 3
MIN_BUFFER_SECONDS = 1.0


def _transcribe_sync(model, audio: np.ndarray) -> list[str]:  # type: ignore[no-untyped-def]
    segments, _ = model.transcribe(audio, beam_size=1, language="en", vad_filter=True)
    results = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            results.append(text)
    return results


class WhisperTTSPipeline(BasePipeline):
    """Speech-to-text pipeline using faster-whisper."""

    def __init__(self, model_size: str = "base", sample_rate: int = 48000) -> None:
        super().__init__()
        self._model_size = model_size
        self._sample_rate = sample_rate
        self._model = None

    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            id="whisper-tts",
            name="Whisper TTS",
            description="Speech-to-text using faster-whisper — streams transcript from audio.",
            output_streams=self._build_output_stream_info(),
        )

    @property
    def output_streams(self) -> list[OutputStreamDescriptor]:
        return [
            OutputStreamDescriptor(
                name="transcript",
                kind=OutputStreamKind.TEXT,
                label="Transcript",
                consumes_audio=True,
            ),
        ]

    async def _do_start(self) -> None:
        if self._model is not None:
            return
        loop = asyncio.get_running_loop()
        self._model = await loop.run_in_executor(None, self._load_model)
        logger.info("Whisper model '%s' loaded", self._model_size)

    def _load_model(self):  # type: ignore[no-untyped-def]
        from faster_whisper import WhisperModel

        return WhisperModel(self._model_size, device="cpu", compute_type="int8")

    async def _do_stop(self) -> None:
        self._model = None

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None,
    ) -> AsyncIterator[bytes]:
        return
        yield  # noqa: F841

    def iter_stream(
        self,
        name: str,
        audio_stream: AsyncIterator[bytes],
        session: Session | None = None,
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        if name == "transcript":
            return self._process_text(audio_stream)
        return None

    async def _process_text(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
        if self._model is None:
            await self.start()

        buffer = np.array([], dtype=np.float32)
        samples_needed = int(WHISPER_SAMPLE_RATE * BUFFER_SECONDS)
        min_samples = int(WHISPER_SAMPLE_RATE * MIN_BUFFER_SECONDS)
        loop = asyncio.get_running_loop()

        async for chunk in audio_stream:
            pcm_int16 = np.frombuffer(chunk, dtype=np.int16)
            pcm_float = pcm_int16.astype(np.float32) / 32768.0
            downsampled = downsample(pcm_float, self._sample_rate, WHISPER_SAMPLE_RATE)
            buffer = np.concatenate([buffer, downsampled])

            if len(buffer) >= samples_needed:
                segment = buffer.copy()
                buffer = np.array([], dtype=np.float32)

                texts = await loop.run_in_executor(
                    None, partial(_transcribe_sync, self._model, segment)
                )
                for text in texts:
                    yield text

        if len(buffer) >= min_samples and self._model is not None:
            texts = await loop.run_in_executor(
                None, partial(_transcribe_sync, self._model, buffer)
            )
            for text in texts:
                yield text
