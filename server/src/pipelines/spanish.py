from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

import numpy as np
import sentencepiece as spm

from src.models import PipelineInfo, Session
from src.pipelines._audio import downsample, synthesize_spanish
from src.pipelines.base import BasePipeline, OutputStreamDescriptor, OutputStreamKind
from src.pipelines.whisper_tts import WHISPER_SAMPLE_RATE, _transcribe_sync

logger = logging.getLogger(__name__)

BUFFER_SECONDS = 3
MIN_BUFFER_SECONDS = 1.0
TRANSLATION_MODEL_ID = "Helsinki-NLP/opus-mt-en-es"


def _translate_sync(
    translator: Any,
    sp_source: Any,
    sp_target: Any,
    text: str,
) -> str:
    tokens: list[str] = sp_source.encode(text, out_type=str) + ["</s>"]
    results = translator.translate_batch([tokens])
    out_tokens = results[0].hypotheses[0]
    return sp_target.decode(out_tokens)  # type: ignore[no-any-return]


class SpanishTranslationPipeline(BasePipeline):
    """English audio in → Spanish audio + EN/ES transcripts out."""

    def __init__(self, whisper_model_size: str = "base", sample_rate: int = 48000) -> None:
        super().__init__()
        self._whisper_model_size = whisper_model_size
        self._sample_rate = sample_rate
        self._whisper_model: Any = None
        self._translator: Any = None
        self._sp_source: Any = None
        self._sp_target: Any = None

    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            id="spanish-translation",
            name="Spanish Translation",
            description="Translates English speech to Spanish audio with transcript.",
            output_streams=self._build_output_stream_info(),
        )

    @property
    def output_streams(self) -> list[OutputStreamDescriptor]:
        return [
            OutputStreamDescriptor(
                name="audio", kind=OutputStreamKind.AUDIO, label="Spanish Audio",
            ),
            OutputStreamDescriptor(
                name="en-transcript",
                kind=OutputStreamKind.TEXT,
                label="English",
                consumes_audio=False,
            ),
            OutputStreamDescriptor(
                name="es-transcript",
                kind=OutputStreamKind.TEXT,
                label="Spanish",
                consumes_audio=False,
            ),
        ]

    async def _do_start(self) -> None:
        loop = asyncio.get_running_loop()
        if self._whisper_model is None:
            self._whisper_model = await loop.run_in_executor(None, self._load_whisper)
            logger.info("Whisper model '%s' loaded", self._whisper_model_size)
        if self._translator is None:
            self._translator, self._sp_source, self._sp_target = await loop.run_in_executor(
                None, self._load_translation
            )
            logger.info("Translation model loaded")

    def _load_whisper(self) -> Any:
        from faster_whisper import WhisperModel

        from src.config import get_settings

        settings = get_settings()
        return WhisperModel(
            self._whisper_model_size,
            device=settings.compute_device,
            compute_type=settings.resolved_compute_type(),
        )

    def _load_translation(self) -> tuple[Any, Any, Any]:
        import ctranslate2
        from huggingface_hub import snapshot_download

        from src.config import get_settings

        settings = get_settings()
        ct2_dir = self._get_ct2_model_dir()
        hf_dir = snapshot_download(TRANSLATION_MODEL_ID)

        translator = ctranslate2.Translator(
            ct2_dir,
            device=settings.compute_device,
            compute_type=settings.resolved_compute_type(),
        )
        sp_src = spm.SentencePieceProcessor()
        sp_src.load(f"{hf_dir}/source.spm")  # type: ignore[attr-defined]
        sp_tgt = spm.SentencePieceProcessor()
        sp_tgt.load(f"{hf_dir}/target.spm")  # type: ignore[attr-defined]
        return translator, sp_src, sp_tgt

    @staticmethod
    def _get_ct2_model_dir() -> str:
        import os

        cache_dir = os.path.join(
            os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
            "sermon_translate",
            "opus-mt-en-es-ct2",
        )
        model_bin = os.path.join(cache_dir, "model.bin")
        if not os.path.exists(model_bin):
            SpanishTranslationPipeline._convert_model(cache_dir)
        return cache_dir

    @staticmethod
    def _convert_model(output_dir: str) -> None:
        import os

        os.makedirs(output_dir, exist_ok=True)
        logger.info("Converting translation model to CTranslate2 (one-time)...")
        try:
            from ctranslate2.converters.transformers import TransformersConverter
        except ImportError as exc:
            raise RuntimeError(
                "Model conversion requires 'torch' and 'transformers' packages. "
                "Install them or provide a pre-converted model at: " + output_dir
            ) from exc

        converter = TransformersConverter(TRANSLATION_MODEL_ID)
        converter.convert(output_dir, quantization="int8", force=True)
        logger.info("Translation model converted to %s", output_dir)

    async def _do_stop(self) -> None:
        self._whisper_model = None
        self._translator = None
        self._sp_source = None
        self._sp_target = None

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None,
    ) -> AsyncIterator[bytes]:
        if self._whisper_model is None or self._translator is None:
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
                async for out in self._process_segment(segment, loop, session):
                    yield out

        if len(buffer) >= min_samples:
            async for out in self._process_segment(buffer, loop, session):
                yield out

        await self._finish_text("en-transcript", session)
        await self._finish_text("es-transcript", session)

    async def _process_segment(
        self,
        audio: np.ndarray,
        loop: asyncio.AbstractEventLoop,
        session: Session | None,
    ) -> AsyncIterator[bytes]:
        texts = await loop.run_in_executor(
            None, partial(_transcribe_sync, self._whisper_model, audio)
        )

        for en_text in texts:
            translate_fn = partial(
                _translate_sync, self._translator, self._sp_source, self._sp_target, en_text
            )
            es_text = await loop.run_in_executor(None, translate_fn)

            await self._publish_text("en-transcript", en_text, session)
            await self._publish_text("es-transcript", es_text, session)

            pcm_bytes = await synthesize_spanish(es_text, self._sample_rate)
            if pcm_bytes:
                yield pcm_bytes

    def iter_stream(
        self,
        name: str,
        audio_stream: AsyncIterator[bytes],
        session: Session | None = None,
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        if name == "en-transcript":
            return self._drain_text(name, session)
        if name == "es-transcript":
            return self._drain_text(name, session)
        return None
