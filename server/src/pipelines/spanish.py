from __future__ import annotations

import asyncio
import io
import logging
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

import av
import numpy as np
import sentencepiece as spm

from src.models import PipelineInfo
from src.pipelines.base import BasePipeline
from src.pipelines.whisper_tts import WHISPER_SAMPLE_RATE, _downsample, _transcribe_sync

logger = logging.getLogger(__name__)

EDGE_TTS_SAMPLE_RATE = 24000
EDGE_TTS_VOICE = "es-ES-AlvaroNeural"
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


def _decode_mp3_to_pcm(mp3_data: bytes, target_rate: int) -> bytes:
    container = av.open(io.BytesIO(mp3_data), format="mp3")
    frames: list[np.ndarray] = []
    for frame in container.decode(audio=0):  # type: ignore[attr-defined]
        frames.append(frame.to_ndarray())

    if not frames:
        return b""

    audio = np.concatenate(frames, axis=1)
    mono = audio[0] if audio.shape[0] > 1 else audio.flatten()
    pcm_float = mono.astype(np.float32)
    resampled = _downsample(pcm_float, EDGE_TTS_SAMPLE_RATE, target_rate)
    pcm_int16 = (resampled * 32767).clip(-32768, 32767).astype(np.int16)
    return pcm_int16.tobytes()


async def _synthesize_spanish(text: str, target_rate: int) -> bytes:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice=EDGE_TTS_VOICE)
    mp3_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_data += chunk.get("data", b"")

    if not mp3_data:
        return b""

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _decode_mp3_to_pcm, mp3_data, target_rate)


class SpanishTranslationPipeline(BasePipeline):
    """English audio in → Spanish audio out, with transcript side-channel."""

    def __init__(self, whisper_model_size: str = "base", sample_rate: int = 48000) -> None:
        self._whisper_model_size = whisper_model_size
        self._sample_rate = sample_rate
        self._whisper_model: Any = None
        self._translator: Any = None
        self._sp_source: Any = None
        self._sp_target: Any = None
        self._text_queue: asyncio.Queue[str | None] = asyncio.Queue()

    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            id="spanish-translation",
            name="Spanish Translation",
            description="Translates English speech to Spanish audio with transcript.",
        )

    async def start(self) -> None:
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

        return WhisperModel(self._whisper_model_size, device="cpu", compute_type="int8")

    def _load_translation(self) -> tuple[Any, Any, Any]:
        import ctranslate2
        from huggingface_hub import snapshot_download

        ct2_dir = self._get_ct2_model_dir()
        hf_dir = snapshot_download(TRANSLATION_MODEL_ID)

        translator = ctranslate2.Translator(ct2_dir, device="cpu", compute_type="int8")
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

    async def stop(self) -> None:
        self._whisper_model = None
        self._translator = None
        self._sp_source = None
        self._sp_target = None
        await self._text_queue.put(None)

    async def process(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        if self._whisper_model is None or self._translator is None:
            await self.start()

        buffer = np.array([], dtype=np.float32)
        samples_needed = int(WHISPER_SAMPLE_RATE * BUFFER_SECONDS)
        min_samples = int(WHISPER_SAMPLE_RATE * MIN_BUFFER_SECONDS)
        loop = asyncio.get_running_loop()

        async for chunk in audio_stream:
            pcm_int16 = np.frombuffer(chunk, dtype=np.int16)
            pcm_float = pcm_int16.astype(np.float32) / 32768.0
            downsampled = _downsample(pcm_float, self._sample_rate, WHISPER_SAMPLE_RATE)
            buffer = np.concatenate([buffer, downsampled])

            if len(buffer) >= samples_needed:
                segment = buffer.copy()
                buffer = np.array([], dtype=np.float32)
                async for out in self._process_segment(segment, loop):
                    yield out

        if len(buffer) >= min_samples:
            async for out in self._process_segment(buffer, loop):
                yield out

        await self._text_queue.put(None)

    async def _process_segment(
        self, audio: np.ndarray, loop: asyncio.AbstractEventLoop
    ) -> AsyncIterator[bytes]:
        texts = await loop.run_in_executor(
            None, partial(_transcribe_sync, self._whisper_model, audio)
        )

        for en_text in texts:
            translate_fn = partial(
                _translate_sync, self._translator, self._sp_source, self._sp_target, en_text
            )
            es_text = await loop.run_in_executor(None, translate_fn)

            await self._text_queue.put(f"[EN] {en_text}")
            await self._text_queue.put(f"[ES] {es_text}")

            pcm_bytes = await _synthesize_spanish(es_text, self._sample_rate)
            if pcm_bytes:
                yield pcm_bytes

    async def process_text(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
        while True:
            text = await self._text_queue.get()
            if text is None:
                return
            yield text
