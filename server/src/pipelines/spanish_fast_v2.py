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

from src.models import PipelineInfo, Session
from src.pipelines._audio import EDGE_TTS_VOICE, downsample
from src.pipelines.base import BasePipeline, OutputStreamDescriptor, OutputStreamKind

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000
BUFFER_SECONDS = 5
MIN_BUFFER_SECONDS = 1.5
TRANSLATION_MODEL_ID = "Helsinki-NLP/opus-mt-en-es"
TTS_CONCURRENCY = 3
EDGE_TTS_SAMPLE_RATE = 24000
TARGET_RATIO = 1.0
MAX_RATE_BOOST = 40


def _transcribe_sync(model: Any, audio: np.ndarray) -> list[str]:
    segments, _ = model.transcribe(audio, beam_size=5, language="en", vad_filter=True)
    return [seg.text.strip() for seg in segments if seg.text.strip()]


def _translate_batch_sync(
    translator: Any,
    sp_source: Any,
    sp_target: Any,
    texts: list[str],
) -> list[str]:
    all_tokens = [sp_source.encode(t, out_type=str) + ["</s>"] for t in texts]
    results = translator.translate_batch(all_tokens)
    return [sp_target.decode(r.hypotheses[0]) for r in results]


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
    resampled = downsample(pcm_float, EDGE_TTS_SAMPLE_RATE, target_rate)
    pcm_int16 = (resampled * 32767).clip(-32768, 32767).astype(np.int16)
    return pcm_int16.tobytes()


async def _synthesize_with_rate(
    text: str,
    target_rate: int,
    rate_pct: int = 0,
) -> bytes:
    import edge_tts

    if not text or not text.strip():
        return b""

    rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
    try:
        communicate = edge_tts.Communicate(text, voice=EDGE_TTS_VOICE, rate=rate_str)
        mp3_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_data += chunk.get("data", b"")
    except Exception:
        logger.exception("edge-tts failed for text: %r", text[:80])
        return b""

    if not mp3_data:
        return b""

    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(None, _decode_mp3_to_pcm, mp3_data, target_rate)
    except Exception:
        logger.exception("MP3 decode failed")
        return b""


class SpanishFastV2Pipeline(BasePipeline):
    """English audio → Spanish audio + transcripts (v2).

    Improvements over v1:
      - Batched Opus-MT translation (all segments from a buffer at once)
      - Batched TTS (all translated segments joined, one Edge TTS call per buffer)
      - Adaptive TTS rate to prevent output falling behind input
      - Concurrent ASR with TTS playback
    """

    def __init__(
        self,
        whisper_model_size: str = "small",
        sample_rate: int = 48000,
    ) -> None:
        super().__init__()
        self._whisper_model_size = whisper_model_size
        self._sample_rate = sample_rate
        self._whisper_model: Any = None
        self._translator: Any = None
        self._sp_source: Any = None
        self._sp_target: Any = None
        self._en_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._es_queue: asyncio.Queue[str | None] = asyncio.Queue()

    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            id="spanish-fast-v2",
            name="Spanish Fast v2 (adaptive rate)",
            description=(
                "Whisper small → batched Opus-MT → Edge TTS with adaptive "
                "rate control to stay in sync with the speaker."
            ),
            output_streams=self._build_output_stream_info(),
        )

    @property
    def output_streams(self) -> list[OutputStreamDescriptor]:
        return [
            OutputStreamDescriptor(
                name="audio", kind=OutputStreamKind.AUDIO, label="Spanish Audio",
            ),
            OutputStreamDescriptor(
                name="en-transcript", kind=OutputStreamKind.TEXT, label="English",
            ),
            OutputStreamDescriptor(
                name="es-transcript", kind=OutputStreamKind.TEXT, label="Spanish",
            ),
        ]

    async def _do_start(self) -> None:
        loop = asyncio.get_running_loop()
        if self._whisper_model is None:
            self._whisper_model = await loop.run_in_executor(
                None, self._load_whisper,
            )
            logger.info("Whisper '%s' loaded", self._whisper_model_size)
        if self._translator is None:
            t, s, d = await loop.run_in_executor(None, self._load_translation)
            self._translator, self._sp_source, self._sp_target = t, s, d
            logger.info("Opus-MT loaded")

    def _load_whisper(self) -> Any:
        from faster_whisper import WhisperModel

        return WhisperModel(
            self._whisper_model_size, device="cpu", compute_type="int8",
        )

    def _load_translation(self) -> tuple[Any, Any, Any]:
        import ctranslate2
        from huggingface_hub import snapshot_download

        ct2_dir = self._get_ct2_model_dir()
        hf_dir = snapshot_download(TRANSLATION_MODEL_ID)
        translator = ctranslate2.Translator(
            ct2_dir, device="cpu", compute_type="int8",
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
            from src.pipelines.spanish import SpanishTranslationPipeline

            SpanishTranslationPipeline._convert_model(cache_dir)
        return cache_dir

    async def _do_stop(self) -> None:
        self._whisper_model = None
        self._translator = None
        self._sp_source = None
        self._sp_target = None
        await self._en_queue.put(None)
        await self._es_queue.put(None)

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None,
    ) -> AsyncIterator[bytes]:
        if self._whisper_model is None or self._translator is None:
            await self.start()

        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        cumulative_input_s = 0.0
        cumulative_output_s = 0.0

        def _compute_rate() -> int:
            if cumulative_input_s < 1.0:
                return 0
            ratio = cumulative_output_s / cumulative_input_s
            if ratio <= TARGET_RATIO:
                return 0
            boost = int((ratio - TARGET_RATIO) * 100)
            return min(boost, MAX_RATE_BOOST)

        async def _process_buffer(segment: np.ndarray, input_s: float) -> None:
            nonlocal cumulative_input_s, cumulative_output_s
            cumulative_input_s += input_s

            en_texts = await loop.run_in_executor(
                None,
                partial(_transcribe_sync, self._whisper_model, segment),
            )
            if not en_texts:
                return

            es_texts = await loop.run_in_executor(
                None,
                partial(
                    _translate_batch_sync,
                    self._translator,
                    self._sp_source,
                    self._sp_target,
                    en_texts,
                ),
            )

            for en, es in zip(en_texts, es_texts, strict=True):
                await self._en_queue.put(en)
                await self._es_queue.put(es)

            joined_es = " ".join(es_texts)
            rate_pct = _compute_rate()
            if rate_pct > 0:
                logger.info(
                    "adaptive rate +%d%% (in=%.1fs, out=%.1fs)",
                    rate_pct, cumulative_input_s, cumulative_output_s,
                )

            pcm = await _synthesize_with_rate(joined_es, self._sample_rate, rate_pct)
            if pcm:
                output_s = len(pcm) / (self._sample_rate * 2)
                cumulative_output_s += output_s
                await audio_queue.put(pcm)

        async def ingest_and_process() -> None:
            buffer = np.array([], dtype=np.float32)
            samples_needed = int(WHISPER_SAMPLE_RATE * BUFFER_SECONDS)
            min_samples = int(WHISPER_SAMPLE_RATE * MIN_BUFFER_SECONDS)

            async for chunk in audio_stream:
                pcm_int16 = np.frombuffer(chunk, dtype=np.int16)
                pcm_float = pcm_int16.astype(np.float32) / 32768.0
                downsampled = downsample(
                    pcm_float, self._sample_rate, WHISPER_SAMPLE_RATE,
                )
                buffer = np.concatenate([buffer, downsampled])

                if len(buffer) >= samples_needed:
                    segment = buffer.copy()
                    buffer = np.array([], dtype=np.float32)
                    await _process_buffer(segment, BUFFER_SECONDS)

            if len(buffer) >= min_samples:
                buf_s = len(buffer) / WHISPER_SAMPLE_RATE
                await _process_buffer(buffer, buf_s)

            await audio_queue.put(None)

        ingest_task = asyncio.create_task(ingest_and_process())

        while True:
            data = await audio_queue.get()
            if data is None:
                break
            yield data

        await ingest_task
        await self._en_queue.put(None)
        await self._es_queue.put(None)

    def iter_stream(
        self, name: str, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        if name == "en-transcript":
            return self._drain_queue(self._en_queue)
        if name == "es-transcript":
            return self._drain_queue(self._es_queue)
        return None
