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

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000
BUFFER_SECONDS = 5
MIN_BUFFER_SECONDS = 1.5
TRANSLATION_MODEL_ID = "Helsinki-NLP/opus-mt-en-es"
TTS_CONCURRENCY = 3


def _transcribe_sync(model: Any, audio: np.ndarray) -> list[str]:
    segments, _ = model.transcribe(audio, beam_size=5, language="en", vad_filter=True)
    results = []
    for seg in segments:
        text = seg.text.strip()
        if text:
            results.append(text)
    return results


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


class SpanishFastPipeline(BasePipeline):
    """English audio → Spanish audio + transcripts.

    Whisper medium (10s buffers) → Opus-MT → Edge TTS with concurrent synthesis.
    Optimised for accuracy and low latency over raw throughput.
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
            id="spanish-fast",
            name="Spanish Fast (Whisper medium + Opus-MT)",
            description=(
                "English → Spanish via Whisper medium ASR, Opus-MT "
                "translation, Edge TTS. 10s buffers, concurrent TTS."
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
        tts_sem = asyncio.Semaphore(TTS_CONCURRENCY)
        next_seq = 0
        pending: dict[int, bytes] = {}
        emit_seq = 0
        emit_lock = asyncio.Lock()

        async def _tts_one(seq: int, text: str) -> None:
            nonlocal emit_seq
            async with tts_sem:
                pcm = await synthesize_spanish(text, self._sample_rate)
            async with emit_lock:
                pending[seq] = pcm if pcm else b""
                while emit_seq in pending:
                    data = pending.pop(emit_seq)
                    if data:
                        await audio_queue.put(data)
                    emit_seq += 1

        async def _process_segments(segments: list[str]) -> list[asyncio.Task[None]]:
            nonlocal next_seq
            tasks: list[asyncio.Task[None]] = []
            for en_text in segments:
                translate_fn = partial(
                    _translate_sync,
                    self._translator,
                    self._sp_source,
                    self._sp_target,
                    en_text,
                )
                es_text = await loop.run_in_executor(None, translate_fn)
                await self._en_queue.put(en_text)
                await self._es_queue.put(es_text)
                seq = next_seq
                next_seq += 1
                tasks.append(asyncio.create_task(_tts_one(seq, es_text)))
            return tasks

        async def ingest_and_process() -> None:
            all_tts: list[asyncio.Task[None]] = []
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
                    texts = await loop.run_in_executor(
                        None,
                        partial(_transcribe_sync, self._whisper_model, segment),
                    )
                    if texts:
                        tasks = await _process_segments(texts)
                        all_tts.extend(tasks)

            if len(buffer) >= min_samples:
                texts = await loop.run_in_executor(
                    None,
                    partial(_transcribe_sync, self._whisper_model, buffer),
                )
                if texts:
                    tasks = await _process_segments(texts)
                    all_tts.extend(tasks)

            if all_tts:
                await asyncio.gather(*all_tts)
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
