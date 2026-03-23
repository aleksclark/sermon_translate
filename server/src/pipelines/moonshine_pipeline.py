"""Moonshine streaming ASR → Opus-MT → Edge TTS pipeline.

No manual audio chunking — Moonshine decides when sentences are complete
and emits them as FINAL events. Translation and TTS fire per-sentence,
producing natural audio at sentence boundaries.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

import numpy as np
import sentencepiece as spm

from src.models import PipelineInfo, Session
from src.pipelines._audio import downsample
from src.pipelines.base import BasePipeline, OutputStreamDescriptor, OutputStreamKind
from src.pipelines.gpu_pipelines import _synthesize_edge_tts

logger = logging.getLogger(__name__)

MOONSHINE_SAMPLE_RATE = 16000
TRANSLATION_MODEL_ID = "Helsinki-NLP/opus-mt-en-es"
TARGET_RATIO = 1.05
MAX_RATE_BOOST = 40
TTS_CONCURRENCY = 3


def _translate_batch_sync(
    translator: Any,
    sp_source: Any,
    sp_target: Any,
    texts: list[str],
) -> list[str]:
    all_tokens = [sp_source.encode(t, out_type=str) + ["</s>"] for t in texts]
    results = translator.translate_batch(all_tokens)
    return [sp_target.decode(r.hypotheses[0]) for r in results]


class MoonshineStreamingPipeline(BasePipeline):
    """Moonshine streaming ASR → Opus-MT → Edge TTS.

    Audio flows in as small chunks (20ms WebRTC frames). Moonshine
    processes them incrementally and emits FINAL sentences when it
    detects natural pause points. Each final sentence is translated
    and synthesised immediately, producing audio at natural sentence
    boundaries with no manual buffering.

    Partial (in-progress) transcriptions are pushed to the
    ``en-partial`` text stream for live UI display.
    """

    def __init__(self, sample_rate: int = 48000) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._moonshine_path: str | None = None
        self._moonshine_arch: Any = None
        self._translator: Any = None
        self._sp_source: Any = None
        self._sp_target: Any = None
        self._en_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._es_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._partial_queue: asyncio.Queue[str | None] = asyncio.Queue()

    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            id="moonshine-streaming",
            name="Moonshine Streaming + Opus-MT",
            description=(
                "True streaming ASR via Moonshine — no manual chunking. "
                "Emits audio at natural sentence boundaries. "
                "Opus-MT translation, Edge TTS synthesis."
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
                name="en-partial", kind=OutputStreamKind.TEXT, label="English (live)",
            ),
            OutputStreamDescriptor(
                name="en-transcript", kind=OutputStreamKind.TEXT, label="English",
            ),
            OutputStreamDescriptor(
                name="es-transcript", kind=OutputStreamKind.TEXT, label="Spanish",
            ),
        ]

    async def _do_start(self) -> None:
        import moonshine_voice as mv

        loop = asyncio.get_running_loop()
        if self._moonshine_path is None:
            path, arch = await loop.run_in_executor(
                None,
                lambda: mv.get_model_for_language("en", mv.ModelArch.MEDIUM_STREAMING),
            )
            self._moonshine_path = path
            self._moonshine_arch = arch
            logger.info("Moonshine model ready: %s", path)
        if self._translator is None:
            self._translator, self._sp_source, self._sp_target = (
                await loop.run_in_executor(None, self._load_translation)
            )
            logger.info("Opus-MT loaded")

    def _load_translation(self) -> tuple[Any, Any, Any]:
        import ctranslate2
        from huggingface_hub import snapshot_download

        from src.pipelines.spanish_fast import SpanishFastPipeline

        ct2_dir = SpanishFastPipeline._get_ct2_model_dir()
        hf_dir = snapshot_download(TRANSLATION_MODEL_ID)
        translator = ctranslate2.Translator(ct2_dir, device="cpu", compute_type="int8")
        sp_src = spm.SentencePieceProcessor()
        sp_src.load(f"{hf_dir}/source.spm")  # type: ignore[attr-defined]
        sp_tgt = spm.SentencePieceProcessor()
        sp_tgt.load(f"{hf_dir}/target.spm")  # type: ignore[attr-defined]
        return translator, sp_src, sp_tgt

    async def _do_stop(self) -> None:
        self._translator = None
        self._sp_source = None
        self._sp_target = None
        await self._en_queue.put(None)
        await self._es_queue.put(None)
        await self._partial_queue.put(None)

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None,
    ) -> AsyncIterator[bytes]:
        if self._moonshine_path is None:
            await self.start()

        import moonshine_voice as mv
        from moonshine_voice.transcriber import TranscriptEventListener

        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
        tts_sem = asyncio.Semaphore(TTS_CONCURRENCY)

        cumul_in_s = 0.0
        cumul_out_s = 0.0
        next_seq = 0
        pending: dict[int, bytes] = {}
        emit_seq = 0
        emit_lock = asyncio.Lock()

        def _compute_rate() -> int:
            if cumul_in_s < 1.0:
                return 0
            ratio = cumul_out_s / cumul_in_s
            if ratio <= TARGET_RATIO:
                return 0
            return min(int((ratio - TARGET_RATIO) * 100), MAX_RATE_BOOST)

        class _Listener(TranscriptEventListener):
            def on_line_text_changed(self, event: Any) -> None:
                loop.call_soon_threadsafe(
                    self._partial_queue.put_nowait, event.line.text,
                )

            def on_line_completed(self, event: Any) -> None:
                text = event.line.text.strip()
                if text:
                    loop.call_soon_threadsafe(sentence_queue.put_nowait, text)

        listener = _Listener()
        listener._partial_queue = self._partial_queue  # type: ignore[attr-defined]

        def _run_moonshine() -> None:
            transcriber = mv.Transcriber(
                self._moonshine_path,
                self._moonshine_arch,
                update_interval=0.3,
            )
            transcriber.add_listener(listener)
            transcriber.start()

            chunk_q_local = chunk_queue
            while True:
                future = asyncio.run_coroutine_threadsafe(
                    chunk_q_local.get(), loop,
                )
                chunk = future.result()
                if chunk is None:
                    break
                transcriber.add_audio(chunk, MOONSHINE_SAMPLE_RATE)

            transcriber.stop()
            loop.call_soon_threadsafe(sentence_queue.put_nowait, None)

        chunk_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

        async def feed_audio() -> None:
            nonlocal cumul_in_s
            async for raw in audio_stream:
                pcm_int16 = np.frombuffer(raw, dtype=np.int16)
                pcm_float = pcm_int16.astype(np.float32) / 32768.0
                ds = downsample(pcm_float, self._sample_rate, MOONSHINE_SAMPLE_RATE)
                cumul_in_s += len(ds) / MOONSHINE_SAMPLE_RATE
                await chunk_queue.put(ds)
            await chunk_queue.put(None)

        async def _tts_one(seq: int, es_text: str) -> None:
            nonlocal emit_seq, cumul_out_s
            rate_pct = _compute_rate()
            async with tts_sem:
                pcm = await _synthesize_edge_tts(
                    es_text, self._sample_rate, rate_pct,
                )
            async with emit_lock:
                pending[seq] = pcm if pcm else b""
                while emit_seq in pending:
                    data = pending.pop(emit_seq)
                    if data:
                        cumul_out_s += len(data) / (self._sample_rate * 2)
                        await audio_queue.put(data)
                    emit_seq += 1

        async def translate_and_tts() -> None:
            nonlocal next_seq
            tts_tasks: list[asyncio.Task[None]] = []
            while True:
                en_text = await sentence_queue.get()
                if en_text is None:
                    break

                await self._en_queue.put(en_text)

                es_texts = await loop.run_in_executor(
                    None,
                    partial(
                        _translate_batch_sync,
                        self._translator,
                        self._sp_source,
                        self._sp_target,
                        [en_text],
                    ),
                )
                es_text = es_texts[0] if es_texts else ""
                if es_text:
                    await self._es_queue.put(es_text)
                    seq = next_seq
                    next_seq += 1
                    tts_tasks.append(
                        asyncio.create_task(_tts_one(seq, es_text)),
                    )

            if tts_tasks:
                await asyncio.gather(*tts_tasks)
            await audio_queue.put(None)

        moonshine_thread = threading.Thread(target=_run_moonshine, daemon=True)
        moonshine_thread.start()

        feed_task = asyncio.create_task(feed_audio())
        tts_task = asyncio.create_task(translate_and_tts())

        while True:
            data = await audio_queue.get()
            if data is None:
                break
            yield data

        await feed_task
        await tts_task
        await self._en_queue.put(None)
        await self._es_queue.put(None)
        await self._partial_queue.put(None)

    def iter_stream(
        self, name: str, audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        if name == "en-transcript":
            return self._drain_queue(self._en_queue)
        if name == "es-transcript":
            return self._drain_queue(self._es_queue)
        if name == "en-partial":
            return self._drain_queue(self._partial_queue)
        return None
