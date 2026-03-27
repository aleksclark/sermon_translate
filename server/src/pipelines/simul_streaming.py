"""SimulStreaming pipeline — true simultaneous ASR + translation.

Uses the AlignAtt attention-guided policy from UFAL's SimulStreaming
(IWSLT 2025 winner) for real-time transcription that emits words as
they become stable — no fixed buffer windows.

Pull architecture: ASR pushes English sentences into a queue.  A worker
loop pulls from it, coalescing and summarising under backpressure, then
translates and synthesises.  The worker can cancel in-flight TTS and
re-synth a shorter version if the output falls behind.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

import numpy as np
import sentencepiece as spm
import torch

from src.models import PipelineInfo, Session
from src.pipelines._audio import SentenceAccumulator, downsample
from src.pipelines.base import BasePipeline, OutputStreamDescriptor, OutputStreamKind
from src.pipelines.gpu_pipelines import _synthesize_edge_tts

logger = logging.getLogger(__name__)

WHISPER_SR = 16000
CHUNK_SECONDS = 1.0
TRANSLATION_MODEL_ID = "Helsinki-NLP/opus-mt-en-es"
MAX_RATE_BOOST = 50
TTS_CONCURRENCY = 3
SILENCE_THRESHOLD = 400
MIN_TAIL_MS = 80
BACKPRESSURE_THRESHOLD_S = 3.0
RESYNTH_LEAD_TIME_S = 1.5


def _trim_trailing_silence(pcm: bytes, sample_rate: int, keep_ms: int = MIN_TAIL_MS) -> bytes:
    arr = np.frombuffer(pcm, dtype=np.int16)
    if len(arr) == 0:
        return pcm
    keep_samples = int(sample_rate * keep_ms / 1000)
    end = len(arr)
    while end > keep_samples and abs(int(arr[end - 1])) < SILENCE_THRESHOLD:
        end -= 1
    end = min(end + keep_samples, len(arr))
    return arr[:end].tobytes()


def _translate_batch_sync(
    translator: Any, sp_src: Any, sp_tgt: Any, texts: list[str],
) -> list[str]:
    all_tokens = [sp_src.encode(t, out_type=str) + ["</s>"] for t in texts]
    results = translator.translate_batch(all_tokens)
    return [sp_tgt.decode(r.hypotheses[0]) for r in results]


def _summarize_english(sentences: list[str]) -> str:
    seen: set[str] = set()
    deduped: list[str] = []
    for s in sentences:
        norm = s.lower().strip()
        if norm not in seen:
            seen.add(norm)
            deduped.append(s)
    return " ".join(deduped)


class SimulStreamingPipeline(BasePipeline):
    """True simultaneous ASR → Opus-MT → Edge TTS with pull-based backpressure."""

    def __init__(self, sample_rate: int = 48000) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._model: Any = None
        self._translator: Any = None
        self._sp_src: Any = None
        self._sp_tgt: Any = None
        self._en_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._es_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._partial_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._coalesced_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._pending_en: asyncio.Queue[str | None] = asyncio.Queue()
        self._cumul_out_s = 0.0
        self._first_emit_wall: float | None = None
        self._active_tts_task: asyncio.Task[None] | None = None

    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            id="simul-streaming",
            name="SimulStreaming (AlignAtt + Opus-MT)",
            description=(
                "True simultaneous transcription via AlignAtt policy "
                "(IWSLT 2025 winner). Pull-based backpressure coalesces "
                "and re-synths when output falls behind."
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
                name="en-coalesced", kind=OutputStreamKind.TEXT, label="To Translator",
            ),
            OutputStreamDescriptor(
                name="es-transcript", kind=OutputStreamKind.TEXT, label="Spanish",
            ),
        ]

    def get_buffer_stats(self) -> tuple[int, float]:
        return self._pending_en.qsize(), self._queued_seconds()

    def _queued_seconds(self) -> float:
        if self._first_emit_wall is None:
            return 0.0
        return self._cumul_out_s - (time.monotonic() - self._first_emit_wall)

    def _compute_rate(self) -> int:
        qs = self._queued_seconds()
        if qs < BACKPRESSURE_THRESHOLD_S:
            return 0
        boost = int(min(qs - 2.0, 10.0) * 5)
        return min(boost, MAX_RATE_BOOST)

    def _get_synth_fn(self) -> Any:
        """Return the async TTS function. Override for voice cloning."""
        return _synthesize_edge_tts

    async def _do_start(self) -> None:
        loop = asyncio.get_running_loop()
        if self._model is None:
            self._model = await loop.run_in_executor(None, self._load_whisper)
            logger.info("SimulStreaming Whisper loaded")
        if self._translator is None:
            self._translator, self._sp_src, self._sp_tgt = await loop.run_in_executor(
                None, self._load_translation,
            )
            logger.info("Opus-MT loaded")

    @staticmethod
    def _load_whisper() -> Any:
        from src.vendor.simulstreaming.whisper.simul_whisper.config import AlignAttConfig
        from src.vendor.simulstreaming.whisper.simul_whisper.simul_whisper import (
            PaddedAlignAttWhisper,
        )

        cfg = AlignAttConfig(
            model_path="large-v3",
            language="en",
            task="transcribe",
            frame_threshold=4,
            audio_max_len=30.0,
            audio_min_len=0.5,
            segment_length=CHUNK_SECONDS,
            decoder_type="greedy",
            beam_size=1,
            never_fire=True,
            logdir=None,
        )
        return PaddedAlignAttWhisper(cfg)

    @staticmethod
    def _load_translation() -> tuple[Any, Any, Any]:
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
        self._model = None
        self._translator = None
        self._sp_src = None
        self._sp_tgt = None
        await self._en_queue.put(None)
        await self._es_queue.put(None)
        await self._partial_queue.put(None)
        await self._coalesced_queue.put(None)
        await self._pending_en.put(None)

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None,
    ) -> AsyncIterator[bytes]:
        if self._model is None:
            await self.start()

        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        acc = SentenceAccumulator()

        self._cumul_out_s = 0.0
        self._first_emit_wall = None
        self._pending_en = asyncio.Queue()

        # --- Worker: pulls English sentences, translates, synthesises ---
        async def translate_tts_worker() -> None:
            while True:
                en = await self._pending_en.get()
                if en is None:
                    break

                # Drain any additional queued sentences
                batch = [en]
                while not self._pending_en.empty():
                    extra = self._pending_en.get_nowait()
                    if extra is None:
                        await self._pending_en.put(None)
                        break
                    batch.append(extra)

                # Under backpressure: coalesce
                qs = self._queued_seconds()
                if qs > BACKPRESSURE_THRESHOLD_S and len(batch) > 1:
                    summary = _summarize_english(batch)
                    logger.info(
                        "backpressure: coalescing %d sentences (queued=%.1fs, pending=%d)",
                        len(batch), qs, self._pending_en.qsize(),
                    )
                    batch = [summary]

                # Translate
                es_texts = await loop.run_in_executor(
                    None,
                    partial(
                        _translate_batch_sync,
                        self._translator, self._sp_src, self._sp_tgt, batch,
                    ),
                )

                joined_en = " ".join(batch)
                joined_es = " ".join(es_texts)
                await self._coalesced_queue.put(joined_en)
                await self._en_queue.put(joined_en)
                await self._es_queue.put(joined_es)

                # Synthesise
                rate_pct = self._compute_rate()
                if rate_pct > 0:
                    logger.info(
                        "adaptive rate +%d%% (queued=%.1fs)", rate_pct, self._queued_seconds(),
                    )
                synth_fn = self._get_synth_fn()
                pcm = await synth_fn(joined_es, self._sample_rate, rate_pct)
                if pcm and rate_pct > 0:
                    pcm = _trim_trailing_silence(pcm, self._sample_rate)
                if pcm:
                    if self._first_emit_wall is None:
                        self._first_emit_wall = time.monotonic()
                    self._cumul_out_s += len(pcm) / (self._sample_rate * 2)
                    await audio_queue.put(pcm)

            await audio_queue.put(None)

        # --- ASR ingest loop ---
        async def ingest_and_process() -> None:
            chunk_samples = int(WHISPER_SR * CHUNK_SECONDS)
            partial_text = ""
            accum = np.array([], dtype=np.float32)

            async for raw in audio_stream:
                pcm_int16 = np.frombuffer(raw, dtype=np.int16)
                pcm_float = pcm_int16.astype(np.float32) / 32768.0
                ds = downsample(pcm_float, self._sample_rate, WHISPER_SR)
                accum = np.concatenate([accum, ds])

                if len(accum) >= chunk_samples:
                    segment = torch.from_numpy(accum)
                    accum = np.array([], dtype=np.float32)
                    self._model.insert_audio(segment)

                    tokens, _ = await loop.run_in_executor(
                        None, self._model.infer, False,
                    )
                    if tokens:
                        text = self._model.tokenizer.decode(tokens)
                        partial_text += text
                        await self._partial_queue.put(partial_text)

                        sentences = acc.push([text])
                        for s in sentences:
                            await self._pending_en.put(s)
                        if sentences:
                            partial_text = ""

            if len(accum) > 0:
                self._model.insert_audio(torch.from_numpy(accum))

            tokens, _ = await loop.run_in_executor(
                None, self._model.infer, True,
            )
            if tokens:
                text = self._model.tokenizer.decode(tokens)
                remaining = acc.push([text]) + acc.flush()
            else:
                remaining = acc.flush()

            for s in remaining:
                await self._pending_en.put(s)

            self._model.refresh_segment(complete=True)
            await self._pending_en.put(None)

        worker_task = asyncio.create_task(translate_tts_worker())
        ingest_task = asyncio.create_task(ingest_and_process())

        while True:
            data = await audio_queue.get()
            if data is None:
                break
            yield data

        await ingest_task
        await worker_task
        await self._en_queue.put(None)
        await self._es_queue.put(None)
        await self._partial_queue.put(None)
        await self._coalesced_queue.put(None)

    def iter_stream(
        self, name: str, audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        if name == "en-transcript":
            return self._drain_queue(self._en_queue)
        if name == "es-transcript":
            return self._drain_queue(self._es_queue)
        if name == "en-partial":
            return self._drain_queue(self._partial_queue)
        if name == "en-coalesced":
            return self._drain_queue(self._coalesced_queue)
        return None
