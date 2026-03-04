from __future__ import annotations

import asyncio
import logging
import threading
from argparse import Namespace
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from src.models import OutputStreamInfo, PipelineInfo, Session
from src.pipelines.base import BasePipeline, OutputStreamDescriptor, OutputStreamKind
from src.pipelines.spanish import EDGE_TTS_VOICE, _decode_mp3_to_pcm
from src.pipelines.whisper_tts import _downsample

logger = logging.getLogger(__name__)

MODEL_SAMPLE_RATE = 16_000
CHUNK_DURATION_MS = 160
CHUNK_SAMPLES = int(MODEL_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

SEGMENT_SECONDS = 10
SEGMENT_CHUNKS = int(SEGMENT_SECONDS * 1000 / CHUNK_DURATION_MS)
STALL_LIMIT = 15
OVERLAP_CHUNKS = 12
EMIT_AFTER_SILENT = 5
MIN_EMIT_WORDS = 4


def _has_cuda() -> bool:
    import torch

    return torch.cuda.is_available()


async def _synthesize_spanish(text: str, target_rate: int) -> bytes:
    import edge_tts

    if not text or not text.strip():
        return b""

    try:
        communicate = edge_tts.Communicate(text, voice=EDGE_TTS_VOICE)
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


def _build_agent_args(tgt_lang: str = "spa", device: str = "cpu") -> Namespace:
    return Namespace(
        unity_model_name="seamless_streaming_unity",
        monotonic_decoder_model_name="seamless_streaming_monotonic_decoder",
        task="s2tt",
        tgt_lang=tgt_lang,
        device=device,
        dtype="fp16" if "cuda" in device else "fp32",
        fp16="cuda" in device,
        sample_rate=MODEL_SAMPLE_RATE,
        shift_size=10,
        window_size=25,
        feature_dim=80,
        denormalize=False,
        min_starting_wait_w2vbert=192,
        max_len_a=0,
        max_len_b=200,
        max_consecutive_write=15,
        min_starting_wait=1,
        no_early_stop=True,
        decision_threshold=0.7,
        decision_method="min",
        p_choose_start_layer=0,
        block_ngrams=True,
        detokenize_only=True,
    )


def _load_agent(tgt_lang: str, device: str) -> Any:
    from src.pipelines._fairseq2_compat import apply as _apply_compat

    _apply_compat()

    from seamless_communication.streaming.agents.seamless_streaming_s2t import (  # type: ignore[import-not-found]
        SeamlessStreamingS2TAgent,
    )

    _apply_compat()

    args = _build_agent_args(tgt_lang, device)
    agent = SeamlessStreamingS2TAgent.from_args(args)
    return agent


def _push_chunk_sync(
    agent: Any,
    states: Any,
    audio: np.ndarray,
    tgt_lang: str,
    finished: bool = False,
) -> str | None:
    import torch
    from simuleval.data.segments import SpeechSegment  # type: ignore[import-not-found]

    segment = SpeechSegment(
        content=audio.astype(np.float32).tolist(),
        sample_rate=MODEL_SAMPLE_RATE,
        tgt_lang=tgt_lang,
        finished=finished,
    )

    with torch.no_grad():
        output = agent.pushpop(segment, states)

    if output.is_empty:
        return None

    content = output.content
    if isinstance(content, str) and content.strip():
        return _dedup_text(content.strip())
    return None


def _flush_agent_sync(agent: Any, states: Any, tgt_lang: str) -> str | None:
    silence = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    return _push_chunk_sync(agent, states, silence, tgt_lang, finished=True)


def _process_segment_sync(
    agent: Any,
    chunks: list[np.ndarray],
    tgt_lang: str,
) -> list[str]:
    states = agent.build_states()
    texts: list[str] = []
    silent_count = 0

    for i, chunk in enumerate(chunks):
        text = _push_chunk_sync(agent, states, chunk, tgt_lang, finished=False)
        if text:
            texts.append(text)
            silent_count = 0
        else:
            silent_count += 1

        if silent_count >= STALL_LIMIT and i > STALL_LIMIT:
            logger.warning("stall detected at chunk %d/%d, flushing segment", i, len(chunks))
            break

    flush = _flush_agent_sync(agent, states, tgt_lang)
    if flush:
        texts.append(flush)

    return texts


def _dedup_text(text: str) -> str | None:
    tokens = text.split()
    for n in (6, 5, 4, 3, 2):
        changed = True
        while changed:
            changed = False
            i = 0
            while i + 2 * n <= len(tokens):
                if tokens[i : i + n] == tokens[i + n : i + 2 * n]:
                    tokens = tokens[: i + n] + tokens[i + 2 * n :]
                    changed = True
                else:
                    i += 1
    result = " ".join(tokens).strip()
    return result or None


def _detokenize(text: str) -> str:
    pieces = text.split()
    words: list[str] = []
    for piece in pieces:
        if piece.startswith("\u2581"):
            words.append(piece[1:])
        elif words:
            words[-1] += piece
        else:
            words.append(piece)
    return " ".join(w for w in words if w)


def _gpu_worker(
    agent: Any,
    tgt_lang: str,
    chunk_queue: asyncio.Queue[np.ndarray | None],
    text_queue: asyncio.Queue[str | None],
    loop: asyncio.AbstractEventLoop,
) -> None:
    import torch

    all_chunks: list[np.ndarray] = []
    seg_start = 0
    states = agent.build_states()
    silent_count = 0
    seg_texts: list[str] = []
    has_produced_text = False

    def _emit_texts() -> None:
        nonlocal seg_texts, has_produced_text
        if not seg_texts:
            return
        sentence = _detokenize(" ".join(seg_texts))
        deduped = _dedup_text(sentence)
        if deduped and any(c.isalpha() for c in deduped):
            loop.call_soon_threadsafe(text_queue.put_nowait, deduped)
        seg_texts = []
        has_produced_text = False

    def _reset_states(new_start: int) -> None:
        nonlocal states, silent_count, seg_start
        _emit_texts()
        states = agent.build_states()
        silent_count = 0
        overlap_start = max(new_start - OVERLAP_CHUNKS, seg_start)
        seg_start = new_start
        with torch.no_grad():
            for chunk in all_chunks[overlap_start:new_start]:
                _push_chunk_sync(agent, states, chunk, tgt_lang, finished=False)

    while True:
        future = asyncio.run_coroutine_threadsafe(chunk_queue.get(), loop)
        chunk = future.result()
        if chunk is None:
            break

        all_chunks.append(chunk)
        idx = len(all_chunks) - 1

        text = _push_chunk_sync(agent, states, chunk, tgt_lang, finished=False)
        if text:
            seg_texts.append(text)
            silent_count = 0
            has_produced_text = True
        else:
            silent_count += 1

        if has_produced_text and silent_count == EMIT_AFTER_SILENT:
            word_count = sum(len(t.split()) for t in seg_texts)
            if word_count >= MIN_EMIT_WORDS:
                _emit_texts()

        chunks_in_seg = idx - seg_start + 1
        if chunks_in_seg >= SEGMENT_CHUNKS:
            flush = _flush_agent_sync(agent, states, tgt_lang)
            if flush:
                seg_texts.append(flush)
            _reset_states(idx + 1)
        elif silent_count >= STALL_LIMIT and chunks_in_seg > STALL_LIMIT:
            logger.warning("stall at chunk %d (seg_start=%d), resetting", idx, seg_start)
            flush = _flush_agent_sync(agent, states, tgt_lang)
            if flush:
                seg_texts.append(flush)
            _reset_states(idx + 1)

    flush = _flush_agent_sync(agent, states, tgt_lang)
    if flush:
        seg_texts.append(flush)
    _emit_texts()
    loop.call_soon_threadsafe(text_queue.put_nowait, None)


class SeamlessStreamingPipeline(BasePipeline):
    """Simultaneous English→Spanish translation using SeamlessStreaming.

    Three-stage pipeline runs concurrently:
      1. GPU worker thread — feeds chunks to the model, emits text
      2. TTS task — synthesizes text to speech (runs on asyncio loop)
      3. Audio yield — streams PCM to the client immediately

    Audio is segmented with fresh decoder states every SEGMENT_SECONDS to
    prevent O(n²) encoder blowup.  OVERLAP_CHUNKS of audio are replayed
    into fresh states so the encoder has context across boundaries.
    """

    def __init__(self, sample_rate: int = 48000) -> None:
        self._sample_rate = sample_rate
        self._agent: Any = None
        self._tgt_lang = "spa"
        self._device = "cuda" if _has_cuda() else "cpu"
        self._es_queue: asyncio.Queue[str | None] = asyncio.Queue()

    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            id="seamless-streaming",
            name="Spanish Streaming (SeamlessStreaming)",
            description=(
                "Simultaneous English→Spanish translation using Meta's "
                "SeamlessStreaming with EMMA monotonic attention. Outputs "
                "only new tokens incrementally — no repeated context."
            ),
            output_streams=[
                OutputStreamInfo(name=s.name, kind=s.kind.value, label=s.label)
                for s in self.output_streams
            ],
        )

    @property
    def output_streams(self) -> list[OutputStreamDescriptor]:
        return [
            OutputStreamDescriptor(
                name="audio", kind=OutputStreamKind.AUDIO, label="Spanish Audio",
            ),
            OutputStreamDescriptor(
                name="es-transcript", kind=OutputStreamKind.TEXT, label="Spanish",
            ),
        ]

    def configure_session(self, session: Session) -> None:
        pass

    async def start(self) -> None:
        if self._agent is not None:
            return
        loop = asyncio.get_running_loop()
        self._agent = await loop.run_in_executor(
            None, _load_agent, self._tgt_lang, self._device,
        )
        logger.info("SeamlessStreaming agent loaded (device=%s)", self._device)

    async def stop(self) -> None:
        self._agent = None
        await self._es_queue.put(None)

    async def process(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        if self._agent is None:
            await self.start()

        loop = asyncio.get_running_loop()
        chunk_queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
        text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        gpu_thread = threading.Thread(
            target=_gpu_worker,
            args=(self._agent, self._tgt_lang, chunk_queue, text_queue, loop),
            daemon=True,
        )
        gpu_thread.start()

        tts_sem = asyncio.Semaphore(3)
        next_seq = 0
        pending_audio: dict[int, bytes] = {}
        emit_seq = 0
        emit_lock = asyncio.Lock()

        async def _tts_one(seq: int, text: str) -> None:
            nonlocal emit_seq
            async with tts_sem:
                pcm = await _synthesize_spanish(text, self._sample_rate)
            if pcm:
                logger.info("TTS produced %d bytes (seq=%d)", len(pcm), seq)
            async with emit_lock:
                if pcm:
                    pending_audio[seq] = pcm
                else:
                    pending_audio[seq] = b""
                while emit_seq in pending_audio:
                    data = pending_audio.pop(emit_seq)
                    if data:
                        await audio_queue.put(data)
                    emit_seq += 1

        async def tts_task() -> None:
            nonlocal next_seq
            tasks: list[asyncio.Task[None]] = []
            while True:
                text = await text_queue.get()
                if text is None:
                    break
                logger.info("segment translated: %r", text[:120])
                await self._es_queue.put(text)
                seq = next_seq
                next_seq += 1
                tasks.append(asyncio.create_task(_tts_one(seq, text)))
            if tasks:
                await asyncio.gather(*tasks)
            await audio_queue.put(None)

        tts = asyncio.create_task(tts_task())

        async def feed_chunks() -> None:
            buffer = np.array([], dtype=np.float32)
            async for raw in audio_stream:
                pcm_int16 = np.frombuffer(raw, dtype=np.int16)
                pcm_float = pcm_int16.astype(np.float32) / 32768.0
                downsampled = _downsample(pcm_float, self._sample_rate, MODEL_SAMPLE_RATE)
                buffer = np.concatenate([buffer, downsampled])
                while len(buffer) >= CHUNK_SAMPLES:
                    await chunk_queue.put(buffer[:CHUNK_SAMPLES].copy())
                    buffer = buffer[CHUNK_SAMPLES:]
            if len(buffer) > 0:
                padded = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
                padded[: len(buffer)] = buffer
                await chunk_queue.put(padded)
            await chunk_queue.put(None)

        feed = asyncio.create_task(feed_chunks())

        while True:
            pcm = await audio_queue.get()
            if pcm is None:
                break
            yield pcm

        await feed
        await tts
        await self._es_queue.put(None)

    def iter_stream(
        self, name: str, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        if name == "es-transcript":
            return self._drain_queue(self._es_queue)
        return None

    @staticmethod
    async def _drain_queue(q: asyncio.Queue[str | None]) -> AsyncIterator[str]:
        while True:
            text = await q.get()
            if text is None:
                return
            yield text
