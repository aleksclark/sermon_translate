"""Pipeline runner — executes pipelines identically to the transport handler."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np

from src.models import Session
from src.pipelines.base import BasePipeline, OutputStreamKind

logger = logging.getLogger(__name__)

PIPELINE_SAMPLE_RATE = 48000
CHUNK_DURATION_MS = 20
CHUNK_SAMPLES = int(PIPELINE_SAMPLE_RATE * CHUNK_DURATION_MS / 1000)


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------


def load_audio(path: Path, target_rate: int = PIPELINE_SAMPLE_RATE) -> list[bytes]:
    """Decode an audio file into 20 ms s16le PCM chunks at *target_rate*."""
    container = av.open(str(path))
    stream = container.streams.audio[0]
    src_rate = stream.sample_rate

    frames: list[np.ndarray] = []
    for frame in container.decode(audio=0):  # type: ignore[attr-defined]
        frames.append(frame.to_ndarray())

    if not frames:
        raise ValueError(f"No audio frames decoded from {path}")

    audio = np.concatenate(frames, axis=1)
    mono = audio[0] if audio.shape[0] > 1 else audio.flatten()

    # Normalise to float32 [-1, 1] regardless of source format
    if mono.dtype == np.int16:
        mono = mono.astype(np.float32) / 32768.0
    elif mono.dtype == np.int32:
        mono = mono.astype(np.float32) / 2147483648.0
    else:
        mono = mono.astype(np.float32)

    if src_rate != target_rate:
        ratio = target_rate / src_rate
        n_out = int(len(mono) * ratio)
        indices = np.linspace(0, len(mono) - 1, n_out).astype(np.int64)
        mono = mono[indices]

    pcm_int16 = (mono * 32767).clip(-32768, 32767).astype(np.int16)
    raw = pcm_int16.tobytes()

    chunk_bytes = CHUNK_SAMPLES * 2  # s16le = 2 bytes per sample
    return [raw[i : i + chunk_bytes] for i in range(0, len(raw), chunk_bytes)]


# ---------------------------------------------------------------------------
# Timestamped output
# ---------------------------------------------------------------------------


@dataclass
class TimestampedText:
    text: str
    elapsed_seconds: float


@dataclass
class RunResult:
    pipeline_id: str
    audio_duration_seconds: float
    wall_seconds: float
    audio_chunks_out: int
    audio_bytes_out: int
    text_streams: dict[str, list[TimestampedText]] = field(default_factory=dict)
    first_audio_seconds: float | None = None
    first_text_seconds: dict[str, float] = field(default_factory=dict)
    output_audio_raw: bytes = b""
    error: str | None = None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def _audio_stream(chunks: list[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


async def run_pipeline(
    pipeline: BasePipeline,
    chunks: list[bytes],
    *,
    session: Session | None = None,
    sample_rate: int = PIPELINE_SAMPLE_RATE,
) -> RunResult:
    """Execute a pipeline the same way the transport handler does.

    Starts ``process()`` and ``iter_stream()`` concurrently, collects
    all outputs with wall-clock timestamps.
    """
    audio_duration = len(chunks) * CHUNK_DURATION_MS / 1000
    result = RunResult(
        pipeline_id=pipeline.info.id,
        audio_duration_seconds=audio_duration,
        wall_seconds=0.0,
        audio_chunks_out=0,
        audio_bytes_out=0,
    )

    try:
        await pipeline.start()
    except (ImportError, ModuleNotFoundError) as exc:
        result.error = f"deps not installed: {exc}"
        return result

    t0 = time.monotonic()

    try:
        audio_queues: list[asyncio.Queue[bytes | None]] = []

        async def queue_iter(q: asyncio.Queue[bytes | None]) -> AsyncIterator[bytes]:
            while True:
                item = await q.get()
                if item is None:
                    return
                yield item

        async def feed_audio() -> None:
            async for chunk in _audio_stream(chunks):
                for q in audio_queues:
                    await q.put(chunk)
            for q in audio_queues:
                await q.put(None)

        audio_parts: list[bytes] = []

        async def collect_audio(stream: AsyncIterator[bytes]) -> None:
            async for chunk in pipeline.process(stream, session=session):
                elapsed = time.monotonic() - t0
                result.audio_chunks_out += 1
                result.audio_bytes_out += len(chunk)
                audio_parts.append(chunk)
                if result.first_audio_seconds is None:
                    result.first_audio_seconds = elapsed

        async def collect_text(name: str, stream: AsyncIterator[bytes]) -> None:
            it = pipeline.iter_stream(name, stream)
            if it is None:
                return
            texts: list[TimestampedText] = []
            async for text in it:
                elapsed = time.monotonic() - t0
                texts.append(TimestampedText(text=text, elapsed_seconds=elapsed))
                if name not in result.first_text_seconds:
                    result.first_text_seconds[name] = elapsed
            result.text_streams[name] = texts

        tasks: list[asyncio.Task] = []  # type: ignore[type-arg]

        has_audio = False
        for desc in pipeline.output_streams:
            if desc.kind == OutputStreamKind.AUDIO and desc.name == "audio":
                has_audio = True
            elif desc.kind == OutputStreamKind.TEXT:
                q: asyncio.Queue[bytes | None] = asyncio.Queue()
                audio_queues.append(q)
                tasks.append(asyncio.create_task(collect_text(desc.name, queue_iter(q))))

        if has_audio:
            q_audio: asyncio.Queue[bytes | None] = asyncio.Queue()
            audio_queues.append(q_audio)
            tasks.append(asyncio.create_task(collect_audio(queue_iter(q_audio))))

        tasks.append(asyncio.create_task(feed_audio()))
        await asyncio.gather(*tasks)

    except Exception as exc:
        result.error = str(exc)
        logger.exception("pipeline %s failed", pipeline.info.id)
    finally:
        result.wall_seconds = round(time.monotonic() - t0, 3)
        result.output_audio_raw = b"".join(audio_parts)
        await pipeline.stop()

    return result


def transcribe_output_audio(
    result: RunResult,
    sample_rate: int = PIPELINE_SAMPLE_RATE,
    whisper_model: str = "medium",
    language: str = "es",
) -> str:
    """Run Whisper on the pipeline's output audio and return the transcript."""
    from faster_whisper import WhisperModel  # noqa: I001

    from src.pipelines._audio import downsample as ds

    if not result.output_audio_raw:
        return ""

    pcm_int16 = np.frombuffer(result.output_audio_raw, dtype=np.int16)
    pcm_float = pcm_int16.astype(np.float32) / 32768.0
    audio_16k = ds(pcm_float, sample_rate, 16000)

    model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        audio_16k, beam_size=5, language=language, vad_filter=True,
    )
    texts = [seg.text.strip() for seg in segments if seg.text.strip()]
    return " ".join(texts)
