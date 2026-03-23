"""Shared audio utilities used across pipelines."""

from __future__ import annotations

import asyncio
import io
import logging
import re

import av
import numpy as np

logger = logging.getLogger(__name__)

EDGE_TTS_SAMPLE_RATE = 24000
EDGE_TTS_VOICE = "es-ES-AlvaroNeural"

_SENTENCE_END = re.compile(r"[.!?][\"\'\)\]]*\s*$")
_CLAUSE_END = re.compile(r",[\"\'\)\]]*\s*$")


def downsample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    ratio = dst_rate / src_rate
    n_samples = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, n_samples).astype(np.int64)
    return audio[indices]


def decode_mp3_to_pcm(mp3_data: bytes, target_rate: int) -> bytes:
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


async def synthesize_spanish(text: str, target_rate: int) -> bytes:
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
        return await loop.run_in_executor(None, decode_mp3_to_pcm, mp3_data, target_rate)
    except Exception:
        logger.exception("MP3 decode failed")
        return b""


class SentenceAccumulator:
    """Buffer Whisper segments and flush at natural pause points.

    Segments ending with sentence-final punctuation (``.``, ``?``, ``!``)
    are always flushed.  Comma-terminated clauses are flushed when
    ``flush_on_comma`` is True (the default), producing more frequent,
    shorter audio chunks that align with natural speech pauses.
    Incomplete trailing text is carried over and prepended to the next
    buffer's output.  Call ``flush()`` at end-of-stream to emit any
    remaining text.
    """

    def __init__(self, *, flush_on_comma: bool = True) -> None:
        self._carry = ""
        self._flush_on_comma = flush_on_comma

    def push(self, segments: list[str]) -> list[str]:
        """Accept new ASR segments, return complete clauses/sentences to emit."""
        if not segments:
            return []

        joined = " ".join(segments)
        text = (self._carry + " " + joined).strip() if self._carry else joined
        self._carry = ""

        split_pat = r"(?<=[.!?,])\s+" if self._flush_on_comma else r"(?<=[.!?])\s+"
        sentences: list[str] = []
        for part in re.split(split_pat, text):
            part = part.strip()
            if not part:
                continue
            if _SENTENCE_END.search(part) or (
                self._flush_on_comma and _CLAUSE_END.search(part)
            ):
                sentences.append(part)
            else:
                self._carry = part

        return sentences

    def flush(self) -> list[str]:
        """Emit any remaining text at end-of-stream."""
        if self._carry.strip():
            text = self._carry.strip()
            self._carry = ""
            return [text]
        return []
