"""Shared audio utilities used across pipelines."""

from __future__ import annotations

import asyncio
import io
import logging

import av
import numpy as np

logger = logging.getLogger(__name__)

EDGE_TTS_SAMPLE_RATE = 24000
EDGE_TTS_VOICE = "es-ES-AlvaroNeural"


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
