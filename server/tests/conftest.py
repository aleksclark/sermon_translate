from __future__ import annotations

import io
from pathlib import Path

import av
import numpy as np
import pytest

FIXTURE_MP3 = Path(__file__).resolve().parent.parent.parent / "e2e" / "fixtures" / "test-speech.mp3"
PIPELINE_SAMPLE_RATE = 48000


@pytest.fixture(scope="session")
def fixture_pcm_chunks() -> list[bytes]:
    """Decode the e2e fixture MP3 into 48 kHz s16le PCM chunks.

    Returns a list of ~20 ms chunks (960 samples at 48 kHz) matching
    what WebRTC delivers to the server pipelines.
    """
    if not FIXTURE_MP3.exists():
        pytest.skip(f"fixture MP3 not found: {FIXTURE_MP3}")

    mp3_data = FIXTURE_MP3.read_bytes()
    container = av.open(io.BytesIO(mp3_data), format="mp3")
    stream = container.streams.audio[0]
    src_rate = stream.sample_rate

    frames: list[np.ndarray] = []
    for frame in container.decode(audio=0):  # type: ignore[attr-defined]
        frames.append(frame.to_ndarray())

    if not frames:
        pytest.skip("fixture MP3 produced no audio frames")

    audio = np.concatenate(frames, axis=1)
    mono = audio[0] if audio.shape[0] > 1 else audio.flatten()

    # Resample to 48 kHz via linear interpolation (same as _downsample)
    ratio = PIPELINE_SAMPLE_RATE / src_rate
    n_out = int(len(mono) * ratio)
    indices = np.linspace(0, len(mono) - 1, n_out).astype(np.int64)
    resampled = mono[indices]

    pcm_int16 = (resampled * 32767).clip(-32768, 32767).astype(np.int16)
    raw = pcm_int16.tobytes()

    # Split into 20 ms chunks (960 samples * 2 bytes)
    chunk_bytes = 960 * 2
    return [raw[i : i + chunk_bytes] for i in range(0, len(raw), chunk_bytes)]
