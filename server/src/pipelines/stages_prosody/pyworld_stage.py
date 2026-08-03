from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pyworld  # type: ignore[import-not-found]

from src.models import MetadataEnvelope, MetadataKind, ProsodyFrame, StageInfo, StageKind


class PyworldProsodyStage:
    """Prosody tracker using pyworld F0 + energy + pause detection."""

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        frame_ms: float = 100.0,
        silence_rms: float = 0.01,
        cache: Any = None,
        **_: object,
    ) -> None:
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._silence_rms = silence_rms
        self._cache = cache

    @property
    def frame_samples(self) -> int:
        return max(1, int(self._sample_rate * self._frame_ms / 1000.0))

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def analyze(
        self, audio_stream: AsyncIterator[bytes], stream_name: str
    ) -> AsyncIterator[MetadataEnvelope]:
        frame_samples = self.frame_samples
        buffer = np.zeros(0, dtype=np.float64)
        sequence = 0
        emitted_ms = 0.0
        hop = frame_samples / self._sample_rate

        async for chunk in audio_stream:
            pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float64) / 32768.0
            buffer = np.concatenate([buffer, pcm])
            while buffer.size >= frame_samples:
                frame, buffer = buffer[:frame_samples], buffer[frame_samples:]
                energy = float(np.sqrt(np.mean(np.square(frame))))
                is_pause = energy < self._silence_rms
                f0_hz = None
                pitch_confidence = 0.0
                if not is_pause:
                    f0, _time = pyworld.dio(
                        frame, self._sample_rate, frame_period=hop * 1000.0
                    )
                    f0 = pyworld.stonemask(frame, f0, _time, self._sample_rate)
                    voiced = f0[f0 > 0]
                    if voiced.size:
                        f0_hz = float(np.median(voiced))
                        pitch_confidence = float(voiced.size / max(1, f0.size))
                yield MetadataEnvelope(
                    stream=stream_name,
                    kind=MetadataKind.PROSODY,
                    sequence=sequence,
                    start_ms=emitted_ms,
                    end_ms=emitted_ms + hop * 1000.0,
                    prosody=ProsodyFrame(
                        energy=energy,
                        is_pause=is_pause,
                        f0_hz=f0_hz,
                        pitch_confidence=pitch_confidence,
                        boundary="pause" if is_pause else None,
                        confidence=1.0,
                    ),
                )
                emitted_ms += hop * 1000.0
                sequence += 1


class PyworldProsodyFactory:
    @property
    def info(self) -> StageInfo:
        return StageInfo(
            id="pyworld-prosody",
            kind=StageKind.PROSODY,
            name="Pyworld Prosody",
            description="pyworld F0 + energy + pause (optional extra).",
            requires_gpu=False,
            default_for_kind=False,
        )

    def create(self, **kwargs: Any) -> PyworldProsodyStage:
        return PyworldProsodyStage(**kwargs)
