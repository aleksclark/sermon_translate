"""Stage protocols for composable translation pipelines.

Each stage is an independent, async-streaming component:
  ASRStage:         audio bytes → transcript strings
  TranslationStage: source strings → target strings
  TTSStage:         text strings → audio bytes
  ProsodyStage:     audio bytes → prosody metadata envelopes
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

import numpy as np

from src.models import MetadataEnvelope, MetadataKind, ProsodyFrame
from src.pipelines._pitch import (
    DEFAULT_F0_MAX_HZ,
    DEFAULT_F0_MIN_HZ,
    UNVOICED,
    PitchTracker,
    YinPitchTracker,
    zero_crossing_rate,
)


@runtime_checkable
class ASRStage(Protocol):
    """Speech-to-text: consumes audio chunks, yields transcript strings."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def transcribe(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]: ...


@runtime_checkable
class TranslationStage(Protocol):
    """Text-to-text: consumes source-language strings, yields target-language strings."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def translate(self, text_stream: AsyncIterator[str]) -> AsyncIterator[str]: ...


@runtime_checkable
class TTSStage(Protocol):
    """Text-to-speech: consumes text strings, yields audio bytes."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def synthesize(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]: ...


@runtime_checkable
class ProsodyStage(Protocol):
    """Prosody generation: consumes audio chunks, yields metadata envelopes.

    Prosody may be derived from the source audio itself, or supplied by a
    translation/TTS model through an instruction channel. Implementations emit
    model-neutral :class:`MetadataEnvelope` objects on a named stream.
    """

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def analyze(
        self, audio_stream: AsyncIterator[bytes], stream_name: str
    ) -> AsyncIterator[MetadataEnvelope]: ...


class BaselineProsodyStage:
    """Dependency-free prosody baseline computed from raw PCM using numpy.

    Emits one :class:`ProsodyFrame` per fixed-size window carrying RMS energy,
    silence-based pause detection, and a real fundamental-frequency estimate
    delegated to a swappable :class:`~src.pipelines._pitch.PitchTracker`
    (:class:`~src.pipelines._pitch.YinPitchTracker` by default). This is
    explicitly a baseline reference, not tied to any translation or TTS model.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        frame_ms: float = 100.0,
        silence_rms: float = 0.01,
        pitch_tracker: PitchTracker | None = None,
        f0_min_hz: float = DEFAULT_F0_MIN_HZ,
        f0_max_hz: float = DEFAULT_F0_MAX_HZ,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if frame_ms <= 0.0:
            raise ValueError("frame_ms must be positive")
        if silence_rms < 0.0:
            raise ValueError("silence_rms must be non-negative")
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._silence_rms = silence_rms
        self._pitch_tracker: PitchTracker = pitch_tracker or YinPitchTracker(
            f0_min=f0_min_hz, f0_max=f0_max_hz
        )

    @property
    def frame_samples(self) -> int:
        return max(1, int(self._sample_rate * self._frame_ms / 1000.0))

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def frame_features(self, frame: np.ndarray) -> ProsodyFrame:
        if frame.size == 0:
            return ProsodyFrame(
                energy=0.0,
                is_pause=True,
                f0_hz=None,
                pitch_confidence=0.0,
                boundary="pause",
                confidence=1.0,
                features={"zero_crossing_rate": 0.0, "voiced": 0.0},
            )

        energy = float(np.sqrt(np.mean(np.square(frame))))
        is_pause = energy < self._silence_rms
        pitch = UNVOICED if is_pause else self._pitch_tracker.estimate(frame, self._sample_rate)
        return ProsodyFrame(
            energy=energy,
            is_pause=is_pause,
            f0_hz=pitch.f0_hz,
            pitch_confidence=pitch.confidence,
            boundary="pause" if is_pause else None,
            confidence=1.0,
            features={
                "zero_crossing_rate": zero_crossing_rate(frame, self._sample_rate),
                "voiced": float(pitch.voiced),
            },
        )

    def _duration_ms(self, samples: int) -> float:
        return samples / self._sample_rate * 1000.0

    def _envelope(
        self, frame: np.ndarray, stream_name: str, sequence: int, start_ms: float
    ) -> MetadataEnvelope:
        return MetadataEnvelope(
            stream=stream_name,
            kind=MetadataKind.PROSODY,
            sequence=sequence,
            start_ms=start_ms,
            end_ms=start_ms + self._duration_ms(frame.size),
            prosody=self.frame_features(frame),
        )

    async def analyze(
        self, audio_stream: AsyncIterator[bytes], stream_name: str
    ) -> AsyncIterator[MetadataEnvelope]:
        frame_samples = self.frame_samples
        buffer = np.zeros(0, dtype=np.float32)
        sequence = 0
        emitted_ms = 0.0

        async for chunk in audio_stream:
            pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            buffer = np.concatenate([buffer, pcm])
            while buffer.size >= frame_samples:
                frame, buffer = buffer[:frame_samples], buffer[frame_samples:]
                envelope = self._envelope(frame, stream_name, sequence, emitted_ms)
                emitted_ms += self._duration_ms(frame.size)
                sequence += 1
                yield envelope

        if buffer.size > 0:
            yield self._envelope(buffer, stream_name, sequence, emitted_ms)
