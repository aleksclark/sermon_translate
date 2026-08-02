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

    Emits one :class:`ProsodyFrame` per fixed-size window with cheap features:
    RMS energy, a zero-crossing-based rough pitch estimate, and silence-based
    pause detection. This is explicitly a baseline reference, not tied to any
    translation or TTS model.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        frame_ms: float = 100.0,
        silence_rms: float = 0.01,
    ) -> None:
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._silence_rms = silence_rms

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def frame_features(self, frame: np.ndarray) -> ProsodyFrame:
        if frame.size == 0:
            return ProsodyFrame(energy=0.0, is_pause=True, f0_hz=0.0, confidence=1.0)
        energy = float(np.sqrt(np.mean(np.square(frame))))
        is_pause = energy < self._silence_rms
        if is_pause:
            f0 = 0.0
        else:
            signs = np.signbit(frame)
            crossings = int(np.count_nonzero(signs[1:] != signs[:-1]))
            duration = frame.size / self._sample_rate
            f0 = (crossings / 2.0) / duration if duration > 0 else 0.0
        return ProsodyFrame(
            energy=energy,
            is_pause=is_pause,
            f0_hz=f0,
            boundary="pause" if is_pause else None,
            confidence=1.0,
        )

    async def analyze(
        self, audio_stream: AsyncIterator[bytes], stream_name: str
    ) -> AsyncIterator[MetadataEnvelope]:
        frame_samples = max(1, int(self._sample_rate * self._frame_ms / 1000.0))
        buffer = np.array([], dtype=np.float32)
        sequence = 0
        emitted_ms = 0.0
        ms_per_frame = frame_samples / self._sample_rate * 1000.0

        async for chunk in audio_stream:
            pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            buffer = np.concatenate([buffer, pcm])
            while buffer.size >= frame_samples:
                frame = buffer[:frame_samples]
                buffer = buffer[frame_samples:]
                start_ms = emitted_ms
                emitted_ms += ms_per_frame
                yield MetadataEnvelope(
                    stream=stream_name,
                    kind=MetadataKind.PROSODY,
                    sequence=sequence,
                    start_ms=start_ms,
                    end_ms=emitted_ms,
                    prosody=self.frame_features(frame),
                )
                sequence += 1

        if buffer.size > 0:
            start_ms = emitted_ms
            end_ms = emitted_ms + buffer.size / self._sample_rate * 1000.0
            yield MetadataEnvelope(
                stream=stream_name,
                kind=MetadataKind.PROSODY,
                sequence=sequence,
                start_ms=start_ms,
                end_ms=end_ms,
                prosody=self.frame_features(buffer),
            )
