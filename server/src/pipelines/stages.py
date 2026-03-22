"""Stage protocols for composable translation pipelines.

Each stage is an independent, async-streaming component:
  ASRStage:         audio bytes → transcript strings
  TranslationStage: source strings → target strings
  TTSStage:         text strings → audio bytes
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


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
