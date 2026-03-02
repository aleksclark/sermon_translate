from __future__ import annotations

import abc
from collections.abc import AsyncIterator

from src.models import PipelineInfo


class BasePipeline(abc.ABC):
    """Base class for all translation pipelines."""

    @property
    @abc.abstractmethod
    def info(self) -> PipelineInfo: ...

    @abc.abstractmethod
    def process(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """Accept an async stream of audio chunks and yield processed chunks."""
        ...

    async def start(self) -> None:  # noqa: B027
        """Called when a session using this pipeline starts."""

    async def stop(self) -> None:  # noqa: B027
        """Called when a session using this pipeline stops."""

    def process_text(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str] | None:  # noqa: B027
        """Accept an async stream of audio chunks and yield transcript strings.

        Return None (the default) if this pipeline does not produce text.
        """
        return None
