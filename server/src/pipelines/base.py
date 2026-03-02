from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum

from src.models import PipelineInfo, Session


class OutputStreamKind(StrEnum):
    AUDIO = "audio"
    TEXT = "text"


@dataclass(frozen=True)
class OutputStreamDescriptor:
    name: str
    kind: OutputStreamKind
    label: str = ""


class BasePipeline(abc.ABC):
    """Base class for all translation pipelines."""

    @property
    @abc.abstractmethod
    def info(self) -> PipelineInfo: ...

    @property
    def output_streams(self) -> list[OutputStreamDescriptor]:
        """Declare the named output streams this pipeline produces.

        Override in subclasses. The default provides a single audio stream
        named ``"audio"`` for backwards compatibility.
        """
        return [OutputStreamDescriptor(name="audio", kind=OutputStreamKind.AUDIO)]

    @abc.abstractmethod
    def process(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        """Accept an async stream of audio chunks and yield processed chunks.

        Audio yielded here is sent on the ``"audio"`` output stream.
        """
        ...

    async def start(self) -> None:  # noqa: B027
        """Called when a session using this pipeline starts."""

    async def stop(self) -> None:  # noqa: B027
        """Called when a session using this pipeline stops."""

    def configure_session(self, session: Session) -> None:  # noqa: B027
        """Called before start() with session settings.

        Pipelines that care about per-session config (e.g. audio_context_seconds)
        should override this.
        """

    def iter_stream(
        self, name: str, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        """Return an async iterator for the named output stream.

        The handler calls this for every stream declared in ``output_streams``
        *except* the default ``"audio"`` stream (which uses ``process()``).
        Return ``None`` if the stream has no data.
        """
        return None
