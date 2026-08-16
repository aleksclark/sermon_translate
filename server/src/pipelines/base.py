from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum

from src.models import MetadataEnvelope, OutputStreamInfo, PipelineInfo, Session


class OutputStreamKind(StrEnum):
    AUDIO = "audio"
    TEXT = "text"
    METADATA = "metadata"


@dataclass(frozen=True)
class OutputStreamDescriptor:
    name: str
    kind: OutputStreamKind
    label: str = ""
    consumes_audio: bool = True


class BasePipeline(abc.ABC):
    """Base class for all translation pipelines."""

    def __init__(self) -> None:
        self._ref_count = 0
        self._ref_lock = asyncio.Lock()
        self._session_text_queues: dict[
            tuple[str | None, str], asyncio.Queue[str | None]
        ] = {}
        self._session_metadata_queues: dict[
            tuple[str | None, str], asyncio.Queue[MetadataEnvelope | None]
        ] = {}

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

    def _build_output_stream_info(self) -> list[OutputStreamInfo]:
        return [
            OutputStreamInfo(name=s.name, kind=s.kind.value, label=s.label)
            for s in self.output_streams
        ]

    @abc.abstractmethod
    def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None,
    ) -> AsyncIterator[bytes]:
        """Accept an async stream of audio chunks and yield processed chunks.

        Audio yielded here is sent on the ``"audio"`` output stream.
        ``session`` carries per-session config (sample_rate, audio_context_seconds, etc.).
        """
        ...

    async def _do_start(self) -> None:  # noqa: B027
        """Subclass hook: load models / allocate resources."""

    async def _do_stop(self) -> None:  # noqa: B027
        """Subclass hook: release models / free resources."""

    async def start(self) -> None:
        async with self._ref_lock:
            if self._ref_count == 0:
                await self._do_start()
            self._ref_count += 1

    async def stop(self) -> None:
        async with self._ref_lock:
            self._ref_count -= 1
            if self._ref_count <= 0:
                self._ref_count = 0
                await self._do_stop()

    def iter_stream(
        self,
        name: str,
        audio_stream: AsyncIterator[bytes],
        session: Session | None = None,
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        """Return an async iterator for the named output stream.

        The handler calls this for every stream declared in ``output_streams``
        *except* the default ``"audio"`` stream (which uses ``process()``).
        Return ``None`` if the stream has no data.
        """
        return None

    def _text_queue(
        self, name: str, session: Session | None
    ) -> asyncio.Queue[str | None]:
        key = (session.id if session is not None else None, name)
        queue = self._session_text_queues.get(key)
        if queue is None:
            queue = asyncio.Queue(maxsize=8)
            self._session_text_queues[key] = queue
        return queue

    async def _publish_text(self, name: str, text: str, session: Session | None) -> None:
        await self._text_queue(name, session).put(text)

    async def _finish_text(self, name: str, session: Session | None) -> None:
        await self._text_queue(name, session).put(None)

    async def _drain_text(self, name: str, session: Session | None) -> AsyncIterator[str]:
        key = (session.id if session is not None else None, name)
        queue = self._text_queue(name, session)
        try:
            while True:
                text = await queue.get()
                if text is None:
                    return
                yield text
        finally:
            if self._session_text_queues.get(key) is queue:
                del self._session_text_queues[key]

    def discard_session_outputs(self, session: Session) -> None:
        keys = [key for key in self._session_text_queues if key[0] == session.id]
        for key in keys:
            del self._session_text_queues[key]
        metadata_keys = [
            key for key in self._session_metadata_queues if key[0] == session.id
        ]
        for key in metadata_keys:
            del self._session_metadata_queues[key]

    def iter_metadata_stream(
        self,
        name: str,
        audio_stream: AsyncIterator[bytes],
        session: Session | None = None,
    ) -> AsyncIterator[MetadataEnvelope] | None:
        """Return an async iterator for the named metadata output stream.

        The handler calls this for every stream declared with
        ``OutputStreamKind.METADATA``. Return ``None`` if the stream has no data.
        Subclasses typically drive it from :meth:`_publish_metadata` /
        :meth:`_finish_metadata` and return :meth:`_drain_metadata`.
        """
        return None

    def _metadata_queue(
        self, name: str, session: Session | None
    ) -> asyncio.Queue[MetadataEnvelope | None]:
        key = (session.id if session is not None else None, name)
        queue = self._session_metadata_queues.get(key)
        if queue is None:
            queue = asyncio.Queue(maxsize=8)
            self._session_metadata_queues[key] = queue
        return queue

    async def _publish_metadata(
        self, name: str, envelope: MetadataEnvelope, session: Session | None
    ) -> None:
        await self._metadata_queue(name, session).put(envelope)

    async def _finish_metadata(self, name: str, session: Session | None) -> None:
        await self._metadata_queue(name, session).put(None)

    async def _drain_metadata(
        self, name: str, session: Session | None
    ) -> AsyncIterator[MetadataEnvelope]:
        key = (session.id if session is not None else None, name)
        queue = self._metadata_queue(name, session)
        try:
            while True:
                envelope = await queue.get()
                if envelope is None:
                    return
                yield envelope
        finally:
            if self._session_metadata_queues.get(key) is queue:
                del self._session_metadata_queues[key]
