from __future__ import annotations

import abc
import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum

from src.models import OutputStreamInfo, PipelineInfo, Session


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

    def __init__(self) -> None:
        self._ref_count = 0
        self._ref_lock = asyncio.Lock()

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
                self._release_gpu()

    @staticmethod
    def _release_gpu() -> None:
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def iter_stream(
        self, name: str, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        """Return an async iterator for the named output stream.

        The handler calls this for every stream declared in ``output_streams``
        *except* the default ``"audio"`` stream (which uses ``process()``).
        Return ``None`` if the stream has no data.
        """
        return None

    def get_buffer_stats(self) -> tuple[int, float]:
        """Return (pending_sentences, queued_audio_seconds).

        Pipelines that track internal buffers should override this.
        """
        return 0, 0.0

    @staticmethod
    async def _drain_queue(q: asyncio.Queue[str | None]) -> AsyncIterator[str]:
        while True:
            text = await q.get()
            if text is None:
                return
            yield text
