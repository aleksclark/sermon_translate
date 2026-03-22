from __future__ import annotations

import asyncio
import collections
import time
from collections.abc import AsyncIterator

from src.models import PipelineInfo, Session
from src.pipelines.base import BasePipeline

DELAY_SECONDS = 5.0


class EchoPipeline(BasePipeline):
    """Echoes audio back to the client after a fixed delay."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            id="echo",
            name="Echo (5s delay)",
            description="Echoes audio back after a 5-second delay. Useful for testing.",
            output_streams=self._build_output_stream_info(),
        )

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None,
    ) -> AsyncIterator[bytes]:
        buffer: collections.deque[tuple[float, bytes]] = collections.deque()

        async def drain() -> AsyncIterator[bytes]:
            while buffer and buffer[0][0] + DELAY_SECONDS <= time.monotonic():
                yield buffer.popleft()[1]

        async for chunk in audio_stream:
            buffer.append((time.monotonic(), chunk))
            async for out in drain():
                yield out

        while buffer:
            wait = buffer[0][0] + DELAY_SECONDS - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            yield buffer.popleft()[1]
