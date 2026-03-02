from __future__ import annotations

from collections.abc import AsyncIterator

from src.pipelines.echo import EchoPipeline


class TestEchoPipelineProcess:
    async def test_echo_returns_same_data(self) -> None:
        pipeline = EchoPipeline()
        chunks = [b"chunk1", b"chunk2", b"chunk3"]

        async def input_stream() -> AsyncIterator[bytes]:
            for c in chunks:
                yield c

        output: list[bytes] = []
        async for out in pipeline.process(input_stream()):
            output.append(out)

        assert output == chunks

    async def test_echo_empty_stream(self) -> None:
        pipeline = EchoPipeline()

        async def input_stream() -> AsyncIterator[bytes]:
            return
            yield  # noqa: unreachable

        output: list[bytes] = []
        async for out in pipeline.process(input_stream()):
            output.append(out)

        assert output == []
