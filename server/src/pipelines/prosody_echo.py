from __future__ import annotations

from collections.abc import AsyncIterator

from src.models import MetadataEnvelope, PipelineInfo, Session
from src.pipelines.base import BasePipeline, OutputStreamDescriptor, OutputStreamKind
from src.pipelines.stages import BaselineProsodyStage

METADATA_STREAM = "prosody"


class ProsodyEchoPipeline(BasePipeline):
    """Echoes audio unchanged while emitting baseline prosody metadata.

    Demonstrates the metadata output channel end-to-end: audio passes through
    on the ``audio`` stream, and per-window prosody frames are published on a
    declared ``METADATA`` stream, forwarded as ``pipeline.event`` transports.
    """

    def __init__(self, sample_rate: int = 48000) -> None:
        super().__init__()
        self._sample_rate = sample_rate

    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            id="prosody-echo",
            name="Prosody Echo",
            description="Echoes audio and streams baseline prosody metadata.",
            output_streams=self._build_output_stream_info(),
        )

    @property
    def output_streams(self) -> list[OutputStreamDescriptor]:
        return [
            OutputStreamDescriptor(name="audio", kind=OutputStreamKind.AUDIO),
            OutputStreamDescriptor(
                name=METADATA_STREAM,
                kind=OutputStreamKind.METADATA,
                label="Prosody",
                consumes_audio=True,
            ),
        ]

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None
    ) -> AsyncIterator[bytes]:
        async for chunk in audio_stream:
            yield chunk

    def iter_metadata_stream(
        self,
        name: str,
        audio_stream: AsyncIterator[bytes],
        session: Session | None = None,
    ) -> AsyncIterator[MetadataEnvelope] | None:
        if name != METADATA_STREAM:
            return None
        sample_rate = session.sample_rate if session is not None else self._sample_rate
        stage = BaselineProsodyStage(sample_rate=sample_rate)
        return stage.analyze(audio_stream, name)
