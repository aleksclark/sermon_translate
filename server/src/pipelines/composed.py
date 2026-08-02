from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from src.models import MetadataEnvelope, PipelineInfo, Session, StageKind, StageSelection
from src.pipelines.base import BasePipeline, OutputStreamDescriptor, OutputStreamKind
from src.pipelines.stage_registry import StageFactory, StageRegistry
from src.pipelines.stages import ASRStage, ProsodyStage, TranslationStage, TTSStage

LISTEN_STREAM = "listen"
TRANSLATE_STREAM = "translate"
PROSODY_STREAM = "prosody"
QUEUE_CAPACITY = 8


async def _queue_bytes(queue: asyncio.Queue[bytes | None]) -> AsyncIterator[bytes]:
    while True:
        item = await queue.get()
        if item is None:
            return
        yield item


async def _queue_text(queue: asyncio.Queue[str | None]) -> AsyncIterator[str]:
    while True:
        item = await queue.get()
        if item is None:
            return
        yield item


async def _tee_audio(
    audio_stream: AsyncIterator[bytes],
    outputs: list[asyncio.Queue[bytes | None]],
) -> None:
    try:
        async for chunk in audio_stream:
            await asyncio.gather(*(queue.put(chunk) for queue in outputs))
    finally:
        await asyncio.gather(*(queue.put(None) for queue in outputs))


class ComposedPipeline(BasePipeline):
    """Wires selectable listen/translate/speak/(prosody) stages into one pipeline."""

    def __init__(self, stage_registry: StageRegistry) -> None:
        super().__init__()
        self._stage_registry = stage_registry

    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            id="composed",
            name="Composed",
            description="Selectable listen, translate, speak, and optional prosody stages.",
            output_streams=self._build_output_stream_info(),
        )

    @property
    def output_streams(self) -> list[OutputStreamDescriptor]:
        return [
            OutputStreamDescriptor(
                name="audio",
                kind=OutputStreamKind.AUDIO,
                label="Translated speech",
            ),
            OutputStreamDescriptor(
                name=LISTEN_STREAM,
                kind=OutputStreamKind.TEXT,
                label="Source transcript",
                consumes_audio=False,
            ),
            OutputStreamDescriptor(
                name=TRANSLATE_STREAM,
                kind=OutputStreamKind.TEXT,
                label="Target transcript",
                consumes_audio=False,
            ),
            OutputStreamDescriptor(
                name=PROSODY_STREAM,
                kind=OutputStreamKind.METADATA,
                label="Prosody",
                consumes_audio=False,
            ),
        ]

    def _require_factory(self, stage_id: str, kind: StageKind) -> StageFactory:
        factory = self._stage_registry.get(stage_id)
        if factory is None:
            raise ValueError(f"Unknown stage: {stage_id}")
        if factory.info.kind != kind:
            raise ValueError(
                f"Stage {stage_id} has kind {factory.info.kind.value}, expected {kind.value}"
            )
        return factory

    def _resolve_selection(self, session: Session | None) -> StageSelection:
        if session is None or session.stages is None:
            raise ValueError("composed pipeline requires session.stages")
        return session.stages

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None
    ) -> AsyncIterator[bytes]:
        selection = self._resolve_selection(session)
        sample_rate = session.sample_rate if session is not None else 48000

        listen_factory = self._require_factory(selection.listen, StageKind.LISTEN)
        translate_factory = self._require_factory(selection.translate, StageKind.TRANSLATE)
        speak_factory = self._require_factory(selection.speak, StageKind.SPEAK)

        listen = listen_factory.create(sample_rate=sample_rate)
        translate = translate_factory.create(sample_rate=sample_rate)
        speak = speak_factory.create(sample_rate=sample_rate)
        assert isinstance(listen, ASRStage)
        assert isinstance(translate, TranslationStage)
        assert isinstance(speak, TTSStage)

        prosody: ProsodyStage | None = None
        if selection.prosody is not None:
            prosody_factory = self._require_factory(selection.prosody, StageKind.PROSODY)
            created = prosody_factory.create(sample_rate=sample_rate)
            assert isinstance(created, ProsodyStage)
            prosody = created

        await listen.start()
        await translate.start()
        await speak.start()
        if prosody is not None:
            await prosody.start()

        listen_audio: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=QUEUE_CAPACITY)
        outputs = [listen_audio]
        prosody_audio: asyncio.Queue[bytes | None] | None = None
        if prosody is not None:
            prosody_audio = asyncio.Queue(maxsize=QUEUE_CAPACITY)
            outputs.append(prosody_audio)

        tee_task = asyncio.create_task(_tee_audio(audio_stream, outputs))
        prosody_task: asyncio.Task[None] | None = None
        if prosody is not None and prosody_audio is not None:
            prosody_task = asyncio.create_task(
                self._run_prosody(prosody, prosody_audio, session)
            )

        source_text: asyncio.Queue[str | None] = asyncio.Queue(maxsize=QUEUE_CAPACITY)
        target_text: asyncio.Queue[str | None] = asyncio.Queue(maxsize=QUEUE_CAPACITY)

        listen_task = asyncio.create_task(
            self._run_listen(listen, listen_audio, source_text, session)
        )
        translate_task = asyncio.create_task(
            self._run_translate(translate, source_text, target_text, session)
        )

        try:
            async for chunk in speak.synthesize(_queue_text(target_text)):
                yield chunk
        finally:
            for task in (listen_task, translate_task, tee_task, prosody_task):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                listen_task,
                translate_task,
                tee_task,
                *([prosody_task] if prosody_task is not None else []),
                return_exceptions=True,
            )
            await self._finish_text(LISTEN_STREAM, session)
            await self._finish_text(TRANSLATE_STREAM, session)
            await self._finish_metadata(PROSODY_STREAM, session)
            await speak.stop()
            await translate.stop()
            await listen.stop()
            if prosody is not None:
                await prosody.stop()

    async def _run_listen(
        self,
        listen: ASRStage,
        audio_queue: asyncio.Queue[bytes | None],
        source_text: asyncio.Queue[str | None],
        session: Session | None,
    ) -> None:
        try:
            async for text in listen.transcribe(_queue_bytes(audio_queue)):
                await self._publish_text(LISTEN_STREAM, text, session)
                await source_text.put(text)
        finally:
            await source_text.put(None)

    async def _run_translate(
        self,
        translate: TranslationStage,
        source_text: asyncio.Queue[str | None],
        target_text: asyncio.Queue[str | None],
        session: Session | None,
    ) -> None:
        try:
            async for text in translate.translate(_queue_text(source_text)):
                await self._publish_text(TRANSLATE_STREAM, text, session)
                await target_text.put(text)
        finally:
            await target_text.put(None)

    async def _run_prosody(
        self,
        prosody: ProsodyStage,
        audio_queue: asyncio.Queue[bytes | None],
        session: Session | None,
    ) -> None:
        async for envelope in prosody.analyze(_queue_bytes(audio_queue), PROSODY_STREAM):
            await self._publish_metadata(PROSODY_STREAM, envelope, session)

    def iter_stream(
        self,
        name: str,
        audio_stream: AsyncIterator[bytes],
        session: Session | None = None,
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        if name in (LISTEN_STREAM, TRANSLATE_STREAM):
            return self._drain_text(name, session)
        return None

    def iter_metadata_stream(
        self,
        name: str,
        audio_stream: AsyncIterator[bytes],
        session: Session | None = None,
    ) -> AsyncIterator[MetadataEnvelope] | None:
        if name == PROSODY_STREAM:
            return self._drain_metadata(name, session)
        return None
