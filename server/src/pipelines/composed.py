from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from src.config import default_model_cache_dir
from src.models import (
    ListenProduct,
    MetadataEnvelope,
    PipelineInfo,
    Session,
    StageKind,
    StageSelection,
    TranslateProduct,
)
from src.pipelines.base import BasePipeline, OutputStreamDescriptor, OutputStreamKind
from src.pipelines.prosody_tokens import ProsodyAligner
from src.pipelines.stage_registry import StageRegistry, create_default_stage_registry
from src.pipelines.stages import ASRStage, ProsodyStage, TranslationStage, TTSStage
from src.runtime.base import StageRuntime
from src.runtime.model_cache import ModelCache

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


async def _queue_listen(
    queue: asyncio.Queue[ListenProduct | None],
) -> AsyncIterator[ListenProduct]:
    while True:
        item = await queue.get()
        if item is None:
            return
        yield item


async def _queue_translate(
    queue: asyncio.Queue[TranslateProduct | None],
) -> AsyncIterator[TranslateProduct]:
    while True:
        item = await queue.get()
        if item is None:
            return
        yield item


async def _queue_metadata(
    queue: asyncio.Queue[MetadataEnvelope | None],
) -> AsyncIterator[MetadataEnvelope]:
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

    def __init__(
        self,
        stage_registry: StageRegistry | None = None,
        *,
        runtime: StageRuntime | None = None,
        cache: ModelCache | None = None,
    ) -> None:
        super().__init__()
        from src.runtime.local import LocalStageRuntime

        self._stage_registry = stage_registry or create_default_stage_registry()
        self._cache = cache or ModelCache(default_model_cache_dir())
        self._runtime = runtime or LocalStageRuntime(self._stage_registry, self._cache)
        self._aligner = ProsodyAligner()

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

    def _resolve_selection(self, session: Session | None) -> StageSelection:
        if session is None or session.stages is None:
            raise ValueError("composed pipeline requires session.stages")
        return session.stages

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None
    ) -> AsyncIterator[bytes]:
        selection = self._resolve_selection(session)
        if session is None:
            raise ValueError("composed pipeline requires session")

        listen_handle = await self._runtime.spawn(
            selection.listen, session, kind=StageKind.LISTEN
        )
        translate_handle = await self._runtime.spawn(
            selection.translate, session, kind=StageKind.TRANSLATE
        )
        speak_handle = await self._runtime.spawn(
            selection.speak, session, kind=StageKind.SPEAK
        )
        listen = listen_handle.stage
        translate = translate_handle.stage
        speak = speak_handle.stage
        assert isinstance(listen, ASRStage)
        assert isinstance(translate, TranslationStage)
        assert isinstance(speak, TTSStage)

        prosody: ProsodyStage | None = None
        prosody_handle = None
        if selection.prosody is not None:
            prosody_handle = await self._runtime.spawn(
                selection.prosody, session, kind=StageKind.PROSODY
            )
            created = prosody_handle.stage
            assert isinstance(created, ProsodyStage)
            prosody = created

        # Start listen/translate/prosody first so ASR can run while speak loads.
        start_tasks = [
            asyncio.create_task(listen_handle.start()),
            asyncio.create_task(translate_handle.start()),
        ]
        if prosody_handle is not None:
            start_tasks.append(asyncio.create_task(prosody_handle.start()))
        await asyncio.gather(*start_tasks)
        speak_start_task = asyncio.create_task(speak_handle.start())

        listen_audio: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=QUEUE_CAPACITY)
        outputs = [listen_audio]
        prosody_audio: asyncio.Queue[bytes | None] | None = None
        if prosody is not None:
            prosody_audio = asyncio.Queue(maxsize=QUEUE_CAPACITY)
            outputs.append(prosody_audio)

        tee_task = asyncio.create_task(_tee_audio(audio_stream, outputs))

        prosody_frames: list[MetadataEnvelope] = []
        prosody_frames_lock = asyncio.Lock()
        prosody_for_translate: asyncio.Queue[MetadataEnvelope | None] | None = None
        prosody_task: asyncio.Task[None] | None = None
        if prosody is not None and prosody_audio is not None:
            prosody_for_translate = asyncio.Queue(maxsize=QUEUE_CAPACITY)
            prosody_task = asyncio.create_task(
                self._run_prosody(
                    prosody,
                    prosody_audio,
                    session,
                    prosody_frames,
                    prosody_frames_lock,
                    prosody_for_translate,
                )
            )

        source_products: asyncio.Queue[ListenProduct | None] = asyncio.Queue(
            maxsize=QUEUE_CAPACITY
        )
        target_products: asyncio.Queue[TranslateProduct | None] = asyncio.Queue(
            maxsize=QUEUE_CAPACITY
        )

        listen_task = asyncio.create_task(
            self._run_listen(
                listen,
                listen_audio,
                source_products,
                session,
                prosody_frames,
                prosody_frames_lock,
            )
        )
        translate_task = asyncio.create_task(
            self._run_translate(
                translate,
                source_products,
                target_products,
                session,
                prosody_for_translate,
            )
        )

        try:
            await speak_start_task
            async for chunk in speak.synthesize(_queue_translate(target_products)):
                yield chunk
        finally:
            if not speak_start_task.done():
                speak_start_task.cancel()
                await asyncio.gather(speak_start_task, return_exceptions=True)
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
            await self._finish_stage_events(session)
            await speak_handle.stop()
            await translate_handle.stop()
            await listen_handle.stop()
            if prosody_handle is not None:
                await prosody_handle.stop()

    async def _run_listen(
        self,
        listen: ASRStage,
        audio_queue: asyncio.Queue[bytes | None],
        source_products: asyncio.Queue[ListenProduct | None],
        session: Session | None,
        prosody_frames: list[MetadataEnvelope],
        prosody_frames_lock: asyncio.Lock,
    ) -> None:
        try:
            async for product in listen.transcribe(_queue_bytes(audio_queue)):
                async with prosody_frames_lock:
                    frames_snapshot = list(prosody_frames)
                aligned = self._aligner.align(product, frames_snapshot)
                await self._publish_text(LISTEN_STREAM, aligned.text, session)
                await self._publish_stage_event(StageKind.LISTEN, aligned, session)
                await source_products.put(aligned)
        finally:
            await source_products.put(None)

    async def _run_translate(
        self,
        translate: TranslationStage,
        source_products: asyncio.Queue[ListenProduct | None],
        target_products: asyncio.Queue[TranslateProduct | None],
        session: Session | None,
        prosody_for_translate: asyncio.Queue[MetadataEnvelope | None] | None,
    ) -> None:
        try:
            prosody_stream = (
                _queue_metadata(prosody_for_translate)
                if prosody_for_translate is not None
                else None
            )
            async for product in translate.translate(
                _queue_listen(source_products),
                prosody=prosody_stream,
            ):
                await self._publish_text(TRANSLATE_STREAM, product.text, session)
                await self._publish_stage_event(StageKind.TRANSLATE, product, session)
                await target_products.put(product)
        finally:
            await target_products.put(None)

    async def _run_prosody(
        self,
        prosody: ProsodyStage,
        audio_queue: asyncio.Queue[bytes | None],
        session: Session | None,
        prosody_frames: list[MetadataEnvelope],
        prosody_frames_lock: asyncio.Lock,
        prosody_for_translate: asyncio.Queue[MetadataEnvelope | None],
    ) -> None:
        try:
            async for envelope in prosody.analyze(_queue_bytes(audio_queue), PROSODY_STREAM):
                async with prosody_frames_lock:
                    prosody_frames.append(envelope)
                await self._publish_metadata(PROSODY_STREAM, envelope, session)
                await prosody_for_translate.put(envelope)
        finally:
            await prosody_for_translate.put(None)

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

    def iter_stage_events(
        self, session: Session | None = None
    ) -> AsyncIterator[dict[str, Any]] | None:
        return self._drain_stage_events(session)
