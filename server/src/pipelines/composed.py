from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

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
from src.pipelines.commit_barrier import DeadlineAwareQueue, QueueItemKind
from src.pipelines.prosody_tokens import ProsodyAligner
from src.pipelines.stage_registry import StageRegistry, create_default_stage_registry
from src.pipelines.stage_session import StageSession, StageSessionConfig
from src.pipelines.stages import ASRStage, ProsodyStage, TranslationStage, TTSStage
from src.runtime.base import StageRuntime
from src.runtime.model_cache import ModelCache
from src.stage_v1.models import StageErrorCode
from src.stage_v1.models import StageKind as StageV1Kind
from src.stage_v1.validation import ValidationError

logger = logging.getLogger(__name__)

LISTEN_STREAM = "listen"
TRANSLATE_STREAM = "translate"
PROSODY_STREAM = "prosody"
QUEUE_CAPACITY = 64
# Long enough that normal passthrough tests never hit deadline under load.
AUDIO_DEADLINE_S = 120.0


async def _queue_bytes(queue: DeadlineAwareQueue[bytes]) -> AsyncIterator[bytes]:
    while True:
        try:
            item = await queue.get()
        except ValidationError as exc:
            if exc.code in (StageErrorCode.CANCELLED, StageErrorCode.DEADLINE_EXCEEDED):
                return
            raise
        yield item


async def _queue_listen(
    queue: DeadlineAwareQueue[ListenProduct],
) -> AsyncIterator[ListenProduct]:
    while True:
        try:
            item = await queue.get()
        except ValidationError as exc:
            if exc.code in (StageErrorCode.CANCELLED, StageErrorCode.DEADLINE_EXCEEDED):
                return
            raise
        yield item


async def _queue_translate(
    queue: DeadlineAwareQueue[TranslateProduct],
) -> AsyncIterator[TranslateProduct]:
    while True:
        try:
            item = await queue.get()
        except ValidationError as exc:
            if exc.code in (StageErrorCode.CANCELLED, StageErrorCode.DEADLINE_EXCEEDED):
                return
            raise
        yield item


async def _queue_metadata(
    queue: DeadlineAwareQueue[MetadataEnvelope],
) -> AsyncIterator[MetadataEnvelope]:
    while True:
        try:
            item = await queue.get()
        except ValidationError as exc:
            if exc.code in (StageErrorCode.CANCELLED, StageErrorCode.DEADLINE_EXCEEDED):
                return
            raise
        yield item


async def _put_deadline(
    queue: DeadlineAwareQueue[Any],
    item: Any,
    *,
    kind: QueueItemKind,
    deadline_at: str,
    bytes_size: int = 0,
    stage_session: StageSession | None = None,
    stream_label: str = "queue",
) -> None:
    """Enqueue with deadline backpressure; emit explicit gap on deadline loss."""
    try:
        await queue.put(
            item,
            kind=kind,
            deadline_at=deadline_at,
            bytes_size=bytes_size,
        )
    except ValidationError as exc:
        if exc.code is StageErrorCode.DEADLINE_EXCEEDED:
            if stage_session is not None:
                stage_session._events.append(  # noqa: SLF001 — surface protocol event
                    {
                        "event_type": "gap",
                        "payload": {
                            "reason": "deadline_exceeded_enqueue",
                            "stream": stream_label,
                            "kind": kind.value,
                            "bytes_size": bytes_size,
                        },
                    }
                )
            logger.warning(
                "stage.v1 gap: deadline exceeded enqueueing %s on %s (%s bytes)",
                kind.value,
                stream_label,
                bytes_size,
            )
            return
        if exc.code is StageErrorCode.CANCELLED:
            return
        raise


async def _tee_audio(
    audio_stream: AsyncIterator[bytes],
    outputs: list[DeadlineAwareQueue[bytes]],
    *,
    stage_session: StageSession,
    deadline_at: str,
) -> None:
    try:
        async for chunk in audio_stream:
            await asyncio.gather(
                *(
                    _put_deadline(
                        queue,
                        chunk,
                        kind=QueueItemKind.AUDIO,
                        deadline_at=deadline_at,
                        bytes_size=len(chunk),
                        stage_session=stage_session,
                        stream_label=f"audio_tee[{index}]",
                    )
                    for index, queue in enumerate(outputs)
                )
            )
    finally:
        await asyncio.gather(*(queue.close() for queue in outputs))


class ComposedPipeline(BasePipeline):
    """Wires selectable listen/translate/speak/(prosody) stages into one pipeline.

    Source audio uses deadline-aware bounded queues (no silent drop-oldest).
    Listen/Translate products route through StageSession commit barriers so only
    newly committed deltas reach the next stage; gaps/drops are explicit events.
    """

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
        self._stage_sessions: dict[str, StageSession] = {}

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

    def _make_stage_session(self, session: Session) -> StageSession:
        cfg = StageSessionConfig(
            audio_queue_capacity=QUEUE_CAPACITY,
            product_queue_capacity=QUEUE_CAPACITY,
            speak_queue_capacity=QUEUE_CAPACITY,
            default_deadline_s=AUDIO_DEADLINE_S,
        )
        stage_session = StageSession(
            session_id=session.id,
            owner_generation=0,
            stage_kind=StageV1Kind.LISTEN,
            stage_id="composed",
            config=cfg,
        )
        self._stage_sessions[session.id] = stage_session
        return stage_session

    def stage_session_for(self, session: Session | None) -> StageSession | None:
        if session is None:
            return None
        return self._stage_sessions.get(session.id)

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None
    ) -> AsyncIterator[bytes]:
        selection = self._resolve_selection(session)
        if session is None:
            raise ValueError("composed pipeline requires session")

        stage_session = self._make_stage_session(session)
        deadline_at = stage_session.default_deadline_at()

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

        # Start stages sequentially: concurrent CUDA model construction races
        # moshi CUDA graphs ("Offset increment outside graph capture").
        await listen_handle.start()
        await translate_handle.start()
        if prosody_handle is not None:
            await prosody_handle.start()
        speak_start_task = asyncio.create_task(speak_handle.start())

        listen_audio: DeadlineAwareQueue[bytes] = DeadlineAwareQueue(
            capacity=QUEUE_CAPACITY, name="listen_audio"
        )
        outputs: list[DeadlineAwareQueue[bytes]] = [listen_audio]
        prosody_audio: DeadlineAwareQueue[bytes] | None = None
        if prosody is not None:
            prosody_audio = DeadlineAwareQueue(capacity=QUEUE_CAPACITY, name="prosody_audio")
            outputs.append(prosody_audio)

        tee_task = asyncio.create_task(
            _tee_audio(
                audio_stream,
                outputs,
                stage_session=stage_session,
                deadline_at=deadline_at,
            )
        )

        prosody_frames: list[MetadataEnvelope] = []
        prosody_frames_lock = asyncio.Lock()
        prosody_for_translate: DeadlineAwareQueue[MetadataEnvelope] | None = None
        prosody_task: asyncio.Task[None] | None = None
        if prosody is not None and prosody_audio is not None:
            prosody_for_translate = DeadlineAwareQueue(
                capacity=QUEUE_CAPACITY, name="prosody_for_translate"
            )
            prosody_task = asyncio.create_task(
                self._run_prosody(
                    prosody,
                    prosody_audio,
                    session,
                    stage_session,
                    deadline_at,
                    prosody_frames,
                    prosody_frames_lock,
                    prosody_for_translate,
                )
            )

        source_products: DeadlineAwareQueue[ListenProduct] = DeadlineAwareQueue(
            capacity=QUEUE_CAPACITY, name="source_products"
        )
        target_products: DeadlineAwareQueue[TranslateProduct] = DeadlineAwareQueue(
            capacity=QUEUE_CAPACITY, name="target_products"
        )

        listen_task = asyncio.create_task(
            self._run_listen(
                listen,
                listen_audio,
                source_products,
                session,
                stage_session,
                deadline_at,
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
                stage_session,
                deadline_at,
                prosody_for_translate,
            )
        )

        utterance_seq = 0
        try:
            await speak_start_task
            async for chunk in speak.synthesize(_queue_translate(target_products)):
                # Ordered publication: one unit per outgoing speak chunk sequence.
                target_span_id = f"speak-{utterance_seq}"
                stage_session.register_publication_unit(
                    utterance_sequence=utterance_seq,
                    target_span_id=target_span_id,
                    deadline_at=deadline_at,
                )
                releases = stage_session.complete_publication(
                    utterance_sequence=utterance_seq,
                    target_span_id=target_span_id,
                    payload={"pcm_bytes": len(chunk)},
                )
                published = False
                for rel in releases:
                    if rel.kind == "gap":
                        await self._publish_stage_event(
                            StageKind.SPEAK,
                            ProtocolSideEvent(
                                event_type="gap",
                                reason="publication_gap",
                                sequence=utterance_seq,
                            ),
                            session,
                        )
                    else:
                        published = True
                        yield chunk
                if not published and not any(r.kind == "gap" for r in releases):
                    # Barrier returned empty (should not happen for first complete);
                    # still emit audio so product path stays live.
                    yield chunk
                utterance_seq += 1
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
            # Flush protocol gap/dropped/error events recorded on the stage session.
            await self._flush_protocol_events(stage_session, session)
            await self._finish_text(LISTEN_STREAM, session)
            await self._finish_text(TRANSLATE_STREAM, session)
            await self._finish_metadata(PROSODY_STREAM, session)
            await self._finish_stage_events(session)
            await speak_handle.stop()
            await translate_handle.stop()
            await listen_handle.stop()
            if prosody_handle is not None:
                await prosody_handle.stop()
            self._stage_sessions.pop(session.id, None)

    async def _flush_protocol_events(
        self, stage_session: StageSession, session: Session | None
    ) -> None:
        for event in stage_session.events:
            event_type = event.get("event_type")
            if event_type not in {"gap", "dropped", "error", "cancelled"}:
                continue
            await self._publish_stage_event(
                StageKind.LISTEN,
                ProtocolSideEvent(
                    event_type=str(event_type),
                    reason=str(event.get("code") or event.get("reason") or event_type),
                    detail=event,
                ),
                session,
            )

    async def _publish_protocol_error(
        self,
        stage: StageKind,
        session: Session | None,
        *,
        code: str,
        message: str,
    ) -> None:
        await self._publish_stage_event(
            stage,
            ProtocolSideEvent(event_type="error", reason=code, message=message),
            session,
        )

    async def _run_listen(
        self,
        listen: ASRStage,
        audio_queue: DeadlineAwareQueue[bytes],
        source_products: DeadlineAwareQueue[ListenProduct],
        session: Session | None,
        stage_session: StageSession,
        deadline_at: str,
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

                # Commit barrier: only newly committed deltas route to translate.
                stage_session.bind_utterance(aligned.utterance_id, aligned.sequence)
                committed_chars = len(aligned.text) if aligned.is_final else 0
                # Per-utterance revision counter (product.sequence is session-global).
                revision = 0 if aligned.is_final else max(0, aligned.sequence)
                try:
                    result = stage_session.observe_listen_product(
                        revision=revision,
                        text=aligned.text,
                        committed_prefix_chars=committed_chars,
                        is_final=aligned.is_final,
                        utterance_id=aligned.utterance_id,
                    )
                except ValidationError as exc:
                    await self._publish_protocol_error(
                        StageKind.LISTEN,
                        session,
                        code=exc.code.value,
                        message=exc.message,
                    )
                    continue

                if result.delta is None:
                    # Uncommitted partial — display-only; do not speak/translate yet.
                    continue

                routed = ListenProduct(
                    sequence=aligned.sequence,
                    utterance_id=aligned.utterance_id,
                    text=result.delta.text,
                    is_final=result.delta.is_final,
                    words=aligned.words,
                    language=aligned.language,
                )
                await _put_deadline(
                    source_products,
                    routed,
                    kind=QueueItemKind.COMMITTED_SPAN,
                    deadline_at=deadline_at,
                    bytes_size=len(routed.text.encode("utf-8")),
                    stage_session=stage_session,
                    stream_label="source_products",
                )
        finally:
            await source_products.close()

    async def _run_translate(
        self,
        translate: TranslationStage,
        source_products: DeadlineAwareQueue[ListenProduct],
        target_products: DeadlineAwareQueue[TranslateProduct],
        session: Session | None,
        stage_session: StageSession,
        deadline_at: str,
        prosody_for_translate: DeadlineAwareQueue[MetadataEnvelope] | None,
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

                source_span_id = product.source_utterance_id
                target_span_id = product.target_utterance_id
                committed_chars = len(product.text) if product.is_final else 0
                # Translate products from stubs are already committed deltas per span.
                revision = 0 if product.is_final else max(0, product.sequence)
                try:
                    result = stage_session.observe_translate_product(
                        source_span_id=source_span_id,
                        target_span_id=target_span_id,
                        revision=revision,
                        text=product.text,
                        committed_prefix_chars=committed_chars,
                        is_final=product.is_final,
                        utterance_id=product.source_utterance_id,
                    )
                except ValidationError as exc:
                    await self._publish_protocol_error(
                        StageKind.TRANSLATE,
                        session,
                        code=exc.code.value,
                        message=exc.message,
                    )
                    continue

                if result.delta is None:
                    continue

                routed = TranslateProduct(
                    sequence=product.sequence,
                    source_utterance_id=product.source_utterance_id,
                    target_utterance_id=product.target_utterance_id,
                    text=result.delta.text,
                    is_final=result.delta.is_final,
                    words=product.words,
                    instructions=product.instructions,
                )
                await _put_deadline(
                    target_products,
                    routed,
                    kind=QueueItemKind.COMMITTED_SPAN,
                    deadline_at=deadline_at,
                    bytes_size=len(routed.text.encode("utf-8")),
                    stage_session=stage_session,
                    stream_label="target_products",
                )
        finally:
            await target_products.close()
            if prosody_for_translate is not None:
                await prosody_for_translate.close()

    async def _run_prosody(
        self,
        prosody: ProsodyStage,
        audio_queue: DeadlineAwareQueue[bytes],
        session: Session | None,
        stage_session: StageSession,
        deadline_at: str,
        prosody_frames: list[MetadataEnvelope],
        prosody_frames_lock: asyncio.Lock,
        prosody_for_translate: DeadlineAwareQueue[MetadataEnvelope],
    ) -> None:
        try:
            async for envelope in prosody.analyze(_queue_bytes(audio_queue), PROSODY_STREAM):
                async with prosody_frames_lock:
                    prosody_frames.append(envelope)
                await self._publish_metadata(PROSODY_STREAM, envelope, session)
                await _put_deadline(
                    prosody_for_translate,
                    envelope,
                    kind=QueueItemKind.OTHER,
                    deadline_at=deadline_at,
                    stage_session=stage_session,
                    stream_label="prosody_for_translate",
                )
        finally:
            await prosody_for_translate.close()

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


class ProtocolSideEvent(BaseModel):
    """stage.v1 gap/dropped/error side-channel carried on the stage-event stream."""

    event_type: str
    reason: str = ""
    message: str = ""
    sequence: int | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
