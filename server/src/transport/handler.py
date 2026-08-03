from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import AsyncIterator

from src.api.deps import get_pipeline_registry, get_server_stats, get_session_store
from src.models import SessionStatus
from src.pipelines.base import OutputStreamKind
from src.transport.base import EventType, TransportConnection, TransportEvent

logger = logging.getLogger(__name__)

BOUNDED_BACKPRESSURE_FANOUT_QUEUE_CAPACITY = 8
PROCESSOR_DRAIN_TIMEOUT_SECONDS = 30
TRANSPORT_CLEANUP_TIMEOUT_SECONDS = 5


async def run_session(transport: TransportConnection, session_id: str) -> None:
    store = get_session_store()
    session = store.get(session_id)
    if session is None:
        await transport.close()
        return

    registry = get_pipeline_registry()
    pipeline = registry.get(session.pipeline_id)
    if pipeline is None:
        await transport.close()
        return

    pipeline_started = False
    start_time = time.monotonic()
    tasks: list[asyncio.Task[object]] = []
    current_fanout: asyncio.Task[None] | None = None

    try:
        try:
            await transport.wait_ready()
        except TimeoutError:
            logger.error("transport not ready for session %s", session_id)
            return

        session.status = SessionStatus.ACTIVE
        await pipeline.start()
        pipeline_started = True
        await transport.send_event(
            TransportEvent(type=EventType.SESSION_START, session_id=session_id)
        )

        stats_tracker = get_server_stats()
        stop_event = asyncio.Event()
        audio_fanout_queues: list[asyncio.Queue[bytes | None]] = []

        async def close_audio_fanout() -> None:
            await asyncio.gather(*(queue.put(None) for queue in audio_fanout_queues))

        async def fanout_chunk(chunk: bytes) -> None:
            await asyncio.gather(*(queue.put(chunk) for queue in audio_fanout_queues))

        async def audio_input() -> None:
            nonlocal current_fanout
            async for chunk in transport.recv_audio():
                session.stats.bytes_received += len(chunk)
                session.stats.chunks_received += 1
                current_fanout = asyncio.create_task(fanout_chunk(chunk))
                try:
                    # Keep one input chunk atomic across all consumers. If AUDIO_END
                    # cancels this receiver, the handler waits for this shielded fanout
                    # before appending end-of-stream sentinels.
                    await asyncio.shield(current_fanout)
                finally:
                    if current_fanout.done():
                        current_fanout = None

        async def queue_iter(queue: asyncio.Queue[bytes | None]) -> AsyncIterator[bytes]:
            while True:
                item = await queue.get()
                if item is None:
                    return
                yield item

        async def empty_stream() -> AsyncIterator[bytes]:
            if False:
                yield b""

        async def forward_audio(stream: AsyncIterator[bytes]) -> None:
            async for chunk in pipeline.process(stream, session=session):
                session.stats.bytes_sent += len(chunk)
                session.stats.chunks_sent += 1
                stats_tracker.total_bytes_processed += len(chunk)
                await transport.send_audio(chunk)

        async def forward_text(name: str, stream: AsyncIterator[bytes]) -> None:
            parameters = inspect.signature(pipeline.iter_stream).parameters
            if "session" in parameters:
                iterator = pipeline.iter_stream(name, stream, session=session)
            else:
                # Keep pipelines implementing the pre-Phase-1 method signature working.
                iterator = pipeline.iter_stream(name, stream)  # type: ignore[call-arg]
            if iterator is None:
                return
            async for text in iterator:
                await transport.send_event(
                    TransportEvent(
                        type=EventType.PIPELINE_EVENT,
                        session_id=session_id,
                        payload={"kind": "transcript", "stream": name, "text": text},
                    )
                )

        async def forward_metadata(name: str, stream: AsyncIterator[bytes]) -> None:
            iterator = pipeline.iter_metadata_stream(name, stream, session=session)
            if iterator is None:
                return
            async for envelope in iterator:
                await transport.send_event(
                    TransportEvent(
                        type=EventType.PIPELINE_EVENT,
                        session_id=session_id,
                        payload={
                            "kind": "metadata",
                            "stream": name,
                            "metadata": envelope.model_dump(),
                        },
                    )
                )

        async def stats_loop() -> None:
            while not stop_event.is_set():
                session.stats.duration_seconds = round(time.monotonic() - start_time, 1)
                session.stats.pipeline_latency_ms = 5000.0
                await transport.send_event(
                    TransportEvent(
                        type=EventType.SESSION_STATS,
                        session_id=session_id,
                        payload=session.stats.model_dump(),
                    )
                )
                await asyncio.sleep(1)

        async def listen_for_terminal() -> EventType | None:
            async for event in transport.recv_event():
                if event.type == EventType.SESSION_STOP:
                    logger.info("client requested stop for session %s", session_id)
                    return event.type
                if event.type == EventType.AUDIO_END:
                    logger.info("client audio ended for session %s", session_id)
                    return event.type
            return None

        processor_tasks: list[asyncio.Task[None]] = []
        has_audio = False
        for descriptor in pipeline.output_streams:
            if descriptor.kind == OutputStreamKind.AUDIO and descriptor.name == "audio":
                has_audio = True
            elif descriptor.kind == OutputStreamKind.TEXT:
                if descriptor.consumes_audio:
                    queue: asyncio.Queue[bytes | None] = asyncio.Queue(
                        maxsize=BOUNDED_BACKPRESSURE_FANOUT_QUEUE_CAPACITY
                    )
                    audio_fanout_queues.append(queue)
                    stream = queue_iter(queue)
                else:
                    stream = empty_stream()
                processor_tasks.append(
                    asyncio.create_task(forward_text(descriptor.name, stream))
                )
            elif descriptor.kind == OutputStreamKind.METADATA:
                if descriptor.consumes_audio:
                    meta_queue: asyncio.Queue[bytes | None] = asyncio.Queue(
                        maxsize=BOUNDED_BACKPRESSURE_FANOUT_QUEUE_CAPACITY
                    )
                    audio_fanout_queues.append(meta_queue)
                    meta_stream = queue_iter(meta_queue)
                else:
                    meta_stream = empty_stream()
                processor_tasks.append(
                    asyncio.create_task(forward_metadata(descriptor.name, meta_stream))
                )

        if has_audio:
            audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(
                maxsize=BOUNDED_BACKPRESSURE_FANOUT_QUEUE_CAPACITY
            )
            audio_fanout_queues.append(audio_queue)
            processor_tasks.append(asyncio.create_task(forward_audio(queue_iter(audio_queue))))

        iter_stage_events = getattr(pipeline, "iter_stage_events", None)
        if iter_stage_events is not None:
            stage_iterator = iter_stage_events(session=session)
            if stage_iterator is not None:
                async def _forward_stage_events(
                    iterator: AsyncIterator[dict[str, object]] = stage_iterator,
                ) -> None:
                    async for payload in iterator:
                        await transport.send_event(
                            TransportEvent(
                                type=EventType.PIPELINE_EVENT,
                                session_id=session_id,
                                payload={
                                    "kind": "stage.product",
                                    "stage": payload["stage"],
                                    "product": payload["product"],
                                },
                            )
                        )

                processor_tasks.append(asyncio.create_task(_forward_stage_events()))

        input_task = asyncio.create_task(audio_input())
        terminal_task = asyncio.create_task(listen_for_terminal())
        stats_task = asyncio.create_task(stats_loop())
        tasks = [input_task, terminal_task, stats_task, *processor_tasks]

        completed, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in completed:
            if task.cancelled():
                continue
            error = task.exception()
            if error is not None:
                raise error

        terminal = terminal_task.result() if terminal_task in completed else None
        stop_event.set()
        if terminal == EventType.SESSION_STOP:
            return

        if input_task in completed or terminal == EventType.AUDIO_END:
            input_task.cancel()
            terminal_task.cancel()
            await asyncio.gather(input_task, terminal_task, return_exceptions=True)
            async with asyncio.timeout(PROCESSOR_DRAIN_TIMEOUT_SECONDS):
                if current_fanout is not None:
                    await current_fanout
                await close_audio_fanout()
                await asyncio.gather(*processor_tasks)
            return

        # A processor, stats sender, or event receiver ended unexpectedly.
        if terminal_task in completed:
            logger.info("transport events ended for session %s", session_id)
        elif stats_task in completed:
            raise RuntimeError("session stats task ended unexpectedly")
    except Exception as exc:
        logger.exception("stream error for session %s", session_id)
        detail = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        try:
            async with asyncio.timeout(TRANSPORT_CLEANUP_TIMEOUT_SECONDS):
                await transport.send_event(
                    TransportEvent(
                        type=EventType.ERROR,
                        session_id=session_id,
                        payload={"detail": detail},
                    )
                )
        except Exception:
            logger.exception("failed to send stream error for session %s", session_id)
    finally:
        if current_fanout is not None and not current_fanout.done():
            current_fanout.cancel()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        session.status = SessionStatus.CLOSED
        session.stats.duration_seconds = round(time.monotonic() - start_time, 1)
        pipeline.discard_session_outputs(session)
        if pipeline_started:
            try:
                await pipeline.stop()
            except Exception:
                logger.exception("failed to stop pipeline for session %s", session_id)
        try:
            if pipeline_started:
                async with asyncio.timeout(TRANSPORT_CLEANUP_TIMEOUT_SECONDS):
                    await transport.send_event(
                        TransportEvent(type=EventType.SESSION_STOP, session_id=session_id)
                    )
        except Exception:
            logger.exception("failed to send stop event for session %s", session_id)
        try:
            async with asyncio.timeout(TRANSPORT_CLEANUP_TIMEOUT_SECONDS):
                await transport.close()
        except Exception:
            logger.exception("failed to close transport for session %s", session_id)
