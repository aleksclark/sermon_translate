from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from starlette.websockets import WebSocket

from src.api.deps import get_pipeline_registry, get_server_stats, get_session_store
from src.models import SessionStatus
from src.pipelines.base import OutputStreamKind
from src.transport.base import EventType, TransportEvent
from src.transport.ws import WebSocketTransport

logger = logging.getLogger(__name__)


async def handle_stream(ws: WebSocket, session_id: str) -> None:
    """Main entrypoint wired to the WS route."""
    store = get_session_store()
    session = store.get(session_id)
    if session is None:
        await ws.close(code=4004, reason="Session not found")
        return

    registry = get_pipeline_registry()
    pipeline = registry.get(session.pipeline_id)
    if pipeline is None:
        await ws.close(code=4004, reason="Pipeline not found")
        return

    await ws.accept()
    transport = WebSocketTransport(ws)
    await transport.start_reader()

    session.status = SessionStatus.ACTIVE
    pipeline.configure_session(session)
    await pipeline.start()
    await transport.send_event(
        TransportEvent(type=EventType.SESSION_START, session_id=session_id)
    )

    stats_tracker = get_server_stats()
    start_time = time.monotonic()

    try:
        audio_queues: list[asyncio.Queue[bytes | None]] = []

        async def audio_input() -> None:
            async for chunk in transport.recv_audio():
                session.stats.bytes_received += len(chunk)
                session.stats.chunks_received += 1
                for q in audio_queues:
                    await q.put(chunk)
            for q in audio_queues:
                await q.put(None)

        async def queue_iter(q: asyncio.Queue[bytes | None]) -> AsyncIterator[bytes]:
            while True:
                item = await q.get()
                if item is None:
                    return
                yield item

        async def forward_audio(stream: AsyncIterator[bytes]) -> None:
            async for chunk in pipeline.process(stream):
                session.stats.bytes_sent += len(chunk)
                session.stats.chunks_sent += 1
                stats_tracker.total_bytes_processed += len(chunk)
                await transport.send_audio(chunk)

        async def forward_text(name: str, stream: AsyncIterator[bytes]) -> None:
            it = pipeline.iter_stream(name, stream)
            if it is None:
                return
            async for text in it:
                await transport.send_event(
                    TransportEvent(
                        type=EventType.PIPELINE_EVENT,
                        session_id=session_id,
                        payload={"kind": "transcript", "stream": name, "text": text},
                    )
                )

        async def stats_loop() -> None:
            while session.status == SessionStatus.ACTIVE:
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

        tasks: list[asyncio.Task | asyncio.Future] = [asyncio.ensure_future(stats_loop())]

        has_audio = False
        for desc in pipeline.output_streams:
            if desc.kind == OutputStreamKind.AUDIO and desc.name == "audio":
                has_audio = True
            elif desc.kind == OutputStreamKind.TEXT:
                q: asyncio.Queue[bytes | None] = asyncio.Queue()
                audio_queues.append(q)
                tasks.append(asyncio.ensure_future(forward_text(desc.name, queue_iter(q))))

        if has_audio:
            q_audio: asyncio.Queue[bytes | None] = asyncio.Queue()
            audio_queues.append(q_audio)
            tasks.append(asyncio.ensure_future(forward_audio(queue_iter(q_audio))))

        tasks.append(asyncio.ensure_future(audio_input()))
        await asyncio.gather(*tasks)
    except Exception:
        logger.exception("stream error for session %s", session_id)
        await transport.send_event(
            TransportEvent(
                type=EventType.ERROR,
                session_id=session_id,
                payload={"detail": "stream error"},
            )
        )
    finally:
        session.status = SessionStatus.CLOSED
        session.stats.duration_seconds = round(time.monotonic() - start_time, 1)
        await pipeline.stop()
        await transport.send_event(
            TransportEvent(type=EventType.SESSION_STOP, session_id=session_id)
        )
        await transport.close()
