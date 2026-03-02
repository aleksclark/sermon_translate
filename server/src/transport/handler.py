from __future__ import annotations

import asyncio
import time

from starlette.websockets import WebSocket

from src.api.deps import get_pipeline_registry, get_server_stats, get_session_store
from src.models import SessionStatus
from src.transport.base import EventType, TransportEvent
from src.transport.ws import WebSocketTransport


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
    await pipeline.start()
    await transport.send_event(
        TransportEvent(type=EventType.SESSION_START, session_id=session_id)
    )

    stats_tracker = get_server_stats()
    start_time = time.monotonic()

    try:
        async def forward_output() -> None:
            async for chunk in pipeline.process(audio_input()):
                session.stats.bytes_sent += len(chunk)
                session.stats.chunks_sent += 1
                stats_tracker.total_bytes_processed += len(chunk)
                await transport.send_audio(chunk)

        async def audio_input():
            async for chunk in transport.recv_audio():
                session.stats.bytes_received += len(chunk)
                session.stats.chunks_received += 1
                yield chunk

        async def forward_text() -> None:
            text_stream = pipeline.process_text(audio_input())
            if text_stream is None:
                return
            async for text in text_stream:
                await transport.send_event(
                    TransportEvent(
                        type=EventType.PIPELINE_EVENT,
                        session_id=session_id,
                        payload={"kind": "transcript", "text": text},
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

        from src.pipelines.base import BasePipeline

        produces_text = type(pipeline).process_text is not BasePipeline.process_text
        tasks = [stats_loop()]
        if produces_text:
            tasks.append(forward_text())
        else:
            tasks.append(forward_output())
        await asyncio.gather(*tasks)
    except Exception:
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
