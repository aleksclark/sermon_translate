from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from src.api.deps import init_deps
from src.api.store import SessionStore
from src.models import PipelineInfo, ServerStatsTracker, Session, SessionCreate, SessionStatus
from src.pipelines import PipelineRegistry
from src.pipelines.base import BasePipeline, OutputStreamDescriptor, OutputStreamKind
from src.transport.base import EventType, TransportConnection, TransportEvent
from src.transport.handler import run_session


class BlockingTransport(TransportConnection):
    def __init__(
        self,
        events: list[TransportEvent] | None = None,
        audio: list[bytes] | None = None,
        ready_error: Exception | None = None,
        send_audio_error: Exception | None = None,
    ) -> None:
        self._events = events or []
        self._audio = audio or []
        self._ready_error = ready_error
        self._send_audio_error = send_audio_error
        self.sent_audio: list[bytes] = []
        self.sent_events: list[TransportEvent] = []
        self.closed = False
        self.audio_cancelled = asyncio.Event()
        self.events_cancelled = asyncio.Event()
        self.audio_queued = asyncio.Event()

    async def wait_ready(self) -> None:
        if self._ready_error is not None:
            raise self._ready_error

    async def recv_audio(self) -> AsyncIterator[bytes]:
        try:
            for chunk in self._audio:
                yield chunk
            self.audio_queued.set()
            await asyncio.Event().wait()
        finally:
            self.audio_cancelled.set()

    async def send_audio(self, data: bytes) -> None:
        if self._send_audio_error is not None:
            raise self._send_audio_error
        self.sent_audio.append(data)

    async def send_event(self, event: TransportEvent) -> None:
        self.sent_events.append(event)

    async def recv_event(self) -> AsyncIterator[TransportEvent]:
        try:
            if self._audio:
                await self.audio_queued.wait()
            for event in self._events:
                yield event
            await asyncio.Event().wait()
        finally:
            self.events_cancelled.set()

    async def close(self) -> None:
        self.closed = True


class SessionEchoPipeline(BasePipeline):
    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(id="test", name="Test", description="Test pipeline")

    @property
    def output_streams(self) -> list[OutputStreamDescriptor]:
        return [
            OutputStreamDescriptor(name="audio", kind=OutputStreamKind.AUDIO),
            OutputStreamDescriptor(
                name="transcript", kind=OutputStreamKind.TEXT, consumes_audio=False
            ),
        ]

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None
    ) -> AsyncIterator[bytes]:
        async for chunk in audio_stream:
            text = f"{session.id}:{chunk.decode()}" if session is not None else chunk.decode()
            await self._publish_text("transcript", text, session)
            yield text.encode()
        await self._finish_text("transcript", session)

    def iter_stream(
        self,
        name: str,
        audio_stream: AsyncIterator[bytes],
        session: Session | None = None,
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        if name == "transcript":
            return self._drain_text(name, session)
        return None


class LegacyTextPipeline(BasePipeline):
    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(id="legacy", name="Legacy", description="Legacy pipeline")

    @property
    def output_streams(self) -> list[OutputStreamDescriptor]:
        return [OutputStreamDescriptor(name="transcript", kind=OutputStreamKind.TEXT)]

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None
    ) -> AsyncIterator[bytes]:
        yield  # pragma: no cover

    def iter_stream(
        self, name: str, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        async def transcribe() -> AsyncIterator[str]:
            async for chunk in audio_stream:
                yield chunk.decode()

        return transcribe() if name == "transcript" else None


class FailingPipeline(SessionEchoPipeline):
    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None
    ) -> AsyncIterator[bytes]:
        async for _ in audio_stream:
            raise RuntimeError("pipeline failed")
        yield b""


@pytest.fixture
def session_setup() -> tuple[SessionStore, PipelineRegistry, SessionEchoPipeline]:
    store = SessionStore()
    registry = PipelineRegistry()
    pipeline = SessionEchoPipeline()
    registry.register(pipeline)
    init_deps(store, registry, ServerStatsTracker())
    return store, registry, pipeline


def create_session(store: SessionStore) -> Session:
    return store.create(SessionCreate(pipeline_id="test"))


@pytest.mark.parametrize("terminal_type", [EventType.AUDIO_END, EventType.SESSION_STOP])
async def test_terminal_event_cancels_blocked_receivers_without_deadlock(
    session_setup: tuple[SessionStore, PipelineRegistry, SessionEchoPipeline],
    terminal_type: EventType,
) -> None:
    store, _, pipeline = session_setup
    session = create_session(store)
    transport = BlockingTransport(
        events=[TransportEvent(type=terminal_type, session_id=session.id)],
        audio=[str(index).encode() for index in range(12)],
    )

    await asyncio.wait_for(run_session(transport, session.id), timeout=1)

    assert transport.closed
    assert transport.audio_cancelled.is_set()
    assert transport.events_cancelled.is_set()
    assert session.status == SessionStatus.CLOSED
    assert pipeline._ref_count == 0
    assert transport.sent_events[-1].type == EventType.SESSION_STOP


async def test_legacy_text_stream_signature_and_audio_consumption_are_supported() -> None:
    store = SessionStore()
    registry = PipelineRegistry()
    pipeline = LegacyTextPipeline()
    registry.register(pipeline)
    init_deps(store, registry, ServerStatsTracker())
    session = store.create(SessionCreate(pipeline_id="legacy"))
    transport = BlockingTransport(
        audio=[b"legacy"],
        events=[TransportEvent(type=EventType.AUDIO_END, session_id=session.id)],
    )

    await asyncio.wait_for(run_session(transport, session.id), timeout=1)

    text = [
        event.payload["text"]
        for event in transport.sent_events
        if event.type == EventType.PIPELINE_EVENT
    ]
    assert text == ["legacy"]
    assert pipeline._ref_count == 0


async def test_pipeline_failure_cleans_up_blocked_transport(
    session_setup: tuple[SessionStore, PipelineRegistry, SessionEchoPipeline],
) -> None:
    store, registry, _ = session_setup
    pipeline = FailingPipeline()
    registry.register(pipeline)
    session = create_session(store)
    transport = BlockingTransport(audio=[b"audio"])

    await asyncio.wait_for(run_session(transport, session.id), timeout=1)

    assert transport.closed
    assert transport.audio_cancelled.is_set()
    assert transport.events_cancelled.is_set()
    assert pipeline._ref_count == 0
    assert EventType.ERROR in [event.type for event in transport.sent_events]


async def test_transport_failure_cleans_up_pipeline(
    session_setup: tuple[SessionStore, PipelineRegistry, SessionEchoPipeline],
) -> None:
    store, _, pipeline = session_setup
    session = create_session(store)
    transport = BlockingTransport(audio=[b"audio"], send_audio_error=RuntimeError("closed"))

    await asyncio.wait_for(run_session(transport, session.id), timeout=1)

    assert transport.closed
    assert transport.audio_cancelled.is_set()
    assert transport.events_cancelled.is_set()
    assert pipeline._ref_count == 0
    assert session.status == SessionStatus.CLOSED


async def test_readiness_timeout_closes_without_starting_pipeline(
    session_setup: tuple[SessionStore, PipelineRegistry, SessionEchoPipeline],
) -> None:
    store, _, pipeline = session_setup
    session = create_session(store)
    transport = BlockingTransport(ready_error=TimeoutError())

    await asyncio.wait_for(run_session(transport, session.id), timeout=1)

    assert transport.closed
    assert pipeline._ref_count == 0
    assert session.status == SessionStatus.CLOSED
    assert transport.sent_events == []


async def test_concurrent_sessions_keep_outputs_isolated(
    session_setup: tuple[SessionStore, PipelineRegistry, SessionEchoPipeline],
) -> None:
    store, _, pipeline = session_setup
    first = create_session(store)
    second = create_session(store)
    first_transport = BlockingTransport(
        audio=[b"first"],
        events=[TransportEvent(type=EventType.AUDIO_END, session_id=first.id)],
    )
    second_transport = BlockingTransport(
        audio=[b"second"],
        events=[TransportEvent(type=EventType.AUDIO_END, session_id=second.id)],
    )

    await asyncio.wait_for(
        asyncio.gather(
            run_session(first_transport, first.id),
            run_session(second_transport, second.id),
        ),
        timeout=1,
    )

    assert first_transport.sent_audio == [f"{first.id}:first".encode()]
    assert second_transport.sent_audio == [f"{second.id}:second".encode()]
    first_text = [
        event.payload["text"]
        for event in first_transport.sent_events
        if event.type == EventType.PIPELINE_EVENT
    ]
    second_text = [
        event.payload["text"]
        for event in second_transport.sent_events
        if event.type == EventType.PIPELINE_EVENT
    ]
    assert first_text == [f"{first.id}:first"]
    assert second_text == [f"{second.id}:second"]
    assert pipeline._session_text_queues == {}
    assert pipeline._ref_count == 0
