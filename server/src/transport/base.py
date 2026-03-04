from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    SESSION_START = "session.start"
    SESSION_STOP = "session.stop"
    SESSION_STATS = "session.stats"
    PIPELINE_EVENT = "pipeline.event"
    AUDIO_END = "audio.end"
    ERROR = "error"


@dataclass
class TransportEvent:
    type: EventType
    session_id: str
    payload: dict[str, Any] = field(default_factory=dict)


class TransportConnection(abc.ABC):
    """Abstract bidirectional connection for audio + events."""

    @abc.abstractmethod
    def recv_audio(self) -> AsyncIterator[bytes]:
        """Yield audio chunks from the client."""
        ...

    @abc.abstractmethod
    async def send_audio(self, data: bytes) -> None:
        """Send an audio chunk to the client."""
        ...

    @abc.abstractmethod
    async def send_event(self, event: TransportEvent) -> None:
        """Send a JSON event to the client."""
        ...

    @abc.abstractmethod
    def recv_event(self) -> AsyncIterator[TransportEvent]:
        """Yield events from the client."""
        ...

    @abc.abstractmethod
    async def wait_ready(self) -> None:
        """Wait until the transport is fully connected."""
        ...

    @abc.abstractmethod
    async def close(self) -> None: ...
