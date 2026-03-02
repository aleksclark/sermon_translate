from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator

from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from .base import EventType, TransportConnection, TransportEvent

AUDIO_PREFIX = b"\x01"
EVENT_PREFIX = b"\x02"


class WebSocketTransport(TransportConnection):
    """WebSocket implementation of the transport abstraction.

    Wire protocol (binary frames):
        0x01 + raw_audio_bytes   -> audio data
        0x02 + utf8_json         -> event envelope
    """

    def __init__(self, ws: WebSocket) -> None:
        self._ws = ws
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._event_queue: asyncio.Queue[TransportEvent] = asyncio.Queue()
        self._closed = False
        self._reader_task: asyncio.Task | None = None

    async def start_reader(self) -> None:
        self._reader_task = asyncio.create_task(self._read_loop())

    async def _read_loop(self) -> None:
        try:
            while not self._closed:
                raw = await self._ws.receive_bytes()
                if len(raw) < 1:
                    continue
                tag = raw[0:1]
                body = raw[1:]
                if tag == AUDIO_PREFIX:
                    await self._audio_queue.put(body)
                elif tag == EVENT_PREFIX:
                    evt = self._parse_event(body)
                    if evt:
                        await self._event_queue.put(evt)
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            self._closed = True
            await self._audio_queue.put(b"")

    @staticmethod
    def _parse_event(data: bytes) -> TransportEvent | None:
        try:
            obj = json.loads(data.decode())
            return TransportEvent(
                type=EventType(obj["type"]),
                session_id=obj.get("session_id", ""),
                payload=obj.get("payload", {}),
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    async def recv_audio(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._audio_queue.get()
            if chunk == b"":
                return
            yield chunk

    async def send_audio(self, data: bytes) -> None:
        if self._closed:
            return
        await self._ws.send_bytes(AUDIO_PREFIX + data)

    async def send_event(self, event: TransportEvent) -> None:
        if self._closed:
            return
        payload = json.dumps(
            {"type": event.type.value, "session_id": event.session_id, "payload": event.payload}
        ).encode()
        await self._ws.send_bytes(EVENT_PREFIX + payload)

    async def recv_event(self) -> AsyncIterator[TransportEvent]:
        while not self._closed:
            try:
                evt = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
                yield evt
            except TimeoutError:
                continue

    async def close(self) -> None:
        self._closed = True
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
        if self._ws.client_state == WebSocketState.CONNECTED:
            with contextlib.suppress(Exception):
                await self._ws.close()
