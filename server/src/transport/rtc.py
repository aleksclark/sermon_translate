from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from aiortc import MediaStreamTrack, RTCPeerConnection
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame

from ._audio_track import OPUS_SAMPLE_RATE, OutputAudioTrack, frame_to_mono_pcm
from .base import EventType, TransportConnection, TransportEvent

logger = logging.getLogger(__name__)

FRAME_DURATION = 0.020  # 20 ms
READY_TIMEOUT = 15.0  # seconds to wait for DC + track

__all__ = ["OPUS_SAMPLE_RATE", "OutputAudioTrack", "WebRTCTransport"]


class WebRTCTransport(TransportConnection):
    """WebRTC implementation of the transport abstraction.

    Audio flows over Opus/RTP tracks.
    Events flow over a reliable ordered DataChannel named ``events``.
    """

    def __init__(self, pc: RTCPeerConnection, sample_rate: int = OPUS_SAMPLE_RATE) -> None:
        self._pc = pc
        self._sample_rate = sample_rate
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._event_queue: asyncio.Queue[TransportEvent] = asyncio.Queue()
        self._output_track = OutputAudioTrack(sample_rate)
        self._closed = False
        self._input_task: asyncio.Task[None] | None = None
        self._dc: Any = None  # RTCDataChannel, set during setup
        self._dc_ready = asyncio.Event()
        self._track_ready = asyncio.Event()

    @property
    def output_track(self) -> OutputAudioTrack:
        return self._output_track

    # ------------------------------------------------------------------
    # Setup helpers (called during signaling, before run_session)
    # ------------------------------------------------------------------

    def setup_data_channel(self, dc: Any) -> None:
        """Attach the DataChannel created during signaling."""
        self._dc = dc

        @dc.on("open")
        def _on_open() -> None:
            logger.info("DataChannel open")
            self._dc_ready.set()

        @dc.on("message")
        def _on_message(message: str | bytes) -> None:
            text = message if isinstance(message, str) else message.decode()
            try:
                obj = json.loads(text)
                evt = TransportEvent(
                    type=EventType(obj["type"]),
                    session_id=obj.get("session_id", ""),
                    payload=obj.get("payload", {}),
                )
                self._event_queue.put_nowait(evt)
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        if dc.readyState == "open":
            self._dc_ready.set()

    def setup_incoming_track(self, track: MediaStreamTrack) -> None:
        """Start reading audio frames from the client's track."""
        self._input_task = asyncio.create_task(self._read_track(track))
        self._track_ready.set()

    async def wait_ready(self, timeout: float = READY_TIMEOUT) -> None:
        """Block until the DataChannel and audio track are available."""
        try:
            await asyncio.wait_for(
                asyncio.gather(self._dc_ready.wait(), self._track_ready.wait()),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning("WebRTC transport ready timeout after %.1fs", timeout)
            self._closed = True
            raise

    async def _read_track(self, track: MediaStreamTrack) -> None:
        try:
            while not self._closed:
                frame: AudioFrame = await track.recv()  # type: ignore[assignment]
                pcm = frame_to_mono_pcm(frame)
                await self._audio_queue.put(pcm)
        except MediaStreamError:
            pass
        except Exception:
            logger.exception("error reading WebRTC audio track")
        finally:
            self._closed = True
            await self._audio_queue.put(b"")

    # ------------------------------------------------------------------
    # TransportConnection interface
    # ------------------------------------------------------------------

    async def recv_audio(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._audio_queue.get()
            if chunk == b"":
                return
            yield chunk

    async def send_audio(self, data: bytes) -> None:
        if self._closed:
            return
        self._output_track.push(data)

    async def send_event(self, event: TransportEvent) -> None:
        if self._closed or self._dc is None:
            return
        if self._dc.readyState != "open":
            return
        payload = json.dumps(
            {"type": event.type.value, "session_id": event.session_id, "payload": event.payload}
        )
        self._dc.send(payload)

    async def recv_event(self) -> AsyncIterator[TransportEvent]:
        while not self._closed:
            try:
                evt = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
                yield evt
            except TimeoutError:
                continue

    async def close(self) -> None:
        self._closed = True
        self._dc_ready.set()
        self._track_ready.set()
        self._output_track.finish()
        if self._input_task and not self._input_task.done():
            self._input_task.cancel()
        await self._pc.close()
