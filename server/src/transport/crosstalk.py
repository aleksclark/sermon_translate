from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from aiortc import MediaStreamTrack, RTCPeerConnection
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame

from ._audio_track import OPUS_SAMPLE_RATE, OutputAudioTrack, frame_to_mono_pcm
from .base import EventType, TransportConnection, TransportEvent
from .crosstalk_client import CrosstalkClient, SignalingSession

logger = logging.getLogger(__name__)

READY_TIMEOUT = 20.0


class CrosstalkTransport(TransportConnection):
    """Transport that bridges a Crosstalk SFU session to the pipeline.

    Received Opus tracks are the translation INPUT (decoded to mono s16le PCM);
    an added OutputAudioTrack carries the translation OUTPUT back to Crosstalk.
    There is no browser DataChannel: events map to internal connection/status
    signals so the shared session handler drains and closes cleanly.
    """

    def __init__(
        self,
        client: CrosstalkClient,
        session_id: str,
        *,
        produce: list[str] | None = None,
        listen: list[str] | None = None,
        sample_rate: int = OPUS_SAMPLE_RATE,
    ) -> None:
        self._client = client
        self._session_id = session_id
        self._produce = produce
        self._listen = listen
        self._sample_rate = sample_rate
        self._output_track = OutputAudioTrack(sample_rate)
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._event_queue: asyncio.Queue[TransportEvent] = asyncio.Queue()
        self._signaling: SignalingSession | None = None
        self._input_tasks: list[asyncio.Task[None]] = []
        self._active_tracks = 0
        self._track_ready = asyncio.Event()
        self._connected = asyncio.Event()
        self._closed = False

    @property
    def output_track(self) -> OutputAudioTrack:
        return self._output_track

    async def connect(self) -> None:
        """Open the media peer, add the output track, and start signaling."""
        signaling = await self._client.open_media_peer(
            self._session_id,
            produce=self._produce,
            listen=self._listen,
            on_ready=self._connected.set,
            configure=self._configure_peer,
        )
        self._signaling = signaling

    def _configure_peer(self, pc: RTCPeerConnection) -> None:
        pc.addTrack(self._output_track)

        @pc.on("track")
        def _on_track(track: MediaStreamTrack) -> None:
            if track.kind == "audio":
                self._active_tracks += 1
                self._track_ready.set()
                self._input_tasks.append(asyncio.create_task(self._read_track(track)))

    async def _read_track(self, track: MediaStreamTrack) -> None:
        try:
            while not self._closed:
                frame: AudioFrame = await track.recv()  # type: ignore[assignment]
                await self._audio_queue.put(frame_to_mono_pcm(frame))
        except MediaStreamError:
            pass
        except Exception:
            logger.exception("error reading crosstalk audio track")
        finally:
            self._active_tracks -= 1
            if self._active_tracks <= 0:
                await self._audio_queue.put(b"")
                await self._event_queue.put(
                    TransportEvent(type=EventType.AUDIO_END, session_id=self._session_id)
                )

    async def wait_ready(self, timeout: float = READY_TIMEOUT) -> None:
        try:
            await asyncio.wait_for(
                asyncio.gather(self._track_ready.wait(), self._connected.wait()),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning("crosstalk transport ready timeout after %.1fs", timeout)
            self._closed = True
            raise

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
        logger.debug(
            "crosstalk transport event %s for session %s", event.type, event.session_id
        )

    async def recv_event(self) -> AsyncIterator[TransportEvent]:
        while not self._closed:
            try:
                event = await asyncio.wait_for(self._event_queue.get(), timeout=0.5)
                yield event
            except TimeoutError:
                continue

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._track_ready.set()
        self._connected.set()
        self._output_track.finish()
        for task in self._input_tasks:
            if not task.done():
                task.cancel()
        if self._signaling is not None:
            await self._signaling.close()
