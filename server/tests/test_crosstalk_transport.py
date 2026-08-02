from __future__ import annotations

import numpy as np
import pytest
from av import AudioFrame

from src.transport.base import EventType, TransportEvent
from src.transport.crosstalk import CrosstalkTransport


class FakeTrack:
    kind = "audio"

    def __init__(self, frames: list[AudioFrame]) -> None:
        self._frames = list(frames)
        self._index = 0

    async def recv(self) -> AudioFrame:
        from aiortc.mediastreams import MediaStreamError

        if self._index >= len(self._frames):
            raise MediaStreamError
        frame = self._frames[self._index]
        self._index += 1
        return frame


class FakePeer:
    def __init__(self) -> None:
        self.added_tracks: list[object] = []
        self._handlers: dict[str, object] = {}

    def addTrack(self, track: object) -> None:  # noqa: N802
        self.added_tracks.append(track)

    def on(self, event: str):
        def deco(fn):
            self._handlers[event] = fn
            return fn

        return deco

    def emit_track(self, track: object) -> None:
        handler = self._handlers.get("track")
        assert handler is not None
        handler(track)  # type: ignore[operator]


class FakeSignaling:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeClient:
    def __init__(self, peer: FakePeer, signaling: FakeSignaling) -> None:
        self._peer = peer
        self._signaling = signaling
        self.opened_with: dict[str, object] = {}
        self.on_ready = None

    async def open_media_peer(
        self,
        session_id: str,
        *,
        produce=None,
        listen=None,
        configure=None,
        on_ready=None,
    ):
        self.opened_with = {
            "session_id": session_id,
            "produce": produce,
            "listen": listen,
        }
        self.on_ready = on_ready
        if configure is not None:
            configure(self._peer)
        return self._signaling


def _mono_frame(value: int = 0, samples: int = 480) -> AudioFrame:
    arr = np.full((1, samples), value, dtype=np.int16)
    frame = AudioFrame.from_ndarray(arr, format="s16", layout="mono")
    frame.sample_rate = 48000
    return frame


async def _make_connected_transport() -> tuple[
    CrosstalkTransport, FakePeer, FakeSignaling, FakeClient
]:
    peer = FakePeer()
    signaling = FakeSignaling()
    client = FakeClient(peer, signaling)
    transport = CrosstalkTransport(
        client,  # type: ignore[arg-type]
        "cs1",
        produce=["type:broadcast"],
        listen=["type:feed"],
    )
    await transport.connect()
    return transport, peer, signaling, client


class TestCrosstalkTransport:
    async def test_connect_adds_output_track_and_passes_selectors(self) -> None:
        transport, peer, _, client = await _make_connected_transport()
        assert transport.output_track in peer.added_tracks
        assert client.opened_with["produce"] == ["type:broadcast"]
        assert client.opened_with["listen"] == ["type:feed"]
        assert client.opened_with["session_id"] == "cs1"

    async def test_send_audio_pushes_to_track(self) -> None:
        transport, _, _, _ = await _make_connected_transport()
        await transport.send_audio(b"\x00" * 100)
        assert transport.output_track.queued_bytes == 1

    async def test_recv_audio_yields_pcm(self) -> None:
        transport, peer, _, _ = await _make_connected_transport()
        peer.emit_track(FakeTrack([_mono_frame(0), _mono_frame(0)]))

        chunks: list[bytes] = []
        async for chunk in transport.recv_audio():
            chunks.append(chunk)
        assert len(chunks) == 2
        assert len(chunks[0]) == 480 * 2

    async def test_recv_audio_downmixes_stereo(self) -> None:
        transport, peer, _, _ = await _make_connected_transport()
        left = np.full(480, 100, dtype=np.int16)
        right = np.full(480, 200, dtype=np.int16)
        interleaved = np.empty(960, dtype=np.int16)
        interleaved[0::2] = left
        interleaved[1::2] = right
        frame = AudioFrame.from_ndarray(interleaved.reshape(1, -1), format="s16", layout="stereo")
        frame.sample_rate = 48000
        peer.emit_track(FakeTrack([frame]))

        chunks: list[bytes] = []
        async for chunk in transport.recv_audio():
            chunks.append(chunk)
        result = np.frombuffer(chunks[0], dtype=np.int16)
        assert result[0] == 150

    async def test_track_end_emits_audio_end_event(self) -> None:
        transport, peer, _, _ = await _make_connected_transport()
        peer.emit_track(FakeTrack([_mono_frame(0)]))

        async for _ in transport.recv_audio():
            pass

        events: list[TransportEvent] = []
        async for evt in transport.recv_event():
            events.append(evt)
            break
        assert events[0].type == EventType.AUDIO_END

    async def test_wait_ready_requires_track_and_connection(self) -> None:
        transport, peer, _, client = await _make_connected_transport()
        with pytest.raises(TimeoutError):
            await transport.wait_ready(timeout=0.05)

        peer.emit_track(FakeTrack([]))
        assert client.on_ready is not None
        client.on_ready()  # type: ignore[misc]
        await transport.wait_ready(timeout=1.0)

    async def test_close_tears_down_signaling(self) -> None:
        transport, _, signaling, _ = await _make_connected_transport()
        await transport.close()
        assert signaling.closed

    async def test_send_event_is_noop(self) -> None:
        transport, _, _, _ = await _make_connected_transport()
        await transport.send_event(
            TransportEvent(type=EventType.SESSION_START, session_id="cs1")
        )
