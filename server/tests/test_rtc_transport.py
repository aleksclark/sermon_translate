from __future__ import annotations

import json

import numpy as np
import pytest
from av import AudioFrame

from src.transport.base import EventType, TransportEvent
from src.transport.rtc import OPUS_SAMPLE_RATE, OutputAudioTrack, WebRTCTransport


class TestOutputAudioTrack:
    async def test_push_and_recv(self) -> None:
        track = OutputAudioTrack(sample_rate=48000)
        pcm = np.zeros(960, dtype=np.int16).tobytes()
        track.push(pcm)
        frame = await track.recv()
        assert isinstance(frame, AudioFrame)
        assert frame.sample_rate == 48000
        assert frame.pts == 0

    async def test_pts_advances(self) -> None:
        track = OutputAudioTrack(sample_rate=48000)
        pcm = np.zeros(960, dtype=np.int16).tobytes()
        track.push(pcm)
        track.push(pcm)
        f1 = await track.recv()
        f2 = await track.recv()
        assert f1.pts == 0
        assert f2.pts == 960

    async def test_finish_raises(self) -> None:
        from aiortc.mediastreams import MediaStreamError

        track = OutputAudioTrack()
        track.finish()
        with pytest.raises(MediaStreamError):
            await track.recv()

    async def test_queued_bytes(self) -> None:
        track = OutputAudioTrack()
        assert track.queued_bytes == 0
        track.push(b"\x00" * 100)
        track.push(b"\x00" * 200)
        assert track.queued_bytes == 2

    async def test_large_blob_chunked_to_20ms(self) -> None:
        track = OutputAudioTrack(sample_rate=48000)
        # Push 3 frames worth of audio as one blob (2880 samples = 60ms)
        pcm = np.zeros(2880, dtype=np.int16).tobytes()
        track.push(pcm)
        track.finish()
        f1 = await track.recv()
        f2 = await track.recv()
        f3 = await track.recv()
        assert f1.pts == 0
        assert f2.pts == 960
        assert f3.pts == 1920
        for f in (f1, f2, f3):
            assert f.samples == 960


class FakeDataChannel:
    def __init__(self) -> None:
        self.readyState = "open"
        self.sent: list[str] = []
        self._handlers: dict[str, object] = {}

    def on(self, event: str):  # noqa: ANN201
        def decorator(fn):  # noqa: ANN001, ANN202
            self._handlers[event] = fn
            return fn
        return decorator

    def send(self, data: str) -> None:
        self.sent.append(data)

    def fire_message(self, text: str) -> None:
        handler = self._handlers.get("message")
        if handler:
            handler(text)  # type: ignore[operator]


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


class FakePeerConnection:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class TestWebRTCTransport:
    def _make_transport(self) -> tuple[WebRTCTransport, FakePeerConnection]:
        pc = FakePeerConnection()
        transport = WebRTCTransport(pc, sample_rate=OPUS_SAMPLE_RATE)  # type: ignore[arg-type]
        return transport, pc

    def test_output_track_property(self) -> None:
        transport, _ = self._make_transport()
        assert isinstance(transport.output_track, OutputAudioTrack)

    async def test_send_audio_pushes_to_track(self) -> None:
        transport, _ = self._make_transport()
        pcm = b"\x00" * 100
        await transport.send_audio(pcm)
        assert transport.output_track.queued_bytes == 1

    async def test_send_event_via_datachannel(self) -> None:
        transport, _ = self._make_transport()
        dc = FakeDataChannel()
        transport.setup_data_channel(dc)
        evt = TransportEvent(type=EventType.SESSION_START, session_id="s1")
        await transport.send_event(evt)
        assert len(dc.sent) == 1
        parsed = json.loads(dc.sent[0])
        assert parsed["type"] == "session.start"

    async def test_send_event_skips_when_closed(self) -> None:
        transport, _ = self._make_transport()
        dc = FakeDataChannel()
        transport.setup_data_channel(dc)
        await transport.close()
        evt = TransportEvent(type=EventType.SESSION_START, session_id="s1")
        await transport.send_event(evt)
        assert len(dc.sent) == 0

    async def test_recv_event_from_datachannel(self) -> None:
        transport, _ = self._make_transport()
        dc = FakeDataChannel()
        transport.setup_data_channel(dc)

        msg = json.dumps({"type": "session.stop", "session_id": "s1", "payload": {}})
        dc.fire_message(msg)

        events: list[TransportEvent] = []
        async for evt in transport.recv_event():
            events.append(evt)
            break
        assert len(events) == 1
        assert events[0].type == EventType.SESSION_STOP

    async def test_recv_audio_from_track(self) -> None:
        transport, _ = self._make_transport()

        arr = np.zeros((1, 480), dtype=np.int16)
        frame = AudioFrame.from_ndarray(arr, format="s16", layout="mono")
        frame.sample_rate = 48000

        fake_track = FakeTrack([frame])
        transport.setup_incoming_track(fake_track)  # type: ignore[arg-type]

        chunks: list[bytes] = []
        async for chunk in transport.recv_audio():
            chunks.append(chunk)
            break

        assert len(chunks) == 1
        assert len(chunks[0]) == 480 * 2

    async def test_recv_audio_downmixes_stereo(self) -> None:
        transport, _ = self._make_transport()

        # Simulate aiortc stereo decode: s16 packed, 480 samples per channel
        left = np.full(480, 100, dtype=np.int16)
        right = np.full(480, 200, dtype=np.int16)
        interleaved = np.empty(960, dtype=np.int16)
        interleaved[0::2] = left
        interleaved[1::2] = right
        arr = interleaved.reshape(1, -1)
        frame = AudioFrame.from_ndarray(arr, format="s16", layout="stereo")
        frame.sample_rate = 48000

        fake_track = FakeTrack([frame])
        transport.setup_incoming_track(fake_track)  # type: ignore[arg-type]

        chunks: list[bytes] = []
        async for chunk in transport.recv_audio():
            chunks.append(chunk)
            break

        assert len(chunks) == 1
        # Downmixed to mono: 480 samples * 2 bytes = 960 bytes
        assert len(chunks[0]) == 480 * 2
        result = np.frombuffer(chunks[0], dtype=np.int16)
        # Mean of 100 and 200 = 150
        assert result[0] == 150

    async def test_close_closes_pc(self) -> None:
        transport, pc = self._make_transport()
        await transport.close()
        assert pc.closed


class TestWebRTCOfferEndpoint:
    """Test the POST /sessions/{id}/offer endpoint via HTTPX."""

    @pytest.fixture
    def app(self):
        from src.main import create_app
        return create_app()

    @pytest.fixture
    async def client(self, app):
        from httpx import ASGITransport, AsyncClient
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c

    async def test_offer_session_not_found(self, client) -> None:  # type: ignore[no-untyped-def]
        r = await client.post(
            "/api/sessions/nonexistent/offer",
            json={"sdp": "v=0\r\n", "type": "offer"},
        )
        assert r.status_code == 404

    async def test_offer_requires_body(self, client) -> None:  # type: ignore[no-untyped-def]
        create = await client.post("/api/sessions", json={"pipeline_id": "echo"})
        sid = create.json()["id"]
        r = await client.post(f"/api/sessions/{sid}/offer")
        assert r.status_code == 422
