from __future__ import annotations

import asyncio
import base64
import json

import httpx
import pytest

from src.transport.crosstalk_client import (
    CrosstalkAuthError,
    CrosstalkClient,
    CrosstalkSSRFError,
    SignalingSession,
    guard_destination,
)

BASE_URL = "https://crosstalk.example.com"


def _make_jwt(exp: float, role: str = "translator", sub: str = "u1") -> str:
    header = base64.urlsafe_b64encode(b"{}").decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"exp": exp, "role": role, "sub": sub}).encode()
    ).decode().rstrip("=")
    return f"{header}.{payload}.sig"


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False
        self._inbound: asyncio.Queue[str] = asyncio.Queue()

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def recv(self) -> str:
        return await self._inbound.get()

    async def close(self) -> None:
        self.closed = True

    def feed(self, message: str) -> None:
        self._inbound.put_nowait(message)


class FakePeer:
    def __init__(self) -> None:
        self.local_sdp = "v=0\r\noffer-sdp"
        self.closed = False
        self.remote: list[tuple[str, str]] = []
        self.handlers: dict[str, object] = {}

    class _Desc:
        def __init__(self, sdp: str) -> None:
            self.sdp = sdp

    @property
    def localDescription(self):  # noqa: N802
        return FakePeer._Desc(self.local_sdp)

    async def createOffer(self):  # noqa: N802
        return FakePeer._Desc(self.local_sdp)

    async def setLocalDescription(self, desc) -> None:  # noqa: N802
        pass

    async def setRemoteDescription(self, desc) -> None:  # noqa: N802
        self.remote.append((desc.type, desc.sdp))

    def on(self, event: str):
        def deco(fn):
            self.handlers[event] = fn
            return fn

        return deco

    async def close(self) -> None:
        self.closed = True


def _make_client(handler, *, ws=None, peer=None, clock=None) -> CrosstalkClient:
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport)

    async def ws_factory(url: str):
        assert ws is not None
        ws.url = url
        return ws

    def peer_factory():
        assert peer is not None
        return peer

    return CrosstalkClient(
        BASE_URL,
        "user",
        "pass",
        http_client=http,
        ws_factory=ws_factory,
        peer_factory=peer_factory,
        clock=clock,
    )


class TestSSRFGuard:
    def test_rejects_non_http_scheme(self) -> None:
        with pytest.raises(CrosstalkSSRFError):
            guard_destination("ftp://host/x", allow_private=False)

    def test_rejects_loopback(self) -> None:
        with pytest.raises(CrosstalkSSRFError):
            guard_destination("http://127.0.0.1/x", allow_private=False)

    def test_rejects_private(self) -> None:
        with pytest.raises(CrosstalkSSRFError):
            guard_destination("http://10.0.0.5/x", allow_private=False)

    def test_allows_private_when_flagged(self) -> None:
        guard_destination("http://127.0.0.1/x", allow_private=True)

    def test_ws_scheme_enforced(self) -> None:
        with pytest.raises(CrosstalkSSRFError):
            guard_destination("https://host/x", allow_private=True, ws=True)
        guard_destination("wss://host/x", allow_private=True, ws=True)

    def test_ctor_blocks_private_base_url(self) -> None:
        with pytest.raises(CrosstalkSSRFError):
            CrosstalkClient(
                "http://127.0.0.1",
                "u",
                "p",
                http_client=httpx.AsyncClient(),
                ws_factory=lambda url: None,  # type: ignore[arg-type,return-value]
                peer_factory=lambda: None,  # type: ignore[arg-type,return-value]
            )


class TestLoginAndDiscovery:
    async def test_login_and_list_sessions(self) -> None:
        clock = FakeClock()
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            if request.url.path == "/api/auth/login":
                return httpx.Response(
                    200,
                    json={
                        "access_token": _make_jwt(clock.now + 900),
                        "refresh_token": "r1",
                    },
                )
            if request.url.path == "/api/sessions":
                assert request.headers["Authorization"].startswith("Bearer ")
                return httpx.Response(
                    200,
                    json={"data": [{"id": "s1", "name": "Main", "description": "d"}]},
                )
            return httpx.Response(404)

        client = _make_client(handler, clock=clock)
        sessions = await client.list_sessions()
        assert [s.id for s in sessions] == ["s1"]
        assert sessions[0].name == "Main"
        assert calls[0] == "/api/auth/login"

    async def test_list_channels_and_sources(self) -> None:
        clock = FakeClock()

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/auth/login":
                return httpx.Response(
                    200,
                    json={"access_token": _make_jwt(clock.now + 900), "refresh_token": "r1"},
                )
            if request.url.path == "/api/sessions/s1/channels":
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {"id": "c1", "name": "Feed", "type": "feed", "session_id": "s1"}
                        ]
                    },
                )
            if request.url.path == "/api/sessions/s1/sources":
                return httpx.Response(
                    200,
                    json={"data": [{"id": "src1", "name": "Booth", "connected": True}]},
                )
            return httpx.Response(404)

        client = _make_client(handler, clock=clock)
        channels = await client.list_channels("s1")
        assert channels[0].type == "feed"
        sources = await client.list_sources("s1")
        assert sources[0].connected is True

    async def test_login_failure_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401)

        client = _make_client(handler, clock=FakeClock())
        with pytest.raises(CrosstalkAuthError):
            await client.list_sessions()


class TestTokenRefresh:
    async def test_refreshes_on_expiry(self) -> None:
        clock = FakeClock(now=1000.0)
        seq: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/api/auth/login":
                seq.append("login")
                return httpx.Response(
                    200,
                    json={"access_token": _make_jwt(clock.now + 60), "refresh_token": "r1"},
                )
            if path == "/api/auth/refresh":
                seq.append("refresh")
                body = json.loads(request.content)
                assert body["refresh_token"] == "r1"
                return httpx.Response(
                    200,
                    json={"access_token": _make_jwt(clock.now + 900), "refresh_token": "r2"},
                )
            if path == "/api/sessions":
                seq.append("sessions")
                return httpx.Response(200, json={"data": []})
            return httpx.Response(404)

        client = _make_client(handler, clock=clock)
        await client.list_sessions()
        clock.now += 100  # push past the 60s access token lifetime + skew
        await client.list_sessions()
        assert seq == ["login", "sessions", "refresh", "sessions"]

    async def test_refresh_falls_back_to_login(self) -> None:
        clock = FakeClock(now=1000.0)
        seq: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/api/auth/login":
                seq.append("login")
                return httpx.Response(
                    200,
                    json={"access_token": _make_jwt(clock.now + 60), "refresh_token": "r1"},
                )
            if path == "/api/auth/refresh":
                seq.append("refresh")
                return httpx.Response(401)
            if path == "/api/sessions":
                seq.append("sessions")
                return httpx.Response(200, json={"data": []})
            return httpx.Response(404)

        client = _make_client(handler, clock=clock)
        await client.list_sessions()
        clock.now += 100
        await client.list_sessions()
        assert seq == ["login", "sessions", "refresh", "login", "sessions"]

    async def test_retries_once_on_401(self) -> None:
        clock = FakeClock(now=1000.0)
        state = {"served_401": False}

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/api/auth/login":
                return httpx.Response(
                    200,
                    json={"access_token": _make_jwt(clock.now + 900), "refresh_token": "r1"},
                )
            if path == "/api/auth/refresh":
                return httpx.Response(
                    200,
                    json={"access_token": _make_jwt(clock.now + 900), "refresh_token": "r2"},
                )
            if path == "/api/sessions":
                if not state["served_401"]:
                    state["served_401"] = True
                    return httpx.Response(401)
                return httpx.Response(200, json={"data": []})
            return httpx.Response(404)

        client = _make_client(handler, clock=clock)
        assert await client.list_sessions() == []


class TestSignalingFlow:
    async def test_offer_sent_and_answer_applied(self) -> None:
        clock = FakeClock()

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/auth/login":
                return httpx.Response(
                    200,
                    json={"access_token": _make_jwt(clock.now + 900), "refresh_token": "r1"},
                )
            return httpx.Response(404)

        ws = FakeWebSocket()
        peer = FakePeer()
        client = _make_client(handler, ws=ws, peer=peer, clock=clock)

        configured: list[FakePeer] = []
        session = await client.open_media_peer(
            "s1",
            produce=["type:broadcast"],
            listen=["type:feed"],
            configure=lambda p: configured.append(p),
        )
        assert isinstance(session, SignalingSession)
        assert configured == [peer]
        assert "produce=type:broadcast" in ws.url
        assert "listen=type:feed" in ws.url
        assert ws.url.startswith("wss://")

        offer = json.loads(ws.sent[0])
        assert offer["type"] == "offer"
        assert offer["sdp"] == peer.local_sdp

        ws.feed(json.dumps({"type": "answer", "sdp": "answer-sdp"}))
        for _ in range(50):
            if peer.remote:
                break
            await asyncio.sleep(0.01)
        assert peer.remote == [("answer", "answer-sdp")]

        await session.close()
        assert ws.closed
        assert peer.closed

    async def test_absent_selectors_omit_query(self) -> None:
        clock = FakeClock()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"access_token": _make_jwt(clock.now + 900), "refresh_token": "r1"},
            )

        ws = FakeWebSocket()
        peer = FakePeer()
        client = _make_client(handler, ws=ws, peer=peer, clock=clock)
        session = await client.open_media_peer("s1", configure=lambda p: None)
        assert "produce=" not in ws.url
        assert "listen=" not in ws.url
        await session.close()
