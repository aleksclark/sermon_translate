from __future__ import annotations

import sys
import types
from typing import Any

import httpx
import pytest

from src.config import Settings
from src.transport.crosstalk_client import CrosstalkError, WebSocketConnection
from src.transport.crosstalk_service import CrosstalkService
from src.transport.websocket_client import (
    MAX_MESSAGE_BYTES,
    WebSocketClientConnection,
    connect_websocket,
    redact_url,
)

WS_URL = "wss://crosstalk.example.com/api/sessions/s1/ws?token=secret-token"


class FakeSocket:
    def __init__(
        self,
        messages: list[str | bytes] | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.sent: list[str] = []
        self.close_calls = 0
        self._messages: list[str | bytes] = list(messages or [])
        self._close_error = close_error

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def recv(self) -> str | bytes:
        return self._messages.pop(0)

    async def close(self) -> None:
        self.close_calls += 1
        if self._close_error is not None:
            raise self._close_error


class RecordingConnect:
    def __init__(self, socket: Any = None, error: Exception | None = None) -> None:
        self.socket = socket
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, uri: str, **kwargs: Any) -> Any:
        self.calls.append((uri, kwargs))
        if self.error is not None:
            raise self.error
        return self.socket


def _install_modern(monkeypatch: pytest.MonkeyPatch, connect: Any) -> None:
    websockets = types.ModuleType("websockets")
    asyncio_mod = types.ModuleType("websockets.asyncio")
    client_mod = types.ModuleType("websockets.asyncio.client")
    client_mod.connect = connect
    asyncio_mod.client = client_mod
    websockets.asyncio = asyncio_mod
    monkeypatch.setitem(sys.modules, "websockets", websockets)
    monkeypatch.setitem(sys.modules, "websockets.asyncio", asyncio_mod)
    monkeypatch.setitem(sys.modules, "websockets.asyncio.client", client_mod)


def _install_legacy(monkeypatch: pytest.MonkeyPatch, connect: Any | None) -> None:
    websockets = types.ModuleType("websockets")
    if connect is not None:
        websockets.connect = connect
    monkeypatch.setitem(sys.modules, "websockets", websockets)
    monkeypatch.setitem(sys.modules, "websockets.asyncio", None)
    monkeypatch.setitem(sys.modules, "websockets.asyncio.client", None)


def _install_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "websockets", None)
    monkeypatch.setitem(sys.modules, "websockets.asyncio", None)
    monkeypatch.setitem(sys.modules, "websockets.asyncio.client", None)


class TestAdapter:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(WebSocketClientConnection(FakeSocket()), WebSocketConnection)

    async def test_send_and_recv_pass_through(self) -> None:
        socket = FakeSocket(messages=["hello", b"raw-bytes"])
        conn = WebSocketClientConnection(socket)

        await conn.send('{"type": "offer"}')
        assert socket.sent == ['{"type": "offer"}']
        assert await conn.recv() == "hello"
        assert await conn.recv() == b"raw-bytes"

    async def test_close_is_idempotent(self) -> None:
        socket = FakeSocket()
        conn = WebSocketClientConnection(socket)

        await conn.close()
        await conn.close()

        assert socket.close_calls == 1
        assert conn.closed is True

    async def test_close_on_already_closed_socket_does_not_raise(self) -> None:
        socket = FakeSocket(close_error=ConnectionResetError("already closed"))
        conn = WebSocketClientConnection(socket)

        await conn.close()

        assert conn.closed is True


class TestConnectWebsocket:
    async def test_uses_modern_asyncio_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socket = FakeSocket()
        connect = RecordingConnect(socket=socket)
        _install_modern(monkeypatch, connect)

        conn = await connect_websocket(WS_URL)

        assert isinstance(conn, WebSocketClientConnection)
        url, kwargs = connect.calls[0]
        assert url == WS_URL
        assert kwargs["max_size"] == MAX_MESSAGE_BYTES
        assert kwargs["open_timeout"] > 0
        assert kwargs["ping_interval"] > 0
        assert kwargs["ping_timeout"] > 0

        await conn.send("ping")
        assert socket.sent == ["ping"]

    async def test_falls_back_to_top_level_connect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        socket = FakeSocket()
        connect = RecordingConnect(socket=socket)
        _install_legacy(monkeypatch, connect)

        conn = await connect_websocket(WS_URL)

        assert isinstance(conn, WebSocketClientConnection)
        assert connect.calls[0][0] == WS_URL

    async def test_connect_failure_maps_to_crosstalk_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        connect = RecordingConnect(error=OSError("connection refused"))
        _install_modern(monkeypatch, connect)

        with pytest.raises(CrosstalkError) as exc:
            await connect_websocket(WS_URL)

        assert "connection refused" in str(exc.value)
        assert "secret-token" not in str(exc.value)

    async def test_missing_package_reports_remedy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_missing(monkeypatch)

        with pytest.raises(CrosstalkError) as exc:
            await connect_websocket(WS_URL)

        message = str(exc.value)
        assert "websockets" in message
        assert "uv sync" in message
        assert "ws_factory" in message

    async def test_package_without_connect_reports_remedy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_legacy(monkeypatch, None)

        with pytest.raises(CrosstalkError):
            await connect_websocket(WS_URL)

    def test_redact_url_strips_query(self) -> None:
        assert redact_url(WS_URL) == "wss://crosstalk.example.com/api/sessions/s1/ws"


def _settings() -> Settings:
    return Settings(
        crosstalk_base_url="http://127.0.0.1:9000",
        crosstalk_username="user",
        crosstalk_password="pass",
        crosstalk_allow_private_hosts=True,
    )


class TestServiceWiring:
    async def test_default_factory_is_the_real_client(self) -> None:
        async with httpx.AsyncClient() as http:
            service = CrosstalkService(_settings(), http_client=http)
            client = service.client()

        assert service.configured() is True
        assert client._ws_factory is connect_websocket

    async def test_injected_factory_is_honored(self) -> None:
        async def fake_factory(url: str) -> WebSocketConnection:
            return WebSocketClientConnection(FakeSocket())

        async with httpx.AsyncClient() as http:
            service = CrosstalkService(_settings(), http_client=http, ws_factory=fake_factory)
            client = service.client()

        assert client._ws_factory is fake_factory

    def test_unconfigured_service_still_raises(self) -> None:
        service = CrosstalkService(Settings())

        assert service.configured() is False
        with pytest.raises(RuntimeError, match="not configured"):
            service.client()
