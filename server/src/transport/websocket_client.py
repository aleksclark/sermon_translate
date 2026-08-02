from __future__ import annotations

import logging
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from .crosstalk_client import CrosstalkError, WebSocketConnection

logger = logging.getLogger(__name__)

OPEN_TIMEOUT_SECONDS = 10.0
CLOSE_TIMEOUT_SECONDS = 5.0
PING_INTERVAL_SECONDS = 20.0
PING_TIMEOUT_SECONDS = 20.0
MAX_MESSAGE_BYTES = 8 * 1024 * 1024

WEBSOCKETS_MISSING_MESSAGE = (
    "the 'websockets' package is required for live Crosstalk media; "
    "install the server dependencies (it ships with uvicorn[standard], e.g. `uv sync`) "
    "or inject a ws_factory into CrosstalkService"
)


class _Connect(Protocol):
    def __call__(self, uri: str, **kwargs: Any) -> Any: ...


def redact_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _load_connect() -> _Connect:
    try:
        import websockets
    except ImportError as exc:
        raise CrosstalkError(WEBSOCKETS_MISSING_MESSAGE) from exc

    connect: Any = None
    try:
        from websockets.asyncio import client as asyncio_client

        connect = asyncio_client.connect
    except ImportError:
        connect = getattr(websockets, "connect", None)
    if connect is None:
        raise CrosstalkError(WEBSOCKETS_MISSING_MESSAGE)
    return connect


class WebSocketClientConnection:
    """Adapts a ``websockets`` asyncio client socket to `WebSocketConnection`."""

    def __init__(self, socket: Any) -> None:
        self._socket = socket
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    async def send(self, data: str) -> None:
        await self._socket.send(data)

    async def recv(self) -> str | bytes:
        return await self._socket.recv()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._socket.close()
        except Exception:
            logger.debug("crosstalk websocket close failed", exc_info=True)


async def connect_websocket(url: str) -> WebSocketConnection:
    """Open a Crosstalk signaling websocket using the `websockets` client."""
    connect = _load_connect()
    try:
        socket = await connect(
            url,
            open_timeout=OPEN_TIMEOUT_SECONDS,
            close_timeout=CLOSE_TIMEOUT_SECONDS,
            ping_interval=PING_INTERVAL_SECONDS,
            ping_timeout=PING_TIMEOUT_SECONDS,
            max_size=MAX_MESSAGE_BYTES,
        )
    except CrosstalkError:
        raise
    except Exception as exc:
        raise CrosstalkError(
            f"crosstalk websocket connect failed: {redact_url(url)}: {exc}"
        ) from exc
    return WebSocketClientConnection(socket)
