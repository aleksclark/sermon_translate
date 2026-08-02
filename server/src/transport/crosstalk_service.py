from __future__ import annotations

import asyncio
import contextlib
import logging

import httpx
from aiortc import RTCPeerConnection

from src.config import Settings, get_settings

from .crosstalk import CrosstalkTransport
from .crosstalk_client import CrosstalkClient, WebSocketConnection
from .handler import run_session
from .ice import build_rtc_configuration

logger = logging.getLogger(__name__)


async def _default_ws_factory(url: str) -> WebSocketConnection:
    raise RuntimeError(
        "no websocket client configured; inject a ws_factory to use live Crosstalk media"
    )


class CrosstalkService:
    """Owns Crosstalk discovery and start/stop of backed translation sessions.

    Keeps HTTP client, peer, and websocket construction out of route handlers.
    Injectable factories keep the service unit-testable.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: CrosstalkClient | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._http = http_client
        self._client = client
        self._runners: dict[str, asyncio.Task[None]] = {}

    def configured(self) -> bool:
        return bool(
            self._client is not None
            or (
                self._settings.crosstalk_base_url
                and self._settings.crosstalk_username
                and self._settings.crosstalk_password
            )
        )

    def _peer_factory(self) -> RTCPeerConnection:
        return RTCPeerConnection(configuration=build_rtc_configuration(self._settings))

    def client(self) -> CrosstalkClient:
        if self._client is not None:
            return self._client
        if not self.configured():
            raise RuntimeError("Crosstalk is not configured")
        if self._http is None:
            self._http = httpx.AsyncClient()
        self._client = CrosstalkClient(
            self._settings.crosstalk_base_url,
            self._settings.crosstalk_username,
            self._settings.crosstalk_password,
            http_client=self._http,
            ws_factory=_default_ws_factory,
            peer_factory=self._peer_factory,
            allow_private_hosts=self._settings.crosstalk_allow_private_hosts,
            request_timeout=self._settings.crosstalk_request_timeout,
        )
        return self._client

    async def start_translation(
        self,
        session_id: str,
        crosstalk_session_id: str,
        *,
        sample_rate: int,
        produce: list[str] | None = None,
        listen: list[str] | None = None,
    ) -> None:
        if session_id in self._runners and not self._runners[session_id].done():
            raise RuntimeError("session already running")
        transport = CrosstalkTransport(
            self.client(),
            crosstalk_session_id,
            produce=produce,
            listen=listen,
            sample_rate=sample_rate,
        )
        await transport.connect()
        self._runners[session_id] = asyncio.create_task(run_session(transport, session_id))

    async def stop_translation(self, session_id: str) -> bool:
        task = self._runners.pop(session_id, None)
        if task is None:
            return False
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        return True
