from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

import httpx
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp

logger = logging.getLogger(__name__)

ACCESS_TOKEN_SKEW_SECONDS = 30.0
DEFAULT_ACCESS_TOKEN_LIFETIME = 900.0


class CrosstalkError(Exception):
    """Base error for Crosstalk client failures."""


class CrosstalkAuthError(CrosstalkError):
    """Authentication with Crosstalk failed."""


class CrosstalkSSRFError(CrosstalkError):
    """A destination was blocked by the SSRF guard."""


@runtime_checkable
class WebSocketConnection(Protocol):
    async def send(self, data: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


WebSocketFactory = Callable[[str], Awaitable[WebSocketConnection]]
PeerFactory = Callable[[], RTCPeerConnection]


@dataclass(frozen=True)
class CrosstalkSession:
    id: str
    name: str
    description: str = ""


@dataclass(frozen=True)
class CrosstalkChannel:
    id: str
    name: str
    type: str
    session_id: str


@dataclass(frozen=True)
class CrosstalkSource:
    id: str
    name: str
    session_id: str
    connected: bool


def _is_blocked_address(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return False
        return ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved:
            return True
    return False


def guard_destination(url: str, *, allow_private: bool, ws: bool = False) -> None:
    parts = urlsplit(url)
    allowed = {"ws", "wss"} if ws else {"http", "https"}
    if parts.scheme not in allowed:
        raise CrosstalkSSRFError(f"disallowed url scheme: {parts.scheme!r}")
    host = parts.hostname
    if not host:
        raise CrosstalkSSRFError("url is missing a host")
    if allow_private:
        return
    if _is_blocked_address(host):
        raise CrosstalkSSRFError(f"destination host is private or loopback: {host}")


def _decode_jwt_exp(token: str) -> float | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        import base64

        raw = base64.urlsafe_b64decode(payload + padding)
        claims = json.loads(raw)
    except (ValueError, json.JSONDecodeError):
        return None
    exp = claims.get("exp")
    if isinstance(exp, (int, float)):
        return float(exp)
    return None


class CrosstalkClient:
    """Typed async client for the Crosstalk REST + signaling protocol.

    The access token is retained server-side only and is never returned to
    callers. httpx client and websocket/peer factories are injected so the
    signaling flow can be exercised without live network access.
    """

    def __init__(
        self,
        base_url: str,
        username: str,
        password: str,
        *,
        http_client: httpx.AsyncClient,
        ws_factory: WebSocketFactory,
        peer_factory: PeerFactory,
        allow_private_hosts: bool = False,
        request_timeout: float = 10.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        guard_destination(base_url, allow_private=allow_private_hosts)
        self._base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self._http = http_client
        self._ws_factory = ws_factory
        self._peer_factory = peer_factory
        self._allow_private = allow_private_hosts
        self._timeout = request_timeout
        self._clock = clock or (lambda: asyncio.get_event_loop().time())
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._access_expiry: float = 0.0
        self._auth_lock = asyncio.Lock()

    def _url(self, path: str) -> str:
        return f"{self._base_url}/{path.lstrip('/')}"

    async def _login(self) -> None:
        resp = await self._http.post(
            self._url("/api/auth/login"),
            json={"username": self._username, "password": self._password},
            timeout=self._timeout,
        )
        if resp.status_code != 200:
            raise CrosstalkAuthError(f"login failed: {resp.status_code}")
        self._store_tokens(resp.json())

    async def _refresh(self) -> None:
        if not self._refresh_token:
            await self._login()
            return
        resp = await self._http.post(
            self._url("/api/auth/refresh"),
            json={"refresh_token": self._refresh_token},
            timeout=self._timeout,
        )
        if resp.status_code != 200:
            await self._login()
            return
        self._store_tokens(resp.json())

    def _store_tokens(self, body: dict[str, Any]) -> None:
        access = body.get("access_token")
        refresh = body.get("refresh_token")
        if not access or not refresh:
            raise CrosstalkAuthError("auth response missing tokens")
        self._access_token = access
        self._refresh_token = refresh
        exp = _decode_jwt_exp(access)
        now = self._clock()
        self._access_expiry = exp if exp is not None else now + DEFAULT_ACCESS_TOKEN_LIFETIME

    def _token_expired(self) -> bool:
        if self._access_token is None:
            return True
        return self._clock() >= self._access_expiry - ACCESS_TOKEN_SKEW_SECONDS

    async def _ensure_token(self) -> str:
        async with self._auth_lock:
            if self._access_token is None:
                await self._login()
            elif self._token_expired():
                await self._refresh()
            assert self._access_token is not None
            return self._access_token

    async def _authed_get(self, path: str) -> Any:
        token = await self._ensure_token()
        url = self._url(path)
        resp = await self._http.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=self._timeout,
        )
        if resp.status_code == 401:
            async with self._auth_lock:
                await self._refresh()
                token = self._access_token
            resp = await self._http.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                timeout=self._timeout,
            )
        if resp.status_code != 200:
            raise CrosstalkError(f"GET {path} failed: {resp.status_code}")
        return resp.json()

    async def list_sessions(self) -> list[CrosstalkSession]:
        body = await self._authed_get("/api/sessions")
        return [
            CrosstalkSession(
                id=item["id"],
                name=item.get("name", ""),
                description=item.get("description", ""),
            )
            for item in body.get("data", [])
        ]

    async def get_session(self, session_id: str) -> CrosstalkSession:
        item = await self._authed_get(f"/api/sessions/{session_id}")
        return CrosstalkSession(
            id=item["id"],
            name=item.get("name", ""),
            description=item.get("description", ""),
        )

    async def list_channels(self, session_id: str) -> list[CrosstalkChannel]:
        body = await self._authed_get(f"/api/sessions/{session_id}/channels")
        return [
            CrosstalkChannel(
                id=item["id"],
                name=item.get("name", ""),
                type=item.get("type", ""),
                session_id=item.get("session_id", session_id),
            )
            for item in body.get("data", [])
        ]

    async def list_sources(self, session_id: str) -> list[CrosstalkSource]:
        body = await self._authed_get(f"/api/sessions/{session_id}/sources")
        return [
            CrosstalkSource(
                id=item["id"],
                name=item.get("name", ""),
                session_id=item.get("session_id", session_id),
                connected=bool(item.get("connected", False)),
            )
            for item in body.get("data", [])
        ]

    def _media_ws_url(
        self,
        session_id: str,
        token: str,
        produce: list[str] | None,
        listen: list[str] | None,
    ) -> str:
        parts = urlsplit(self._base_url)
        scheme = "wss" if parts.scheme == "https" else "ws"
        query = [f"token={token}"]
        if produce is not None:
            query.append(f"produce={','.join(produce)}")
        if listen is not None:
            query.append(f"listen={','.join(listen)}")
        path = f"{parts.path.rstrip('/')}/api/sessions/{session_id}/ws"
        return urlunsplit((scheme, parts.netloc, path, "&".join(query), ""))

    async def open_media_peer(
        self,
        session_id: str,
        *,
        produce: list[str] | None = None,
        listen: list[str] | None = None,
        configure: Callable[[RTCPeerConnection], None] | None = None,
        on_ready: Callable[[], None] | None = None,
    ) -> SignalingSession:
        """Open the media WebSocket and drive the client-offer handshake.

        ``configure`` runs against the fresh peer (add tracks, register the
        ``track`` handler) before the offer is created so added tracks appear
        in the SDP. Returns a live SignalingSession that must be closed when the
        transport tears down.
        """
        token = await self._ensure_token()
        ws_url = self._media_ws_url(session_id, token, produce, listen)
        guard_destination(ws_url, allow_private=self._allow_private, ws=True)
        pc = self._peer_factory()
        if configure is not None:
            configure(pc)
        ws = await self._ws_factory(ws_url)
        session = SignalingSession(pc, ws)
        await session.start(on_ready=on_ready)
        return session


class SignalingSession:
    """Runs the client-offer signaling loop against an aiortc peer.

    The client is the offerer: it sends its SDP offer, applies the server's
    answer, and exchanges trickled ICE candidates as ``candidate`` messages.
    """

    def __init__(self, pc: RTCPeerConnection, ws: WebSocketConnection) -> None:
        self._pc = pc
        self._ws = ws
        self._reader: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def pc(self) -> RTCPeerConnection:
        return self._pc

    async def start(self, *, on_ready: Callable[[], None] | None = None) -> None:
        if on_ready is not None:

            @self._pc.on("connectionstatechange")
            def _on_state() -> None:  # pragma: no cover - event glue
                if self._pc.connectionState == "connected":
                    on_ready()

        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        await self._send({"type": "offer", "sdp": self._pc.localDescription.sdp})
        self._reader = asyncio.create_task(self._read_loop())

    async def _send(self, message: dict[str, Any]) -> None:
        if self._closed:
            return
        await self._ws.send(json.dumps(message))

    async def _read_loop(self) -> None:
        try:
            while not self._closed:
                raw = await self._ws.recv()
                text = raw if isinstance(raw, str) else raw.decode()
                await self._handle(json.loads(text))
        except (asyncio.CancelledError, ConnectionError):
            raise
        except Exception:
            logger.exception("crosstalk signaling read loop error")

    async def _handle(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "answer":
            await self._pc.setRemoteDescription(
                RTCSessionDescription(sdp=msg["sdp"], type="answer")
            )
        elif kind == "offer":
            await self._pc.setRemoteDescription(
                RTCSessionDescription(sdp=msg["sdp"], type="offer")
            )
            answer = await self._pc.createAnswer()
            await self._pc.setLocalDescription(answer)
            await self._send({"type": "answer", "sdp": self._pc.localDescription.sdp})
        elif kind == "candidate":
            init = msg.get("candidate")
            if not init or not init.get("candidate"):
                return
            sdp = init["candidate"]
            if sdp.startswith("candidate:"):
                sdp = sdp[len("candidate:") :]
            candidate = candidate_from_sdp(sdp)
            candidate.sdpMid = init.get("sdpMid")
            candidate.sdpMLineIndex = init.get("sdpMLineIndex")
            await self._pc.addIceCandidate(candidate)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
        try:
            await self._ws.close()
        except Exception:
            logger.debug("error closing crosstalk websocket", exc_info=True)
        await self._pc.close()
