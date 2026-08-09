#!/usr/bin/env python3
"""Remote stage.v1 product conformance harness.

Checks:
  - HTTP /health/live and /health/ready
  - WebSocket hello → accepted with subprotocol stage.v1 + auth header
  - Optional production expectation: unauthenticated upgrade is rejected

Exit non-zero on failure. Writes a JSON report to --output when set.
Never prints the auth token.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

# Allow `uv run python scripts/...` from server/
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from src.stage_v1.auth import STAGE_V1_SUBPROTOCOL  # noqa: E402
from src.stage_v1.models import (  # noqa: E402
    BASELINE_SAMPLE_RATE_HZ,
    SCHEMA_VERSION,
    EventType,
    StageErrorCode,
    StageKind,
    parse_event,
    parse_event_json,
)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""
    status_code: int | None = None


@dataclass
class Report:
    ok: bool
    base_url: str
    stage_kind: str
    stage_id: str
    checks: list[CheckResult] = field(default_factory=list)
    error: str | None = None
    finished_at: str = ""

    def add(self, name: str, ok: bool, detail: str = "", status_code: int | None = None) -> None:
        self.checks.append(
            CheckResult(name=name, ok=ok, detail=detail, status_code=status_code)
        )
        if not ok:
            self.ok = False


def _http_base(ws_base: str) -> str:
    parsed = urlparse(ws_base)
    scheme = parsed.scheme.lower()
    if scheme in {"ws", "http"}:
        http_scheme = "http"
    elif scheme in {"wss", "https"}:
        http_scheme = "https"
    else:
        raise ValueError(f"unsupported scheme in base-url: {scheme!r}")
    # Strip path; health lives at root.
    return urlunparse((http_scheme, parsed.netloc, "", "", "", ""))


def _ws_stream_url(ws_base: str) -> str:
    parsed = urlparse(ws_base)
    scheme = parsed.scheme.lower()
    if scheme in {"http"}:
        scheme = "ws"
    elif scheme in {"https"}:
        scheme = "wss"
    path = parsed.path.rstrip("/")
    if path.endswith("/stage/v1/stream"):
        stream_path = path
    elif path:
        stream_path = f"{path}/stage/v1/stream"
    else:
        stream_path = "/stage/v1/stream"
    return urlunparse((scheme, parsed.netloc, stream_path, "", "", ""))


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _deadline() -> str:
    return (datetime.now(UTC) + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S.%f")[
        :-3
    ] + "Z"


def _hello_json(*, stage_kind: str, stage_id: str, session_id: str) -> str:
    kind = StageKind(stage_kind)
    data = {
        "schema_version": SCHEMA_VERSION,
        "event_type": EventType.HELLO.value,
        "message_id": str(uuid4()),
        "event_sequence": 0,
        "created_at": _now(),
        "correlation_id": f"corr-conformance-{uuid4()}",
        "session_id": session_id,
        "owner_generation": 1,
        "stage_kind": kind.value,
        "stage_id": stage_id,
        "attempt_id": str(uuid4()),
        "cancel_id": str(uuid4()),
        "deadline_at": _deadline(),
        "payload": {
            "audio_formats": [
                {
                    "codec": "pcm_s16le",
                    "sample_rate_hz": BASELINE_SAMPLE_RATE_HZ,
                    "channels": 1,
                }
            ],
            "limits_requested": {
                "max_frame_bytes": 65536,
                "max_inflight_events": 32,
            },
        },
    }
    return parse_event(data).model_dump_json(exclude_none=True)


def _auth_headers(token: str | None, *, trust_proxy_header: bool) -> dict[str, str]:
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if trust_proxy_header:
        headers["X-Forwarded-Proto"] = "https"
    return headers


async def _http_get_json(url: str) -> tuple[int, Any]:
    import httpx

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url)
        try:
            body: Any = resp.json()
        except Exception:
            body = {"raw": resp.text[:500]}
        return resp.status_code, body


async def _ws_hello_accepted(
    stream_url: str,
    *,
    token: str | None,
    stage_kind: str,
    stage_id: str,
    trust_proxy_header: bool,
    expect_reject: bool = False,
) -> tuple[bool, str]:
    import websockets
    from websockets.exceptions import ConnectionClosed

    headers = _auth_headers(token, trust_proxy_header=trust_proxy_header)
    try:
        async with websockets.connect(
            stream_url,
            subprotocols=[STAGE_V1_SUBPROTOCOL],  # type: ignore[list-item]
            additional_headers=headers,
            open_timeout=10,
            close_timeout=5,
            max_size=2**20,
        ) as ws:
            if expect_reject:
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                if not isinstance(raw, str):
                    return False, "expected text error frame on auth reject"
                env = parse_event_json(raw)
                if env.event_type != EventType.ERROR:
                    return False, f"expected ERROR, got {env.event_type}"
                code = env.payload.get("code") if isinstance(env.payload, dict) else None
                if code != StageErrorCode.AUTHENTICATION_FAILED.value:
                    return False, f"expected AUTHENTICATION_FAILED, got {code!r}"
                return True, "auth reject received"

            await ws.send(
                _hello_json(
                    stage_kind=stage_kind,
                    stage_id=stage_id,
                    session_id=f"conformance-{uuid4()}",
                )
            )
            raw = await asyncio.wait_for(ws.recv(), timeout=15)
            if not isinstance(raw, str):
                return False, "expected text accepted frame"
            env = parse_event_json(raw)
            if env.event_type == EventType.ERROR:
                code = env.payload.get("code") if isinstance(env.payload, dict) else None
                msg = env.payload.get("message") if isinstance(env.payload, dict) else ""
                return False, f"server error code={code} message={msg}"
            if env.event_type != EventType.ACCEPTED:
                return False, f"expected ACCEPTED, got {env.event_type}"
            if env.stage_id != stage_id:
                return False, f"stage_id mismatch: {env.stage_id!r} != {stage_id!r}"
            # Drain optional window (best-effort).
            try:
                maybe = await asyncio.wait_for(ws.recv(), timeout=1.0)
                if isinstance(maybe, str):
                    w = parse_event_json(maybe)
                    if w.event_type not in {EventType.WINDOW, EventType.OPENED}:
                        return False, f"unexpected post-accept event {w.event_type}"
            except TimeoutError:
                pass
            boot = None
            if isinstance(env.payload, dict):
                boot = env.payload.get("boot_id")
            return True, f"accepted boot_id={boot}"
    except ConnectionClosed as exc:
        if expect_reject:
            return True, f"connection closed on reject ({exc.code})"
        return False, f"connection closed: code={exc.code} reason={exc.reason!r}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


async def run_conformance(args: argparse.Namespace) -> Report:
    report = Report(
        ok=True,
        base_url=args.base_url,
        stage_kind=args.stage_kind,
        stage_id=args.stage_id,
    )
    try:
        http_base = _http_base(args.base_url)
        stream_url = _ws_stream_url(args.base_url)
    except ValueError as exc:
        report.ok = False
        report.error = str(exc)
        return report

    # Health live
    try:
        code, body = await _http_get_json(f"{http_base}/health/live")
        ok = code == 200 and (
            isinstance(body, dict) and body.get("status") in {"live", "ok", None} or code == 200
        )
        if code == 200:
            ok = True
        report.add("health.live", ok, detail=str(body)[:300], status_code=code)
    except Exception as exc:
        report.add("health.live", False, detail=f"{type(exc).__name__}: {exc}")

    # Health ready
    try:
        code, body = await _http_get_json(f"{http_base}/health/ready")
        ok = code == 200
        detail = str(body)[:300]
        if isinstance(body, dict):
            detail = json.dumps(
                {
                    k: body.get(k)
                    for k in (
                        "status",
                        "stage_id",
                        "boot_id",
                        "model_loaded",
                        "model_warm",
                        "loader_invocation_count",
                    )
                    if k in body
                }
            )
        report.add("health.ready", ok, detail=detail, status_code=code)
    except Exception as exc:
        report.add("health.ready", False, detail=f"{type(exc).__name__}: {exc}")

    # Authenticated hello → accepted
    ok, detail = await _ws_hello_accepted(
        stream_url,
        token=args.token,
        stage_kind=args.stage_kind,
        stage_id=args.stage_id,
        trust_proxy_header=args.trust_proxy_header,
        expect_reject=False,
    )
    report.add("ws.hello_accepted", ok, detail=detail)

    # Production expectation: reject without auth
    if args.expect_auth_reject:
        ok, detail = await _ws_hello_accepted(
            stream_url,
            token=None,
            stage_kind=args.stage_kind,
            stage_id=args.stage_id,
            trust_proxy_header=args.trust_proxy_header,
            expect_reject=True,
        )
        report.add("ws.unauthenticated_reject", ok, detail=detail)

    report.finished_at = _now()
    return report


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="stage.v1 remote product conformance")
    p.add_argument(
        "--base-url",
        required=True,
        help="ws(s)://host:port or http(s)://host:port (path optional)",
    )
    p.add_argument(
        "--token",
        default=None,
        help="STAGE_AUTH_TOKEN value (never logged)",
    )
    p.add_argument("--stage-kind", default="listen", choices=[k.value for k in StageKind])
    p.add_argument("--stage-id", default="whisper-listen")
    p.add_argument(
        "--output",
        default=None,
        help="Write JSON report to this path",
    )
    p.add_argument(
        "--expect-auth-reject",
        action="store_true",
        help="Also assert unauthenticated upgrade is rejected (production expectation)",
    )
    p.add_argument(
        "--trust-proxy-header",
        action="store_true",
        help="Send X-Forwarded-Proto: https (local production mode behind plain TCP)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = asyncio.run(run_conformance(args))
    except Exception as exc:
        # Never include token in crash output.
        err = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        text = json.dumps(err, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(text)
        print(text, file=sys.stderr)
        return 2

    payload = {
        "ok": report.ok,
        "base_url": report.base_url,
        "stage_kind": report.stage_kind,
        "stage_id": report.stage_id,
        "finished_at": report.finished_at,
        "error": report.error,
        "checks": [asdict(c) for c in report.checks],
        # Explicitly omit token.
    }
    text = json.dumps(payload, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")

    # Human summary without secrets
    status = "PASS" if report.ok else "FAIL"
    print(f"stage.v1 remote conformance: {status}")
    for c in report.checks:
        mark = "ok" if c.ok else "FAIL"
        sc = f" status={c.status_code}" if c.status_code is not None else ""
        print(f"  [{mark}] {c.name}{sc}: {c.detail}")
    if report.error:
        print(f"  error: {report.error}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
