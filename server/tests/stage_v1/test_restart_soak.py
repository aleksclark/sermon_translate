"""Restart / reconnect soak + production auth fail-closed (stage.v1).

In-process ASGI coverage with fake loaders (no GPU / model download):
- warm host survives session close; sequential sessions keep loader_count==1
- client reconnect uses a fresh attempt_id; stale fence (old cancel/attempt) rejects
- production empty-token boot refuse; wrong token upgrade reject before open_session
- short soak: N sequential sessions on a warm host without reload
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.models import ListenProduct, WordSpan
from src.pipelines.stages_listen.whisper import WhisperListenStage, WhisperLoadedModel
from src.runtime.worker import create_worker_app
from src.stage_v1.adapters import build_whisper_listen_host
from src.stage_v1.auth import (
    STAGE_V1_SUBPROTOCOL,
    StageV1AuthConfig,
    StageV1Mode,
)
from src.stage_v1.models import (
    BASELINE_SAMPLE_RATE_HZ,
    SCHEMA_VERSION,
    AcceptedPayload,
    EventType,
    StageErrorCode,
    StageKind,
    parse_event,
    parse_event_json,
)
from src.stage_v1.validation import Fence, ValidationError, check_fence


def _deadline(hours: float = 1.0) -> str:
    return (datetime.now(UTC) + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.%f")[
        :-3
    ] + "Z"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _hello_json(
    *,
    stage_id: str = "whisper-listen",
    session_id: str = "sess-soak-1",
    attempt_id: str | None = None,
    cancel_id: str | None = None,
    owner_generation: int = 1,
) -> str:
    attempt = attempt_id or str(uuid4())
    cancel = cancel_id or str(uuid4())
    data = {
        "schema_version": SCHEMA_VERSION,
        "event_type": EventType.HELLO.value,
        "message_id": str(uuid4()),
        "event_sequence": 0,
        "created_at": _now(),
        "correlation_id": f"corr-{uuid4()}",
        "session_id": session_id,
        "owner_generation": owner_generation,
        "stage_kind": StageKind.LISTEN.value,
        "stage_id": stage_id,
        "attempt_id": attempt,
        "cancel_id": cancel,
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


def _cancel_json(
    *,
    accepted: Any,
    attempt_id: str | None = None,
    cancel_id: str | None = None,
    event_sequence: int = 1,
) -> str:
    data = {
        "schema_version": SCHEMA_VERSION,
        "event_type": EventType.CANCEL.value,
        "message_id": str(uuid4()),
        "event_sequence": event_sequence,
        "created_at": _now(),
        "correlation_id": accepted.correlation_id,
        "session_id": accepted.session_id,
        "owner_generation": accepted.owner_generation,
        "stage_kind": StageKind.LISTEN.value,
        "stage_id": accepted.stage_id,
        "attempt_id": attempt_id if attempt_id is not None else accepted.attempt_id,
        "cancel_id": cancel_id if cancel_id is not None else accepted.cancel_id,
        "stage_instance_id": accepted.stage_instance_id,
        "deadline_at": _deadline(),
        "payload": {
            "scope": "attempt",
            "reason": "stale_probe",
            "attempt_id": attempt_id if attempt_id is not None else accepted.attempt_id,
            "session_id": accepted.session_id,
        },
    }
    return parse_event(data).model_dump_json(exclude_none=True)


def _fake_whisper_loaded() -> WhisperLoadedModel:
    return WhisperLoadedModel(model=object(), model_size="tiny", revision="tiny")


async def _scripted_transcribe(
    self: WhisperListenStage, audio_stream: AsyncIterator[bytes]
) -> AsyncIterator[ListenProduct]:
    saw_audio = False
    async for chunk in audio_stream:
        if chunk and not saw_audio:
            saw_audio = True
            yield ListenProduct(
                sequence=0,
                utterance_id="utt-1",
                text="hello world",
                is_final=False,
                words=[
                    WordSpan(text="hello", start_ms=0.0, end_ms=200.0, conf=0.9),
                    WordSpan(text="world", start_ms=200.0, end_ms=400.0, conf=0.9),
                ],
                language="en",
            )
    yield ListenProduct(
        sequence=1,
        utterance_id="utt-1",
        text="hello world",
        is_final=True,
        words=[
            WordSpan(text="hello", start_ms=0.0, end_ms=200.0, conf=0.9),
            WordSpan(text="world", start_ms=200.0, end_ms=400.0, conf=0.9),
        ],
        language="en",
    )


def _build_warm_listen_app(
    *,
    auth: StageV1AuthConfig | None = None,
    boot_id: str = "boot-soak-test",
    loads: dict[str, int] | None = None,
    max_sessions: int = 2,
) -> tuple[Any, dict[str, int]]:
    counter = loads if loads is not None else {"n": 0}

    def loader() -> WhisperLoadedModel:
        counter["n"] += 1
        return _fake_whisper_loaded()

    host = build_whisper_listen_host(
        model_loader=loader,
        max_sessions=max_sessions,
        boot_id=boot_id,
        model_size="tiny",
        local_dev=True,
    )
    cfg = auth or StageV1AuthConfig(
        mode=StageV1Mode.DEV,
        auth_token="",
        trust_proxy=False,
        allow_loopback_without_auth=True,
    )
    app = create_worker_app(
        "whisper-listen",
        max_sessions=max_sessions,
        host=host,
        auth=cfg,
        warm=True,
    )
    return app, counter


def _drain_until_accepted(ws: Any) -> Any:
    raw = ws.receive_text()
    env = parse_event_json(raw)
    assert env.event_type == EventType.ACCEPTED, env
    # optional window / opened
    return env


def test_production_empty_token_boot_refuse() -> None:
    auth = StageV1AuthConfig(
        mode=StageV1Mode.PRODUCTION,
        auth_token="",
        trust_proxy=False,
        allow_loopback_without_auth=False,
    )
    with pytest.raises(RuntimeError, match="STAGE_AUTH_TOKEN"):
        auth.validate_boot()
    with pytest.raises(RuntimeError, match="STAGE_AUTH_TOKEN"):
        create_worker_app("passthrough-listen", auth=auth, warm=False)


def test_production_wrong_token_rejects_before_open_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        WhisperListenStage, "transcribe", _scripted_transcribe, raising=True
    )
    loads = {"n": 0}
    auth = StageV1AuthConfig(
        mode=StageV1Mode.PRODUCTION,
        auth_token="prod-secret",
        trust_proxy=True,
        allow_loopback_without_auth=False,
    )
    app, _ = _build_warm_listen_app(auth=auth, loads=loads, boot_id="boot-auth")

    with TestClient(app) as client:
        with client.websocket_connect(
            "/stage/v1/stream",
            subprotocols=[STAGE_V1_SUBPROTOCOL],
            headers={
                "Authorization": "Bearer wrong-token",
                "X-Forwarded-Proto": "https",
            },
        ) as ws:
            err = parse_event_json(ws.receive_text())
            assert err.event_type == EventType.ERROR
            assert err.payload["code"] == StageErrorCode.AUTHENTICATION_FAILED.value
            # Must never reach accepted / open_session fence.
            assert err.session_id == "unauthorized"
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()

        # Correct token still works after reject path.
        with client.websocket_connect(
            "/stage/v1/stream",
            subprotocols=[STAGE_V1_SUBPROTOCOL],
            headers={
                "Authorization": "Bearer prod-secret",
                "X-Forwarded-Proto": "https",
            },
        ) as ws:
            ws.send_text(_hello_json(session_id="auth-ok"))
            accepted = parse_event_json(ws.receive_text())
            assert accepted.event_type == EventType.ACCEPTED

    assert loads["n"] == 1


def test_host_survives_session_close_second_session_loader_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        WhisperListenStage, "transcribe", _scripted_transcribe, raising=True
    )
    app, counter = _build_warm_listen_app(boot_id="boot-two")
    boot_ids: list[str] = []

    with TestClient(app) as client:
        for i in range(2):
            with client.websocket_connect(
                "/stage/v1/stream",
                subprotocols=[STAGE_V1_SUBPROTOCOL],
            ) as ws:
                attempt = str(uuid4())
                ws.send_text(
                    _hello_json(session_id=f"sess-close-{i}", attempt_id=attempt)
                )
                accepted = _drain_until_accepted(ws)
                payload = AcceptedPayload.model_validate(accepted.payload)
                boot_ids.append(payload.boot_id)
                _ = ws.receive_text()  # window

    assert counter["n"] == 1
    assert boot_ids == ["boot-two", "boot-two"]


def test_reconnect_new_attempt_stale_fence_rejects_old_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After disconnect, a new connection gets a new attempt; old cancel_id is stale."""
    monkeypatch.setattr(
        WhisperListenStage, "transcribe", _scripted_transcribe, raising=True
    )
    app, counter = _build_warm_listen_app(boot_id="boot-reconnect")

    old_attempt = str(uuid4())
    old_cancel = str(uuid4())
    session_id = "reconnect-sess"

    with TestClient(app) as client:
        # Session 1
        with client.websocket_connect(
            "/stage/v1/stream",
            subprotocols=[STAGE_V1_SUBPROTOCOL],
        ) as ws:
            ws.send_text(
                _hello_json(
                    session_id=session_id,
                    attempt_id=old_attempt,
                    cancel_id=old_cancel,
                )
            )
            accepted1 = _drain_until_accepted(ws)
            assert accepted1.attempt_id == old_attempt
            assert accepted1.cancel_id == old_cancel
            _ = ws.receive_text()
            # disconnect closes the context

        # Session 2 — fresh attempt/cancel (reconnect)
        new_attempt = str(uuid4())
        new_cancel = str(uuid4())
        assert new_attempt != old_attempt
        assert new_cancel != old_cancel

        with client.websocket_connect(
            "/stage/v1/stream",
            subprotocols=[STAGE_V1_SUBPROTOCOL],
        ) as ws:
            ws.send_text(
                _hello_json(
                    session_id=session_id,
                    attempt_id=new_attempt,
                    cancel_id=new_cancel,
                )
            )
            accepted2 = _drain_until_accepted(ws)
            assert accepted2.attempt_id == new_attempt
            assert accepted2.cancel_id == new_cancel
            assert accepted2.payload["boot_id"] == "boot-reconnect"
            _ = ws.receive_text()

            # Stale fence: cancel carrying old attempt/cancel must be rejected.
            ws.send_text(
                _cancel_json(
                    accepted=accepted2,
                    attempt_id=old_attempt,
                    cancel_id=old_cancel,
                    event_sequence=1,
                )
            )
            err = parse_event_json(ws.receive_text())
            assert err.event_type == EventType.ERROR
            assert err.payload["code"] == StageErrorCode.STALE_FENCE.value

            # Pure fence helper also rejects old IDs against active fence.
            active = Fence(
                session_id=accepted2.session_id,
                owner_generation=accepted2.owner_generation,
                stage_kind=StageKind.LISTEN.value,
                stage_id=accepted2.stage_id,
                attempt_id=accepted2.attempt_id,
                cancel_id=accepted2.cancel_id,
                stage_instance_id=accepted2.stage_instance_id,
            )
            stale_env = parse_event_json(
                _cancel_json(
                    accepted=accepted2,
                    attempt_id=old_attempt,
                    cancel_id=old_cancel,
                )
            )
            with pytest.raises(ValidationError) as ei:
                check_fence(active, stale_env, require_instance=False)
            assert ei.value.code == StageErrorCode.STALE_FENCE

    assert counter["n"] == 1


def test_short_soak_five_sequential_sessions_no_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        WhisperListenStage, "transcribe", _scripted_transcribe, raising=True
    )
    n_sessions = 5
    app, counter = _build_warm_listen_app(
        boot_id="boot-soak-5",
        max_sessions=1,
    )
    attempts: list[str] = []

    with TestClient(app) as client:
        for i in range(n_sessions):
            with client.websocket_connect(
                "/stage/v1/stream",
                subprotocols=[STAGE_V1_SUBPROTOCOL],
            ) as ws:
                attempt = str(uuid4())
                attempts.append(attempt)
                ws.send_text(
                    _hello_json(session_id=f"soak-{i}", attempt_id=attempt)
                )
                accepted = _drain_until_accepted(ws)
                assert accepted.event_type == EventType.ACCEPTED
                assert accepted.payload["boot_id"] == "boot-soak-5"
                assert accepted.attempt_id == attempt
                _ = ws.receive_text()

    assert counter["n"] == 1
    assert len(set(attempts)) == n_sessions


def test_stale_fence_after_cancel_on_same_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel acknowledges once; further work on that attempt is not re-admitted."""
    monkeypatch.setattr(
        WhisperListenStage, "transcribe", _scripted_transcribe, raising=True
    )
    app, counter = _build_warm_listen_app(boot_id="boot-cancel")

    with TestClient(app) as client, client.websocket_connect(
        "/stage/v1/stream",
        subprotocols=[STAGE_V1_SUBPROTOCOL],
    ) as ws:
        attempt = str(uuid4())
        cancel = str(uuid4())
        ws.send_text(
            _hello_json(
                session_id="cancel-sess",
                attempt_id=attempt,
                cancel_id=cancel,
            )
        )
        accepted = _drain_until_accepted(ws)
        _ = ws.receive_text()

        ws.send_text(_cancel_json(accepted=accepted, event_sequence=1))
        cancelled = parse_event_json(ws.receive_text())
        assert cancelled.event_type == EventType.CANCELLED
        assert cancelled.payload.get("disposed") is True

        # Post-cancel: either explicit ERROR (cancelled/stale) or clean close.
        # Do not re-open a second session on this connection.
        open_data = {
            "schema_version": SCHEMA_VERSION,
            "event_type": EventType.OPEN.value,
            "message_id": str(uuid4()),
            "event_sequence": 2,
            "created_at": _now(),
            "correlation_id": accepted.correlation_id,
            "session_id": accepted.session_id,
            "owner_generation": accepted.owner_generation,
            "stage_kind": StageKind.LISTEN.value,
            "stage_id": accepted.stage_id,
            "attempt_id": accepted.attempt_id,
            "cancel_id": accepted.cancel_id,
            "stage_instance_id": accepted.stage_instance_id,
            "deadline_at": _deadline(),
            "payload": {},
        }
        ws.send_text(parse_event(open_data).model_dump_json(exclude_none=True))
        try:
            late = parse_event_json(ws.receive_text())
            assert late.event_type == EventType.ERROR
            assert late.payload["code"] in {
                StageErrorCode.CANCELLED.value,
                StageErrorCode.STALE_FENCE.value,
            }
        except WebSocketDisconnect:
            # Server may tear down the attempt connection after cancel.
            pass

    assert counter["n"] == 1
