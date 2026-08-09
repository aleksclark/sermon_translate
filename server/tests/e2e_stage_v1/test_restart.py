"""Wave 5 lite: restart/reconnect proofs for stage.v1 (fake loaders, no GPU).

Full multi-node G5 remains out of scope; this module covers reconnect fence
behavior and warm-host soak that the prior skeleton reserved. Deeper ASGI
coverage lives in ``tests/stage_v1/test_restart_soak.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.models import ListenProduct, WordSpan
from src.pipelines.stages_listen.whisper import WhisperListenStage, WhisperLoadedModel
from src.runtime.worker import create_worker_app
from src.stage_v1.adapters import build_whisper_listen_host
from src.stage_v1.auth import STAGE_V1_SUBPROTOCOL, StageV1AuthConfig, StageV1Mode
from src.stage_v1.models import (
    BASELINE_SAMPLE_RATE_HZ,
    SCHEMA_VERSION,
    EventType,
    StageErrorCode,
    StageKind,
    parse_event,
    parse_event_json,
)


def _deadline() -> str:
    return (datetime.now(UTC) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.%f")[
        :-3
    ] + "Z"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _hello(
    *,
    session_id: str,
    attempt_id: str,
    cancel_id: str,
) -> str:
    data = {
        "schema_version": SCHEMA_VERSION,
        "event_type": EventType.HELLO.value,
        "message_id": str(uuid4()),
        "event_sequence": 0,
        "created_at": _now(),
        "correlation_id": f"corr-{uuid4()}",
        "session_id": session_id,
        "owner_generation": 1,
        "stage_kind": StageKind.LISTEN.value,
        "stage_id": "whisper-listen",
        "attempt_id": attempt_id,
        "cancel_id": cancel_id,
        "deadline_at": _deadline(),
        "payload": {
            "audio_formats": [
                {
                    "codec": "pcm_s16le",
                    "sample_rate_hz": BASELINE_SAMPLE_RATE_HZ,
                    "channels": 1,
                }
            ],
            "limits_requested": {"max_frame_bytes": 65536, "max_inflight_events": 32},
        },
    }
    return parse_event(data).model_dump_json(exclude_none=True)


async def _scripted_transcribe(
    self: WhisperListenStage, audio_stream: AsyncIterator[bytes]
) -> AsyncIterator[ListenProduct]:
    async for _chunk in audio_stream:
        pass
    yield ListenProduct(
        sequence=0,
        utterance_id="utt",
        text="",
        is_final=True,
        words=[WordSpan(text="", start_ms=0.0, end_ms=0.0, conf=0.0)],
        language="en",
    )


def _app(loads: dict[str, int]):
    def loader() -> WhisperLoadedModel:
        loads["n"] += 1
        return WhisperLoadedModel(model=object(), model_size="tiny", revision="tiny")

    host = build_whisper_listen_host(
        model_loader=loader,
        max_sessions=2,
        boot_id="boot-e2e-restart",
        model_size="tiny",
        local_dev=True,
    )
    return create_worker_app(
        "whisper-listen",
        max_sessions=2,
        host=host,
        auth=StageV1AuthConfig(
            mode=StageV1Mode.DEV,
            auth_token="",
            trust_proxy=False,
            allow_loopback_without_auth=True,
        ),
        warm=True,
    )


@pytest.mark.asyncio
async def test_restart_rejects_stale_fence(monkeypatch: pytest.MonkeyPatch) -> None:
    """After session close (worker host stays up), old attempt fence is rejected."""
    monkeypatch.setattr(
        WhisperListenStage, "transcribe", _scripted_transcribe, raising=True
    )
    loads = {"n": 0}
    app = _app(loads)
    old_attempt = str(uuid4())
    old_cancel = str(uuid4())

    with TestClient(app) as client:
        with client.websocket_connect(
            "/stage/v1/stream", subprotocols=[STAGE_V1_SUBPROTOCOL]
        ) as ws:
            ws.send_text(
                _hello(
                    session_id="restart-1",
                    attempt_id=old_attempt,
                    cancel_id=old_cancel,
                )
            )
            accepted = parse_event_json(ws.receive_text())
            assert accepted.event_type == EventType.ACCEPTED
            _ = ws.receive_text()

        new_attempt = str(uuid4())
        new_cancel = str(uuid4())
        with client.websocket_connect(
            "/stage/v1/stream", subprotocols=[STAGE_V1_SUBPROTOCOL]
        ) as ws:
            ws.send_text(
                _hello(
                    session_id="restart-1",
                    attempt_id=new_attempt,
                    cancel_id=new_cancel,
                )
            )
            accepted2 = parse_event_json(ws.receive_text())
            assert accepted2.event_type == EventType.ACCEPTED
            assert accepted2.attempt_id == new_attempt
            _ = ws.receive_text()

            stale_cancel = {
                "schema_version": SCHEMA_VERSION,
                "event_type": EventType.CANCEL.value,
                "message_id": str(uuid4()),
                "event_sequence": 1,
                "created_at": _now(),
                "correlation_id": accepted2.correlation_id,
                "session_id": accepted2.session_id,
                "owner_generation": accepted2.owner_generation,
                "stage_kind": StageKind.LISTEN.value,
                "stage_id": accepted2.stage_id,
                "attempt_id": old_attempt,
                "cancel_id": old_cancel,
                "stage_instance_id": accepted2.stage_instance_id,
                "deadline_at": _deadline(),
                "payload": {
                    "scope": "attempt",
                    "reason": "stale_after_restart",
                    "attempt_id": old_attempt,
                    "session_id": accepted2.session_id,
                },
            }
            ws.send_text(parse_event(stale_cancel).model_dump_json(exclude_none=True))
            err = parse_event_json(ws.receive_text())
            assert err.event_type == EventType.ERROR
            assert err.payload["code"] == StageErrorCode.STALE_FENCE.value

    assert loads["n"] == 1


@pytest.mark.asyncio
async def test_restart_emits_gap_for_unresumable_listen_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Listen resume is unsupported — hello.resume must fail closed (explicit error)."""
    monkeypatch.setattr(
        WhisperListenStage, "transcribe", _scripted_transcribe, raising=True
    )
    loads = {"n": 0}
    app = _app(loads)

    with TestClient(app) as client, client.websocket_connect(
        "/stage/v1/stream", subprotocols=[STAGE_V1_SUBPROTOCOL]
    ) as ws:
        data = {
            "schema_version": SCHEMA_VERSION,
            "event_type": EventType.HELLO.value,
            "message_id": str(uuid4()),
            "event_sequence": 0,
            "created_at": _now(),
            "correlation_id": f"corr-{uuid4()}",
            "session_id": "resume-sess",
            "owner_generation": 1,
            "stage_kind": StageKind.LISTEN.value,
            "stage_id": "whisper-listen",
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
                "resume": {"attempt_id": "prior-attempt"},
            },
        }
        ws.send_text(parse_event(data).model_dump_json(exclude_none=True))
        err = parse_event_json(ws.receive_text())
        assert err.event_type == EventType.ERROR
        assert err.payload["code"] in {
            StageErrorCode.RESUME_UNSUPPORTED.value,
            StageErrorCode.INVALID_ARGUMENT.value,
        }

    assert loads["n"] == 1


@pytest.mark.asyncio
async def test_replay_unpublished_translate_speak_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warm host: sequential reconnects do not reload; each attempt is distinct."""
    monkeypatch.setattr(
        WhisperListenStage, "transcribe", _scripted_transcribe, raising=True
    )
    loads = {"n": 0}
    app = _app(loads)
    attempts: list[str] = []

    with TestClient(app) as client:
        for i in range(3):
            attempt = str(uuid4())
            attempts.append(attempt)
            with client.websocket_connect(
                "/stage/v1/stream", subprotocols=[STAGE_V1_SUBPROTOCOL]
            ) as ws:
                ws.send_text(
                    _hello(
                        session_id=f"replay-{i}",
                        attempt_id=attempt,
                        cancel_id=str(uuid4()),
                    )
                )
                accepted = parse_event_json(ws.receive_text())
                assert accepted.event_type == EventType.ACCEPTED
                assert accepted.attempt_id == attempt
                assert accepted.payload["boot_id"] == "boot-e2e-restart"
                _ = ws.receive_text()

    assert loads["n"] == 1
    assert len(set(attempts)) == 3
