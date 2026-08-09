"""Live /stage/v1/stream WebSocket server tests (C1/C2/C3).

Covers:
- hello → accepted handshake with stage.v1 subprotocol
- production auth reject (no token / bad token) before session allocation
- warm host: loader once, pre-EOS product with scripted model
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import numpy as np
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
    authorize_stage_upgrade,
    load_stage_v1_auth_config,
)
from src.stage_v1.framing import encode_binary_frame
from src.stage_v1.models import (
    BASELINE_SAMPLE_RATE_HZ,
    SCHEMA_VERSION,
    AcceptedPayload,
    AudioFormat,
    BinaryAudioPayload,
    EventType,
    StageErrorCode,
    StageKind,
    parse_event_json,
)


def _deadline(hours: float = 1.0) -> str:
    return (datetime.now(UTC) + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.%f")[
        :-3
    ] + "Z"


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _hello_json(
    *,
    stage_id: str = "whisper-listen",
    session_id: str = "sess-stream-1",
    attempt_id: str | None = None,
    cancel_id: str | None = None,
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
        "owner_generation": 1,
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
    from src.stage_v1.models import parse_event

    return parse_event(data).model_dump_json(exclude_none=True)


def _fake_whisper_loaded() -> WhisperLoadedModel:
    return WhisperLoadedModel(model=object(), model_size="tiny", revision="tiny")


async def _scripted_transcribe(
    self: WhisperListenStage, audio_stream: AsyncIterator[bytes]
) -> AsyncIterator[ListenProduct]:
    """Emit a committed mid-stream product before EOS, then a final."""
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
    boot_id: str = "boot-stream-test",
    loads: dict[str, int] | None = None,
) -> tuple[Any, dict[str, int]]:
    counter = loads if loads is not None else {"n": 0}

    def loader() -> WhisperLoadedModel:
        counter["n"] += 1
        return _fake_whisper_loaded()

    host = build_whisper_listen_host(
        model_loader=loader,
        max_sessions=2,
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
        max_sessions=2,
        host=host,
        auth=cfg,
        warm=True,
    )
    return app, counter


def test_authorize_production_requires_token_and_wss() -> None:
    cfg = StageV1AuthConfig(
        mode=StageV1Mode.PRODUCTION,
        auth_token="secret-token",
        trust_proxy=True,
        allow_loopback_without_auth=False,
    )
    # plain ws + no auth
    d = authorize_stage_upgrade(
        config=cfg,
        headers={},
        url_scheme="ws",
        client_host="10.0.0.5",
    )
    assert d.ok is False
    assert d.code == StageErrorCode.AUTHENTICATION_FAILED

    # wss but missing token
    d = authorize_stage_upgrade(
        config=cfg,
        headers={},
        url_scheme="wss",
        client_host="10.0.0.5",
    )
    assert d.ok is False
    assert d.code == StageErrorCode.AUTHENTICATION_FAILED

    # trusted proxy https + bearer ok
    d = authorize_stage_upgrade(
        config=cfg,
        headers={
            "Authorization": "Bearer secret-token",
            "X-Forwarded-Proto": "https",
        },
        url_scheme="ws",
        client_host="10.0.0.5",
    )
    assert d.ok is True

    # bad token
    d = authorize_stage_upgrade(
        config=cfg,
        headers={"X-Stage-Auth": "wrong", "X-Forwarded-Proto": "https"},
        url_scheme="ws",
        client_host="10.0.0.5",
    )
    assert d.ok is False


def test_production_empty_token_refuses_boot() -> None:
    cfg = StageV1AuthConfig(
        mode=StageV1Mode.PRODUCTION,
        auth_token="",
        trust_proxy=False,
        allow_loopback_without_auth=False,
    )
    with pytest.raises(RuntimeError, match="STAGE_AUTH_TOKEN"):
        cfg.validate_boot()


def test_load_stage_v1_auth_config_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAGE_V1_MODE", "production")
    monkeypatch.setenv("STAGE_AUTH_TOKEN", "tok")
    monkeypatch.setenv("STAGE_TRUST_PROXY", "1")
    cfg = load_stage_v1_auth_config()
    assert cfg.mode == StageV1Mode.PRODUCTION
    assert cfg.auth_token == "tok"
    assert cfg.trust_proxy is True
    assert cfg.allow_loopback_without_auth is False


def test_stream_hello_accepted_loopback_dev(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        WhisperListenStage, "transcribe", _scripted_transcribe, raising=True
    )
    app, counter = _build_warm_listen_app()
    with TestClient(app) as client, client.websocket_connect(
        "/stage/v1/stream",
        subprotocols=[STAGE_V1_SUBPROTOCOL],
    ) as ws:
        ws.send_text(_hello_json())
        accepted_raw = ws.receive_text()
        accepted = parse_event_json(accepted_raw)
        assert accepted.event_type == EventType.ACCEPTED
        payload = AcceptedPayload.model_validate(accepted.payload)
        assert payload.boot_id == "boot-stream-test"
        assert payload.stage_instance_id
        assert accepted.stage_id == "whisper-listen"
        # optional window
        maybe = ws.receive_text()
        window = parse_event_json(maybe)
        assert window.event_type in {EventType.WINDOW, EventType.OPENED}
    assert counter["n"] == 1


def test_stream_production_auth_reject_before_session(
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
    app, _ = _build_warm_listen_app(auth=auth, loads=loads)

    with TestClient(app) as client:
        # Missing auth + no secure proxy header → reject
        with client.websocket_connect(
            "/stage/v1/stream",
            subprotocols=[STAGE_V1_SUBPROTOCOL],
        ) as ws:
            err_raw = ws.receive_text()
            err = parse_event_json(err_raw)
            assert err.event_type == EventType.ERROR
            assert err.payload["code"] == StageErrorCode.AUTHENTICATION_FAILED.value
            with pytest.raises(WebSocketDisconnect):
                ws.receive_text()

        # Bad token with trusted proxy still rejects
        with client.websocket_connect(
            "/stage/v1/stream",
            subprotocols=[STAGE_V1_SUBPROTOCOL],
            headers={
                "Authorization": "Bearer wrong",
                "X-Forwarded-Proto": "https",
            },
        ) as ws:
            err_raw = ws.receive_text()
            err = parse_event_json(err_raw)
            assert err.event_type == EventType.ERROR
            assert err.payload["code"] == StageErrorCode.AUTHENTICATION_FAILED.value

        # Valid token + proxy proto accepts handshake
        with client.websocket_connect(
            "/stage/v1/stream",
            subprotocols=[STAGE_V1_SUBPROTOCOL],
            headers={
                "Authorization": "Bearer prod-secret",
                "X-Forwarded-Proto": "https",
            },
        ) as ws:
            ws.send_text(_hello_json())
            accepted = parse_event_json(ws.receive_text())
            assert accepted.event_type == EventType.ACCEPTED

    # Model still loaded once at worker startup (not per failed auth).
    assert loads["n"] == 1


def test_stream_pre_eos_product_with_scripted_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        WhisperListenStage, "transcribe", _scripted_transcribe, raising=True
    )
    app, counter = _build_warm_listen_app(boot_id="boot-pre-eos")

    pcm = (np.zeros(320, dtype=np.int16)).tobytes()  # 20 ms @ 16k

    with TestClient(app) as client, client.websocket_connect(
        "/stage/v1/stream",
        subprotocols=[STAGE_V1_SUBPROTOCOL],
    ) as ws:
        hello = _hello_json(session_id="pre-eos-sess")
        ws.send_text(hello)
        accepted = parse_event_json(ws.receive_text())
        assert accepted.event_type == EventType.ACCEPTED
        # drain window
        window = parse_event_json(ws.receive_text())
        assert window.event_type == EventType.WINDOW

        # send one audio frame (binary STG1)
        audio_env = {
            "schema_version": SCHEMA_VERSION,
            "event_type": EventType.LISTEN_AUDIO.value,
            "message_id": str(uuid4()),
            "event_sequence": 1,
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
            "payload": BinaryAudioPayload(
                stream_id="source:main",
                media_sequence=0,
                start_sample=0,
                sample_count=320,
                payload_bytes=len(pcm),
                format=AudioFormat(sample_rate_hz=BASELINE_SAMPLE_RATE_HZ, channels=1),
            ).model_dump(mode="json"),
        }
        from src.stage_v1.models import parse_event

        frame = encode_binary_frame(parse_event(audio_env), pcm)
        ws.send_bytes(frame)

        # Expect product before EOS
        product_raw = ws.receive_text()
        product = parse_event_json(product_raw)
        assert product.event_type == EventType.LISTEN_PRODUCT
        assert product.payload["text"] == "hello world"
        assert product.payload["committed_prefix_chars"] == len("hello world")
        assert product.payload["is_final"] is False

        # EOS
        eos = {
            "schema_version": SCHEMA_VERSION,
            "event_type": EventType.EOS.value,
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
            "payload": {
                "stream_id": "source:main",
                "last_media_sequence": 0,
                "last_sample_end": 320,
            },
        }
        ws.send_text(parse_event(eos).model_dump_json(exclude_none=True))

        final_raw = ws.receive_text()
        final = parse_event_json(final_raw)
        assert final.event_type == EventType.LISTEN_PRODUCT
        assert final.payload["is_final"] is True

    # Warm: single loader invocation across startup + one session.
    assert counter["n"] == 1


def test_stream_two_sessions_loader_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        WhisperListenStage, "transcribe", _scripted_transcribe, raising=True
    )
    app, counter = _build_warm_listen_app(boot_id="boot-two-sess")

    with TestClient(app) as client:
        for i in range(2):
            with client.websocket_connect(
                "/stage/v1/stream",
                subprotocols=[STAGE_V1_SUBPROTOCOL],
            ) as ws:
                ws.send_text(_hello_json(session_id=f"sess-{i}"))
                accepted = parse_event_json(ws.receive_text())
                assert accepted.event_type == EventType.ACCEPTED
                assert accepted.payload["boot_id"] == "boot-two-sess"
                # drain window then close
                _ = ws.receive_text()

    assert counter["n"] == 1


def test_create_worker_app_production_empty_token_raises() -> None:
    auth = StageV1AuthConfig(
        mode=StageV1Mode.PRODUCTION,
        auth_token="",
        trust_proxy=False,
        allow_loopback_without_auth=False,
    )
    with pytest.raises(RuntimeError, match="STAGE_AUTH_TOKEN"):
        create_worker_app("passthrough-listen", auth=auth, warm=False)
