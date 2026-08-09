"""HTTP health endpoints: live / startup / ready semantics."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.stage_v1.health import build_health_router, health_status_code, mount_health_routes
from src.stage_v1.host import StageHost, StageHostError
from src.stage_v1.models import ArtifactDigestStatus, StageKind


def _host(
    *,
    artifact_digest: str = "sha256:" + "cd" * 32,
    expected: str | None = None,
    max_sessions: int = 1,
) -> StageHost:
    return StageHost(
        stage_kind=StageKind.TRANSLATE,
        stage_id="test-translate",
        stage_version="1.2.3",
        model_loader=lambda: {"ok": True},
        canary=lambda _m: True,
        max_sessions=max_sessions,
        code_git_sha="abc123",
        model_provider_id="opus-mt",
        model_revision="opus-mt-en-es@1",
        model_artifact_digest=artifact_digest,
        model_artifact_status=ArtifactDigestStatus.VERIFIED,
        expected_artifact_digest=expected,
        boot_id="boot-health",
        stage_instance_id="inst-health",
        local_dev=True,
    )


@pytest.mark.asyncio
async def test_readiness_503_before_warmup_200_after() -> None:
    host = _host()
    app = FastAPI()
    mount_health_routes(app, host)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        live = await client.get("/health/live")
        assert live.status_code == 200
        assert live.json()["status"] == "live"

        startup = await client.get("/health/startup")
        assert startup.status_code == 503

        ready = await client.get("/health/ready")
        assert ready.status_code == 503
        body = ready.json()
        assert body["model_loaded"] is False
        assert body["model_warm"] is False

        await host.load()
        startup2 = await client.get("/health/startup")
        assert startup2.status_code == 200

        ready_loaded = await client.get("/health/ready")
        assert ready_loaded.status_code == 503  # not warm yet

        await host.warmup()
        ready_ok = await client.get("/health/ready")
        assert ready_ok.status_code == 200
        payload = ready_ok.json()
        assert payload["status"] == "ready"
        assert payload["stage_kind"] == "translate"
        assert payload["stage_id"] == "test-translate"
        assert payload["stage_version"] == "1.2.3"
        assert payload["stage_instance_id"] == "inst-health"
        assert payload["boot_id"] == "boot-health"
        assert payload["model_loaded"] is True
        assert payload["model_warm"] is True
        assert payload["last_canary_ok"] is True
        assert payload["provenance_id"] is not None
        assert payload["provenance_id"].startswith("sha256:")
        assert payload["max_sessions"] == 1
        assert payload["active_sessions"] == 0
        assert payload["available_capacity"] == 1
        assert "provenance" in payload


@pytest.mark.asyncio
async def test_digest_mismatch_yields_503() -> None:
    host = _host(
        artifact_digest="sha256:" + "00" * 32,
        expected="sha256:" + "ff" * 32,
    )
    app = FastAPI()
    app.include_router(build_health_router(host))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with pytest.raises(StageHostError):
            await host.load()

        assert host.digest_mismatch is True
        assert health_status_code(host, "startup") == 503
        assert health_status_code(host, "ready") == 503

        startup = await client.get("/health/startup")
        assert startup.status_code == 503
        assert startup.json()["digest_mismatch"] is True

        ready = await client.get("/health/ready")
        assert ready.status_code == 503
        assert ready.json()["digest_mismatch"] is True


@pytest.mark.asyncio
async def test_ready_503_when_no_capacity_or_draining() -> None:
    host = _host(max_sessions=1)
    await host.load()
    await host.warmup()
    assert health_status_code(host, "ready") == 200

    s = await host.open_session()
    assert host.available_capacity == 0
    assert health_status_code(host, "ready") == 503

    await host.close_session(s.session_state_id)
    assert health_status_code(host, "ready") == 200

    host.begin_drain(reason="rollout")
    assert health_status_code(host, "ready") == 503
    assert health_status_code(host, "live") == 200
    assert health_status_code(host, "startup") == 200


@pytest.mark.asyncio
async def test_healthz_is_liveness_alias_not_readiness() -> None:
    host = _host()
    app = FastAPI()
    mount_health_routes(app, host)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Cold host: healthz still ok (compat liveness), ready is 503
        hz = await client.get("/healthz")
        assert hz.status_code == 200
        assert hz.text == "ok"
        ready = await client.get("/health/ready")
        assert ready.status_code == 503
