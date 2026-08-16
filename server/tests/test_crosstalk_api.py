from __future__ import annotations

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api import crosstalk_router
from src.api.deps import init_deps
from src.api.store import SessionStore
from src.models import ServerStatsTracker, SessionCreate
from src.pipelines import create_default_registry
from src.transport.crosstalk_client import CrosstalkChannel, CrosstalkSession


class FakeClient:
    async def list_sessions(self) -> list[CrosstalkSession]:
        return [CrosstalkSession(id="cs1", name="Main", description="d")]

    async def list_channels(self, session_id: str) -> list[CrosstalkChannel]:
        return [CrosstalkChannel(id="c1", name="Feed", type="feed", session_id=session_id)]


class FakeService:
    def __init__(self, configured: bool = True) -> None:
        self._configured = configured
        self.started: list[tuple[str, str]] = []
        self.stopped: list[str] = []

    def configured(self) -> bool:
        return self._configured

    def client(self) -> FakeClient:
        return FakeClient()

    async def start_translation(self, session_id, crosstalk_session_id, *, sample_rate, **_):
        self.started.append((session_id, crosstalk_session_id))

    async def stop_translation(self, session_id: str) -> bool:
        if session_id in self.stopped:
            return False
        self.stopped.append(session_id)
        return True


def _make_app(service: FakeService) -> tuple[FastAPI, SessionStore]:
    app = FastAPI()
    store = SessionStore()
    init_deps(store, create_default_registry(), ServerStatsTracker(), service)
    app.include_router(crosstalk_router)
    return app, store


@pytest.fixture
async def client_and_store():
    service = FakeService()
    app, store = _make_app(service)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c, store, service


class TestCrosstalkDiscovery:
    async def test_list_sessions(self, client_and_store) -> None:
        c, _, _ = client_and_store
        r = await c.get("/api/crosstalk/sessions")
        assert r.status_code == 200
        assert r.json()[0]["id"] == "cs1"

    async def test_list_channels(self, client_and_store) -> None:
        c, _, _ = client_and_store
        r = await c.get("/api/crosstalk/sessions/cs1/channels")
        assert r.status_code == 200
        assert r.json()[0]["type"] == "feed"

    async def test_discovery_503_when_unconfigured(self) -> None:
        service = FakeService(configured=False)
        app, _ = _make_app(service)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/api/crosstalk/sessions")
            assert r.status_code == 503


class TestCrosstalkTranslationLifecycle:
    async def test_start_requires_crosstalk_source(self, client_and_store) -> None:
        c, store, _ = client_and_store
        session = store.create(SessionCreate(pipeline_id="echo"))
        r = await c.post(f"/api/crosstalk/translations/{session.id}/start")
        assert r.status_code == 400

    async def test_start_and_stop(self, client_and_store) -> None:
        c, store, service = client_and_store
        session = store.create(
            SessionCreate(
                pipeline_id="echo",
                audio_source="crosstalk",
                crosstalk_session_id="cs1",
            )
        )
        r = await c.post(f"/api/crosstalk/translations/{session.id}/start")
        assert r.status_code == 202
        assert service.started == [(session.id, "cs1")]

        r = await c.post(f"/api/crosstalk/translations/{session.id}/stop")
        assert r.status_code == 200
        assert session.id in service.stopped

    async def test_start_unknown_session_404(self, client_and_store) -> None:
        c, _, _ = client_and_store
        r = await c.post("/api/crosstalk/translations/missing/start")
        assert r.status_code == 404

    async def test_stop_when_not_running_404(self, client_and_store) -> None:
        c, _, service = client_and_store
        service.stopped.append("already")
        r = await c.post("/api/crosstalk/translations/already/stop")
        assert r.status_code == 404
