from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestStatsEndpoint:
    async def test_get_stats(self, client: AsyncClient) -> None:
        r = await client.get("/api/stats")
        assert r.status_code == 200
        data = r.json()
        assert "uptime_seconds" in data
        assert data["active_sessions"] == 0


class TestPipelinesEndpoint:
    async def test_list_pipelines(self, client: AsyncClient) -> None:
        r = await client.get("/api/pipelines")
        assert r.status_code == 200
        data = r.json()
        assert len(data) >= 1
        assert data[0]["id"] == "echo"
        assert any(p["id"] == "composed" for p in data)


class TestStagesEndpoint:
    async def test_list_stages(self, client: AsyncClient) -> None:
        r = await client.get("/api/stages")
        assert r.status_code == 200
        data = r.json()
        ids = {s["id"] for s in data}
        assert {
            "passthrough-listen",
            "passthrough-translate",
            "passthrough-speak",
            "baseline-prosody",
        } <= ids

    async def test_list_stages_filter_kind(self, client: AsyncClient) -> None:
        r = await client.get("/api/stages", params={"kind": "listen"})
        assert r.status_code == 200
        data = r.json()
        assert data
        assert all(s["kind"] == "listen" for s in data)


class TestSessionsEndpoint:
    async def test_create_session(self, client: AsyncClient) -> None:
        r = await client.post("/api/sessions", json={"pipeline_id": "echo", "label": "test"})
        assert r.status_code == 201
        data = r.json()
        assert data["pipeline_id"] == "echo"
        assert data["label"] == "test"
        assert data["status"] == "created"
        assert data["stages"] is None

    async def test_create_session_bad_pipeline(self, client: AsyncClient) -> None:
        r = await client.post("/api/sessions", json={"pipeline_id": "nope"})
        assert r.status_code == 400

    async def test_create_composed_session(self, client: AsyncClient) -> None:
        r = await client.post(
            "/api/sessions",
            json={
                "pipeline_id": "composed",
                "stages": {
                    "listen": "passthrough-listen",
                    "translate": "passthrough-translate",
                    "speak": "passthrough-speak",
                    "prosody": "baseline-prosody",
                },
            },
        )
        assert r.status_code == 201
        data = r.json()
        assert data["pipeline_id"] == "composed"
        assert data["stages"]["listen"] == "passthrough-listen"
        assert data["stages"]["prosody"] == "baseline-prosody"

    async def test_create_composed_requires_stages(self, client: AsyncClient) -> None:
        r = await client.post("/api/sessions", json={"pipeline_id": "composed"})
        assert r.status_code == 400

    async def test_create_composed_rejects_unknown_stage(self, client: AsyncClient) -> None:
        r = await client.post(
            "/api/sessions",
            json={
                "pipeline_id": "composed",
                "stages": {
                    "listen": "missing-listen",
                    "translate": "passthrough-translate",
                    "speak": "passthrough-speak",
                },
            },
        )
        assert r.status_code == 400

    async def test_create_composed_rejects_wrong_kind(self, client: AsyncClient) -> None:
        r = await client.post(
            "/api/sessions",
            json={
                "pipeline_id": "composed",
                "stages": {
                    "listen": "passthrough-translate",
                    "translate": "passthrough-translate",
                    "speak": "passthrough-speak",
                },
            },
        )
        assert r.status_code == 400

    async def test_legacy_pipeline_rejects_stages(self, client: AsyncClient) -> None:
        r = await client.post(
            "/api/sessions",
            json={
                "pipeline_id": "echo",
                "stages": {
                    "listen": "passthrough-listen",
                    "translate": "passthrough-translate",
                    "speak": "passthrough-speak",
                },
            },
        )
        assert r.status_code == 400

    async def test_list_sessions(self, client: AsyncClient) -> None:
        await client.post("/api/sessions", json={"pipeline_id": "echo"})
        r = await client.get("/api/sessions")
        assert r.status_code == 200
        assert len(r.json()) >= 1

    async def test_get_session(self, client: AsyncClient) -> None:
        create = await client.post("/api/sessions", json={"pipeline_id": "echo"})
        sid = create.json()["id"]
        r = await client.get(f"/api/sessions/{sid}")
        assert r.status_code == 200
        assert r.json()["id"] == sid

    async def test_get_session_not_found(self, client: AsyncClient) -> None:
        r = await client.get("/api/sessions/nonexistent")
        assert r.status_code == 404

    async def test_update_session(self, client: AsyncClient) -> None:
        create = await client.post("/api/sessions", json={"pipeline_id": "echo"})
        sid = create.json()["id"]
        r = await client.patch(f"/api/sessions/{sid}", json={"label": "renamed"})
        assert r.status_code == 200
        assert r.json()["label"] == "renamed"

    async def test_update_session_status(self, client: AsyncClient) -> None:
        create = await client.post("/api/sessions", json={"pipeline_id": "echo"})
        sid = create.json()["id"]
        r = await client.patch(f"/api/sessions/{sid}", json={"status": "closed"})
        assert r.status_code == 200
        assert r.json()["status"] == "closed"

    async def test_delete_session(self, client: AsyncClient) -> None:
        create = await client.post("/api/sessions", json={"pipeline_id": "echo"})
        sid = create.json()["id"]
        r = await client.delete(f"/api/sessions/{sid}")
        assert r.status_code == 204
        r2 = await client.get(f"/api/sessions/{sid}")
        assert r2.status_code == 404

    async def test_delete_session_not_found(self, client: AsyncClient) -> None:
        r = await client.delete("/api/sessions/nonexistent")
        assert r.status_code == 404
