from __future__ import annotations

from src.api.store import SessionStore
from src.models import SessionCreate, SessionStatus, SessionUpdate


class TestSessionStore:
    def test_create_and_get(self) -> None:
        store = SessionStore()
        session = store.create(SessionCreate(pipeline_id="echo", label="test"))
        assert session.pipeline_id == "echo"
        assert session.label == "test"
        assert store.get(session.id) is session

    def test_list_all(self) -> None:
        store = SessionStore()
        store.create(SessionCreate(pipeline_id="echo"))
        store.create(SessionCreate(pipeline_id="echo"))
        assert len(store.list_all()) == 2

    def test_update(self) -> None:
        store = SessionStore()
        session = store.create(SessionCreate(pipeline_id="echo"))
        updated = store.update(session.id, SessionUpdate(label="renamed"))
        assert updated is not None
        assert updated.label == "renamed"

    def test_update_status(self) -> None:
        store = SessionStore()
        session = store.create(SessionCreate(pipeline_id="echo"))
        store.update(session.id, SessionUpdate(status=SessionStatus.ACTIVE))
        assert session.status == SessionStatus.ACTIVE
        assert store.active_count() == 1

    def test_update_nonexistent(self) -> None:
        store = SessionStore()
        assert store.update("nope", SessionUpdate(label="x")) is None

    def test_delete(self) -> None:
        store = SessionStore()
        session = store.create(SessionCreate(pipeline_id="echo"))
        assert store.delete(session.id) is True
        assert store.get(session.id) is None

    def test_delete_nonexistent(self) -> None:
        store = SessionStore()
        assert store.delete("nope") is False
