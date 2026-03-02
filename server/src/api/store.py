from __future__ import annotations

from src.models import Session, SessionCreate, SessionStatus, SessionUpdate


class SessionStore:
    """In-memory session storage."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, req: SessionCreate) -> Session:
        session = Session(
            pipeline_id=req.pipeline_id,
            label=req.label,
            sample_rate=req.sample_rate,
            channels=req.channels,
            audio_context_seconds=req.audio_context_seconds,
        )
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def list_all(self) -> list[Session]:
        return list(self._sessions.values())

    def update(self, session_id: str, req: SessionUpdate) -> Session | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if req.label is not None:
            session.label = req.label
        if req.status is not None:
            session.status = req.status
        return session

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def active_count(self) -> int:
        return sum(
            1 for s in self._sessions.values() if s.status == SessionStatus.ACTIVE
        )
