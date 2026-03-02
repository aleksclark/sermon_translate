from __future__ import annotations

from src.models import Session, SessionCreate, SessionStats, SessionStatus, SessionUpdate


class TestSessionCreate:
    def test_defaults(self) -> None:
        sc = SessionCreate(pipeline_id="echo")
        assert sc.sample_rate == 48000
        assert sc.channels == 1
        assert sc.label == ""

    def test_custom_values(self) -> None:
        sc = SessionCreate(pipeline_id="echo", sample_rate=16000, channels=2, label="test")
        assert sc.sample_rate == 16000
        assert sc.channels == 2
        assert sc.label == "test"


class TestSession:
    def test_default_fields(self) -> None:
        s = Session(pipeline_id="echo")
        assert s.status == SessionStatus.CREATED
        assert s.id  # non-empty
        assert s.created_at > 0
        assert isinstance(s.stats, SessionStats)

    def test_stats_defaults_zero(self) -> None:
        stats = SessionStats()
        assert stats.bytes_received == 0
        assert stats.bytes_sent == 0
        assert stats.duration_seconds == 0.0


class TestSessionUpdate:
    def test_partial_update(self) -> None:
        u = SessionUpdate(status=SessionStatus.ACTIVE)
        assert u.status == SessionStatus.ACTIVE
        assert u.label is None

    def test_empty_update(self) -> None:
        u = SessionUpdate()
        assert u.status is None
        assert u.label is None
