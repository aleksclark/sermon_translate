"""G2: ordered publication barrier + commit barrier integration tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.pipelines.commit_barrier import (
    PublicationBarrier,
    new_fence,
    rfc3339_deadline_from_now,
)
from src.pipelines.stage_session import StageSession, StageSessionConfig
from src.stage_v1.models import StageErrorCode, StageKind
from src.stage_v1.validation import ValidationError


def _fence(**kw: object):
    return new_fence(session_id="s1", stage_kind="speak", stage_id="tts", **kw)  # type: ignore[arg-type]


class TestPublicationBarrierOrdering:
    def test_in_order_completion_releases_immediately(self) -> None:
        barrier = PublicationBarrier()
        fence = _fence()
        barrier.set_active_fence(fence)
        barrier.register(utterance_sequence=0, target_span_id="t0", fence=fence)
        barrier.register(utterance_sequence=1, target_span_id="t1", fence=fence)

        rel = barrier.complete(
            utterance_sequence=0,
            target_span_id="t0",
            payload={"pcm": b"a"},
            fence=fence,
        )
        assert len(rel) == 1
        assert rel[0].kind == "product"
        assert rel[0].unit.target_span_id == "t0"

        rel2 = barrier.complete(
            utterance_sequence=1,
            target_span_id="t1",
            payload={"pcm": b"b"},
            fence=fence,
        )
        assert len(rel2) == 1
        assert rel2[0].unit.target_span_id == "t1"

    def test_out_of_order_completion_held_until_earlier_ready(self) -> None:
        barrier = PublicationBarrier()
        fence = _fence()
        barrier.set_active_fence(fence)
        barrier.register(utterance_sequence=0, target_span_id="t0", fence=fence)
        barrier.register(utterance_sequence=1, target_span_id="t1", fence=fence)
        barrier.register(utterance_sequence=2, target_span_id="t2", fence=fence)

        # Complete later units first
        held = barrier.complete(
            utterance_sequence=2,
            target_span_id="t2",
            payload="c",
            fence=fence,
        )
        assert held == []
        held = barrier.complete(
            utterance_sequence=1,
            target_span_id="t1",
            payload="b",
            fence=fence,
        )
        assert held == []

        # Completing earliest drains all three in order
        released = barrier.complete(
            utterance_sequence=0,
            target_span_id="t0",
            payload="a",
            fence=fence,
        )
        assert [r.unit.target_span_id for r in released] == ["t0", "t1", "t2"]
        assert [r.unit.payload for r in released] == ["a", "b", "c"]
        assert all(r.kind == "product" for r in released)

    def test_earlier_unit_failure_emits_gap_then_advances(self) -> None:
        barrier = PublicationBarrier()
        fence = _fence()
        barrier.set_active_fence(fence)
        barrier.register(utterance_sequence=0, target_span_id="t0", fence=fence)
        barrier.register(utterance_sequence=1, target_span_id="t1", fence=fence)

        # Later unit ready first — held
        assert (
            barrier.complete(
                utterance_sequence=1,
                target_span_id="t1",
                payload="ok",
                fence=fence,
            )
            == []
        )

        # Earlier fails → gap then product
        released = barrier.fail(
            utterance_sequence=0,
            target_span_id="t0",
            reason="inference_failed",
            fence=fence,
        )
        assert len(released) == 2
        assert released[0].kind == "gap"
        assert released[0].gap is not None
        assert released[0].gap.reason == "inference_failed"
        assert released[0].unit.target_span_id == "t0"
        assert released[1].kind == "product"
        assert released[1].unit.target_span_id == "t1"
        assert released[1].unit.payload == "ok"

    def test_published_unit_immutable_idempotent(self) -> None:
        barrier = PublicationBarrier()
        fence = _fence()
        barrier.set_active_fence(fence)
        barrier.register(utterance_sequence=0, target_span_id="t0", fence=fence)
        barrier.complete(
            utterance_sequence=0, target_span_id="t0", payload="first", fence=fence
        )
        again = barrier.complete(
            utterance_sequence=0, target_span_id="t0", payload="second", fence=fence
        )
        assert again == []
        assert len(barrier.released) == 1
        assert barrier.released[0].unit.payload == "first"

    def test_stale_fence_rejected(self) -> None:
        barrier = PublicationBarrier()
        active = _fence(attempt_id="a1", cancel_id="c1", stage_instance_id="i1")
        stale = _fence(attempt_id="a2", cancel_id="c2", stage_instance_id="i2")
        barrier.set_active_fence(active)
        barrier.register(utterance_sequence=0, target_span_id="t0", fence=active)
        with pytest.raises(ValidationError) as ei:
            barrier.complete(
                utterance_sequence=0,
                target_span_id="t0",
                payload=b"x",
                fence=stale,
            )
        assert ei.value.code == StageErrorCode.STALE_FENCE

    def test_cancel_never_releases_pending(self) -> None:
        barrier = PublicationBarrier()
        fence = _fence()
        barrier.set_active_fence(fence)
        barrier.register(utterance_sequence=0, target_span_id="t0", fence=fence)
        barrier.register(utterance_sequence=1, target_span_id="t1", fence=fence)
        barrier.complete(
            utterance_sequence=1, target_span_id="t1", payload="late", fence=fence
        )
        barrier.cancel()
        with pytest.raises(ValidationError) as ei:
            barrier.complete(
                utterance_sequence=0, target_span_id="t0", payload="a", fence=fence
            )
        assert ei.value.code == StageErrorCode.STALE_FENCE
        assert barrier.released == []

    def test_deadline_exceeded_before_publish_becomes_gap(self) -> None:
        barrier = PublicationBarrier()
        fence = _fence()
        barrier.set_active_fence(fence)
        past = (datetime.now(UTC) - timedelta(seconds=1)).strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )[:-3] + "Z"
        barrier.register(
            utterance_sequence=0,
            target_span_id="t0",
            fence=fence,
            deadline_at=past,
        )
        # complete checks deadline at complete time too
        with pytest.raises(ValidationError) as ei:
            barrier.complete(
                utterance_sequence=0,
                target_span_id="t0",
                payload="x",
                fence=fence,
            )
        assert ei.value.code == StageErrorCode.DEADLINE_EXCEEDED


class TestStageSessionCommitBarrier:
    @pytest.mark.asyncio
    async def test_session_routes_listen_commits_only(self) -> None:
        session = StageSession(
            session_id="sess",
            stage_kind=StageKind.LISTEN,
            stage_id="whisper",
        )
        session.bind_utterance("utt-1", 0)
        session.observe_listen_product(
            revision=0, text="Hello there", committed_prefix_chars=0
        )
        assert session.pending_translate_deltas() == []
        session.observe_listen_product(
            revision=1, text="Hello there", committed_prefix_chars=5
        )
        deltas = session.pending_translate_deltas()
        assert len(deltas) == 1
        assert deltas[0].text == "Hello"

    @pytest.mark.asyncio
    async def test_session_publication_order(self) -> None:
        session = StageSession(
            session_id="sess",
            stage_kind=StageKind.SPEAK,
            stage_id="tts",
            config=StageSessionConfig(default_deadline_s=5.0),
        )
        session.register_publication_unit(utterance_sequence=0, target_span_id="a")
        session.register_publication_unit(utterance_sequence=1, target_span_id="b")
        # out of order
        r = session.complete_publication(
            utterance_sequence=1, target_span_id="b", payload=b"B"
        )
        assert r == []
        r = session.complete_publication(
            utterance_sequence=0, target_span_id="a", payload=b"A"
        )
        assert [x.unit.target_span_id for x in r] == ["a", "b"]
        published = [e for e in session.events if e["event_type"] == "published"]
        assert [e["target_span_id"] for e in published] == ["a", "b"]

    def test_future_deadline_helper(self) -> None:
        dl = rfc3339_deadline_from_now(10)
        assert dl.endswith("Z")
