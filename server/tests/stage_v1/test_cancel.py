"""G2: cancel acknowledgement, fencing, late-after-cancel rejection."""

from __future__ import annotations

import pytest

from src.pipelines.commit_barrier import (
    CancelController,
    PublicationBarrier,
    new_fence,
)
from src.pipelines.stage_session import StageSession
from src.stage_v1.models import CancelScope, StageErrorCode, StageKind
from src.stage_v1.validation import Fence, ValidationError


class TestCancelController:
    def test_cancel_emits_once_and_disposes(self) -> None:
        fence = new_fence(session_id="s")
        disposed: list[str] = []
        ctrl = CancelController(active_fence=fence)
        ctrl.on_dispose(lambda: disposed.append("yes"))

        first = ctrl.cancel(reason="user_stop", scope=CancelScope.ATTEMPT)
        assert first is not None
        assert first["disposed"] is True
        assert first["reason"] == "user_stop"
        assert first["scope"] == "attempt"
        assert ctrl.cancelled is True
        assert ctrl.disposed is True
        assert ctrl.cancelled_emitted is True
        assert disposed == ["yes"]

        second = ctrl.cancel(reason="again")
        assert second is None
        assert disposed == ["yes"]  # dispose once

    def test_stop_admission_after_cancel(self) -> None:
        fence = new_fence(session_id="s")
        ctrl = CancelController(active_fence=fence)
        ctrl.cancel(reason="x")
        with pytest.raises(ValidationError) as ei:
            ctrl.check_admission()
        assert ei.value.code == StageErrorCode.CANCELLED

    def test_late_product_stale_fence(self) -> None:
        fence = new_fence(session_id="s", attempt_id="a1", cancel_id="c1")
        ctrl = CancelController(active_fence=fence)
        ctrl.cancel(reason="timeout")
        with pytest.raises(ValidationError) as ei:
            ctrl.accept_late_product(fence)
        assert ei.value.code == StageErrorCode.STALE_FENCE

    def test_mismatched_fence_rejected_before_cancel(self) -> None:
        active = new_fence(session_id="s", attempt_id="a1")
        other = new_fence(session_id="s", attempt_id="a2")
        ctrl = CancelController(active_fence=active)
        with pytest.raises(ValidationError) as ei:
            ctrl.check_fence(other)
        assert ei.value.code == StageErrorCode.STALE_FENCE

    def test_rotate_fence_retains_stale(self) -> None:
        old = new_fence(session_id="s", attempt_id="old")
        ctrl = CancelController(active_fence=old)
        ctrl.cancel(reason="disconnect")
        fresh = new_fence(session_id="s", attempt_id="new")
        ctrl.rotate_fence(fresh)
        assert ctrl.cancelled is False
        with pytest.raises(ValidationError) as ei:
            ctrl.check_fence(old)
        assert ei.value.code == StageErrorCode.STALE_FENCE
        ctrl.check_fence(fresh)  # ok


class TestPublicationBarrierCancelFence:
    def test_barrier_rejects_after_cancel(self) -> None:
        barrier = PublicationBarrier()
        fence = new_fence(session_id="s", stage_kind="speak")
        barrier.set_active_fence(fence)
        barrier.register(utterance_sequence=0, target_span_id="t0", fence=fence)
        barrier.cancel()
        with pytest.raises(ValidationError) as ei:
            barrier.complete(
                utterance_sequence=0,
                target_span_id="t0",
                payload=b"pcm",
                fence=fence,
            )
        assert ei.value.code == StageErrorCode.STALE_FENCE

    def test_provider_returns_after_cancel_is_fenced(self) -> None:
        """Cancellation proof: model returns after cancel; output fenced."""
        session = StageSession(
            session_id="s",
            stage_kind=StageKind.SPEAK,
            stage_id="tts",
        )
        session.register_publication_unit(utterance_sequence=0, target_span_id="span-1")
        cancelled = session.cancel(reason="orchestrator_cancel")
        assert cancelled is not None
        assert cancelled["disposed"] is True

        # Deliberate late provider product with same fence IDs
        late_fence = session.fence  # same IDs but cancelled
        with pytest.raises(ValidationError) as ei:
            session.complete_publication(
                utterance_sequence=0,
                target_span_id="span-1",
                payload=b"STALE_AUDIO",
                fence=late_fence,
            )
        assert ei.value.code == StageErrorCode.STALE_FENCE
        assert not any(e["event_type"] == "published" for e in session.events)

    def test_cancelled_emitted_exactly_once_on_session(self) -> None:
        session = StageSession(session_id="s")
        p1 = session.cancel(reason="r1")
        p2 = session.cancel(reason="r2")
        assert p1 is not None
        assert p2 is None
        cancelled_events = [e for e in session.events if e["event_type"] == "cancelled"]
        assert len(cancelled_events) == 1

    @pytest.mark.asyncio
    async def test_cancel_stops_audio_admission(self) -> None:
        session = StageSession(session_id="s")
        await session.cancel_async(reason="stop")
        with pytest.raises(ValidationError) as ei:
            await session.put_audio(b"\x00\x01", deadline_at="2099-01-01T00:00:00.000Z")
        assert ei.value.code in (
            StageErrorCode.CANCELLED,
            StageErrorCode.STALE_FENCE,
        )

    def test_fresh_attempt_rejects_old_fence_products(self) -> None:
        session = StageSession(session_id="s", stage_kind=StageKind.TRANSLATE)
        old_fence = session.fence
        session.cancel(reason="ws_drop")
        new = session.open_fresh_attempt()
        assert new.attempt_id != old_fence.attempt_id
        with pytest.raises(ValidationError) as ei:
            session.reject_late_product(old_fence)
        # After rotate, cancel_ctrl is not cancelled; but old is in stale list
        # reject_late_product uses accept_late_product which checks cancelled OR fence
        # After rotate, cancelled is False — check_fence should still reject stale.
        # open_fresh_attempt rotates; accept_late_product only fails if cancelled.
        # Use check_inbound_fence instead:
        with pytest.raises(ValidationError) as ei2:
            session.check_inbound_fence(old_fence)
        assert ei2.value.code == StageErrorCode.STALE_FENCE
        # silence unused if first path differed
        _ = ei

    def test_check_inbound_matches_active(self) -> None:
        session = StageSession(session_id="s")
        session.check_inbound_fence(session.fence)
        bad = Fence(
            session_id=session.session_id,
            owner_generation=session.owner_generation,
            stage_kind=session.fence.stage_kind,
            stage_id=session.stage_id,
            attempt_id="other-attempt",
            cancel_id=session.cancel_id,
            stage_instance_id=session.stage_instance_id,
        )
        with pytest.raises(ValidationError) as ei:
            session.check_inbound_fence(bad)
        assert ei.value.code == StageErrorCode.STALE_FENCE
