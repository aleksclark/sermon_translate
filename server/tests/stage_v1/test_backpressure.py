"""G2: deadline-aware bounded queues, overload, no silent drop, memory bound."""

from __future__ import annotations

import asyncio
import resource
from datetime import UTC, datetime, timedelta

import pytest

from src.pipelines.commit_barrier import (
    PROTECTED_KINDS,
    DeadlineAwareQueue,
    QueueItemKind,
    rfc3339_deadline_from_now,
)
from src.pipelines.stage_session import StageSession, StageSessionConfig
from src.stage_v1.models import DropReason, StageErrorCode
from src.stage_v1.validation import ValidationError


def _past_deadline() -> str:
    return (datetime.now(UTC) - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S.%f")[
        :-3
    ] + "Z"


def _soon_deadline(ms: float = 50) -> str:
    return rfc3339_deadline_from_now(ms / 1000.0)


class TestDeadlineAwareQueue:
    @pytest.mark.asyncio
    async def test_put_get_roundtrip(self) -> None:
        q: DeadlineAwareQueue[bytes] = DeadlineAwareQueue(capacity=2, name="t")
        await q.put(b"a", kind=QueueItemKind.AUDIO, deadline_at=rfc3339_deadline_from_now(5))
        item = await q.get(deadline_at=rfc3339_deadline_from_now(5))
        assert item == b"a"

    @pytest.mark.asyncio
    async def test_put_waits_until_deadline_then_exceeded(self) -> None:
        q: DeadlineAwareQueue[int] = DeadlineAwareQueue(capacity=1, name="tiny")
        await q.put(1, kind=QueueItemKind.AUDIO, deadline_at=rfc3339_deadline_from_now(5))
        with pytest.raises(ValidationError) as ei:
            await q.put(
                2,
                kind=QueueItemKind.AUDIO,
                deadline_at=_soon_deadline(80),
            )
        assert ei.value.code == StageErrorCode.DEADLINE_EXCEEDED
        assert q.qsize == 1
        assert q.high_water_items == 1

    @pytest.mark.asyncio
    async def test_expired_deadline_before_enqueue(self) -> None:
        q: DeadlineAwareQueue[int] = DeadlineAwareQueue(capacity=4, name="t")
        with pytest.raises(ValidationError) as ei:
            await q.put(1, kind=QueueItemKind.AUDIO, deadline_at=_past_deadline())
        assert ei.value.code == StageErrorCode.DEADLINE_EXCEEDED

    @pytest.mark.asyncio
    async def test_never_silently_drops_protected_kinds(self) -> None:
        q: DeadlineAwareQueue[str] = DeadlineAwareQueue(capacity=1, name="t")
        await q.put("a", kind=QueueItemKind.AUDIO, deadline_at=rfc3339_deadline_from_now(5))
        for kind in PROTECTED_KINDS:
            with pytest.raises(ValidationError) as ei:
                await q.put(
                    f"x-{kind}",
                    kind=kind,
                    deadline_at=_soon_deadline(40),
                )
            assert ei.value.code == StageErrorCode.DEADLINE_EXCEEDED
        # Original still present — no silent drop
        assert await q.get(deadline_at=rfc3339_deadline_from_now(5)) == "a"

    @pytest.mark.asyncio
    async def test_coalesce_uncommitted_revisions_emits_dropped(self) -> None:
        q: DeadlineAwareQueue[str] = DeadlineAwareQueue(capacity=2, name="partials")
        d1 = await q.put(
            "Hel",
            kind=QueueItemKind.UNCOMMITTED_REVISION,
            deadline_at=rfc3339_deadline_from_now(5),
            revision=0,
            coalesce_key="utt-1",
        )
        assert d1 is None
        d2 = await q.put(
            "Hello",
            kind=QueueItemKind.UNCOMMITTED_REVISION,
            deadline_at=rfc3339_deadline_from_now(5),
            revision=1,
            coalesce_key="utt-1",
        )
        assert d2 is not None
        assert d2.reason is DropReason.SUPERSEDED_UNCOMMITTED
        assert d2.revision_start == 0
        assert q.qsize == 1
        assert await q.get(deadline_at=rfc3339_deadline_from_now(5)) == "Hello"
        assert len(q.dropped_events) == 1

    @pytest.mark.asyncio
    async def test_producer_blocks_until_consumer_drains(self) -> None:
        q: DeadlineAwareQueue[int] = DeadlineAwareQueue(capacity=1, name="bp")
        await q.put(1, kind=QueueItemKind.AUDIO, deadline_at=rfc3339_deadline_from_now(5))

        started = asyncio.Event()
        done = asyncio.Event()

        async def producer() -> None:
            started.set()
            await q.put(
                2,
                kind=QueueItemKind.AUDIO,
                deadline_at=rfc3339_deadline_from_now(2),
            )
            done.set()

        task = asyncio.create_task(producer())
        await started.wait()
        await asyncio.sleep(0.05)
        assert not done.is_set()
        assert await q.get(deadline_at=rfc3339_deadline_from_now(5)) == 1
        await asyncio.wait_for(done.wait(), timeout=1.0)
        assert await q.get(deadline_at=rfc3339_deadline_from_now(5)) == 2
        await task
        assert q.high_water_items <= 1

    @pytest.mark.asyncio
    async def test_high_water_never_exceeds_capacity(self) -> None:
        cap = 3
        q: DeadlineAwareQueue[bytes] = DeadlineAwareQueue(capacity=cap, name="hw")
        for i in range(cap):
            await q.put(
                bytes([i]),
                kind=QueueItemKind.AUDIO,
                deadline_at=rfc3339_deadline_from_now(5),
                bytes_size=1,
            )
        assert q.high_water_items == cap
        assert q.qsize == cap
        with pytest.raises(ValidationError):
            await q.put(
                b"x",
                kind=QueueItemKind.AUDIO,
                deadline_at=_soon_deadline(30),
                bytes_size=1,
            )
        assert q.high_water_items <= cap


class TestOverloadAndMemoryBound:
    @pytest.mark.asyncio
    async def test_tiny_capacity_overload_session_audio(self) -> None:
        session = StageSession(
            session_id="overload",
            config=StageSessionConfig(
                audio_queue_capacity=2,
                max_frame_bytes=64,
                default_deadline_s=0.05,
            ),
        )
        frame = b"\x00\x01" * 32  # 64 bytes
        await session.put_audio(frame, deadline_at=rfc3339_deadline_from_now(2))
        await session.put_audio(frame, deadline_at=rfc3339_deadline_from_now(2))
        with pytest.raises(ValidationError) as ei:
            await session.put_audio(frame, deadline_at=_soon_deadline(40))
        assert ei.value.code == StageErrorCode.DEADLINE_EXCEEDED
        snap = session.snapshot()
        assert snap["audio_high_water"] <= 2
        assert snap["audio_high_water"] == 2

    @pytest.mark.asyncio
    async def test_memory_bound_observation_under_overload(self) -> None:
        """Derive upper bound from capacities; observe growth stays under bound.

        Anti-cheat: not merely 'metric exists' — fill to capacity and assert
        high-water bytes ≤ capacity-derived bound and RSS delta is sane.
        """
        capacity = 4
        max_frame = 1024
        session = StageSession(
            session_id="mem",
            config=StageSessionConfig(
                audio_queue_capacity=capacity,
                product_queue_capacity=capacity,
                speak_queue_capacity=capacity,
                max_frame_bytes=max_frame,
                default_deadline_s=0.1,
            ),
        )
        bound = session.memory_bound_bytes()
        assert bound == capacity * max_frame * 3

        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        frame = b"\xff" * max_frame
        # Fill audio queue to capacity
        for _ in range(capacity):
            await session.put_audio(frame, deadline_at=rfc3339_deadline_from_now(2))

        # Attempt overload — must not grow past capacity
        with pytest.raises(ValidationError):
            await session.put_audio(frame, deadline_at=_soon_deadline(50))

        snap = session.snapshot()
        assert snap["audio_high_water"] <= capacity
        assert snap["audio_high_water_bytes"] <= capacity * max_frame
        assert snap["audio_high_water_bytes"] <= bound
        assert snap["memory_bound_audio_bytes"] == capacity * max_frame

        rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is KB on Linux. Growth should be far below an unbounded buffer
        # of e.g. 1000 frames; allow generous headroom for allocator noise.
        growth_kb = max(0, rss_after - rss_before)
        unbounded_would_be_kb = (1000 * max_frame) // 1024
        assert growth_kb < unbounded_would_be_kb

    @pytest.mark.asyncio
    async def test_committed_span_not_dropped_on_full_queue(self) -> None:
        q: DeadlineAwareQueue[str] = DeadlineAwareQueue(capacity=1, name="spans")
        await q.put(
            "committed-1",
            kind=QueueItemKind.COMMITTED_SPAN,
            deadline_at=rfc3339_deadline_from_now(5),
        )
        with pytest.raises(ValidationError) as ei:
            await q.put(
                "committed-2",
                kind=QueueItemKind.COMMITTED_SPAN,
                deadline_at=_soon_deadline(40),
            )
        assert ei.value.code == StageErrorCode.DEADLINE_EXCEEDED
        assert await q.get(deadline_at=rfc3339_deadline_from_now(5)) == "committed-1"

    @pytest.mark.asyncio
    async def test_eos_cancel_error_gap_protected(self) -> None:
        q: DeadlineAwareQueue[str] = DeadlineAwareQueue(capacity=1, name="ctrl")
        await q.put("hold", kind=QueueItemKind.AUDIO, deadline_at=rfc3339_deadline_from_now(5))
        for kind, label in (
            (QueueItemKind.EOS, "eos"),
            (QueueItemKind.CANCEL, "cancel"),
            (QueueItemKind.ERROR, "error"),
            (QueueItemKind.GAP, "gap"),
            (QueueItemKind.SPEAK, "speak"),
        ):
            with pytest.raises(ValidationError) as ei:
                await q.put(label, kind=kind, deadline_at=_soon_deadline(30))
            assert ei.value.code == StageErrorCode.DEADLINE_EXCEEDED, kind

    @pytest.mark.asyncio
    async def test_no_task_leak_on_deadline(self) -> None:
        q: DeadlineAwareQueue[int] = DeadlineAwareQueue(capacity=1, name="leak")
        await q.put(0, kind=QueueItemKind.AUDIO, deadline_at=rfc3339_deadline_from_now(5))
        before = len(asyncio.all_tasks())
        for _ in range(5):
            with pytest.raises(ValidationError):
                await q.put(1, kind=QueueItemKind.AUDIO, deadline_at=_soon_deadline(20))
        await asyncio.sleep(0)  # flush callbacks
        after = len(asyncio.all_tasks())
        # Allow small variance but no growing leak of 5+ tasks
        assert after <= before + 2
