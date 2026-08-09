"""Commit barriers, revision ledgers, ordered publication, and deadline queues.

Pure/stateful orchestration primitives for stage.v1 Wave 2 (G2):

- RevisionLedger — monotonic product revisions + commit prefixes per scope
- CommittedDeltaRouter — only newly committed deltas route downstream
- PublicationBarrier — hold out-of-order completions; gap on earlier failure
- DeadlineAwareQueue — bounded put that waits only until deadline
- CancelController — stop admission, dispose, single cancelled, stale fence
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from src.stage_v1.models import (
    CancelScope,
    DroppedPayload,
    DropReason,
    GapPayload,
    StageErrorCode,
    StageKind,
)
from src.stage_v1.validation import (
    DedupeResult,
    Fence,
    RevisionState,
    ValidationError,
    check_deadline,
    parse_rfc3339,
)


@dataclass(frozen=True, slots=True)
class CommittedDelta:
    """Newly committed text span crossing the previous commit boundary."""

    scope_key: str
    stage_kind: StageKind
    revision: int
    char_start: int
    char_end: int
    text: str
    is_final: bool
    source_span_id: str | None = None
    target_span_id: str | None = None
    utterance_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def span_id(self) -> str:
        if self.target_span_id is not None:
            return self.target_span_id
        if self.source_span_id is not None:
            return self.source_span_id
        return f"{self.scope_key}:{self.char_start}:{self.char_end}"


@dataclass(frozen=True, slots=True)
class DroppedRevision:
    """Explicit coalescing of superseded uncommitted revisions."""

    scope_key: str
    stage_kind: StageKind
    revision_start: int
    revision_end: int
    reason: DropReason = DropReason.SUPERSEDED_UNCOMMITTED
    utterance_id: str | None = None
    source_span_id: str | None = None

    def to_payload(self) -> DroppedPayload:
        return DroppedPayload(
            reason=self.reason,
            revision_start=self.revision_start,
            revision_end=self.revision_end,
            stage_kind=self.stage_kind,
            utterance_id=self.utterance_id,
            source_span_id=self.source_span_id,
        )


@dataclass(frozen=True, slots=True)
class RevisionObserveResult:
    dedupe: DedupeResult
    delta: CommittedDelta | None = None
    dropped: DroppedRevision | None = None
    failed: bool = False
    error: ValidationError | None = None


@dataclass
class _ScopeLedger:
    state: RevisionState = field(default_factory=RevisionState)
    stage_kind: StageKind = StageKind.LISTEN
    utterance_id: str | None = None
    source_span_id: str | None = None
    target_span_id: str | None = None
    failed: bool = False
    fail_code: StageErrorCode | None = None
    fail_message: str | None = None
    last_uncommitted_revision: int | None = None
    pending_uncommitted_start: int | None = None


@dataclass
class RevisionLedger:
    """Per-scope product revision + commit-prefix ledger.

    Wraps RevisionState and surfaces newly committed deltas for routing.
    COMMIT_RETRACTION fails the utterance/scope.
    """

    _scopes: dict[str, _ScopeLedger] = field(default_factory=dict)

    def scope(
        self,
        scope_key: str,
        *,
        stage_kind: StageKind = StageKind.LISTEN,
        utterance_id: str | None = None,
        source_span_id: str | None = None,
        target_span_id: str | None = None,
    ) -> _ScopeLedger:
        entry = self._scopes.get(scope_key)
        if entry is None:
            entry = _ScopeLedger(
                stage_kind=stage_kind,
                utterance_id=utterance_id,
                source_span_id=source_span_id,
                target_span_id=target_span_id,
            )
            self._scopes[scope_key] = entry
        else:
            if utterance_id is not None:
                entry.utterance_id = utterance_id
            if source_span_id is not None:
                entry.source_span_id = source_span_id
            if target_span_id is not None:
                entry.target_span_id = target_span_id
        return entry

    def get_state(self, scope_key: str) -> RevisionState | None:
        entry = self._scopes.get(scope_key)
        return entry.state if entry is not None else None

    def is_failed(self, scope_key: str) -> bool:
        entry = self._scopes.get(scope_key)
        return bool(entry and entry.failed)

    def observe(
        self,
        scope_key: str,
        *,
        revision: int,
        text: str,
        committed_prefix_chars: int,
        is_final: bool = False,
        stage_kind: StageKind = StageKind.LISTEN,
        utterance_id: str | None = None,
        source_span_id: str | None = None,
        target_span_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        fail_on_error: bool = True,
    ) -> RevisionObserveResult:
        entry = self.scope(
            scope_key,
            stage_kind=stage_kind,
            utterance_id=utterance_id,
            source_span_id=source_span_id,
            target_span_id=target_span_id,
        )
        if entry.failed:
            return RevisionObserveResult(
                dedupe=DedupeResult.NEW,
                failed=True,
                error=ValidationError(
                    entry.fail_code or StageErrorCode.COMMIT_RETRACTION,
                    entry.fail_message or "scope already failed",
                ),
            )

        prev_committed = entry.state.committed_prefix_chars
        prev_uncommitted_start = entry.pending_uncommitted_start
        prev_last_uncommitted = entry.last_uncommitted_revision

        try:
            dedupe = entry.state.observe(
                revision=revision,
                text=text,
                committed_prefix_chars=committed_prefix_chars,
                is_final=is_final,
            )
        except ValidationError as exc:
            if fail_on_error and exc.code in (
                StageErrorCode.COMMIT_RETRACTION,
                StageErrorCode.SEQUENCE_GAP,
                StageErrorCode.DUPLICATE_CONFLICT,
            ):
                entry.failed = True
                entry.fail_code = exc.code
                entry.fail_message = exc.message
            return RevisionObserveResult(dedupe=DedupeResult.NEW, failed=True, error=exc)

        if dedupe is DedupeResult.IDEMPOTENT:
            return RevisionObserveResult(dedupe=dedupe)

        new_committed = entry.state.committed_prefix_chars
        delta: CommittedDelta | None = None
        dropped: DroppedRevision | None = None

        # Coalesce superseded uncommitted revisions when commit advances past them.
        if (
            new_committed > prev_committed
            and prev_uncommitted_start is not None
            and prev_last_uncommitted is not None
            and prev_uncommitted_start <= prev_last_uncommitted
        ):
            # Uncommitted-only coalescing when commit advances past pending partials.
            drop_end = revision - 1 if new_committed > prev_committed else prev_last_uncommitted
            if drop_end >= prev_uncommitted_start and drop_end >= 0:
                dropped = DroppedRevision(
                    scope_key=scope_key,
                    stage_kind=entry.stage_kind,
                    revision_start=prev_uncommitted_start,
                    revision_end=drop_end,
                    utterance_id=entry.utterance_id,
                    source_span_id=entry.source_span_id,
                )

        if new_committed > prev_committed:
            delta = CommittedDelta(
                scope_key=scope_key,
                stage_kind=entry.stage_kind,
                revision=revision,
                char_start=prev_committed,
                char_end=new_committed,
                text=text[prev_committed:new_committed],
                is_final=is_final and new_committed == len(text),
                source_span_id=entry.source_span_id,
                target_span_id=entry.target_span_id,
                utterance_id=entry.utterance_id,
                metadata=dict(metadata or {}),
            )
            entry.pending_uncommitted_start = None
            entry.last_uncommitted_revision = None
        else:
            # Uncommitted partial — track for possible coalescing.
            if entry.pending_uncommitted_start is None:
                entry.pending_uncommitted_start = revision
            entry.last_uncommitted_revision = revision

        return RevisionObserveResult(dedupe=dedupe, delta=delta, dropped=dropped)


@dataclass
class CommittedDeltaRouter:
    """Routes only newly committed deltas to translate/speak consumers.

    Cumulative partials that do not advance the commit boundary produce no
    downstream request. Optional coalescing emits DroppedRevision events.
    """

    ledger: RevisionLedger = field(default_factory=RevisionLedger)
    _routed_spans: list[CommittedDelta] = field(default_factory=list)
    _dropped: list[DroppedRevision] = field(default_factory=list)
    _span_counter: int = 0

    @property
    def routed(self) -> list[CommittedDelta]:
        return list(self._routed_spans)

    @property
    def dropped_events(self) -> list[DroppedRevision]:
        return list(self._dropped)

    def _next_span_id(self, prefix: str) -> str:
        self._span_counter += 1
        return f"{prefix}-{self._span_counter:04d}"

    def observe_listen_product(
        self,
        *,
        utterance_id: str,
        revision: int,
        text: str,
        committed_prefix_chars: int,
        is_final: bool = False,
        metadata: dict[str, Any] | None = None,
        assign_source_span_id: bool = True,
    ) -> RevisionObserveResult:
        scope_key = f"listen:{utterance_id}"
        result = self.ledger.observe(
            scope_key,
            revision=revision,
            text=text,
            committed_prefix_chars=committed_prefix_chars,
            is_final=is_final,
            stage_kind=StageKind.LISTEN,
            utterance_id=utterance_id,
            metadata=metadata,
        )
        if result.dropped is not None:
            self._dropped.append(result.dropped)
        if result.delta is not None:
            delta = result.delta
            if assign_source_span_id and delta.source_span_id is None:
                span_id = self._next_span_id(f"src-{utterance_id[:8]}")
                delta = CommittedDelta(
                    scope_key=delta.scope_key,
                    stage_kind=delta.stage_kind,
                    revision=delta.revision,
                    char_start=delta.char_start,
                    char_end=delta.char_end,
                    text=delta.text,
                    is_final=delta.is_final,
                    source_span_id=span_id,
                    target_span_id=delta.target_span_id,
                    utterance_id=delta.utterance_id,
                    metadata=delta.metadata,
                )
                result = RevisionObserveResult(
                    dedupe=result.dedupe,
                    delta=delta,
                    dropped=result.dropped,
                    failed=result.failed,
                    error=result.error,
                )
            self._routed_spans.append(delta)
        return result

    def observe_translate_product(
        self,
        *,
        source_span_id: str,
        target_span_id: str,
        revision: int,
        text: str,
        committed_prefix_chars: int,
        is_final: bool = False,
        utterance_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RevisionObserveResult:
        scope_key = f"translate:{source_span_id}"
        result = self.ledger.observe(
            scope_key,
            revision=revision,
            text=text,
            committed_prefix_chars=committed_prefix_chars,
            is_final=is_final,
            stage_kind=StageKind.TRANSLATE,
            utterance_id=utterance_id,
            source_span_id=source_span_id,
            target_span_id=target_span_id,
            metadata=metadata,
        )
        if result.dropped is not None:
            self._dropped.append(result.dropped)
        if result.delta is not None:
            self._routed_spans.append(result.delta)
        return result

    def translate_requests_from_listen(self) -> list[CommittedDelta]:
        """Deltas that should become translate.request units."""
        return [d for d in self._routed_spans if d.stage_kind is StageKind.LISTEN]

    def speak_requests_from_translate(self) -> list[CommittedDelta]:
        """Deltas that should become speak.request units."""
        return [d for d in self._routed_spans if d.stage_kind is StageKind.TRANSLATE]


class PublicationUnitStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    PUBLISHED = "published"
    GAPPED = "gapped"
    CANCELLED = "cancelled"


@dataclass
class PublicationUnit:
    """One ordered publication unit (utterance_sequence + target_span)."""

    utterance_sequence: int
    target_span_id: str
    fence: Fence | None = None
    payload: Any = None
    status: PublicationUnitStatus = PublicationUnitStatus.PENDING
    fail_reason: str | None = None
    deadline_at: str | None = None

    @property
    def key(self) -> tuple[int, str]:
        return (self.utterance_sequence, self.target_span_id)


@dataclass(frozen=True, slots=True)
class PublicationRelease:
    """Released publication: either a successful unit or an explicit gap."""

    kind: str  # "product" | "gap"
    unit: PublicationUnit
    gap: GapPayload | None = None


@dataclass
class PublicationBarrier:
    """Ordered publication barrier for same-session out-of-order completion.

    Holds later completions until earlier utterance_sequence / target-span
    units are ready. If an earlier unit fails, emits gap before advancing.
    Never releases cancelled/stale fence audio.
    """

    _next_sequence: int = 0
    _units: dict[tuple[int, str], PublicationUnit] = field(default_factory=dict)
    _order: list[tuple[int, str]] = field(default_factory=list)
    _sequence_to_keys: dict[int, list[tuple[int, str]]] = field(default_factory=dict)
    _published_keys: set[tuple[int, str]] = field(default_factory=set)
    _active_fence: Fence | None = None
    _cancelled: bool = False
    _released: list[PublicationRelease] = field(default_factory=list)

    def set_active_fence(self, fence: Fence | None) -> None:
        self._active_fence = fence

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def released(self) -> list[PublicationRelease]:
        return list(self._released)

    @property
    def next_expected_sequence(self) -> int:
        return self._next_sequence

    def register(
        self,
        *,
        utterance_sequence: int,
        target_span_id: str,
        fence: Fence | None = None,
        deadline_at: str | None = None,
    ) -> PublicationUnit:
        key = (utterance_sequence, target_span_id)
        if key in self._units:
            return self._units[key]
        unit = PublicationUnit(
            utterance_sequence=utterance_sequence,
            target_span_id=target_span_id,
            fence=fence,
            deadline_at=deadline_at,
        )
        self._units[key] = unit
        self._order.append(key)
        self._sequence_to_keys.setdefault(utterance_sequence, []).append(key)
        return unit

    def complete(
        self,
        *,
        utterance_sequence: int,
        target_span_id: str,
        payload: Any,
        fence: Fence | None = None,
        now: datetime | None = None,
    ) -> list[PublicationRelease]:
        """Mark a unit ready and drain any contiguous ready prefix."""
        if self._cancelled:
            raise ValidationError(
                StageErrorCode.STALE_FENCE,
                "publication barrier cancelled; refusing completion",
            )

        key = (utterance_sequence, target_span_id)
        unit = self._units.get(key)
        if unit is None:
            unit = self.register(
                utterance_sequence=utterance_sequence,
                target_span_id=target_span_id,
                fence=fence,
            )

        if key in self._published_keys:
            # Immutable/idempotent: already published — ignore duplicate.
            return []

        if unit.status in (
            PublicationUnitStatus.PUBLISHED,
            PublicationUnitStatus.GAPPED,
            PublicationUnitStatus.CANCELLED,
        ):
            return []

        # Fence check
        check_fence_pair = fence if fence is not None else unit.fence
        active = self._active_fence
        if (
            active is not None
            and check_fence_pair is not None
            and not active.matches(check_fence_pair, require_instance=True)
        ):
            raise ValidationError(
                StageErrorCode.STALE_FENCE,
                "completion fence does not match active publication fence",
            )

        if unit.deadline_at is not None:
            check_deadline(unit.deadline_at, now=now)

        unit.payload = payload
        unit.fence = check_fence_pair
        unit.status = PublicationUnitStatus.READY
        return self._drain(now=now)

    def fail(
        self,
        *,
        utterance_sequence: int,
        target_span_id: str,
        reason: str,
        fence: Fence | None = None,
    ) -> list[PublicationRelease]:
        """Mark earlier unit failed; drain will emit gap then continue."""
        if self._cancelled:
            raise ValidationError(
                StageErrorCode.STALE_FENCE,
                "publication barrier cancelled; refusing fail",
            )

        key = (utterance_sequence, target_span_id)
        unit = self._units.get(key)
        if unit is None:
            unit = self.register(
                utterance_sequence=utterance_sequence,
                target_span_id=target_span_id,
                fence=fence,
            )
        if key in self._published_keys:
            return []
        unit.status = PublicationUnitStatus.GAPPED
        unit.fail_reason = reason
        unit.fence = fence if fence is not None else unit.fence
        return self._drain()

    def cancel(self) -> None:
        """Cancel barrier: never release further audio from this fence."""
        self._cancelled = True
        for unit in self._units.values():
            if unit.status in (
                PublicationUnitStatus.PENDING,
                PublicationUnitStatus.READY,
            ):
                unit.status = PublicationUnitStatus.CANCELLED

    def accept_product(
        self,
        *,
        fence: Fence,
        utterance_sequence: int | None = None,
        target_span_id: str | None = None,
    ) -> None:
        """Orchestrator-side stale-product gate before barrier release."""
        if self._cancelled:
            raise ValidationError(
                StageErrorCode.STALE_FENCE,
                "barrier cancelled; product discarded",
            )
        if self._active_fence is not None and not self._active_fence.matches(
            fence, require_instance=True
        ):
            raise ValidationError(
                StageErrorCode.STALE_FENCE,
                "product fence does not match active attempt",
            )
        if (
            utterance_sequence is not None
            and target_span_id is not None
            and (utterance_sequence, target_span_id) in self._published_keys
        ):
            # Already published — idempotent accept, no re-release.
            return

    def _drain(self, *, now: datetime | None = None) -> list[PublicationRelease]:
        if self._cancelled:
            return []

        released: list[PublicationRelease] = []
        while True:
            # Find lowest registered sequence >= next_sequence that we know about.
            pending_seqs = sorted(
                seq for seq in self._sequence_to_keys if seq >= self._next_sequence
            )
            if not pending_seqs:
                break
            seq = pending_seqs[0]
            if seq != self._next_sequence:
                # Hole in registration — wait for earlier unit to be registered.
                break

            keys = self._sequence_to_keys.get(seq, [])
            if not keys:
                break

            # For a sequence, require all registered keys to be terminal (ready/gapped).
            statuses = [self._units[k].status for k in keys]
            if any(s is PublicationUnitStatus.PENDING for s in statuses):
                break
            if any(s is PublicationUnitStatus.CANCELLED for s in statuses):
                break

            for key in keys:
                unit = self._units[key]
                if key in self._published_keys:
                    continue
                if unit.status is PublicationUnitStatus.READY:
                    if unit.deadline_at is not None:
                        try:
                            check_deadline(unit.deadline_at, now=now)
                        except ValidationError:
                            unit.status = PublicationUnitStatus.GAPPED
                            unit.fail_reason = "deadline_exceeded_before_publish"
                            gap = GapPayload(
                                reason=unit.fail_reason,
                                utterance_id=None,
                                target_span_id=unit.target_span_id,
                            )
                            unit.status = PublicationUnitStatus.GAPPED
                            self._published_keys.add(key)
                            rel = PublicationRelease(kind="gap", unit=unit, gap=gap)
                            released.append(rel)
                            self._released.append(rel)
                            continue
                    unit.status = PublicationUnitStatus.PUBLISHED
                    self._published_keys.add(key)
                    rel = PublicationRelease(kind="product", unit=unit)
                    released.append(rel)
                    self._released.append(rel)
                elif unit.status is PublicationUnitStatus.GAPPED:
                    gap = GapPayload(
                        reason=unit.fail_reason or "unit_failed",
                        utterance_id=None,
                        target_span_id=unit.target_span_id,
                    )
                    self._published_keys.add(key)
                    rel = PublicationRelease(kind="gap", unit=unit, gap=gap)
                    released.append(rel)
                    self._released.append(rel)

            self._next_sequence = seq + 1

        return released


class QueueItemKind(StrEnum):
    """Semantic kinds for backpressure policy."""

    AUDIO = "audio"
    COMMITTED_SPAN = "committed_span"
    SPEAK = "speak"
    EOS = "eos"
    CANCEL = "cancel"
    ERROR = "error"
    GAP = "gap"
    DROPPED = "dropped"
    UNCOMMITTED_REVISION = "uncommitted_revision"
    OTHER = "other"


# Kinds that MUST NOT be silently dropped.
PROTECTED_KINDS = frozenset(
    {
        QueueItemKind.AUDIO,
        QueueItemKind.COMMITTED_SPAN,
        QueueItemKind.SPEAK,
        QueueItemKind.EOS,
        QueueItemKind.CANCEL,
        QueueItemKind.ERROR,
        QueueItemKind.GAP,
    }
)


@dataclass(frozen=True, slots=True)
class QueueItem[T]:
    value: T
    kind: QueueItemKind = QueueItemKind.OTHER
    deadline_at: str | None = None
    enqueued_at_mono: float = 0.0
    bytes_size: int = 0
    revision: int | None = None
    coalesce_key: str | None = None


@dataclass
class DeadlineAwareQueue[T]:
    """Bounded asyncio queue; put waits only until deadline.

    - Never silently drops protected kinds (audio, committed spans, speak,
      eos, cancel, errors, gaps).
    - Permitted coalescing: superseded uncommitted revisions emit dropped.
    - Timeout / expired deadline → DEADLINE_EXCEEDED ValidationError.
    """

    capacity: int
    max_bytes: int | None = None
    name: str = "queue"

    _queue: asyncio.Queue[QueueItem[T] | None] = field(init=False)
    _items: int = field(default=0, init=False)
    _bytes: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)
    _high_water_items: int = field(default=0, init=False)
    _high_water_bytes: int = field(default=0, init=False)
    _dropped: list[DroppedRevision] = field(default_factory=list, init=False)
    _coalesce_index: dict[str, int] = field(default_factory=dict, init=False)
    # Pending uncommitted items keyed by coalesce_key for in-queue replacement.
    _uncommitted_slots: dict[str, QueueItem[T]] = field(default_factory=dict, init=False)
    _getters_woken: int = field(default=0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _not_full: asyncio.Condition = field(init=False)
    _not_empty: asyncio.Condition = field(init=False)
    _buffer: list[QueueItem[T] | None] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._queue = asyncio.Queue(maxsize=self.capacity)
        self._not_full = asyncio.Condition()
        self._not_empty = asyncio.Condition()

    @property
    def qsize(self) -> int:
        return self._items

    @property
    def byte_size(self) -> int:
        return self._bytes

    @property
    def high_water_items(self) -> int:
        return self._high_water_items

    @property
    def high_water_bytes(self) -> int:
        return self._high_water_bytes

    @property
    def capacity_items(self) -> int:
        return self.capacity

    @property
    def dropped_events(self) -> list[DroppedRevision]:
        return list(self._dropped)

    @property
    def closed(self) -> bool:
        return self._closed

    def memory_bound_bytes(self, *, max_frame_bytes: int) -> int:
        """Upper bound from configured capacity and frame limits."""
        per_item = max_frame_bytes
        if self.max_bytes is not None:
            return min(self.capacity * per_item, self.max_bytes)
        return self.capacity * per_item

    async def put(
        self,
        value: T,
        *,
        kind: QueueItemKind = QueueItemKind.OTHER,
        deadline_at: str | None = None,
        bytes_size: int = 0,
        revision: int | None = None,
        coalesce_key: str | None = None,
        now: datetime | None = None,
        stage_kind: StageKind = StageKind.LISTEN,
        utterance_id: str | None = None,
    ) -> DroppedRevision | None:
        """Enqueue with deadline-bounded wait. Returns dropped event if coalesced."""
        terminal = (QueueItemKind.CANCEL, QueueItemKind.ERROR, QueueItemKind.EOS)
        if self._closed and kind not in terminal:
            raise ValidationError(
                StageErrorCode.CANCELLED,
                f"queue {self.name} closed; refusing {kind}",
            )

        check_deadline(deadline_at, now=now)

        item = QueueItem(
            value=value,
            kind=kind,
            deadline_at=deadline_at,
            enqueued_at_mono=time.monotonic(),
            bytes_size=max(0, bytes_size),
            revision=revision,
            coalesce_key=coalesce_key,
        )

        # Coalesce superseded uncommitted revisions in-place when possible.
        dropped: DroppedRevision | None = None
        if (
            kind is QueueItemKind.UNCOMMITTED_REVISION
            and coalesce_key is not None
            and revision is not None
        ):
            async with self._not_full:
                existing = self._uncommitted_slots.get(coalesce_key)
                if existing is not None and existing.revision is not None:
                    # Replace in buffer if still present.
                    replaced = False
                    for i, buffered in enumerate(self._buffer):
                        if (
                            buffered is not None
                            and buffered.coalesce_key == coalesce_key
                            and buffered.kind is QueueItemKind.UNCOMMITTED_REVISION
                        ):
                            old_rev = buffered.revision or 0
                            self._bytes -= buffered.bytes_size
                            self._bytes += item.bytes_size
                            self._buffer[i] = item
                            self._uncommitted_slots[coalesce_key] = item
                            dropped = DroppedRevision(
                                scope_key=coalesce_key,
                                stage_kind=stage_kind,
                                revision_start=old_rev,
                                revision_end=old_rev,
                                utterance_id=utterance_id,
                            )
                            self._dropped.append(dropped)
                            replaced = True
                            break
                    if replaced:
                        return dropped

        timeout = self._remaining_seconds(deadline_at, now=now)

        async with self._not_full:
            while self._is_full(item) and not self._closed:
                if timeout is not None and timeout <= 0:
                    raise ValidationError(
                        StageErrorCode.DEADLINE_EXCEEDED,
                        f"deadline exceeded waiting to enqueue on {self.name}",
                    )
                try:
                    await asyncio.wait_for(self._not_full.wait(), timeout=timeout)
                except TimeoutError as exc:
                    raise ValidationError(
                        StageErrorCode.DEADLINE_EXCEEDED,
                        f"deadline exceeded waiting to enqueue on {self.name}",
                    ) from exc
                # Re-check deadline after wake.
                check_deadline(deadline_at)
                timeout = self._remaining_seconds(deadline_at)

            if self._closed and kind not in (
                QueueItemKind.CANCEL,
                QueueItemKind.ERROR,
                QueueItemKind.EOS,
            ):
                raise ValidationError(
                    StageErrorCode.CANCELLED,
                    f"queue {self.name} closed during put",
                )

            # Protected kinds never drop; if still full after wait, that is an error
            # only when capacity is zero-like — with correct wait, we only proceed
            # when space exists OR item is terminal that we must accept by policy.
            if self._is_full(item):
                if kind in PROTECTED_KINDS:
                    # Last resort: still full after deadline wait path should have
                    # raised; if closed-full, surface resource exhaustion.
                    raise ValidationError(
                        StageErrorCode.RESOURCE_EXHAUSTED,
                        f"queue {self.name} full; refusing silent drop of {kind}",
                    )
                if kind is QueueItemKind.UNCOMMITTED_REVISION and revision is not None:
                    dropped = DroppedRevision(
                        scope_key=coalesce_key or self.name,
                        stage_kind=stage_kind,
                        revision_start=revision,
                        revision_end=revision,
                        utterance_id=utterance_id,
                    )
                    self._dropped.append(dropped)
                    return dropped
                raise ValidationError(
                    StageErrorCode.RESOURCE_EXHAUSTED,
                    f"queue {self.name} full",
                )

            self._buffer.append(item)
            self._items += 1
            self._bytes += item.bytes_size
            self._high_water_items = max(self._high_water_items, self._items)
            self._high_water_bytes = max(self._high_water_bytes, self._bytes)
            if kind is QueueItemKind.UNCOMMITTED_REVISION and coalesce_key is not None:
                self._uncommitted_slots[coalesce_key] = item

        async with self._not_empty:
            self._not_empty.notify()
        return dropped

    async def get(self, *, deadline_at: str | None = None, now: datetime | None = None) -> T:
        check_deadline(deadline_at, now=now)
        timeout = self._remaining_seconds(deadline_at, now=now)

        async with self._not_empty:
            while not self._buffer and not self._closed:
                if timeout is not None and timeout <= 0:
                    raise ValidationError(
                        StageErrorCode.DEADLINE_EXCEEDED,
                        f"deadline exceeded waiting to dequeue from {self.name}",
                    )
                try:
                    await asyncio.wait_for(self._not_empty.wait(), timeout=timeout)
                except TimeoutError as exc:
                    raise ValidationError(
                        StageErrorCode.DEADLINE_EXCEEDED,
                        f"deadline exceeded waiting to dequeue from {self.name}",
                    ) from exc
                check_deadline(deadline_at)
                timeout = self._remaining_seconds(deadline_at)

            if not self._buffer:
                raise ValidationError(
                    StageErrorCode.CANCELLED,
                    f"queue {self.name} closed and empty",
                )

            item = self._buffer.pop(0)
            if item is None:
                raise ValidationError(
                    StageErrorCode.CANCELLED,
                    f"queue {self.name} sentinel",
                )
            self._items = max(0, self._items - 1)
            self._bytes = max(0, self._bytes - item.bytes_size)
            if (
                item.kind is QueueItemKind.UNCOMMITTED_REVISION
                and item.coalesce_key is not None
                and self._uncommitted_slots.get(item.coalesce_key) is item
            ):
                self._uncommitted_slots.pop(item.coalesce_key, None)

        async with self._not_full:
            self._not_full.notify()
        return item.value

    async def get_item(
        self, *, deadline_at: str | None = None, now: datetime | None = None
    ) -> QueueItem[T]:
        check_deadline(deadline_at, now=now)
        timeout = self._remaining_seconds(deadline_at, now=now)

        async with self._not_empty:
            while not self._buffer and not self._closed:
                if timeout is not None and timeout <= 0:
                    raise ValidationError(
                        StageErrorCode.DEADLINE_EXCEEDED,
                        f"deadline exceeded waiting to dequeue from {self.name}",
                    )
                try:
                    await asyncio.wait_for(self._not_empty.wait(), timeout=timeout)
                except TimeoutError as exc:
                    raise ValidationError(
                        StageErrorCode.DEADLINE_EXCEEDED,
                        f"deadline exceeded waiting to dequeue from {self.name}",
                    ) from exc
                check_deadline(deadline_at)
                timeout = self._remaining_seconds(deadline_at)

            if not self._buffer:
                raise ValidationError(
                    StageErrorCode.CANCELLED,
                    f"queue {self.name} closed and empty",
                )
            item = self._buffer.pop(0)
            if item is None:
                raise ValidationError(
                    StageErrorCode.CANCELLED,
                    f"queue {self.name} sentinel",
                )
            self._items = max(0, self._items - 1)
            self._bytes = max(0, self._bytes - item.bytes_size)
            if (
                item.kind is QueueItemKind.UNCOMMITTED_REVISION
                and item.coalesce_key is not None
                and self._uncommitted_slots.get(item.coalesce_key) is item
            ):
                self._uncommitted_slots.pop(item.coalesce_key, None)

        async with self._not_full:
            self._not_full.notify()
        return item

    async def close(self) -> None:
        async with self._not_full:
            self._closed = True
            self._not_full.notify_all()
        async with self._not_empty:
            self._not_empty.notify_all()

    def _is_full(self, item: QueueItem[T]) -> bool:
        if self._items >= self.capacity:
            return True
        if self.max_bytes is None:
            return False
        if self._bytes + item.bytes_size <= self.max_bytes:
            return False
        # Full when adding would exceed byte budget (empty queue still accepts
        # a single oversized frame only if it alone fits — otherwise block).
        return self._items > 0 or item.bytes_size > self.max_bytes

    @staticmethod
    def _remaining_seconds(
        deadline_at: str | None, *, now: datetime | None = None
    ) -> float | None:
        if deadline_at is None:
            return None
        current = now or datetime.now(UTC)
        remaining = (parse_rfc3339(deadline_at) - current).total_seconds()
        return max(0.0, remaining)


@dataclass
class CancelController:
    """Attempt/utterance/session cancel with single cancelled emission + fencing."""

    active_fence: Fence
    scope: CancelScope = CancelScope.ATTEMPT
    _cancelled: bool = False
    _cancelled_emitted: bool = False
    _disposed: bool = False
    _cancel_reason: str | None = None
    _stale_fences: list[Fence] = field(default_factory=list)
    _on_dispose: list[Callable[[], None]] = field(default_factory=list)
    _admission_stopped: bool = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def disposed(self) -> bool:
        return self._disposed

    @property
    def cancelled_emitted(self) -> bool:
        return self._cancelled_emitted

    @property
    def admission_stopped(self) -> bool:
        return self._admission_stopped

    def on_dispose(self, callback: Callable[[], None]) -> None:
        self._on_dispose.append(callback)

    def stop_admission(self) -> None:
        self._admission_stopped = True

    def check_admission(self, candidate: Fence | None = None) -> None:
        if self._admission_stopped or self._cancelled:
            raise ValidationError(
                StageErrorCode.CANCELLED,
                "admission stopped for cancelled scope",
            )
        if candidate is not None:
            self.check_fence(candidate)

    def check_fence(self, candidate: Fence, *, require_instance: bool = True) -> None:
        if self._cancelled:
            # After cancel, active fence is stale for new products.
            raise ValidationError(
                StageErrorCode.STALE_FENCE,
                "fence belongs to cancelled attempt",
            )
        if not self.active_fence.matches(candidate, require_instance=require_instance):
            raise ValidationError(
                StageErrorCode.STALE_FENCE,
                "event fence does not match active attempt",
            )
        for stale in self._stale_fences:
            if stale.matches(candidate, require_instance=require_instance):
                raise ValidationError(
                    StageErrorCode.STALE_FENCE,
                    "event fence matches retained stale fence",
                )

    def accept_late_product(self, candidate: Fence) -> None:
        """Reject late products after cancel (orchestrator discard path)."""
        if self._cancelled:
            raise ValidationError(
                StageErrorCode.STALE_FENCE,
                "late product after cancel discarded",
            )
        self.check_fence(candidate)

    def cancel(
        self,
        *,
        reason: str,
        scope: CancelScope | None = None,
    ) -> dict[str, Any] | None:
        """Cancel scope. Returns cancelled payload once; subsequent calls return None."""
        if scope is not None:
            self.scope = scope
        self._admission_stopped = True
        self._cancelled = True
        self._cancel_reason = reason

        if not self._disposed:
            for cb in list(self._on_dispose):
                cb()
            self._disposed = True

        # Retain stale fence so reconnect cannot accept old products.
        self._stale_fences.append(self.active_fence)

        if self._cancelled_emitted:
            return None
        self._cancelled_emitted = True
        return {
            "scope": self.scope.value
            if isinstance(self.scope, CancelScope)
            else str(self.scope),
            "reason": reason,
            "disposed": True,
        }

    def rotate_fence(self, new_fence: Fence) -> None:
        """Open a fresh attempt fence; previous becomes stale."""
        if not any(
            s.matches(self.active_fence, require_instance=True) for s in self._stale_fences
        ):
            self._stale_fences.append(self.active_fence)
        self.active_fence = new_fence
        self._cancelled = False
        self._cancelled_emitted = False
        self._disposed = False
        self._admission_stopped = False
        self._cancel_reason = None


def new_fence(
    *,
    session_id: str,
    owner_generation: int = 0,
    stage_kind: str = "listen",
    stage_id: str = "stage",
    attempt_id: str | None = None,
    cancel_id: str | None = None,
    stage_instance_id: str | None = None,
) -> Fence:
    return Fence(
        session_id=session_id,
        owner_generation=owner_generation,
        stage_kind=stage_kind,
        stage_id=stage_id,
        attempt_id=attempt_id or str(uuid4()),
        cancel_id=cancel_id or str(uuid4()),
        stage_instance_id=stage_instance_id or str(uuid4()),
    )


def rfc3339_deadline_from_now(seconds: float) -> str:
    from datetime import timedelta

    dt = datetime.now(UTC) + timedelta(seconds=seconds)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
