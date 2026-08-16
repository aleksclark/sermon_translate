"""StageSession: coordinates fence IDs, commit barriers, cancel, and queues.

Wiring helper for stage.v1 orchestration. Pure enough that ComposedPipeline
(or a later orchestrator) can adopt it without owning transport details.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from src.pipelines.commit_barrier import (
    CancelController,
    CommittedDelta,
    CommittedDeltaRouter,
    DeadlineAwareQueue,
    PublicationBarrier,
    PublicationRelease,
    QueueItemKind,
    RevisionLedger,
    RevisionObserveResult,
    new_fence,
    rfc3339_deadline_from_now,
)
from src.stage_v1.models import CancelScope, GapPayload, StageErrorCode, StageKind
from src.stage_v1.validation import Fence, ValidationError, check_deadline


@dataclass
class StageSessionConfig:
    """Queue and capacity knobs for a stage session attempt."""

    audio_queue_capacity: int = 32
    product_queue_capacity: int = 32
    speak_queue_capacity: int = 32
    max_inflight_bytes: int | None = None
    max_frame_bytes: int = 65_536
    default_deadline_s: float = 30.0


@dataclass
class StageSession:
    """Per-attempt session coordinating revisions, publication, cancel, queues.

    Owns:
    - active fence (session/attempt/cancel/instance)
    - RevisionLedger + CommittedDeltaRouter
    - PublicationBarrier
    - CancelController
    - deadline-aware bounded queues for audio / products / speak
    """

    session_id: str
    owner_generation: int = 0
    stage_kind: StageKind = StageKind.LISTEN
    stage_id: str = "stage"
    attempt_id: str = field(default_factory=lambda: str(uuid4()))
    cancel_id: str = field(default_factory=lambda: str(uuid4()))
    stage_instance_id: str = field(default_factory=lambda: str(uuid4()))
    config: StageSessionConfig = field(default_factory=StageSessionConfig)

    fence: Fence = field(init=False)
    ledger: RevisionLedger = field(default_factory=RevisionLedger)
    router: CommittedDeltaRouter = field(init=False)
    publication: PublicationBarrier = field(default_factory=PublicationBarrier)
    cancel_ctrl: CancelController = field(init=False)

    audio_in: DeadlineAwareQueue[bytes] = field(init=False)
    product_out: DeadlineAwareQueue[Any] = field(init=False)
    speak_out: DeadlineAwareQueue[Any] = field(init=False)

    utterance_id: str | None = None
    utterance_sequence: int | None = None
    _events: list[dict[str, Any]] = field(default_factory=list)
    _closed: bool = False

    def __post_init__(self) -> None:
        self.fence = Fence(
            session_id=self.session_id,
            owner_generation=self.owner_generation,
            stage_kind=(
                self.stage_kind.value
                if isinstance(self.stage_kind, StageKind)
                else str(self.stage_kind)
            ),
            stage_id=self.stage_id,
            attempt_id=self.attempt_id,
            cancel_id=self.cancel_id,
            stage_instance_id=self.stage_instance_id,
        )
        self.router = CommittedDeltaRouter(ledger=self.ledger)
        self.cancel_ctrl = CancelController(active_fence=self.fence)
        self.publication.set_active_fence(self.fence)

        cfg = self.config
        max_bytes = cfg.max_inflight_bytes
        self.audio_in = DeadlineAwareQueue(
            capacity=cfg.audio_queue_capacity,
            max_bytes=max_bytes,
            name="audio_in",
        )
        self.product_out = DeadlineAwareQueue(
            capacity=cfg.product_queue_capacity,
            max_bytes=max_bytes,
            name="product_out",
        )
        self.speak_out = DeadlineAwareQueue(
            capacity=cfg.speak_queue_capacity,
            max_bytes=max_bytes,
            name="speak_out",
        )

        def _dispose_queues() -> None:
            # Synchronously mark closed; async close awaited by cancel_async.
            self.audio_in._closed = True  # noqa: SLF001 — intentional dispose
            self.product_out._closed = True  # noqa: SLF001
            self.speak_out._closed = True  # noqa: SLF001
            self.publication.cancel()

        self.cancel_ctrl.on_dispose(_dispose_queues)

    # --- fence / identity -------------------------------------------------

    def bind_utterance(self, utterance_id: str, utterance_sequence: int) -> None:
        self.utterance_id = utterance_id
        self.utterance_sequence = utterance_sequence

    def check_inbound_fence(self, candidate: Fence, *, require_instance: bool = True) -> None:
        self.cancel_ctrl.check_fence(candidate, require_instance=require_instance)

    def check_outbound_product(self, candidate: Fence) -> None:
        """Reject late/stale products before publication."""
        self.cancel_ctrl.accept_late_product(candidate)
        self.publication.accept_product(fence=candidate)

    # --- deadlines --------------------------------------------------------

    def default_deadline_at(self) -> str:
        return rfc3339_deadline_from_now(self.config.default_deadline_s)

    def ensure_deadline(self, deadline_at: str | None, *, now: datetime | None = None) -> str:
        dl = deadline_at or self.default_deadline_at()
        check_deadline(dl, now=now)
        return dl

    # --- audio admission --------------------------------------------------

    async def put_audio(
        self,
        pcm: bytes,
        *,
        deadline_at: str | None = None,
        now: datetime | None = None,
    ) -> None:
        self.cancel_ctrl.check_admission(self.fence)
        dl = self.ensure_deadline(deadline_at, now=now)
        await self.audio_in.put(
            pcm,
            kind=QueueItemKind.AUDIO,
            deadline_at=dl,
            bytes_size=len(pcm),
            now=now,
        )

    async def get_audio(self, *, deadline_at: str | None = None) -> bytes:
        dl = deadline_at or self.default_deadline_at()
        return await self.audio_in.get(deadline_at=dl)

    # --- listen product / commit routing ----------------------------------

    def observe_listen_product(
        self,
        *,
        revision: int,
        text: str,
        committed_prefix_chars: int,
        is_final: bool = False,
        utterance_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RevisionObserveResult:
        self.cancel_ctrl.check_admission()
        uid = utterance_id or self.utterance_id
        if uid is None:
            raise ValidationError(
                StageErrorCode.INVALID_ARGUMENT,
                "utterance_id required for listen product",
            )
        result = self.router.observe_listen_product(
            utterance_id=uid,
            revision=revision,
            text=text,
            committed_prefix_chars=committed_prefix_chars,
            is_final=is_final,
            metadata=metadata,
        )
        if result.failed and result.error is not None:
            self._events.append(
                {
                    "event_type": "error",
                    "code": result.error.code.value,
                    "message": result.error.message,
                    "utterance_id": uid,
                }
            )
            raise result.error
        if result.dropped is not None:
            self._events.append(
                {
                    "event_type": "dropped",
                    "payload": result.dropped.to_payload().model_dump(mode="json"),
                }
            )
        if result.delta is not None:
            self._events.append(
                {
                    "event_type": "committed_delta",
                    "direction": "listen_to_translate",
                    "delta": {
                        "source_span_id": result.delta.source_span_id,
                        "char_start": result.delta.char_start,
                        "char_end": result.delta.char_end,
                        "text": result.delta.text,
                        "revision": result.delta.revision,
                        "is_final": result.delta.is_final,
                    },
                }
            )
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
        self.cancel_ctrl.check_admission()
        result = self.router.observe_translate_product(
            source_span_id=source_span_id,
            target_span_id=target_span_id,
            revision=revision,
            text=text,
            committed_prefix_chars=committed_prefix_chars,
            is_final=is_final,
            utterance_id=utterance_id or self.utterance_id,
            metadata=metadata,
        )
        if result.failed and result.error is not None:
            self._events.append(
                {
                    "event_type": "error",
                    "code": result.error.code.value,
                    "message": result.error.message,
                }
            )
            raise result.error
        if result.dropped is not None:
            self._events.append(
                {
                    "event_type": "dropped",
                    "payload": result.dropped.to_payload().model_dump(mode="json"),
                }
            )
        if result.delta is not None:
            self._events.append(
                {
                    "event_type": "committed_delta",
                    "direction": "translate_to_speak",
                    "delta": {
                        "source_span_id": result.delta.source_span_id,
                        "target_span_id": result.delta.target_span_id,
                        "char_start": result.delta.char_start,
                        "char_end": result.delta.char_end,
                        "text": result.delta.text,
                        "revision": result.delta.revision,
                        "is_final": result.delta.is_final,
                    },
                }
            )
        return result

    def pending_translate_deltas(self) -> list[CommittedDelta]:
        return self.router.translate_requests_from_listen()

    def pending_speak_deltas(self) -> list[CommittedDelta]:
        return self.router.speak_requests_from_translate()

    # --- ordered publication ----------------------------------------------

    def register_publication_unit(
        self,
        *,
        utterance_sequence: int,
        target_span_id: str,
        deadline_at: str | None = None,
    ) -> None:
        self.publication.register(
            utterance_sequence=utterance_sequence,
            target_span_id=target_span_id,
            fence=self.fence,
            deadline_at=deadline_at,
        )

    def complete_publication(
        self,
        *,
        utterance_sequence: int,
        target_span_id: str,
        payload: Any,
        fence: Fence | None = None,
        now: datetime | None = None,
    ) -> list[PublicationRelease]:
        cand = fence or self.fence
        self.check_outbound_product(cand)
        releases = self.publication.complete(
            utterance_sequence=utterance_sequence,
            target_span_id=target_span_id,
            payload=payload,
            fence=cand,
            now=now,
        )
        for rel in releases:
            self._record_release(rel)
        return releases

    def fail_publication(
        self,
        *,
        utterance_sequence: int,
        target_span_id: str,
        reason: str,
    ) -> list[PublicationRelease]:
        releases = self.publication.fail(
            utterance_sequence=utterance_sequence,
            target_span_id=target_span_id,
            reason=reason,
            fence=self.fence,
        )
        for rel in releases:
            self._record_release(rel)
        return releases

    def _record_release(self, rel: PublicationRelease) -> None:
        if rel.kind == "gap":
            gap = rel.gap or GapPayload(
                reason=rel.unit.fail_reason or "unit_failed",
                target_span_id=rel.unit.target_span_id,
            )
            self._events.append(
                {
                    "event_type": "gap",
                    "payload": gap.model_dump(mode="json"),
                    "utterance_sequence": rel.unit.utterance_sequence,
                    "target_span_id": rel.unit.target_span_id,
                }
            )
        else:
            self._events.append(
                {
                    "event_type": "published",
                    "utterance_sequence": rel.unit.utterance_sequence,
                    "target_span_id": rel.unit.target_span_id,
                    "payload": rel.unit.payload,
                }
            )

    async def put_speak_payload(
        self,
        payload: Any,
        *,
        deadline_at: str | None = None,
        bytes_size: int = 0,
        now: datetime | None = None,
    ) -> None:
        self.cancel_ctrl.check_admission()
        dl = self.ensure_deadline(deadline_at, now=now)
        await self.speak_out.put(
            payload,
            kind=QueueItemKind.SPEAK,
            deadline_at=dl,
            bytes_size=bytes_size,
            now=now,
        )

    # --- cancel -----------------------------------------------------------

    def cancel(
        self,
        *,
        reason: str = "cancel",
        scope: CancelScope = CancelScope.ATTEMPT,
    ) -> dict[str, Any] | None:
        """Stop admission, dispose attempt state, emit cancelled once."""
        payload = self.cancel_ctrl.cancel(reason=reason, scope=scope)
        if payload is not None:
            self._events.append({"event_type": "cancelled", "payload": payload})
        return payload

    async def cancel_async(
        self,
        *,
        reason: str = "cancel",
        scope: CancelScope = CancelScope.ATTEMPT,
    ) -> dict[str, Any] | None:
        payload = self.cancel(reason=reason, scope=scope)
        await self.audio_in.close()
        await self.product_out.close()
        await self.speak_out.close()
        return payload

    def reject_late_product(self, candidate: Fence) -> None:
        """Explicit late-after-cancel rejection (STALE_FENCE)."""
        self.cancel_ctrl.accept_late_product(candidate)

    # --- reconnect / new attempt ------------------------------------------

    def open_fresh_attempt(
        self,
        *,
        attempt_id: str | None = None,
        cancel_id: str | None = None,
        stage_instance_id: str | None = None,
    ) -> Fence:
        """Fence rotation after cancel/disconnect; previous fence stays stale."""
        new = new_fence(
            session_id=self.session_id,
            owner_generation=self.owner_generation,
            stage_kind=(
                self.stage_kind.value
                if isinstance(self.stage_kind, StageKind)
                else str(self.stage_kind)
            ),
            stage_id=self.stage_id,
            attempt_id=attempt_id,
            cancel_id=cancel_id,
            stage_instance_id=stage_instance_id,
        )
        self.attempt_id = new.attempt_id
        self.cancel_id = new.cancel_id
        self.stage_instance_id = new.stage_instance_id or self.stage_instance_id
        self.fence = new
        self.cancel_ctrl.rotate_fence(new)
        self.publication = PublicationBarrier()
        self.publication.set_active_fence(new)
        # Fresh queues for the new attempt.
        cfg = self.config
        self.audio_in = DeadlineAwareQueue(
            capacity=cfg.audio_queue_capacity,
            max_bytes=cfg.max_inflight_bytes,
            name="audio_in",
        )
        self.product_out = DeadlineAwareQueue(
            capacity=cfg.product_queue_capacity,
            max_bytes=cfg.max_inflight_bytes,
            name="product_out",
        )
        self.speak_out = DeadlineAwareQueue(
            capacity=cfg.speak_queue_capacity,
            max_bytes=cfg.max_inflight_bytes,
            name="speak_out",
        )

        def _dispose_queues() -> None:
            self.audio_in._closed = True  # noqa: SLF001
            self.product_out._closed = True  # noqa: SLF001
            self.speak_out._closed = True  # noqa: SLF001
            self.publication.cancel()

        self.cancel_ctrl.on_dispose(_dispose_queues)
        return new

    # --- observability ----------------------------------------------------

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "attempt_id": self.attempt_id,
            "cancel_id": self.cancel_id,
            "stage_instance_id": self.stage_instance_id,
            "cancelled": self.cancel_ctrl.cancelled,
            "admission_stopped": self.cancel_ctrl.admission_stopped,
            "audio_qsize": self.audio_in.qsize,
            "audio_high_water": self.audio_in.high_water_items,
            "audio_high_water_bytes": self.audio_in.high_water_bytes,
            "audio_capacity": self.audio_in.capacity_items,
            "speak_qsize": self.speak_out.qsize,
            "speak_high_water": self.speak_out.high_water_items,
            "publication_next": self.publication.next_expected_sequence,
            "publication_released": len(self.publication.released),
            "routed_deltas": len(self.router.routed),
            "dropped": len(self.router.dropped_events),
            "memory_bound_audio_bytes": self.audio_in.memory_bound_bytes(
                max_frame_bytes=self.config.max_frame_bytes
            ),
        }

    def memory_bound_bytes(self) -> int:
        """Capacity-derived upper bound across session queues."""
        frame = self.config.max_frame_bytes
        return (
            self.audio_in.memory_bound_bytes(max_frame_bytes=frame)
            + self.product_out.memory_bound_bytes(max_frame_bytes=frame)
            + self.speak_out.memory_bound_bytes(max_frame_bytes=frame)
        )


__all__ = [
    "StageSession",
    "StageSessionConfig",
]
