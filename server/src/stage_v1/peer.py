from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from src.stage_v1.models import (
    DEFAULT_MAX_FRAME_BYTES,
    AcceptedPayload,
    ArtifactDigestStatus,
    AudioFormat,
    EventEnvelope,
    EventType,
    HelloPayload,
    LimitsAdvertised,
    ProvenanceBlock,
    StageErrorCode,
    StageKind,
    parse_event,
)
from src.stage_v1.provenance import provenance_id_from_block
from src.stage_v1.validation import (
    Fence,
    MessageIdTracker,
    ValidationError,
    check_deadline,
    check_fence,
    check_schema_version,
)


class PeerMode(StrEnum):
    NORMAL = "normal"
    DELAY = "delay"
    REORDER = "reorder"
    FAIL = "fail"
    EMIT_AFTER_CANCEL = "emit_after_cancel"


@dataclass
class ScriptedResponse:
    """A pre-scripted outbound event the peer will emit after a trigger."""

    after_inbound_types: frozenset[str] | None = None
    delay_s: float = 0.0
    event: dict[str, Any] | EventEnvelope | None = None
    binary_pcm: bytes | None = None
    fail_code: StageErrorCode | None = None


@dataclass
class CreditWindow:
    stream_id: str
    available_events: int
    available_bytes: int
    credit_epoch: int = 0
    oldest_queue_age_ms: int = 0

    def consume(self, *, events: int = 1, bytes_: int = 0) -> None:
        if events > self.available_events or bytes_ > self.available_bytes:
            raise ValidationError(
                StageErrorCode.RESOURCE_EXHAUSTED,
                f"credit exceeded on stream={self.stream_id}",
            )
        self.available_events -= events
        self.available_bytes -= bytes_

    def grant(self, *, events: int = 0, bytes_: int = 0) -> None:
        self.available_events += events
        self.available_bytes += bytes_
        self.credit_epoch += 1


@dataclass
class ScriptedStagePeer:
    """In-memory scripted stage peer for conformance tests.

    Obeys application credit windows and can deliberately delay, reorder,
    fail, or emit product after cancel for fencing tests.
    """

    stage_kind: StageKind = StageKind.LISTEN
    stage_id: str = "scripted-listen"
    stage_version: str = "0.1.0"
    model_revision: str = "scripted-1"
    model_artifact_digest: str = "sha256:" + ("ab" * 32)
    boot_id: str = field(default_factory=lambda: str(uuid4()))
    stage_instance_id: str = field(default_factory=lambda: str(uuid4()))
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES
    initial_event_credits: int = 32
    initial_byte_credits: int = DEFAULT_MAX_FRAME_BYTES * 32
    mode: PeerMode = PeerMode.NORMAL
    scripted: list[ScriptedResponse] = field(default_factory=list)

    _outbound: asyncio.Queue[EventEnvelope | tuple[EventEnvelope, bytes]] = field(
        default_factory=asyncio.Queue
    )
    _inbound_log: list[EventEnvelope] = field(default_factory=list)
    _message_ids: MessageIdTracker = field(default_factory=MessageIdTracker)
    _windows: dict[str, CreditWindow] = field(default_factory=dict)
    _active_fence: Fence | None = None
    _cancelled: bool = False
    _accepted: bool = False
    _outbound_seq: int = 0
    _pending_reorder: deque[EventEnvelope | tuple[EventEnvelope, bytes]] = field(
        default_factory=deque
    )
    _late_after_cancel: list[EventEnvelope | tuple[EventEnvelope, bytes]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        self._windows["default"] = CreditWindow(
            stream_id="default",
            available_events=self.initial_event_credits,
            available_bytes=self.initial_byte_credits,
        )

    @property
    def accepted(self) -> bool:
        return self._accepted

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def inbound_log(self) -> list[EventEnvelope]:
        return list(self._inbound_log)

    def window(self, stream_id: str = "default") -> CreditWindow:
        if stream_id not in self._windows:
            self._windows[stream_id] = CreditWindow(
                stream_id=stream_id,
                available_events=self.initial_event_credits,
                available_bytes=self.initial_byte_credits,
            )
        return self._windows[stream_id]

    async def send(self, event: EventEnvelope | dict[str, Any], pcm: bytes | None = None) -> None:
        """Orchestrator → peer inbound path."""
        envelope = event if isinstance(event, EventEnvelope) else parse_event(event)
        self._inbound_log.append(envelope)

        if envelope.event_type == EventType.HELLO:
            await self._handle_hello(envelope)
            return

        if not self._accepted:
            await self._emit_error(
                StageErrorCode.INVALID_ARGUMENT,
                "hello required before other events",
                base=envelope,
            )
            return

        if self._cancelled and envelope.event_type not in (EventType.CANCEL,):
            # Reject further data on cancelled fence.
            await self._emit_error(
                StageErrorCode.CANCELLED,
                "attempt cancelled",
                base=envelope,
            )
            return

        if self._active_fence is not None and envelope.event_type not in (EventType.HELLO,):
            try:
                require_instance = envelope.event_type != EventType.OPEN
                if envelope.stage_instance_id is not None:
                    check_fence(self._active_fence, envelope, require_instance=require_instance)
            except ValidationError as exc:
                await self._emit_error(exc.code, exc.message, base=envelope)
                return

        try:
            check_deadline(envelope.deadline_at)
        except ValidationError as exc:
            await self._emit_error(exc.code, exc.message, base=envelope)
            return

        # Credit accounting for data-bearing frames.
        stream_id = "default"
        nbytes = len(pcm) if pcm is not None else 0
        if isinstance(envelope.payload, dict) and "stream_id" in envelope.payload:
            stream_id = str(envelope.payload["stream_id"])
        try:
            self.window(stream_id).consume(events=1, bytes_=nbytes)
        except ValidationError as exc:
            await self._emit_error(exc.code, exc.message, base=envelope)
            return

        if envelope.event_type == EventType.CANCEL:
            await self._handle_cancel(envelope)
            return

        await self._run_scripts(envelope)

        if self.mode == PeerMode.FAIL:
            await self._emit_error(
                StageErrorCode.INFERENCE_FAILED, "scripted failure", base=envelope
            )

    async def recv(
        self,
        *,
        timeout: float | None = 5.0,
    ) -> EventEnvelope | tuple[EventEnvelope, bytes]:
        if timeout is None:
            return await self._outbound.get()
        return await asyncio.wait_for(self._outbound.get(), timeout=timeout)

    def inject_late(self, event: EventEnvelope | dict[str, Any], pcm: bytes | None = None) -> None:
        """Queue a late product (e.g. after cancel) for fencing tests."""
        envelope = event if isinstance(event, EventEnvelope) else parse_event(event)
        item: EventEnvelope | tuple[EventEnvelope, bytes]
        item = (envelope, pcm) if pcm is not None else envelope
        if self.mode == PeerMode.EMIT_AFTER_CANCEL or self._cancelled:
            self._late_after_cancel.append(item)
        else:
            self._outbound.put_nowait(item)

    async def flush_late_after_cancel(self) -> int:
        """Emit deliberately stale products after cancel (orchestrator must discard)."""
        count = 0
        while self._late_after_cancel:
            item = self._late_after_cancel.pop(0)
            await self._outbound.put(item)
            count += 1
        return count

    async def _handle_hello(self, envelope: EventEnvelope) -> None:
        try:
            check_schema_version(str(envelope.schema_version))
        except ValidationError as exc:
            await self._emit_error(exc.code, exc.message, base=envelope, terminal=True)
            return

        hello = HelloPayload.model_validate(envelope.payload)
        if hello.resume is not None:
            await self._emit_error(
                StageErrorCode.RESUME_UNSUPPORTED,
                "resume unsupported in baseline stage.v1",
                base=envelope,
                terminal=True,
            )
            return

        # Bind peer identity to the negotiated hello (one kind/id per connection).
        self.stage_kind = envelope.stage_kind
        self.stage_id = envelope.stage_id

        max_frame = min(
            self.max_frame_bytes,
            hello.limits_requested.max_frame_bytes or self.max_frame_bytes,
        )
        provenance = ProvenanceBlock(
            stage_id=self.stage_id,
            stage_version=self.stage_version,
            code_git_sha="0" * 40,
            container_image_digest=None,
            model_provider_id="scripted",
            model_revision=self.model_revision,
            model_artifact_digest=self.model_artifact_digest,
            model_artifact_status=ArtifactDigestStatus.VERIFIED,
            runtime_versions={"python": "3.12"},
            hardware_class="test",
            boot_id=self.boot_id,
        )
        prov_id = provenance_id_from_block(provenance)
        accepted_payload = AcceptedPayload(
            stage_version=self.stage_version,
            model_revision=self.model_revision,
            model_artifact_digest=self.model_artifact_digest,
            stage_instance_id=self.stage_instance_id,
            boot_id=self.boot_id,
            capabilities=["scripted"],
            audio_formats=[AudioFormat()],
            limits=LimitsAdvertised(max_frame_bytes=max_frame),
            provenance=provenance,
            provenance_id=prov_id,
        )
        out = self._base_envelope(
            EventType.ACCEPTED,
            envelope,
            accepted_payload.model_dump(mode="json"),
            stage_instance_id=self.stage_instance_id,
            stage_version=self.stage_version,
            model_revision=self.model_revision,
            model_artifact_digest=self.model_artifact_digest,
            provenance_id=prov_id,
        )
        self._active_fence = Fence.from_envelope(out)
        self._accepted = True
        await self._emit(out)
        # Advertise initial window.
        win = self.window("default")
        window_env = self._base_envelope(
            EventType.WINDOW,
            envelope,
            {
                "stream_id": "default",
                "available_events": win.available_events,
                "available_bytes": win.available_bytes,
                "credit_epoch": win.credit_epoch,
                "oldest_queue_age_ms": win.oldest_queue_age_ms,
            },
            stage_instance_id=self.stage_instance_id,
            stage_version=self.stage_version,
            model_revision=self.model_revision,
            model_artifact_digest=self.model_artifact_digest,
            provenance_id=prov_id,
        )
        await self._emit(window_env)

    async def _handle_cancel(self, envelope: EventEnvelope) -> None:
        self._cancelled = True
        cancelled = self._base_envelope(
            EventType.CANCELLED,
            envelope,
            {
                "scope": envelope.payload.get("scope", "attempt"),
                "reason": envelope.payload.get("reason", "cancel"),
                "disposed": True,
            },
            stage_instance_id=self.stage_instance_id,
            stage_version=self.stage_version,
            model_revision=self.model_revision,
            model_artifact_digest=self.model_artifact_digest,
            provenance_id=envelope.provenance_id,
        )
        await self._emit(cancelled)
        if self.mode == PeerMode.EMIT_AFTER_CANCEL:
            await self.flush_late_after_cancel()

    async def _run_scripts(self, inbound: EventEnvelope) -> None:
        remaining: list[ScriptedResponse] = []
        for script in self.scripted:
            types = script.after_inbound_types
            if types is not None and inbound.event_type.value not in types:
                remaining.append(script)
                continue
            if script.fail_code is not None:
                await self._emit_error(script.fail_code, "scripted fail", base=inbound)
                continue
            if script.event is None:
                continue
            if script.delay_s > 0 or self.mode == PeerMode.DELAY:
                await asyncio.sleep(script.delay_s or 0.01)
            env = (
                script.event
                if isinstance(script.event, EventEnvelope)
                else parse_event(script.event)
            )
            item: EventEnvelope | tuple[EventEnvelope, bytes]
            item = (env, script.binary_pcm) if script.binary_pcm is not None else env
            if self.mode == PeerMode.REORDER:
                self._pending_reorder.append(item)
                if len(self._pending_reorder) >= 2:
                    second = self._pending_reorder.pop()
                    first = self._pending_reorder.popleft()
                    await self._outbound.put(second)
                    await self._outbound.put(first)
            else:
                await self._emit_item(item)
        self.scripted = remaining

    def _base_envelope(
        self,
        event_type: EventType,
        base: EventEnvelope,
        payload: dict[str, Any],
        **overrides: Any,
    ) -> EventEnvelope:
        now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        data: dict[str, Any] = {
            "schema_version": "stage.v1",
            "event_type": event_type.value,
            "message_id": str(uuid4()),
            "event_sequence": self._outbound_seq,
            "created_at": now,
            "correlation_id": base.correlation_id,
            "session_id": base.session_id,
            "owner_generation": base.owner_generation,
            "stage_kind": (
                self.stage_kind.value
                if isinstance(self.stage_kind, StageKind)
                else str(self.stage_kind)
            ),
            "stage_id": self.stage_id,
            "attempt_id": base.attempt_id,
            "cancel_id": base.cancel_id,
            "payload": payload,
        }
        data.update({k: v for k, v in overrides.items() if v is not None})
        self._outbound_seq += 1
        return parse_event(data)

    async def _emit(self, envelope: EventEnvelope) -> None:
        await self._outbound.put(envelope)

    async def _emit_item(self, item: EventEnvelope | tuple[EventEnvelope, bytes]) -> None:
        await self._outbound.put(item)

    async def _emit_error(
        self,
        code: StageErrorCode,
        message: str,
        *,
        base: EventEnvelope,
        terminal: bool = False,
    ) -> None:
        env = self._base_envelope(
            EventType.ERROR,
            base,
            {
                "code": code.value,
                "message": message,
                "retryable": False,
                "scope": "connection" if terminal else "attempt",
            },
            stage_instance_id=self.stage_instance_id if self._accepted else None,
            stage_version=self.stage_version if self._accepted else None,
            model_revision=self.model_revision if self._accepted else None,
            model_artifact_digest=self.model_artifact_digest if self._accepted else None,
        )
        await self._emit(env)
