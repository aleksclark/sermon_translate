from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from src.stage_v1.models import (
    SCHEMA_VERSION,
    BinaryAudioPayload,
    EventEnvelope,
    EventType,
    StageErrorCode,
)
from src.stage_v1.provenance import canonical_json_bytes


class ValidationError(Exception):
    def __init__(self, code: StageErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class DedupeResult(StrEnum):
    NEW = "new"
    IDEMPOTENT = "idempotent"


@dataclass(frozen=True, slots=True)
class Fence:
    session_id: str
    owner_generation: int
    stage_kind: str
    stage_id: str
    attempt_id: str
    cancel_id: str
    stage_instance_id: str | None = None

    @classmethod
    def from_envelope(cls, envelope: EventEnvelope) -> Fence:
        kind = envelope.stage_kind
        stage_kind = kind.value if isinstance(kind, StrEnum) else str(kind)
        return cls(
            session_id=envelope.session_id,
            owner_generation=envelope.owner_generation,
            stage_kind=stage_kind,
            stage_id=envelope.stage_id,
            attempt_id=envelope.attempt_id,
            cancel_id=envelope.cancel_id,
            stage_instance_id=envelope.stage_instance_id,
        )

    def matches(self, other: Fence, *, require_instance: bool = True) -> bool:
        base = (
            self.session_id == other.session_id
            and self.owner_generation == other.owner_generation
            and self.stage_kind == other.stage_kind
            and self.stage_id == other.stage_id
            and self.attempt_id == other.attempt_id
            and self.cancel_id == other.cancel_id
        )
        if not base:
            return False
        if not require_instance:
            return True
        return self.stage_instance_id == other.stage_instance_id


def parse_rfc3339(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def check_schema_version(schema_version: str) -> None:
    if schema_version != SCHEMA_VERSION:
        raise ValidationError(
            StageErrorCode.VERSION_UNSUPPORTED,
            f"unsupported schema_version={schema_version!r}",
        )


def check_deadline(deadline_at: str | None, *, now: datetime | None = None) -> None:
    if deadline_at is None:
        return
    current = now or datetime.now(UTC)
    if parse_rfc3339(deadline_at) <= current:
        raise ValidationError(
            StageErrorCode.DEADLINE_EXCEEDED,
            f"deadline_at={deadline_at} has expired",
        )


def check_fence(active: Fence, candidate: EventEnvelope, *, require_instance: bool = True) -> None:
    cand = Fence.from_envelope(candidate)
    if not active.matches(cand, require_instance=require_instance):
        raise ValidationError(
            StageErrorCode.STALE_FENCE,
            "event fence does not match active attempt",
        )


@dataclass
class MessageIdTracker:
    _seen: dict[str, bytes] = field(default_factory=dict)

    def observe(self, message_id: str, canonical: bytes) -> DedupeResult:
        prior = self._seen.get(message_id)
        if prior is None:
            self._seen[message_id] = canonical
            return DedupeResult.NEW
        if prior == canonical:
            return DedupeResult.IDEMPOTENT
        raise ValidationError(
            StageErrorCode.DUPLICATE_CONFLICT,
            f"message_id={message_id} reused with different canonical bytes",
        )

    def observe_envelope(self, envelope: EventEnvelope) -> DedupeResult:
        data = envelope.model_dump(mode="json", exclude_none=True)
        return self.observe(envelope.message_id, canonical_json_bytes(data))


@dataclass
class EventSequenceTracker:
    """Strictly increasing event_sequence per connection direction, starting at 0."""

    next_expected: int = 0
    _started: bool = False

    def observe(self, sequence: int) -> None:
        if not self._started:
            if sequence != 0:
                raise ValidationError(
                    StageErrorCode.SEQUENCE_GAP,
                    f"event_sequence must start at 0, got {sequence}",
                )
            self._started = True
            self.next_expected = 1
            return
        if sequence != self.next_expected:
            raise ValidationError(
                StageErrorCode.SEQUENCE_GAP,
                f"expected event_sequence={self.next_expected}, got {sequence}",
            )
        self.next_expected = sequence + 1


@dataclass
class MediaClock:
    """media_sequence and start_sample monotonicity per stream_id."""

    stream_id: str
    next_media_sequence: int = 0
    next_start_sample: int = 0
    format_fingerprint: tuple[str, int, int] | None = None
    frames_seen: int = 0

    def observe_frame(self, audio: BinaryAudioPayload) -> None:
        if audio.stream_id != self.stream_id:
            raise ValidationError(
                StageErrorCode.INVALID_ARGUMENT,
                f"stream_id mismatch: tracker={self.stream_id} frame={audio.stream_id}",
            )

        fp = (audio.format.codec, audio.format.sample_rate_hz, audio.format.channels)
        if self.format_fingerprint is None:
            if audio.format.codec == "pcm_s16le" and audio.format.channels != 1:
                raise ValidationError(
                    StageErrorCode.UNSUPPORTED_FORMAT,
                    f"baseline pcm_s16le requires channels=1, got {audio.format.channels}",
                )
            self.format_fingerprint = fp
        elif fp != self.format_fingerprint:
            raise ValidationError(
                StageErrorCode.UNSUPPORTED_FORMAT,
                f"format changed mid-stream from {self.format_fingerprint} to {fp}",
            )

        if audio.media_sequence != self.next_media_sequence:
            raise ValidationError(
                StageErrorCode.SEQUENCE_GAP,
                f"expected media_sequence={self.next_media_sequence}, got {audio.media_sequence}",
            )

        if audio.start_sample != self.next_start_sample:
            raise ValidationError(
                StageErrorCode.SEQUENCE_GAP,
                f"expected start_sample={self.next_start_sample}, got {audio.start_sample}",
            )

        if audio.format.codec == "pcm_s16le":
            expected = audio.sample_count * 2 * audio.format.channels
            if audio.payload_bytes != expected:
                raise ValidationError(
                    StageErrorCode.INVALID_ARGUMENT,
                    f"payload_bytes={audio.payload_bytes} != sample_count*2*channels={expected}",
                )

        self.next_media_sequence = audio.media_sequence + 1
        self.next_start_sample = audio.start_sample + audio.sample_count
        self.frames_seen += 1

    def apply_gap(
        self,
        *,
        end_sample: int | None = None,
        skip_to_media_sequence: int | None = None,
        skip_to_start_sample: int | None = None,
    ) -> None:
        if end_sample is not None:
            if end_sample < self.next_start_sample:
                raise ValidationError(
                    StageErrorCode.INVALID_ARGUMENT,
                    f"gap end_sample={end_sample} before "
                    f"next_start_sample={self.next_start_sample}",
                )
            self.next_start_sample = end_sample
        if skip_to_start_sample is not None:
            if skip_to_start_sample < self.next_start_sample:
                raise ValidationError(
                    StageErrorCode.INVALID_ARGUMENT,
                    f"gap skip_to_start_sample={skip_to_start_sample} regresses clock",
                )
            self.next_start_sample = skip_to_start_sample
        if skip_to_media_sequence is not None:
            if skip_to_media_sequence < self.next_media_sequence:
                raise ValidationError(
                    StageErrorCode.INVALID_ARGUMENT,
                    f"gap media_sequence regression to {skip_to_media_sequence}",
                )
            self.next_media_sequence = skip_to_media_sequence


@dataclass
class RevisionState:
    """Product revision + commit-prefix ledger for one product scope."""

    next_revision: int = 0
    committed_prefix_chars: int = 0
    committed_text_prefix: str = ""
    last_text: str | None = None
    last_revision: int | None = None
    finalized: bool = False
    _revision_bytes: dict[int, bytes] = field(default_factory=dict)

    def observe(
        self,
        *,
        revision: int,
        text: str,
        committed_prefix_chars: int,
        is_final: bool,
    ) -> DedupeResult:
        canonical = canonical_json_bytes(
            {
                "committed_prefix_chars": committed_prefix_chars,
                "is_final": is_final,
                "revision": revision,
                "text": text,
            }
        )

        if revision in self._revision_bytes:
            if self._revision_bytes[revision] == canonical:
                return DedupeResult.IDEMPOTENT
            raise ValidationError(
                StageErrorCode.DUPLICATE_CONFLICT,
                f"revision={revision} reused with different product bytes",
            )

        if self.finalized:
            raise ValidationError(
                StageErrorCode.INVALID_ARGUMENT,
                "product scope already finalized",
            )

        if revision != self.next_revision:
            raise ValidationError(
                StageErrorCode.SEQUENCE_GAP,
                f"expected revision={self.next_revision}, got {revision}",
            )

        if committed_prefix_chars < self.committed_prefix_chars:
            raise ValidationError(
                StageErrorCode.COMMIT_RETRACTION,
                f"committed_prefix_chars regressed "
                f"{self.committed_prefix_chars} -> {committed_prefix_chars}",
            )

        if committed_prefix_chars > len(text):
            raise ValidationError(
                StageErrorCode.INVALID_ARGUMENT,
                "committed_prefix_chars exceeds text length",
            )

        new_prefix = text[:committed_prefix_chars]
        if not new_prefix.startswith(self.committed_text_prefix):
            raise ValidationError(
                StageErrorCode.COMMIT_RETRACTION,
                "committed prefix bytes changed (commit retraction)",
            )

        if is_final and committed_prefix_chars != len(text):
            raise ValidationError(
                StageErrorCode.INVALID_ARGUMENT,
                "is_final requires committed_prefix_chars == len(text)",
            )

        self._revision_bytes[revision] = canonical
        self.next_revision = revision + 1
        self.committed_prefix_chars = committed_prefix_chars
        self.committed_text_prefix = new_prefix
        self.last_text = text
        self.last_revision = revision
        if is_final:
            self.finalized = True
        return DedupeResult.NEW


def validate_handshake_hello(envelope: EventEnvelope) -> None:
    if envelope.event_type != EventType.HELLO:
        raise ValidationError(
            StageErrorCode.INVALID_ARGUMENT,
            f"expected hello, got {envelope.event_type}",
        )
    check_schema_version(str(envelope.schema_version))
    if envelope.event_sequence != 0:
        raise ValidationError(
            StageErrorCode.SEQUENCE_GAP,
            "hello event_sequence must be 0",
        )
    resume = envelope.payload.get("resume") if isinstance(envelope.payload, dict) else None
    if resume is not None:
        raise ValidationError(
            StageErrorCode.RESUME_UNSUPPORTED,
            "hello.payload.resume is unsupported in baseline stage.v1",
        )


def envelope_from_mapping(data: dict[str, Any]) -> EventEnvelope:
    return EventEnvelope.model_validate(data)
