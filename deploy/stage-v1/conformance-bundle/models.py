from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "stage.v1"
PROSODY_SCHEMA_VERSION = "prosody.v1"
JS_SAFE_INTEGER_MAX = (1 << 53) - 1
DEFAULT_MAX_FRAME_BYTES = 65_536
MAX_HEADER_BYTES = 16 * 1024
MAX_TEXT_FRAME_BYTES = 64 * 1024
BASELINE_CODEC = "pcm_s16le"
BASELINE_SAMPLE_RATE_HZ = 16_000
BASELINE_CHANNELS = 1


class StageErrorCode(StrEnum):
    VERSION_UNSUPPORTED = "VERSION_UNSUPPORTED"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    AUTHORIZATION_FAILED = "AUTHORIZATION_FAILED"
    INVALID_ARGUMENT = "INVALID_ARGUMENT"
    FRAME_TOO_LARGE = "FRAME_TOO_LARGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    UNSUPPORTED_CAPABILITY = "UNSUPPORTED_CAPABILITY"
    DUPLICATE_CONFLICT = "DUPLICATE_CONFLICT"
    SEQUENCE_GAP = "SEQUENCE_GAP"
    COMMIT_RETRACTION = "COMMIT_RETRACTION"
    STALE_FENCE = "STALE_FENCE"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    CANCELLED = "CANCELLED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    INTERNAL = "INTERNAL"
    RESUME_UNSUPPORTED = "RESUME_UNSUPPORTED"


class EventType(StrEnum):
    HELLO = "hello"
    ACCEPTED = "accepted"
    OPEN = "open"
    OPENED = "opened"
    LISTEN_AUDIO = "listen.audio"
    LISTEN_PRODUCT = "listen.product"
    TRANSLATE_REQUEST = "translate.request"
    TRANSLATE_PRODUCT = "translate.product"
    SPEAK_REQUEST = "speak.request"
    SPEAK_AUDIO = "speak.audio"
    SPEAK_COMPLETE = "speak.complete"
    WINDOW = "window"
    ACK = "ack"
    EOS = "eos"
    CANCEL = "cancel"
    CANCELLED = "cancelled"
    GAP = "gap"
    DROPPED = "dropped"
    ERROR = "error"
    HEALTH = "health"
    DRAINING = "draining"


class StageKind(StrEnum):
    LISTEN = "listen"
    TRANSLATE = "translate"
    SPEAK = "speak"
    PROSODY = "prosody"


class CancelScope(StrEnum):
    UTTERANCE = "utterance"
    ATTEMPT = "attempt"
    SESSION = "session"


class ErrorScope(StrEnum):
    EVENT = "event"
    SPAN = "span"
    ATTEMPT = "attempt"
    CONNECTION = "connection"
    UTTERANCE = "utterance"
    SESSION = "session"


class TimingKind(StrEnum):
    MODEL = "model"
    CHUNK = "chunk"
    UNAVAILABLE = "unavailable"


class AlignmentKind(StrEnum):
    MODEL = "model"
    HUMAN = "human"
    HEURISTIC = "heuristic"
    UNAVAILABLE = "unavailable"


class ProsodyStatus(StrEnum):
    APPLIED = "applied"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


class ArtifactDigestStatus(StrEnum):
    VERIFIED = "verified"
    PROVIDER_MANAGED = "provider_managed"
    UNAVAILABLE = "unavailable"


class DropReason(StrEnum):
    SUPERSEDED_UNCOMMITTED = "superseded_uncommitted"


class StageModel(BaseModel):
    """Base model: ignore unknown additive fields under the same major version."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class AudioFormat(StageModel):
    codec: str = BASELINE_CODEC
    sample_rate_hz: int = BASELINE_SAMPLE_RATE_HZ
    channels: int = BASELINE_CHANNELS

    @field_validator("channels")
    @classmethod
    def _channels_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("channels must be >= 1")
        return value

    @field_validator("sample_rate_hz")
    @classmethod
    def _sample_rate_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("sample_rate_hz must be >= 1")
        return value


class LimitsRequested(StageModel):
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES
    max_inflight_events: int = 32
    max_inflight_bytes: int | None = None
    max_queue_age_ms: int | None = None


class LimitsAdvertised(StageModel):
    max_sessions: int = 1
    max_inflight_events: int = 32
    max_inflight_bytes: int = DEFAULT_MAX_FRAME_BYTES * 32
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES
    input_queue_capacity: int = 32
    max_queue_age_ms: int = 5_000


class ProvenanceBlock(StageModel):
    stage_id: str
    stage_version: str
    code_git_sha: str
    container_image_digest: str | None = None
    model_provider_id: str
    model_revision: str
    model_artifact_digest: str
    model_artifact_status: ArtifactDigestStatus
    runtime_versions: dict[str, str] = Field(default_factory=dict)
    prompt_digest: str | None = None
    glossary_digest: str | None = None
    voice_digest: str | None = None
    stage_config_digest: str | None = None
    hardware_class: str | None = None
    boot_id: str


class EventEnvelope(StageModel):
    schema_version: Literal["stage.v1"] = SCHEMA_VERSION
    event_type: EventType
    message_id: str
    event_sequence: int = Field(ge=0, le=JS_SAFE_INTEGER_MAX)
    created_at: str
    correlation_id: str
    session_id: str = Field(min_length=1, max_length=128)
    owner_generation: int = Field(ge=0)
    stage_kind: StageKind
    stage_id: str
    attempt_id: str
    cancel_id: str
    stage_version: str | None = None
    model_revision: str | None = None
    model_artifact_digest: str | None = None
    stage_instance_id: str | None = None
    utterance_id: str | None = None
    utterance_sequence: int | None = Field(default=None, ge=0, le=JS_SAFE_INTEGER_MAX)
    deadline_at: str | None = None
    traceparent: str | None = None
    tracestate: str | None = None
    provenance_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _utterance_fields_coupled(self) -> EventEnvelope:
        has_id = self.utterance_id is not None
        has_seq = self.utterance_sequence is not None
        if has_id != has_seq:
            raise ValueError("utterance_id and utterance_sequence must be supplied together")
        return self

    def fence_tuple(self) -> tuple[str, int, str, str, str, str, str | None]:
        return (
            self.session_id,
            self.owner_generation,
            self.stage_kind.value
            if isinstance(self.stage_kind, StageKind)
            else str(self.stage_kind),
            self.stage_id,
            self.attempt_id,
            self.cancel_id,
            self.stage_instance_id,
        )

    def canonical_bytes(self) -> bytes:
        """Canonical JSON bytes for message_id idempotency comparison."""
        from src.stage_v1.provenance import canonical_json_bytes

        data = self.model_dump(mode="json", exclude_none=True)
        return canonical_json_bytes(data)


class HelloPayload(StageModel):
    audio_formats: list[AudioFormat] = Field(default_factory=list)
    limits_requested: LimitsRequested = Field(default_factory=LimitsRequested)
    resume: dict[str, Any] | None = None
    capabilities_requested: list[str] = Field(default_factory=list)


class AcceptedPayload(StageModel):
    stage_version: str
    model_revision: str
    model_artifact_digest: str
    stage_instance_id: str
    boot_id: str
    capabilities: list[str] = Field(default_factory=list)
    audio_formats: list[AudioFormat] = Field(default_factory=list)
    limits: LimitsAdvertised = Field(default_factory=LimitsAdvertised)
    provenance: ProvenanceBlock
    provenance_id: str


class OpenPayload(StageModel):
    utterance_id: str | None = None
    utterance_sequence: int | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class OpenedPayload(StageModel):
    ready: bool = True
    config_echo: dict[str, Any] = Field(default_factory=dict)


class WordTiming(StageModel):
    text: str
    start_sample: int = Field(ge=0)
    end_sample: int = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    timing_kind: TimingKind = TimingKind.UNAVAILABLE


class ListenProductPayload(StageModel):
    revision: int = Field(ge=0)
    text: str
    committed_prefix_chars: int = Field(ge=0)
    is_final: bool = False
    language: str = "en"
    source_start_sample: int = Field(ge=0)
    source_end_sample: int = Field(ge=0)
    words: list[WordTiming] = Field(default_factory=list)
    confidence: float | None = None

    @model_validator(mode="after")
    def _commit_bounds(self) -> ListenProductPayload:
        if self.committed_prefix_chars > len(self.text):
            raise ValueError("committed_prefix_chars exceeds text length")
        if self.is_final and self.committed_prefix_chars != len(self.text):
            raise ValueError("is_final requires committed_prefix_chars == len(text)")
        if self.source_end_sample < self.source_start_sample:
            raise ValueError("source_end_sample must be >= source_start_sample")
        return self


class TranslateRequestPayload(StageModel):
    source_span_id: str
    source_revision: int = Field(ge=0)
    source_char_start: int = Field(ge=0)
    source_char_end: int = Field(ge=0)
    text: str
    source_language: str = "en"
    target_language: str = "es"
    preceding_source_context: str = ""
    preceding_target_context: str = ""
    sermon_notes: str | None = None
    sermon_notes_ref: str | None = None
    sermon_notes_revision: str | None = None
    glossary: list[dict[str, str]] = Field(default_factory=list)
    glossary_ref: str | None = None
    glossary_revision: str | None = None
    prompt_revision: str | None = None

    @model_validator(mode="after")
    def _span_bounds(self) -> TranslateRequestPayload:
        if self.source_char_end < self.source_char_start:
            raise ValueError("source_char_end must be >= source_char_start")
        return self


class TranslateProductPayload(StageModel):
    source_span_id: str
    target_span_id: str
    revision: int = Field(ge=0)
    text: str
    committed_prefix_chars: int = Field(ge=0)
    is_final: bool = False
    source_char_start: int = Field(ge=0)
    source_char_end: int = Field(ge=0)
    target_language: str = "es"
    alignment_kind: AlignmentKind = AlignmentKind.UNAVAILABLE
    alignment_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    terminology: list[dict[str, Any]] = Field(default_factory=list)
    prompt_revision: str | None = None
    glossary_revision: str | None = None

    @model_validator(mode="after")
    def _commit_bounds(self) -> TranslateProductPayload:
        if self.committed_prefix_chars > len(self.text):
            raise ValueError("committed_prefix_chars exceeds text length")
        if self.is_final and self.committed_prefix_chars != len(self.text):
            raise ValueError("is_final requires committed_prefix_chars == len(text)")
        if self.source_char_end < self.source_char_start:
            raise ValueError("source_char_end must be >= source_char_start")
        return self


class ProsodyOverall(StageModel):
    rate: float = Field(default=1.0, ge=0.5, le=2.0)
    pitch_semitones: float = Field(default=0.0, ge=-12.0, le=12.0)
    energy: float = Field(default=1.0, ge=0.0, le=2.0)
    style: str = "neutral"


class ProsodyMarker(StageModel):
    target_char_start: int | None = Field(default=None, ge=0)
    target_char_end: int | None = Field(default=None, ge=0)
    after_target_char: int | None = Field(default=None, ge=0)
    emphasis: float | None = Field(default=None, ge=0.0, le=1.0)
    pause_ms: int | None = Field(default=None, ge=0, le=5000)


class ProsodyV1(StageModel):
    schema_version: Literal["prosody.v1"] = PROSODY_SCHEMA_VERSION
    overall: ProsodyOverall = Field(default_factory=ProsodyOverall)
    markers: list[ProsodyMarker] = Field(default_factory=list)
    alignment_kind: AlignmentKind = AlignmentKind.UNAVAILABLE
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class SpeakRequestPayload(StageModel):
    target_span_id: str
    text: str
    target_language: str = "es"
    publication_order: int = Field(ge=0)
    output_format: AudioFormat = Field(default_factory=AudioFormat)
    voice_id: str
    voice_revision: str
    voice_config_digest: str
    prosody: ProsodyV1 | None = None
    prosody_required: bool = False


class ProsodyReport(StageModel):
    prosody_status: ProsodyStatus
    consumed_fields: list[str] = Field(default_factory=list)
    ignored_fields: list[str] = Field(default_factory=list)


class SpeakCompletePayload(StageModel):
    target_span_id: str
    chunk_count: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    duration_ms: float = Field(ge=0.0)
    is_final: bool = True
    prosody_report: ProsodyReport | None = None


class BinaryAudioPayload(StageModel):
    """Payload fields required inside a binary audio frame header."""

    stream_id: str
    media_sequence: int = Field(ge=0, le=JS_SAFE_INTEGER_MAX)
    start_sample: int = Field(ge=0)
    sample_count: int = Field(ge=0)
    payload_bytes: int = Field(ge=0)
    format: AudioFormat = Field(default_factory=AudioFormat)
    capture_time: str | None = None
    discontinuity: bool = False
    target_span_id: str | None = None
    audio_chunk_sequence: int | None = Field(default=None, ge=0)


class WindowPayload(StageModel):
    stream_id: str
    available_events: int = Field(ge=0)
    available_bytes: int = Field(ge=0)
    credit_epoch: int = Field(ge=0)
    oldest_queue_age_ms: int = Field(ge=0)


class AckPayload(StageModel):
    stream_id: str | None = None
    event_sequence: int | None = Field(default=None, ge=0)
    media_sequence: int | None = Field(default=None, ge=0)


class EosPayload(StageModel):
    stream_id: str
    last_media_sequence: int | None = Field(default=None, ge=0)
    last_sample_end: int | None = Field(default=None, ge=0)
    utterance_id: str | None = None


class CancelPayload(StageModel):
    scope: CancelScope
    reason: str
    utterance_id: str | None = None
    attempt_id: str | None = None
    session_id: str | None = None


class CancelledPayload(StageModel):
    scope: CancelScope
    reason: str
    disposed: bool = True


class GapPayload(StageModel):
    stream_id: str | None = None
    reason: str
    start_sample: int | None = Field(default=None, ge=0)
    end_sample: int | None = Field(default=None, ge=0)
    media_sequence_start: int | None = Field(default=None, ge=0)
    media_sequence_end: int | None = Field(default=None, ge=0)
    utterance_id: str | None = None
    target_span_id: str | None = None


class DroppedPayload(StageModel):
    reason: DropReason | str
    revision_start: int = Field(ge=0)
    revision_end: int = Field(ge=0)
    stage_kind: StageKind | None = None
    utterance_id: str | None = None
    source_span_id: str | None = None


class ErrorPayload(StageModel):
    code: StageErrorCode
    message: str
    retryable: bool = False
    scope: ErrorScope = ErrorScope.ATTEMPT
    retry_after_ms: int | None = Field(default=None, ge=0)
    affected_message_id: str | None = None
    affected_span_id: str | None = None
    affected_utterance_id: str | None = None


class HealthPayload(StageModel):
    status: str
    stage_kind: StageKind | None = None
    stage_id: str | None = None
    stage_version: str | None = None
    stage_instance_id: str | None = None
    boot_id: str | None = None
    active_sessions: int = Field(default=0, ge=0)
    max_sessions: int = Field(default=1, ge=0)
    model_loaded: bool = False
    model_warm: bool = False
    draining: bool = False
    last_canary_at: str | None = None
    last_canary_ok: bool | None = None
    provenance_id: str | None = None
    limits: LimitsAdvertised | None = None


class DrainingPayload(StageModel):
    reason: str
    grace_ms: int = Field(default=0, ge=0)
    active_sessions: int = Field(default=0, ge=0)


PAYLOAD_TYPES: dict[EventType, type[StageModel] | None] = {
    EventType.HELLO: HelloPayload,
    EventType.ACCEPTED: AcceptedPayload,
    EventType.OPEN: OpenPayload,
    EventType.OPENED: OpenedPayload,
    EventType.LISTEN_AUDIO: BinaryAudioPayload,
    EventType.LISTEN_PRODUCT: ListenProductPayload,
    EventType.TRANSLATE_REQUEST: TranslateRequestPayload,
    EventType.TRANSLATE_PRODUCT: TranslateProductPayload,
    EventType.SPEAK_REQUEST: SpeakRequestPayload,
    EventType.SPEAK_AUDIO: BinaryAudioPayload,
    EventType.SPEAK_COMPLETE: SpeakCompletePayload,
    EventType.WINDOW: WindowPayload,
    EventType.ACK: AckPayload,
    EventType.EOS: EosPayload,
    EventType.CANCEL: CancelPayload,
    EventType.CANCELLED: CancelledPayload,
    EventType.GAP: GapPayload,
    EventType.DROPPED: DroppedPayload,
    EventType.ERROR: ErrorPayload,
    EventType.HEALTH: HealthPayload,
    EventType.DRAINING: DrainingPayload,
}


def parse_payload(
    event_type: EventType | str, payload: dict[str, Any]
) -> StageModel | dict[str, Any]:
    et = EventType(event_type) if not isinstance(event_type, EventType) else event_type
    model_cls = PAYLOAD_TYPES.get(et)
    if model_cls is None:
        return payload
    return model_cls.model_validate(payload)


def parse_event(data: dict[str, Any]) -> EventEnvelope:
    return EventEnvelope.model_validate(data)


def parse_event_json(raw: str | bytes) -> EventEnvelope:
    return EventEnvelope.model_validate_json(raw)
