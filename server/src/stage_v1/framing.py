from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any

from src.stage_v1.models import (
    DEFAULT_MAX_FRAME_BYTES,
    MAX_HEADER_BYTES,
    MAX_TEXT_FRAME_BYTES,
    BinaryAudioPayload,
    EventEnvelope,
    EventType,
    StageErrorCode,
    parse_event,
)

MAGIC = b"STG1"
_HEADER_LEN_STRUCT = struct.Struct(">I")


class FramingError(Exception):
    def __init__(self, code: StageErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class DecodedBinaryFrame:
    envelope: EventEnvelope
    audio_payload: BinaryAudioPayload
    pcm: bytes
    header_bytes: bytes


def encode_binary_frame(
    envelope: EventEnvelope | dict[str, Any],
    pcm: bytes,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> bytes:
    """Encode a STG1 binary audio frame.

    Layout: magic(4) + BE u32 header_len + UTF-8 JSON header + raw PCM.
    """
    if isinstance(envelope, EventEnvelope):
        header_obj = envelope.model_dump(mode="json", exclude_none=True)
        event_type = envelope.event_type
        payload = envelope.payload
    else:
        header_obj = dict(envelope)
        event_type = header_obj.get("event_type")
        payload = header_obj.get("payload") or {}

    if event_type not in (
        EventType.LISTEN_AUDIO,
        EventType.SPEAK_AUDIO,
        "listen.audio",
        "speak.audio",
    ):
        raise FramingError(
            StageErrorCode.INVALID_ARGUMENT,
            f"binary frames require listen.audio or speak.audio, got {event_type!r}",
        )

    audio = BinaryAudioPayload.model_validate(payload)
    if audio.payload_bytes != len(pcm):
        raise FramingError(
            StageErrorCode.INVALID_ARGUMENT,
            f"payload_bytes={audio.payload_bytes} does not match pcm length {len(pcm)}",
        )
    if len(pcm) > max_frame_bytes:
        raise FramingError(
            StageErrorCode.FRAME_TOO_LARGE,
            f"pcm payload {len(pcm)} exceeds max_frame_bytes={max_frame_bytes}",
        )

    # Keep payload consistent with validated audio model.
    header_obj["payload"] = audio.model_dump(mode="json", exclude_none=True)
    header_bytes = (
        EventEnvelope.model_validate(header_obj).model_dump_json(exclude_none=True).encode("utf-8")
    )
    # Prefer canonical sorted form for golden stability when callers dump dicts.
    if not isinstance(envelope, EventEnvelope):
        from src.stage_v1.provenance import canonical_json_bytes

        header_bytes = canonical_json_bytes(
            parse_event(header_obj).model_dump(mode="json", exclude_none=True)
        )

    if len(header_bytes) > MAX_HEADER_BYTES:
        raise FramingError(
            StageErrorCode.FRAME_TOO_LARGE,
            f"header {len(header_bytes)} exceeds MAX_HEADER_BYTES={MAX_HEADER_BYTES}",
        )

    return MAGIC + _HEADER_LEN_STRUCT.pack(len(header_bytes)) + header_bytes + pcm


def encode_binary_frame_from_parts(
    *,
    envelope: EventEnvelope,
    pcm: bytes,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
    canonical_header: bool = True,
) -> bytes:
    """Encode with optional canonical header JSON (for golden fixtures)."""
    audio = BinaryAudioPayload.model_validate(envelope.payload)
    if audio.payload_bytes != len(pcm):
        raise FramingError(
            StageErrorCode.INVALID_ARGUMENT,
            f"payload_bytes={audio.payload_bytes} does not match pcm length {len(pcm)}",
        )
    if len(pcm) > max_frame_bytes:
        raise FramingError(
            StageErrorCode.FRAME_TOO_LARGE,
            f"pcm payload {len(pcm)} exceeds max_frame_bytes={max_frame_bytes}",
        )

    data = envelope.model_dump(mode="json", exclude_none=True)
    data["payload"] = audio.model_dump(mode="json", exclude_none=True)
    if canonical_header:
        from src.stage_v1.provenance import canonical_json_bytes

        header_bytes = canonical_json_bytes(data)
    else:
        header_bytes = (
            EventEnvelope.model_validate(data).model_dump_json(exclude_none=True).encode("utf-8")
        )

    if len(header_bytes) > MAX_HEADER_BYTES:
        raise FramingError(
            StageErrorCode.FRAME_TOO_LARGE,
            f"header {len(header_bytes)} exceeds MAX_HEADER_BYTES={MAX_HEADER_BYTES}",
        )
    return MAGIC + _HEADER_LEN_STRUCT.pack(len(header_bytes)) + header_bytes + pcm


def decode_binary_frame(
    data: bytes,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> DecodedBinaryFrame:
    if len(data) < 8:
        raise FramingError(StageErrorCode.INVALID_ARGUMENT, "frame shorter than magic+length")

    magic = data[0:4]
    if magic != MAGIC:
        raise FramingError(
            StageErrorCode.INVALID_ARGUMENT, f"bad magic {magic!r}, expected {MAGIC!r}"
        )

    (header_len,) = _HEADER_LEN_STRUCT.unpack(data[4:8])
    if header_len > MAX_HEADER_BYTES:
        raise FramingError(
            StageErrorCode.FRAME_TOO_LARGE,
            f"header_len={header_len} exceeds MAX_HEADER_BYTES={MAX_HEADER_BYTES}",
        )
    if len(data) < 8 + header_len:
        raise FramingError(
            StageErrorCode.INVALID_ARGUMENT,
            f"truncated frame: need {8 + header_len} bytes, have {len(data)}",
        )

    header_bytes = data[8 : 8 + header_len]
    pcm = data[8 + header_len :]
    if len(pcm) > max_frame_bytes:
        raise FramingError(
            StageErrorCode.FRAME_TOO_LARGE,
            f"pcm payload {len(pcm)} exceeds max_frame_bytes={max_frame_bytes}",
        )

    try:
        envelope = parse_event(
            __import__("json").loads(header_bytes.decode("utf-8")),
        )
    except Exception as exc:  # noqa: BLE001 — map all header parse failures
        raise FramingError(
            StageErrorCode.INVALID_ARGUMENT, f"malformed header JSON: {exc}"
        ) from exc

    if envelope.event_type not in (EventType.LISTEN_AUDIO, EventType.SPEAK_AUDIO):
        raise FramingError(
            StageErrorCode.INVALID_ARGUMENT,
            "binary frame event_type must be listen.audio or speak.audio, "
            f"got {envelope.event_type}",
        )

    try:
        audio = BinaryAudioPayload.model_validate(envelope.payload)
    except Exception as exc:  # noqa: BLE001
        raise FramingError(
            StageErrorCode.INVALID_ARGUMENT, f"invalid audio payload: {exc}"
        ) from exc

    if audio.payload_bytes != len(pcm):
        raise FramingError(
            StageErrorCode.INVALID_ARGUMENT,
            f"payload_bytes={audio.payload_bytes} does not match pcm length {len(pcm)}",
        )

    expected_bytes = _expected_pcm_bytes(audio)
    if expected_bytes is not None and expected_bytes != len(pcm):
        raise FramingError(
            StageErrorCode.INVALID_ARGUMENT,
            f"sample_count/format imply {expected_bytes} bytes, got {len(pcm)}",
        )

    return DecodedBinaryFrame(
        envelope=envelope,
        audio_payload=audio,
        pcm=pcm,
        header_bytes=header_bytes,
    )


def _expected_pcm_bytes(audio: BinaryAudioPayload) -> int | None:
    fmt = audio.format
    if fmt.codec != "pcm_s16le":
        return None
    bytes_per_sample = 2 * fmt.channels
    return audio.sample_count * bytes_per_sample


def validate_text_frame_size(raw: str | bytes) -> None:
    size = len(raw.encode("utf-8") if isinstance(raw, str) else raw)
    if size > MAX_TEXT_FRAME_BYTES:
        raise FramingError(
            StageErrorCode.FRAME_TOO_LARGE,
            f"text frame {size} exceeds MAX_TEXT_FRAME_BYTES={MAX_TEXT_FRAME_BYTES}",
        )
