from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from src.stage_v1.framing import (
    MAGIC,
    MAX_HEADER_BYTES,
    FramingError,
    decode_binary_frame,
    encode_binary_frame_from_parts,
    validate_text_frame_size,
)
from src.stage_v1.models import (
    DEFAULT_MAX_FRAME_BYTES,
    MAX_TEXT_FRAME_BYTES,
    BinaryAudioPayload,
    EventEnvelope,
    EventType,
    StageErrorCode,
    parse_event,
)
from src.stage_v1.provenance import sha256_hex
from src.stage_v1.validation import MediaClock, ValidationError

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "stage_v1"


def _audio_envelope(
    *,
    media_sequence: int = 0,
    start_sample: int = 0,
    sample_count: int = 320,
    pcm_len: int | None = None,
    event_type: str = "listen.audio",
    stream_id: str = "source:main",
    channels: int = 1,
    extra_payload: dict | None = None,
) -> tuple[EventEnvelope, bytes]:
    nbytes = pcm_len if pcm_len is not None else sample_count * 2 * channels
    pcm = bytes([(i * 3) % 256 for i in range(nbytes)])
    payload: dict = {
        "stream_id": stream_id,
        "media_sequence": media_sequence,
        "start_sample": start_sample,
        "sample_count": sample_count,
        "payload_bytes": nbytes,
        "format": {"codec": "pcm_s16le", "sample_rate_hz": 16000, "channels": channels},
        "capture_time": "2026-08-08T16:00:00.240Z",
        "discontinuity": False,
    }
    if extra_payload:
        payload.update(extra_payload)
    env = parse_event(
        {
            "schema_version": "stage.v1",
            "event_type": event_type,
            "message_id": "0198bbbb-cccc-7ddd-8eee-ffffffffffff",
            "event_sequence": media_sequence,
            "created_at": "2026-08-08T16:00:00.000Z",
            "correlation_id": "run-0198",
            "session_id": "product-session-id",
            "owner_generation": 7,
            "stage_kind": "listen" if event_type.startswith("listen") else "speak",
            "stage_id": "whisper-listen",
            "attempt_id": "0198ffff-1111-7222-8333-444444444444",
            "cancel_id": "0198ffff-5555-7666-8777-888888888888",
            "stage_instance_id": "0198inst-0000-7000-8000-000000000001",
            "stage_version": "1.0.0",
            "model_revision": "rev-1",
            "model_artifact_digest": "sha256:" + ("ab" * 32),
            "utterance_id": "0198utt-0000-7000-8000-000000000001",
            "utterance_sequence": 0,
            "deadline_at": "2099-01-01T00:00:00.000Z",
            "payload": payload,
        }
    )
    return env, pcm


class TestBinaryEncodeDecode:
    def test_round_trip(self) -> None:
        env, pcm = _audio_envelope()
        frame = encode_binary_frame_from_parts(envelope=env, pcm=pcm)
        assert frame[:4] == MAGIC
        decoded = decode_binary_frame(frame)
        assert decoded.pcm == pcm
        assert decoded.envelope.event_type == EventType.LISTEN_AUDIO
        assert decoded.audio_payload.media_sequence == 0
        assert decoded.audio_payload.sample_count == 320
        assert decoded.audio_payload.payload_bytes == len(pcm)

    def test_speak_audio_extra_fields(self) -> None:
        env, pcm = _audio_envelope(
            event_type="speak.audio",
            stream_id="translated:span-1",
            extra_payload={"target_span_id": "span-1", "audio_chunk_sequence": 0},
        )
        frame = encode_binary_frame_from_parts(envelope=env, pcm=pcm)
        decoded = decode_binary_frame(frame)
        assert decoded.envelope.event_type == EventType.SPEAK_AUDIO
        assert decoded.audio_payload.target_span_id == "span-1"
        assert decoded.audio_payload.audio_chunk_sequence == 0

    def test_header_length_big_endian(self) -> None:
        env, pcm = _audio_envelope()
        frame = encode_binary_frame_from_parts(envelope=env, pcm=pcm)
        (header_len,) = struct.unpack(">I", frame[4:8])
        assert header_len > 0
        assert frame[8 : 8 + header_len]
        assert frame[8 + header_len :] == pcm


class TestMalformedFrames:
    def test_bad_magic(self) -> None:
        env, pcm = _audio_envelope()
        frame = bytearray(encode_binary_frame_from_parts(envelope=env, pcm=pcm))
        frame[0:4] = b"XXXX"
        with pytest.raises(FramingError) as ei:
            decode_binary_frame(bytes(frame))
        assert ei.value.code == StageErrorCode.INVALID_ARGUMENT
        assert "magic" in ei.value.message.lower()

    def test_truncated_frame(self) -> None:
        with pytest.raises(FramingError) as ei:
            decode_binary_frame(b"STG1\x00\x00")
        assert ei.value.code == StageErrorCode.INVALID_ARGUMENT

    def test_header_length_overflow_vs_buffer(self) -> None:
        # Claim huge header but buffer is small
        frame = MAGIC + struct.pack(">I", 1000) + b"{}"
        with pytest.raises(FramingError) as ei:
            decode_binary_frame(frame)
        assert ei.value.code in (StageErrorCode.INVALID_ARGUMENT, StageErrorCode.FRAME_TOO_LARGE)

    def test_malformed_header_json(self) -> None:
        bad_header = b"not-json"
        frame = MAGIC + struct.pack(">I", len(bad_header)) + bad_header + b"\x00" * 10
        with pytest.raises(FramingError) as ei:
            decode_binary_frame(frame)
        assert ei.value.code == StageErrorCode.INVALID_ARGUMENT

    def test_payload_bytes_mismatch(self) -> None:
        env, pcm = _audio_envelope(sample_count=320)
        # Force wrong payload_bytes in envelope
        data = env.model_dump(mode="json", exclude_none=True)
        data["payload"]["payload_bytes"] = len(pcm) - 2
        bad_env = parse_event(data)
        with pytest.raises(FramingError) as ei:
            encode_binary_frame_from_parts(envelope=bad_env, pcm=pcm)
        assert ei.value.code == StageErrorCode.INVALID_ARGUMENT

    def test_decode_payload_bytes_mismatch(self) -> None:
        env, pcm = _audio_envelope()
        frame = encode_binary_frame_from_parts(envelope=env, pcm=pcm)
        # Truncate PCM by 2 bytes without updating header
        with pytest.raises(FramingError) as ei:
            decode_binary_frame(frame[:-2])
        assert ei.value.code == StageErrorCode.INVALID_ARGUMENT


class TestOversizedFrames:
    def test_oversized_header(self) -> None:
        env, pcm = _audio_envelope()
        # Build a frame with a header larger than MAX_HEADER_BYTES
        huge = b"{" + b"a" * (MAX_HEADER_BYTES + 10) + b"}"
        frame = MAGIC + struct.pack(">I", len(huge)) + huge + pcm
        with pytest.raises(FramingError) as ei:
            decode_binary_frame(frame)
        assert ei.value.code == StageErrorCode.FRAME_TOO_LARGE

    def test_oversized_pcm_payload(self) -> None:
        big = DEFAULT_MAX_FRAME_BYTES + 1
        sample_count = big // 2
        env, pcm = _audio_envelope(sample_count=sample_count, pcm_len=big)
        with pytest.raises(FramingError) as ei:
            encode_binary_frame_from_parts(
                envelope=env, pcm=pcm, max_frame_bytes=DEFAULT_MAX_FRAME_BYTES
            )
        assert ei.value.code == StageErrorCode.FRAME_TOO_LARGE

    def test_text_frame_limit(self) -> None:
        ok = "x" * MAX_TEXT_FRAME_BYTES
        validate_text_frame_size(ok)
        with pytest.raises(FramingError) as ei:
            validate_text_frame_size(ok + "y")
        assert ei.value.code == StageErrorCode.FRAME_TOO_LARGE


class TestMediaMonotonicityWithFrames:
    def test_sequence_and_sample_clock(self) -> None:
        clock = MediaClock(stream_id="source:main")
        for i in range(5):
            env, pcm = _audio_envelope(media_sequence=i, start_sample=i * 320, sample_count=320)
            frame = encode_binary_frame_from_parts(envelope=env, pcm=pcm)
            decoded = decode_binary_frame(frame)
            clock.observe_frame(decoded.audio_payload)
        assert clock.next_start_sample == 5 * 320

    def test_gap_then_continue(self) -> None:
        clock = MediaClock(stream_id="source:main")
        env0, pcm0 = _audio_envelope(media_sequence=0, start_sample=0)
        clock.observe_frame(
            decode_binary_frame(
                encode_binary_frame_from_parts(envelope=env0, pcm=pcm0)
            ).audio_payload
        )
        # gap samples 320..959
        clock.apply_gap(skip_to_start_sample=960, skip_to_media_sequence=2)
        env2, pcm2 = _audio_envelope(media_sequence=2, start_sample=960)
        data = env2.model_dump(mode="json", exclude_none=True)
        data["payload"]["discontinuity"] = True
        env2 = parse_event(data)
        clock.observe_frame(
            decode_binary_frame(
                encode_binary_frame_from_parts(envelope=env2, pcm=pcm2)
            ).audio_payload
        )

    def test_noncontiguous_without_gap_fails(self) -> None:
        clock = MediaClock(stream_id="source:main")
        env0, pcm0 = _audio_envelope(media_sequence=0, start_sample=0)
        clock.observe_frame(BinaryAudioPayload.model_validate(env0.payload))
        env2, _pcm2 = _audio_envelope(media_sequence=2, start_sample=640)
        with pytest.raises(ValidationError) as ei:
            clock.observe_frame(BinaryAudioPayload.model_validate(env2.payload))
        assert ei.value.code == StageErrorCode.SEQUENCE_GAP


class TestGoldenBinaryVectors:
    def test_golden_frame_sha(self) -> None:
        path = FIXTURES / "binary" / "listen_audio_frame.stg1"
        if not path.exists():
            pytest.skip("fixtures not generated yet")
        raw = path.read_bytes()
        decoded = decode_binary_frame(raw)
        # Re-encode with canonical header and compare payload identity
        re = encode_binary_frame_from_parts(envelope=decoded.envelope, pcm=decoded.pcm)
        assert decode_binary_frame(re).pcm == decoded.pcm
        meta_path = FIXTURES / "binary" / "listen_audio_frame.meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            assert meta["sha256"] == sha256_hex(raw)
            assert meta["pcm_sha256"] == sha256_hex(decoded.pcm)
