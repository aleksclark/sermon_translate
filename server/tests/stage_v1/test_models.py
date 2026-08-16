from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError as PydanticValidationError

from src.stage_v1.framing import decode_binary_frame
from src.stage_v1.models import (
    SCHEMA_VERSION,
    AudioFormat,
    BinaryAudioPayload,
    EventEnvelope,
    EventType,
    HelloPayload,
    ListenProductPayload,
    StageErrorCode,
    StageKind,
    parse_event,
    parse_payload,
)
from src.stage_v1.peer import PeerMode, ScriptedStagePeer
from src.stage_v1.provenance import canonical_json_bytes, provenance_id_from_block, sha256_hex
from src.stage_v1.validation import (
    DedupeResult,
    Fence,
    MediaClock,
    MessageIdTracker,
    RevisionState,
    ValidationError,
    check_deadline,
    check_fence,
    check_schema_version,
    validate_handshake_hello,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "stage_v1"


def _hello_dict(**overrides: object) -> dict:
    base = {
        "schema_version": "stage.v1",
        "event_type": "hello",
        "message_id": "0198aaaa-bbbb-7ccc-8ddd-eeeeeeeeeeee",
        "event_sequence": 0,
        "created_at": "2026-08-08T16:00:00.000Z",
        "correlation_id": "run-0198",
        "session_id": "product-session-id",
        "owner_generation": 7,
        "stage_kind": "listen",
        "stage_id": "whisper-listen",
        "attempt_id": "0198ffff-1111-7222-8333-444444444444",
        "cancel_id": "0198ffff-5555-7666-8777-888888888888",
        "deadline_at": "2099-01-01T00:00:00.000Z",
        "traceparent": None,
        "payload": {
            "audio_formats": [
                {"codec": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1},
            ],
            "limits_requested": {"max_frame_bytes": 65536, "max_inflight_events": 32},
        },
    }
    base.update(overrides)
    return base


class TestHandshakeVersion:
    def test_accepts_stage_v1(self) -> None:
        env = parse_event(_hello_dict())
        validate_handshake_hello(env)
        assert env.schema_version == SCHEMA_VERSION

    def test_rejects_unknown_major(self) -> None:
        with pytest.raises(PydanticValidationError):
            parse_event(_hello_dict(schema_version="stage.v2"))

    def test_check_schema_version_rejects_other(self) -> None:
        with pytest.raises(ValidationError) as ei:
            check_schema_version("stage.v99")
        assert ei.value.code == StageErrorCode.VERSION_UNSUPPORTED

    def test_resume_rejected(self) -> None:
        data = _hello_dict()
        data["payload"] = {**data["payload"], "resume": {"attempt_id": "x"}}  # type: ignore[index]
        env = parse_event(data)
        with pytest.raises(ValidationError) as ei:
            validate_handshake_hello(env)
        assert ei.value.code == StageErrorCode.RESUME_UNSUPPORTED


class TestEnvelopeRoundTrip:
    def test_round_trip_json(self) -> None:
        env = parse_event(_hello_dict())
        raw = env.model_dump_json(exclude_none=True)
        again = EventEnvelope.model_validate_json(raw)
        assert again.event_type == EventType.HELLO
        assert again.session_id == "product-session-id"
        assert again.payload["limits_requested"]["max_frame_bytes"] == 65536

    def test_canonical_stable(self) -> None:
        env = parse_event(_hello_dict())
        a = canonical_json_bytes(env.model_dump(mode="json", exclude_none=True))
        b = canonical_json_bytes(env.model_dump(mode="json", exclude_none=True))
        assert a == b
        assert sha256_hex(a) == sha256_hex(b)

    def test_hello_payload_typed(self) -> None:
        env = parse_event(_hello_dict())
        payload = parse_payload(env.event_type, env.payload)
        assert isinstance(payload, HelloPayload)
        assert payload.limits_requested.max_frame_bytes == 65536


class TestUnknownAdditiveFields:
    def test_unknown_envelope_field_ignored(self) -> None:
        data = _hello_dict()
        data["future_field"] = {"nested": True}
        env = parse_event(data)
        assert env.event_type == EventType.HELLO
        dumped = env.model_dump(mode="json")
        assert "future_field" not in dumped

    def test_unknown_payload_field_ignored(self) -> None:
        data = _hello_dict()
        data["payload"] = {**data["payload"], "experimental_flag": 1}  # type: ignore[index]
        env = parse_event(data)
        payload = HelloPayload.model_validate(env.payload)
        assert (
            not hasattr(payload, "experimental_flag")
            or "experimental_flag" not in payload.model_dump()
        )


class TestMessageIdDedupe:
    def test_duplicate_identical_idempotent(self) -> None:
        env = parse_event(_hello_dict())
        tracker = MessageIdTracker()
        canonical = canonical_json_bytes(env.model_dump(mode="json", exclude_none=True))
        assert tracker.observe(env.message_id, canonical) == DedupeResult.NEW
        assert tracker.observe(env.message_id, canonical) == DedupeResult.IDEMPOTENT

    def test_duplicate_conflict(self) -> None:
        env = parse_event(_hello_dict())
        tracker = MessageIdTracker()
        canonical = canonical_json_bytes(env.model_dump(mode="json", exclude_none=True))
        tracker.observe(env.message_id, canonical)
        other = canonical_json_bytes(
            {**env.model_dump(mode="json", exclude_none=True), "event_sequence": 1}
        )
        with pytest.raises(ValidationError) as ei:
            tracker.observe(env.message_id, other)
        assert ei.value.code == StageErrorCode.DUPLICATE_CONFLICT


class TestFenceAndDeadline:
    def test_fence_match(self) -> None:
        env = parse_event(
            _hello_dict(
                stage_instance_id="0198inst-0000-7000-8000-000000000001",
                stage_version="1.0.0",
            )
        )
        # promote to accepted-like fields
        data = env.model_dump(mode="json", exclude_none=True)
        data["event_type"] = "listen.product"
        data["stage_instance_id"] = "0198inst-0000-7000-8000-000000000001"
        data["stage_version"] = "1.0.0"
        data["payload"] = {
            "revision": 0,
            "text": "hi",
            "committed_prefix_chars": 2,
            "is_final": True,
            "language": "en",
            "source_start_sample": 0,
            "source_end_sample": 320,
        }
        product = parse_event(data)
        active = Fence.from_envelope(product)
        check_fence(active, product)

    def test_stale_fence(self) -> None:
        env = parse_event(
            _hello_dict(
                stage_instance_id="aaaa",
                stage_version="1.0.0",
            )
        )
        data = env.model_dump(mode="json", exclude_none=True)
        data["event_type"] = "listen.product"
        data["stage_instance_id"] = "aaaa"
        data["stage_version"] = "1.0.0"
        data["payload"] = {
            "revision": 0,
            "text": "hi",
            "committed_prefix_chars": 2,
            "is_final": True,
            "language": "en",
            "source_start_sample": 0,
            "source_end_sample": 320,
        }
        product = parse_event(data)
        active = Fence.from_envelope(product)
        stale_data = dict(data)
        stale_data["cancel_id"] = "different-cancel"
        stale = parse_event(stale_data)
        with pytest.raises(ValidationError) as ei:
            check_fence(active, stale)
        assert ei.value.code == StageErrorCode.STALE_FENCE

    def test_deadline_exceeded(self) -> None:
        past = (datetime.now(UTC) - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S.%f")[
            :-3
        ] + "Z"
        with pytest.raises(ValidationError) as ei:
            check_deadline(past)
        assert ei.value.code == StageErrorCode.DEADLINE_EXCEEDED

    def test_deadline_ok_future(self) -> None:
        future = (datetime.now(UTC) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.%f")[
            :-3
        ] + "Z"
        check_deadline(future)


class TestRevisionAndCommit:
    def test_monotonic_revisions(self) -> None:
        state = RevisionState()
        assert (
            state.observe(revision=0, text="Hel", committed_prefix_chars=0, is_final=False)
            == DedupeResult.NEW
        )
        assert (
            state.observe(revision=1, text="Hello", committed_prefix_chars=3, is_final=False)
            == DedupeResult.NEW
        )
        assert state.committed_text_prefix == "Hel"

    def test_duplicate_revision_idempotent(self) -> None:
        state = RevisionState()
        state.observe(revision=0, text="Hi", committed_prefix_chars=2, is_final=True)
        assert (
            state.observe(revision=0, text="Hi", committed_prefix_chars=2, is_final=True)
            == DedupeResult.IDEMPOTENT
        )

    def test_duplicate_revision_conflict(self) -> None:
        state = RevisionState()
        state.observe(revision=0, text="Hi", committed_prefix_chars=0, is_final=False)
        with pytest.raises(ValidationError) as ei:
            state.observe(revision=0, text="Ho", committed_prefix_chars=0, is_final=False)
        assert ei.value.code == StageErrorCode.DUPLICATE_CONFLICT

    def test_skipped_revision_gap(self) -> None:
        state = RevisionState()
        state.observe(revision=0, text="A", committed_prefix_chars=0, is_final=False)
        with pytest.raises(ValidationError) as ei:
            state.observe(revision=2, text="ABC", committed_prefix_chars=1, is_final=False)
        assert ei.value.code == StageErrorCode.SEQUENCE_GAP

    def test_commit_retraction_rejected(self) -> None:
        state = RevisionState()
        state.observe(revision=0, text="Hello", committed_prefix_chars=5, is_final=False)
        with pytest.raises(ValidationError) as ei:
            state.observe(revision=1, text="Yello", committed_prefix_chars=5, is_final=False)
        assert ei.value.code == StageErrorCode.COMMIT_RETRACTION

    def test_commit_prefix_regression_rejected(self) -> None:
        state = RevisionState()
        state.observe(revision=0, text="Hello", committed_prefix_chars=5, is_final=False)
        with pytest.raises(ValidationError) as ei:
            state.observe(revision=1, text="Hello!", committed_prefix_chars=3, is_final=False)
        assert ei.value.code == StageErrorCode.COMMIT_RETRACTION

    def test_listen_product_payload_finality(self) -> None:
        with pytest.raises(PydanticValidationError):
            ListenProductPayload(
                revision=0,
                text="Hello",
                committed_prefix_chars=3,
                is_final=True,
                source_start_sample=0,
                source_end_sample=100,
            )


class TestMediaClock:
    def test_contiguous_frames(self) -> None:
        clock = MediaClock(stream_id="source:main")
        for i in range(3):
            audio = BinaryAudioPayload(
                stream_id="source:main",
                media_sequence=i,
                start_sample=i * 320,
                sample_count=320,
                payload_bytes=640,
                format=AudioFormat(),
            )
            clock.observe_frame(audio)
        assert clock.next_start_sample == 960
        assert clock.next_media_sequence == 3

    def test_sequence_gap(self) -> None:
        clock = MediaClock(stream_id="source:main")
        clock.observe_frame(
            BinaryAudioPayload(
                stream_id="source:main",
                media_sequence=0,
                start_sample=0,
                sample_count=320,
                payload_bytes=640,
            )
        )
        with pytest.raises(ValidationError) as ei:
            clock.observe_frame(
                BinaryAudioPayload(
                    stream_id="source:main",
                    media_sequence=2,
                    start_sample=640,
                    sample_count=320,
                    payload_bytes=640,
                )
            )
        assert ei.value.code == StageErrorCode.SEQUENCE_GAP

    def test_explicit_gap_advances_clock(self) -> None:
        clock = MediaClock(stream_id="source:main")
        clock.observe_frame(
            BinaryAudioPayload(
                stream_id="source:main",
                media_sequence=0,
                start_sample=0,
                sample_count=320,
                payload_bytes=640,
            )
        )
        clock.apply_gap(skip_to_start_sample=960, skip_to_media_sequence=2)
        clock.observe_frame(
            BinaryAudioPayload(
                stream_id="source:main",
                media_sequence=2,
                start_sample=960,
                sample_count=320,
                payload_bytes=640,
                discontinuity=True,
            )
        )
        assert clock.next_media_sequence == 3


class TestProvenance:
    def test_provenance_id_stable(self) -> None:
        from src.stage_v1.models import ArtifactDigestStatus, ProvenanceBlock

        block = ProvenanceBlock(
            stage_id="whisper-listen",
            stage_version="1.0.0",
            code_git_sha="b137f834be9d108b0ed620deafba3afbab6fca73",
            model_provider_id="faster-whisper",
            model_revision="large-v3",
            model_artifact_digest="sha256:" + ("cd" * 32),
            model_artifact_status=ArtifactDigestStatus.VERIFIED,
            boot_id="boot-1",
        )
        a = provenance_id_from_block(block)
        b = provenance_id_from_block(block.model_dump(mode="json", exclude_none=True))
        assert a == b
        assert a.startswith("sha256:")
        assert len(a) == len("sha256:") + 64


class TestGoldenFixtures:
    def test_hello_fixture_loads(self) -> None:
        path = FIXTURES / "json" / "hello.json"
        if not path.exists():
            pytest.skip("fixtures not generated yet")
        data = json.loads(path.read_text())
        env = parse_event(data)
        assert env.event_type == EventType.HELLO

    def test_binary_fixture_roundtrip(self) -> None:
        path = FIXTURES / "binary" / "listen_audio_frame.stg1"
        if not path.exists():
            pytest.skip("fixtures not generated yet")
        raw = path.read_bytes()
        decoded = decode_binary_frame(raw)
        assert decoded.envelope.event_type == EventType.LISTEN_AUDIO
        assert len(decoded.pcm) == decoded.audio_payload.payload_bytes

    def test_manifest_present(self) -> None:
        path = FIXTURES / "MANIFEST.sha256.json"
        if not path.exists():
            pytest.skip("fixtures not generated yet")
        manifest = json.loads(path.read_text())
        assert manifest["algorithm"] == "sha256"
        assert len(manifest["files"]) >= 1
        for entry in manifest["files"]:
            file_path = FIXTURES / entry["path"]
            assert file_path.exists(), entry["path"]
            assert sha256_hex(file_path.read_bytes()) == entry["sha256"]


@pytest.mark.asyncio
class TestScriptedPeer:
    async def test_handshake_accept(self) -> None:
        peer = ScriptedStagePeer(stage_kind=StageKind.LISTEN)
        await peer.send(_hello_dict())
        accepted = await peer.recv()
        assert isinstance(accepted, EventEnvelope)
        assert accepted.event_type == EventType.ACCEPTED
        window = await peer.recv()
        assert isinstance(window, EventEnvelope)
        assert window.event_type == EventType.WINDOW

    async def test_version_reject(self) -> None:
        with pytest.raises(PydanticValidationError):
            parse_event(_hello_dict(schema_version="other.v1"))

    async def test_cancel_then_late_product_mode(self) -> None:
        peer = ScriptedStagePeer(mode=PeerMode.EMIT_AFTER_CANCEL)
        await peer.send(_hello_dict())
        _ = await peer.recv()  # accepted
        _ = await peer.recv()  # window
        late = parse_event(
            {
                **_hello_dict(
                    event_type="listen.product",
                    event_sequence=5,
                    message_id="0198late-0000-7000-8000-000000000099",
                    stage_instance_id=peer.stage_instance_id,
                    stage_version=peer.stage_version,
                    model_revision=peer.model_revision,
                    model_artifact_digest=peer.model_artifact_digest,
                ),
                "payload": {
                    "revision": 0,
                    "text": "stale",
                    "committed_prefix_chars": 5,
                    "is_final": True,
                    "language": "en",
                    "source_start_sample": 0,
                    "source_end_sample": 320,
                },
            }
        )
        peer.inject_late(late)
        cancel = parse_event(
            {
                **_hello_dict(
                    event_type="cancel",
                    event_sequence=1,
                    message_id="0198canc-0000-7000-8000-000000000001",
                    stage_instance_id=peer.stage_instance_id,
                    stage_version=peer.stage_version,
                ),
                "payload": {"scope": "attempt", "reason": "test"},
            }
        )
        await peer.send(cancel)
        cancelled = await peer.recv()
        assert isinstance(cancelled, EventEnvelope)
        assert cancelled.event_type == EventType.CANCELLED
        late_out = await peer.recv()
        assert isinstance(late_out, EventEnvelope)
        assert late_out.event_type == EventType.LISTEN_PRODUCT
        # Orchestrator must reject via fence after cancel — peer still emits for test harness.
        assert peer.cancelled is True
