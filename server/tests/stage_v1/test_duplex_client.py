"""G1 pre-EOS full-duplex client tests (anti-cheat latches).

Source iterators withhold EOS until product/audio is observed. Tests also
assert sender+receiver tasks are simultaneously live during the exchange.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import pytest

from src.stage_v1.client import (
    PeerTransport,
    StageV1Client,
    duplex_tasks_live,
)
from src.stage_v1.models import (
    EventEnvelope,
    EventType,
    ListenProductPayload,
    SpeakCompletePayload,
    SpeakRequestPayload,
    StageKind,
    TranslateProductPayload,
    TranslateRequestPayload,
    parse_event,
)
from src.stage_v1.peer import ScriptedResponse, ScriptedStagePeer


def _deadline(hours: float = 1.0) -> str:
    return (datetime.now(UTC) + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _pcm_frame(*, samples: int = 320) -> bytes:
    # 20 ms mono s16le @ 16 kHz
    return b"\x01\x00" * samples


def _base_ids() -> dict[str, Any]:
    return {
        "session_id": "duplex-session",
        "correlation_id": f"corr-{uuid4()}",
        "attempt_id": str(uuid4()),
        "cancel_id": str(uuid4()),
    }


def _listen_product_event(peer: ScriptedStagePeer, base: dict[str, Any]) -> EventEnvelope:
    return parse_event(
        {
            "schema_version": "stage.v1",
            "event_type": "listen.product",
            "message_id": str(uuid4()),
            "event_sequence": 10,
            "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "correlation_id": base["correlation_id"],
            "session_id": base["session_id"],
            "owner_generation": 0,
            "stage_kind": "listen",
            "stage_id": peer.stage_id,
            "attempt_id": base["attempt_id"],
            "cancel_id": base["cancel_id"],
            "stage_instance_id": peer.stage_instance_id,
            "stage_version": peer.stage_version,
            "model_revision": peer.model_revision,
            "model_artifact_digest": peer.model_artifact_digest,
            "provenance_id": "sha256:" + ("11" * 32),
            "utterance_id": base.get("utterance_id", str(uuid4())),
            "utterance_sequence": 0,
            "deadline_at": _deadline(),
            "payload": {
                "revision": 0,
                "text": "hello",
                "committed_prefix_chars": 5,
                "is_final": True,
                "language": "en",
                "source_start_sample": 0,
                "source_end_sample": 320,
            },
        }
    )


def _translate_product_event(peer: ScriptedStagePeer, base: dict[str, Any]) -> EventEnvelope:
    return parse_event(
        {
            "schema_version": "stage.v1",
            "event_type": "translate.product",
            "message_id": str(uuid4()),
            "event_sequence": 10,
            "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "correlation_id": base["correlation_id"],
            "session_id": base["session_id"],
            "owner_generation": 0,
            "stage_kind": "translate",
            "stage_id": peer.stage_id,
            "attempt_id": base["attempt_id"],
            "cancel_id": base["cancel_id"],
            "stage_instance_id": peer.stage_instance_id,
            "stage_version": peer.stage_version,
            "model_revision": peer.model_revision,
            "model_artifact_digest": peer.model_artifact_digest,
            "provenance_id": "sha256:" + ("22" * 32),
            "utterance_id": base.get("utterance_id", str(uuid4())),
            "utterance_sequence": 0,
            "deadline_at": _deadline(),
            "payload": {
                "source_span_id": "span-0",
                "target_span_id": "tgt-0",
                "revision": 0,
                "text": "hola",
                "committed_prefix_chars": 4,
                "is_final": True,
                "source_char_start": 0,
                "source_char_end": 5,
                "target_language": "es",
            },
        }
    )


def _speak_audio_event(
    peer: ScriptedStagePeer, base: dict[str, Any], *, pcm: bytes
) -> tuple[EventEnvelope, bytes]:
    sample_count = len(pcm) // 2
    env = parse_event(
        {
            "schema_version": "stage.v1",
            "event_type": "speak.audio",
            "message_id": str(uuid4()),
            "event_sequence": 10,
            "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "correlation_id": base["correlation_id"],
            "session_id": base["session_id"],
            "owner_generation": 0,
            "stage_kind": "speak",
            "stage_id": peer.stage_id,
            "attempt_id": base["attempt_id"],
            "cancel_id": base["cancel_id"],
            "stage_instance_id": peer.stage_instance_id,
            "stage_version": peer.stage_version,
            "model_revision": peer.model_revision,
            "model_artifact_digest": peer.model_artifact_digest,
            "provenance_id": "sha256:" + ("33" * 32),
            "utterance_id": base.get("utterance_id", str(uuid4())),
            "utterance_sequence": 0,
            "deadline_at": _deadline(),
            "payload": {
                "stream_id": "translated:main",
                "media_sequence": 0,
                "start_sample": 0,
                "sample_count": sample_count,
                "payload_bytes": len(pcm),
                "format": {"codec": "pcm_s16le", "sample_rate_hz": 16000, "channels": 1},
                "target_span_id": "tgt-0",
                "audio_chunk_sequence": 0,
                "discontinuity": False,
            },
        }
    )
    return env, pcm


def _speak_complete_event(peer: ScriptedStagePeer, base: dict[str, Any]) -> EventEnvelope:
    return parse_event(
        {
            "schema_version": "stage.v1",
            "event_type": "speak.complete",
            "message_id": str(uuid4()),
            "event_sequence": 11,
            "created_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "correlation_id": base["correlation_id"],
            "session_id": base["session_id"],
            "owner_generation": 0,
            "stage_kind": "speak",
            "stage_id": peer.stage_id,
            "attempt_id": base["attempt_id"],
            "cancel_id": base["cancel_id"],
            "stage_instance_id": peer.stage_instance_id,
            "stage_version": peer.stage_version,
            "model_revision": peer.model_revision,
            "model_artifact_digest": peer.model_artifact_digest,
            "provenance_id": "sha256:" + ("33" * 32),
            "utterance_id": base.get("utterance_id", str(uuid4())),
            "utterance_sequence": 0,
            "deadline_at": _deadline(),
            "payload": {
                "target_span_id": "tgt-0",
                "chunk_count": 1,
                "sample_count": 320,
                "duration_ms": 20.0,
                "is_final": True,
                "prosody_report": {
                    "prosody_status": "unsupported",
                    "consumed_fields": [],
                    "ignored_fields": [],
                },
            },
        }
    )


async def _latched_audio(
    frames: list[bytes],
    *,
    release_eos: asyncio.Event,
    eos_released: asyncio.Event,
    started: asyncio.Event | None = None,
) -> AsyncIterator[bytes]:
    if started is not None:
        started.set()
    for frame in frames:
        yield frame
        # Yield control so sender/receiver can interleave.
        await asyncio.sleep(0)
    # Anti-cheat: do not finish iterator (EOS) until product observed.
    await release_eos.wait()
    eos_released.set()


async def _latched_requests(
    items: list[Any],
    *,
    release_eos: asyncio.Event,
    eos_released: asyncio.Event,
) -> AsyncIterator[Any]:
    for item in items:
        yield item
        await asyncio.sleep(0)
    await release_eos.wait()
    eos_released.set()


@pytest.mark.asyncio
async def test_listen_product_arrives_before_source_eos() -> None:
    ids = _base_ids()
    utterance_id = str(uuid4())
    ids["utterance_id"] = utterance_id

    peer = ScriptedStagePeer(
        stage_kind=StageKind.LISTEN,
        stage_id="scripted-listen",
        scripted=[],  # filled after peer identity known? identity is fixed at init
    )
    # Script product after first listen.audio; peer fills stage_instance on accept.
    # Use a placeholder event rebuilt after handshake via custom peer script timing:
    transport = PeerTransport(peer)
    client = StageV1Client(
        transport,
        session_id=ids["session_id"],
        stage_kind=StageKind.LISTEN,
        stage_id="scripted-listen",
        correlation_id=ids["correlation_id"],
        attempt_id=ids["attempt_id"],
        cancel_id=ids["cancel_id"],
        default_deadline_s=5.0,
    )

    # Install script that emits product on first audio using peer identity after start.
    product_holder: dict[str, EventEnvelope] = {}

    async def _install_and_run() -> ListenProductPayload:
        await client.start()
        assert duplex_tasks_live(client)
        product_holder["event"] = _listen_product_event(peer, ids)
        peer.scripted.append(
            ScriptedResponse(
                after_inbound_types=frozenset({"listen.audio"}),
                delay_s=0.01,
                event=product_holder["event"],
            )
        )

        release_eos = asyncio.Event()
        eos_released = asyncio.Event()
        frames = [_pcm_frame(), _pcm_frame()]

        async def _consume() -> ListenProductPayload:
            assert duplex_tasks_live(client)
            agen = client.listen(
                _latched_audio(frames, release_eos=release_eos, eos_released=eos_released),
                utterance_id=utterance_id,
                deadline_at=_deadline(),
            )
            first = await agen.__anext__()
            # Product observed — only now may EOS proceed.
            assert not eos_released.is_set(), "source EOS released before product (cheat)"
            assert duplex_tasks_live(client)
            release_eos.set()
            # Finish generator (final product already yielded).
            async for _ in agen:
                pass
            return first

        product = await asyncio.wait_for(_consume(), timeout=3.0)
        assert product.committed_prefix_chars > 0
        assert product.text
        assert eos_released.is_set()
        assert client.outbound_high_water <= 32
        assert client.inbound_high_water <= 32
        return product

    try:
        product = await _install_and_run()
        assert isinstance(product, ListenProductPayload)
        assert product.committed_prefix_chars >= 5
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_translate_product_arrives_before_request_eos() -> None:
    ids = _base_ids()
    utterance_id = str(uuid4())
    ids["utterance_id"] = utterance_id

    peer = ScriptedStagePeer(stage_kind=StageKind.TRANSLATE, stage_id="scripted-translate")
    transport = PeerTransport(peer)
    client = StageV1Client(
        transport,
        session_id=ids["session_id"],
        stage_kind=StageKind.TRANSLATE,
        stage_id="scripted-translate",
        correlation_id=ids["correlation_id"],
        attempt_id=ids["attempt_id"],
        cancel_id=ids["cancel_id"],
        default_deadline_s=5.0,
    )

    release_eos = asyncio.Event()
    eos_released = asyncio.Event()

    try:
        await client.start()
        assert duplex_tasks_live(client)
        peer.scripted.append(
            ScriptedResponse(
                after_inbound_types=frozenset({"translate.request"}),
                delay_s=0.01,
                event=_translate_product_event(peer, ids),
            )
        )

        req = TranslateRequestPayload(
            source_span_id="span-0",
            source_revision=0,
            source_char_start=0,
            source_char_end=5,
            text="hello",
            source_language="en",
            target_language="es",
        )

        async def _consume() -> TranslateProductPayload:
            agen = client.translate(
                _latched_requests([req], release_eos=release_eos, eos_released=eos_released),
                utterance_id=utterance_id,
                deadline_at=_deadline(),
            )
            first = await asyncio.wait_for(agen.__anext__(), timeout=3.0)
            assert not eos_released.is_set(), "request EOS released before product (cheat)"
            assert duplex_tasks_live(client)
            release_eos.set()
            # Exhaust generator cleanly.
            async for _ in agen:
                pass
            return first

        product = await _consume()
        assert product.text == "hola"
        assert product.is_final is True
        assert product.target_language == "es"
        assert eos_released.is_set()
        assert duplex_tasks_live(client) or client.sender_task is not None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_speak_audio_arrives_before_request_stream_eos() -> None:
    ids = _base_ids()
    utterance_id = str(uuid4())
    ids["utterance_id"] = utterance_id
    pcm = _pcm_frame()

    peer = ScriptedStagePeer(stage_kind=StageKind.SPEAK, stage_id="scripted-speak")
    transport = PeerTransport(peer)
    client = StageV1Client(
        transport,
        session_id=ids["session_id"],
        stage_kind=StageKind.SPEAK,
        stage_id="scripted-speak",
        correlation_id=ids["correlation_id"],
        attempt_id=ids["attempt_id"],
        cancel_id=ids["cancel_id"],
        default_deadline_s=5.0,
    )

    release_eos = asyncio.Event()
    eos_released = asyncio.Event()

    try:
        await client.start()
        assert duplex_tasks_live(client)
        audio_env, audio_pcm = _speak_audio_event(peer, ids, pcm=pcm)
        peer.scripted.append(
            ScriptedResponse(
                after_inbound_types=frozenset({"speak.request"}),
                delay_s=0.01,
                event=audio_env,
                binary_pcm=audio_pcm,
            )
        )
        # complete after second trigger (eos) or same request — emit complete via second script
        peer.scripted.append(
            ScriptedResponse(
                after_inbound_types=frozenset({"speak.request"}),
                delay_s=0.02,
                event=_speak_complete_event(peer, ids),
            )
        )

        req = SpeakRequestPayload(
            target_span_id="tgt-0",
            text="hola",
            target_language="es",
            publication_order=0,
            voice_id="test-voice",
            voice_revision="1",
            voice_config_digest="sha256:" + ("aa" * 32),
        )

        saw_audio = False

        async def _consume() -> bytes:
            nonlocal saw_audio
            agen = client.speak(
                _latched_requests([req], release_eos=release_eos, eos_released=eos_released),
                utterance_id=utterance_id,
                deadline_at=_deadline(),
            )
            first = await asyncio.wait_for(agen.__anext__(), timeout=3.0)
            if isinstance(first, SpeakCompletePayload):
                raise AssertionError("speak.complete before speak.audio")
            chunk, env = first
            assert env.event_type == EventType.SPEAK_AUDIO
            assert len(chunk) > 0
            assert not eos_released.is_set(), "request EOS released before speak.audio (cheat)"
            assert duplex_tasks_live(client)
            saw_audio = True
            release_eos.set()
            # Drain complete
            async for item in agen:
                if isinstance(item, SpeakCompletePayload):
                    break
            return chunk

        chunk = await _consume()
        assert saw_audio
        assert len(chunk) == len(pcm)
        assert eos_released.is_set()
    finally:
        await client.close()
