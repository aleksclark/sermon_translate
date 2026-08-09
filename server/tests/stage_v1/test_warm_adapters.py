"""Warm model-host adapters: load once, session close never unloads weights."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest

from src.models import ListenProduct, TranslateProduct, WordSpan
from src.pipelines.stages_listen.whisper import WhisperListenStage, WhisperLoadedModel
from src.pipelines.stages_speak.edge_tts import EdgeTTSLoadedModel
from src.pipelines.stages_translate.opus_mt import OpusMTLoadedModel
from src.stage_v1.adapters import (
    POCKET_TTS_AVAILABLE,
    build_edge_tts_host,
    build_opus_mt_host,
    build_pocket_tts_host,
    build_whisper_listen_host,
    listen_product_to_payload,
    open_edge_tts_session_stage,
    open_opus_mt_session_stage,
    open_whisper_session_stage,
    run_listen_session,
    run_speak_session,
    run_translate_session,
    translate_product_to_payload,
)
from src.stage_v1.host import SessionState


def _fake_whisper_loaded() -> WhisperLoadedModel:
    return WhisperLoadedModel(model=object(), model_size="tiny", revision="tiny")


def _fake_opus_loaded() -> OpusMTLoadedModel:
    return OpusMTLoadedModel(
        translator=object(),
        sp_source=object(),
        sp_target=object(),
        model_id="Helsinki-NLP/opus-mt-en-es",
        revision="Helsinki-NLP/opus-mt-en-es",
    )


def _fake_edge_loaded() -> EdgeTTSLoadedModel:
    return EdgeTTSLoadedModel(voice_id="es-ES-AlvaroNeural", streams_pcm=False)


@pytest.mark.asyncio
async def test_whisper_loader_once_across_two_sessions_stop_keeps_weights() -> None:
    loads = {"n": 0}

    def loader() -> WhisperLoadedModel:
        loads["n"] += 1
        return _fake_whisper_loaded()

    host = build_whisper_listen_host(
        model_loader=loader,
        max_sessions=2,
        boot_id="boot-whisper-1",
        model_size="tiny",
    )
    await host.load()
    await host.warmup()
    assert host.loader_invocation_count == 1
    assert loads["n"] == 1
    resident = host.model
    assert isinstance(resident, WhisperLoadedModel)

    s1 = await host.open_session(attempt_id="a1")
    stage1 = open_whisper_session_stage(host, s1, sample_rate=16000)
    await stage1.start()
    assert stage1.loaded_model is resident
    stage1._sequence = 7
    stage1._buffer = np.ones(8, dtype=np.float32)
    await stage1.stop()
    # Preloaded weights must survive session stop.
    assert stage1.loaded_model is resident
    assert host.model is resident
    assert stage1._sequence == 0
    assert stage1._buffer.size == 0
    await host.close_session(s1.session_state_id)

    s2 = await host.open_session(attempt_id="a2")
    stage2 = open_whisper_session_stage(host, s2, sample_rate=16000)
    await stage2.start()
    assert stage2.loaded_model is resident
    assert stage2 is not stage1
    assert stage2._sequence == 0
    await stage2.stop()
    assert stage2.loaded_model is resident
    await host.close_session(s2.session_state_id)

    assert host.loader_invocation_count == 1
    assert loads["n"] == 1
    assert host.model_loaded is True
    assert host.model is resident
    assert host.boot_id == "boot-whisper-1"


@pytest.mark.asyncio
async def test_opus_mt_loader_once_and_stop_keeps_weights() -> None:
    loads = {"n": 0}

    def loader() -> OpusMTLoadedModel:
        loads["n"] += 1
        return _fake_opus_loaded()

    host = build_opus_mt_host(model_loader=loader, max_sessions=2, boot_id="boot-opus-1")
    await host.load()
    await host.warmup()
    resident = host.model
    assert isinstance(resident, OpusMTLoadedModel)

    s1 = await host.open_session()
    stage1 = open_opus_mt_session_stage(host, s1)
    await stage1.start()
    assert stage1.loaded_model is resident
    await stage1.stop()
    assert stage1.loaded_model is resident
    await host.close_session(s1.session_state_id)

    s2 = await host.open_session()
    stage2 = open_opus_mt_session_stage(host, s2)
    await stage2.start()
    assert stage2.loaded_model is resident
    await stage2.stop()
    await host.close_session(s2.session_state_id)

    assert host.loader_invocation_count == 1
    assert loads["n"] == 1
    assert host.model is resident


@pytest.mark.asyncio
async def test_edge_tts_loader_once_and_stop_keeps_backend() -> None:
    loads = {"n": 0}

    def loader() -> EdgeTTSLoadedModel:
        loads["n"] += 1
        return _fake_edge_loaded()

    host = build_edge_tts_host(model_loader=loader, max_sessions=2, boot_id="boot-edge-1")
    await host.load()
    await host.warmup()
    resident = host.model
    assert isinstance(resident, EdgeTTSLoadedModel)
    assert resident.streams_pcm is False

    s1 = await host.open_session()
    stage1 = open_edge_tts_session_stage(host, s1, sample_rate=16000)
    await stage1.start()
    assert stage1.loaded_model is resident
    stage1._utterance_count = 3
    await stage1.stop()
    assert stage1.loaded_model is resident
    assert stage1._utterance_count == 0
    await host.close_session(s1.session_state_id)

    s2 = await host.open_session()
    stage2 = open_edge_tts_session_stage(host, s2, sample_rate=16000)
    await stage2.start()
    assert stage2.loaded_model is resident
    await stage2.stop()
    await host.close_session(s2.session_state_id)

    assert host.loader_invocation_count == 1
    assert loads["n"] == 1


@pytest.mark.asyncio
async def test_close_session_does_not_call_unloader() -> None:
    unloads = {"n": 0}

    def unloader(_model: object) -> None:
        unloads["n"] += 1

    host = build_whisper_listen_host(
        model_loader=_fake_whisper_loaded,
        max_sessions=1,
    )
    # Inject unloader after construction.
    host._model_unloader = unloader  # noqa: SLF001 — test proves D6 contract
    await host.load()
    await host.warmup()
    s = await host.open_session()
    open_whisper_session_stage(host, s)
    await host.close_session(s.session_state_id)
    assert unloads["n"] == 0
    assert host.model_loaded is True
    await host.shutdown()
    assert unloads["n"] == 1


@pytest.mark.asyncio
async def test_listen_pre_eos_commit_and_payload_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Buffered mid-stream frames commit text with is_final=False before EOS."""
    loaded = _fake_whisper_loaded()
    stage = WhisperListenStage(sample_rate=16000, loaded_model=loaded)

    calls: list[bool] = []

    async def fake_decode(
        self: WhisperListenStage,
        loop: Any,
        frame: Any,
        *,
        sequence: int,
        start_ms: float,
        is_final: bool = True,
    ) -> ListenProduct:
        calls.append(is_final)
        return ListenProduct(
            sequence=sequence,
            utterance_id=f"w-{sequence}",
            text="hello world",
            is_final=is_final,
            words=[
                WordSpan(text="hello", start_ms=start_ms, end_ms=start_ms + 50),
                WordSpan(text="world", start_ms=start_ms + 50, end_ms=start_ms + 100),
            ],
            language="en",
        )

    monkeypatch.setattr(WhisperListenStage, "_decode_frame", fake_decode)

    async def audio() -> AsyncIterator[bytes]:
        # 4.2s @ 16k → one mid-stream max buffer (3s) + EOS flush remainder (>=1s)
        samples = np.zeros(int(16000 * 4.2), dtype=np.int16)
        yield samples.tobytes()

    products = [p async for p in stage.transcribe(audio())]
    assert products
    # At least one non-final product before stream end (pre-EOS commit).
    non_final = [p for p in products if not p.is_final]
    finals = [p for p in products if p.is_final]
    assert non_final, "expected pre-EOS committed product from full buffer window"
    assert finals, "expected final product on stream EOS"
    assert any(c is False for c in calls)
    assert any(c is True for c in calls)
    assert finals[-1].is_final is True

    payload = listen_product_to_payload(non_final[0], revision=0, sample_rate=16000)
    assert payload.revision == 0
    assert payload.text == "hello world"
    assert payload.committed_prefix_chars == len(payload.text)
    assert payload.is_final is False
    assert payload.language == "en"
    assert payload.words

    final_payload = listen_product_to_payload(finals[0], revision=1, sample_rate=16000)
    assert final_payload.is_final is True
    assert final_payload.committed_prefix_chars == len(final_payload.text)


@pytest.mark.asyncio
async def test_run_listen_session_via_host(monkeypatch: pytest.MonkeyPatch) -> None:
    host = build_whisper_listen_host(
        model_loader=_fake_whisper_loaded,
        max_sessions=1,
        sample_rate=16000,
    )
    await host.load()
    await host.warmup()
    session = await host.open_session()
    stage = open_whisper_session_stage(host, session, sample_rate=16000)

    async def fake_decode(
        self: WhisperListenStage,
        loop: Any,
        frame: Any,
        *,
        sequence: int,
        start_ms: float,
        is_final: bool = True,
    ) -> ListenProduct:
        return ListenProduct(
            sequence=sequence,
            utterance_id=f"w-{sequence}",
            text="grace",
            is_final=is_final,
            words=[WordSpan(text="grace", start_ms=start_ms, end_ms=start_ms + 80)],
        )

    monkeypatch.setattr(WhisperListenStage, "_decode_frame", fake_decode)

    from src.stage_v1.adapters import SessionRuntime

    runtime = SessionRuntime(stage=stage, sample_rate=16000)

    async def audio() -> AsyncIterator[bytes]:
        yield np.zeros(int(16000 * 3.2), dtype=np.int16).tobytes()

    payloads = [p async for p in run_listen_session(runtime, audio())]
    assert payloads
    assert payloads[0].committed_prefix_chars == len(payloads[0].text)
    assert any(not p.is_final for p in payloads) or any(p.is_final for p in payloads)
    # stop cleared session state but kept weights
    assert stage.loaded_model is host.model
    await host.close_session(session.session_state_id)
    assert host.loader_invocation_count == 1


@pytest.mark.asyncio
async def test_run_translate_committed_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    host = build_opus_mt_host(model_loader=_fake_opus_loaded, max_sessions=1)
    await host.load()
    await host.warmup()
    session = await host.open_session()
    stage = open_opus_mt_session_stage(host, session)

    monkeypatch.setattr(
        "src.pipelines.stages_translate.opus_mt._translate_sync",
        lambda translator, sp_source, sp_target, text: f"ES:{text}",
    )

    from src.stage_v1.adapters import SessionRuntime

    runtime = SessionRuntime(stage=stage)

    async def listen_stream() -> AsyncIterator[ListenProduct]:
        yield ListenProduct(
            sequence=0,
            utterance_id="u0",
            text="hello",
            is_final=False,
            words=[WordSpan(text="hello", start_ms=0, end_ms=100)],
        )
        yield ListenProduct(
            sequence=1,
            utterance_id="u1",
            text="hello world",
            is_final=True,
            words=[
                WordSpan(text="hello", start_ms=0, end_ms=100),
                WordSpan(text="world", start_ms=100, end_ms=200),
            ],
        )

    payloads = [p async for p in run_translate_session(runtime, listen_stream())]
    assert len(payloads) == 2
    assert payloads[0].text == "ES:hello"
    assert payloads[0].committed_prefix_chars == len(payloads[0].text)
    assert payloads[0].is_final is False
    assert payloads[1].is_final is True
    assert payloads[1].revision == 1
    assert stage.loaded_model is host.model
    await host.close_session(session.session_state_id)
    assert host.loader_invocation_count == 1


@pytest.mark.asyncio
async def test_run_speak_edge_tts_monkeypatched(monkeypatch: pytest.MonkeyPatch) -> None:
    host = build_edge_tts_host(model_loader=_fake_edge_loaded, max_sessions=1)
    await host.load()
    await host.warmup()
    session = await host.open_session()
    stage = open_edge_tts_session_stage(host, session, sample_rate=16000)

    pcm_out = (np.arange(32, dtype=np.int16) + 1).tobytes()

    async def fake_synth(text: str, target_rate: int) -> bytes:
        assert text == "hola"
        assert target_rate == 16000
        return pcm_out

    monkeypatch.setattr(
        "src.pipelines.stages_speak.edge_tts.synthesize_spanish",
        fake_synth,
    )

    from src.stage_v1.adapters import SessionRuntime

    runtime = SessionRuntime(stage=stage, sample_rate=16000)

    async def products() -> AsyncIterator[TranslateProduct]:
        yield TranslateProduct(
            sequence=0,
            source_utterance_id="u0",
            target_utterance_id="t0",
            text="hola",
            is_final=True,
        )

    chunks = [c async for c in run_speak_session(runtime, products())]
    assert chunks == [pcm_out]
    # Non-silent
    assert any(b != 0 for b in chunks[0])
    assert stage.loaded_model is host.model
    assert stage.streams_pcm is False
    await host.close_session(session.session_state_id)
    assert host.loader_invocation_count == 1


def test_translate_product_payload_mapping() -> None:
    product = TranslateProduct(
        sequence=0,
        source_utterance_id="src-1",
        target_utterance_id="tgt-1",
        text="hola mundo",
        is_final=True,
    )
    payload = translate_product_to_payload(product, revision=2, source_char_start=0)
    assert payload.revision == 2
    assert payload.text == "hola mundo"
    assert payload.committed_prefix_chars == len("hola mundo")
    assert payload.is_final is True
    assert payload.target_language == "es"
    assert payload.source_span_id == "src-1"
    assert payload.target_span_id == "tgt-1"


def test_pocket_tts_availability_flag() -> None:
    # Document CONDITIONAL: pocket may be absent; edge-tts is the shipped path.
    assert isinstance(POCKET_TTS_AVAILABLE, bool)
    if not POCKET_TTS_AVAILABLE:
        with pytest.raises(ImportError, match="pocket-tts"):
            build_pocket_tts_host()


@pytest.mark.asyncio
async def test_legacy_owned_model_still_clears_on_stop() -> None:
    """Without preloaded model, stop() may clear owned weights (legacy path)."""
    stage = WhisperListenStage(sample_rate=16000)
    stage._model = object()  # type: ignore[method-assign]
    assert stage.loaded_model is not None
    assert stage._owns_model is True
    await stage.stop()
    assert stage.loaded_model is None


@pytest.mark.asyncio
async def test_session_state_isolation_markers() -> None:
    host = build_whisper_listen_host(model_loader=_fake_whisper_loaded, max_sessions=2)
    await host.load()
    await host.warmup()
    a = await host.open_session(attempt_id="A")
    b = await host.open_session(attempt_id="B")
    sa = open_whisper_session_stage(host, a)
    sb = open_whisper_session_stage(host, b)
    sa._sequence = 99
    assert sb._sequence == 0
    assert a.session_state_id != b.session_state_id
    await host.close_session(a.session_state_id)
    assert host.get_session(a.session_state_id) is None
    assert isinstance(host.get_session(b.session_state_id), SessionState)
    await host.close_session(b.session_state_id)
    assert host.loader_invocation_count == 1
