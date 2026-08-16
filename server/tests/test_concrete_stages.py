from __future__ import annotations

from typing import Any

import pytest

from src.models import ListenProduct, StageKind, TranslateProduct, WordSpan
from src.pipelines import create_default_stage_registry
from src.pipelines.stages_listen.whisper import WhisperListenStage
from src.pipelines.stages_speak.edge_tts import EdgeTTSSpeakStage
from src.pipelines.stages_speak.qwen3_client import Qwen3TTSClientStage
from src.pipelines.stages_translate.opus_mt import OpusMTTranslateStage


def test_concrete_stages_registered() -> None:
    registry = create_default_stage_registry()
    ids = {info.id for info in registry.list_all()}
    assert "whisper-listen" in ids
    assert "opus-mt-en-es" in ids
    assert "edge-tts-es" in ids
    assert "qwen3-tts-0.6b" in ids
    assert "passthrough-listen" in ids
    assert "baseline-prosody" in ids

    defaults = {
        info.kind: info.id
        for info in registry.list_all()
        if info.default_for_kind
    }
    assert defaults[StageKind.LISTEN] == "whisper-listen"
    assert defaults[StageKind.TRANSLATE] == "opus-mt-en-es"
    assert defaults[StageKind.SPEAK] == "edge-tts-es"
    assert defaults[StageKind.PROSODY] == "baseline-prosody"


@pytest.mark.asyncio
async def test_whisper_listen_protocol_with_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = WhisperListenStage(sample_rate=16000)

    class _FakeModel:
        pass

    async def fake_start(self: WhisperListenStage) -> None:
        self._model = _FakeModel()

    monkeypatch.setattr(WhisperListenStage, "start", fake_start)

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
            text="hello",
            is_final=is_final,
            words=[WordSpan(text="hello", start_ms=start_ms, end_ms=start_ms + 100)],
        )

    monkeypatch.setattr(WhisperListenStage, "_decode_frame", fake_decode)
    await stage.start()

    async def audio():
        import numpy as np

        samples = np.zeros(int(16000 * 3.1), dtype=np.int16)
        yield samples.tobytes()

    products = [p async for p in stage.transcribe(audio())]
    assert products
    assert products[0].text == "hello"


@pytest.mark.asyncio
async def test_opus_mt_protocol_with_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = OpusMTTranslateStage(sample_rate=16000)

    async def fake_start(self: OpusMTTranslateStage) -> None:
        self._translator = object()
        self._sp_source = object()
        self._sp_target = object()

    monkeypatch.setattr(OpusMTTranslateStage, "start", fake_start)
    monkeypatch.setattr(
        "src.pipelines.stages_translate.opus_mt._translate_sync",
        lambda translator, sp_source, sp_target, text: f"ES:{text}",
    )

    async def products():
        yield ListenProduct(
            sequence=0,
            utterance_id="u1",
            text="hello",
            is_final=True,
            words=[WordSpan(text="hello", start_ms=0, end_ms=100)],
        )

    out = [p async for p in stage.translate(products())]
    assert len(out) == 1
    assert out[0].text == "ES:hello"
    assert out[0].instructions is not None
    assert out[0].instructions.markers


@pytest.mark.asyncio
async def test_edge_tts_protocol_with_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    stage = EdgeTTSSpeakStage(sample_rate=16000)

    async def fake_synth(text: str, target_rate: int) -> bytes:
        return b"\x00\x01" * 8

    monkeypatch.setattr(
        "src.pipelines.stages_speak.edge_tts.synthesize_spanish",
        fake_synth,
    )

    async def products():
        yield TranslateProduct(
            sequence=0,
            source_utterance_id="u1",
            target_utterance_id="t1",
            text="hola",
            is_final=True,
        )

    chunks = [c async for c in stage.synthesize(products())]
    assert chunks and chunks[0] == b"\x00\x01" * 8


@pytest.mark.asyncio
async def test_qwen_requires_url() -> None:
    stage = Qwen3TTSClientStage(sample_rate=16000, ws_url="")
    with pytest.raises(RuntimeError, match="QWEN3_TTS_WS_URL"):
        await stage.start()
