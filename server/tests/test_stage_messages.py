from __future__ import annotations

from src.models import (
    ListenProduct,
    ProsodyToken,
    SpeakProduct,
    SynthesisInstructions,
    TranslateProduct,
    WordSpan,
)


class TestStageMessages:
    def test_listen_product_round_trip(self) -> None:
        product = ListenProduct(
            sequence=1,
            utterance_id="utt-1",
            text="hello world",
            is_final=True,
            words=[
                WordSpan(
                    text="hello",
                    start_ms=0.0,
                    end_ms=200.0,
                    conf=0.9,
                    prosody=ProsodyToken(
                        pitch_median=10,
                        pitch_range=2,
                        pitch_slope=16,
                        duration=6,
                        energy=8,
                        f0_hz=140.0,
                        energy_rms=0.2,
                        start_ms=0.0,
                        end_ms=200.0,
                    ),
                ),
                WordSpan(text="world", start_ms=200.0, end_ms=400.0),
            ],
            language="en",
        )
        restored = ListenProduct.model_validate(product.model_dump())
        assert restored == product
        assert restored.words[0].prosody is not None
        assert restored.words[0].prosody.pitch_median == 10

    def test_translate_product_carries_instructions(self) -> None:
        product = TranslateProduct(
            sequence=0,
            source_utterance_id="utt-1",
            target_utterance_id="tgt-utt-1",
            text="hola",
            words=[WordSpan(text="hola", start_ms=0.0, end_ms=180.0)],
            instructions=SynthesisInstructions(
                markers=[{"word": "hola", "prosody": {"pitch_median": 4}}]
            ),
        )
        restored = TranslateProduct.model_validate_json(product.model_dump_json())
        assert restored.instructions is not None
        assert restored.instructions.markers[0]["word"] == "hola"

    def test_speak_product_keeps_pcm_bytes(self) -> None:
        product = SpeakProduct(
            sequence=0,
            target_utterance_id="tgt-1",
            pcm=b"\x00\x01\x02\x03",
            sample_rate=16000,
            start_ms=0.0,
            end_ms=50.0,
        )
        restored = SpeakProduct.model_validate(product.model_dump())
        assert restored.pcm == b"\x00\x01\x02\x03"
        assert restored.sample_rate == 16000
