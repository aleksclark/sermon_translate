from __future__ import annotations

from src.models import (
    METADATA_SCHEMA_VERSION,
    MetadataEnvelope,
    MetadataKind,
    ProsodyFrame,
    SynthesisInstructions,
)


class TestMetadataModels:
    def test_prosody_envelope_round_trip(self) -> None:
        envelope = MetadataEnvelope(
            stream="prosody",
            kind=MetadataKind.PROSODY,
            sequence=3,
            start_ms=100.0,
            end_ms=200.0,
            prosody=ProsodyFrame(
                f0_hz=120.5,
                energy=0.42,
                is_pause=False,
                confidence=1.0,
                features={"spectral_flatness": 0.1},
            ),
        )
        dumped = envelope.model_dump()
        restored = MetadataEnvelope.model_validate(dumped)
        assert restored == envelope
        assert restored.schema_version == METADATA_SCHEMA_VERSION
        assert restored.prosody is not None
        assert restored.prosody.features["spectral_flatness"] == 0.1

    def test_instructions_envelope_round_trip(self) -> None:
        envelope = MetadataEnvelope(
            stream="instructions",
            kind=MetadataKind.INSTRUCTIONS,
            sequence=0,
            instructions=SynthesisInstructions(
                hints={"emotion": "calm", "rate": 0.9},
                markers=[{"word": "peace", "emphasis": 0.8}],
            ),
        )
        restored = MetadataEnvelope.model_validate_json(envelope.model_dump_json())
        assert restored == envelope
        assert restored.instructions is not None
        assert restored.instructions.hints["emotion"] == "calm"

    def test_open_payload_survives_round_trip(self) -> None:
        envelope = MetadataEnvelope(
            stream="prosody",
            kind=MetadataKind.PROSODY,
            sequence=1,
            payload={"future_model_field": [1, 2, 3]},
        )
        restored = MetadataEnvelope.model_validate(envelope.model_dump())
        assert restored.payload["future_model_field"] == [1, 2, 3]
