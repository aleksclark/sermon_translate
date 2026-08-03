from __future__ import annotations

from src.models import (
    ListenProduct,
    MetadataEnvelope,
    MetadataKind,
    ProsodyFrame,
    WordSpan,
)
from src.pipelines.prosody_tokens import (
    DEFAULT_N_BINS,
    ProsodyAligner,
    quantize_prosody,
    quantize_value,
)


class TestQuantizeProsody:
    def test_determinism(self) -> None:
        kwargs = {
            "f0_values": [120.0, 125.0, 130.0, 128.0],
            "energy_values": [0.2, 0.25, 0.22],
            "duration_ms": 240.0,
        }
        first = quantize_prosody(**kwargs)
        second = quantize_prosody(**kwargs)
        assert first == second

    def test_bin_bounds(self) -> None:
        token = quantize_prosody(
            f0_values=[50.0, 400.0],
            energy_values=[0.0, 1.0],
            duration_ms=1000.0,
            n_bins=DEFAULT_N_BINS,
        )
        for value in (
            token.pitch_median,
            token.pitch_range,
            token.pitch_slope,
            token.duration,
            token.energy,
        ):
            assert 0 <= value < DEFAULT_N_BINS

    def test_silence_defaults(self) -> None:
        token = quantize_prosody(
            f0_values=[],
            energy_values=[0.0, 0.0],
            duration_ms=100.0,
        )
        assert token.pitch_median == 0
        assert token.pitch_range == 0
        assert token.energy == 0
        assert token.f0_hz is None
        assert token.energy_rms == 0.0
        assert 0 <= token.duration < DEFAULT_N_BINS
        assert token.pitch_slope == DEFAULT_N_BINS // 2

    def test_quantize_value_edges(self) -> None:
        assert quantize_value(0.0, 0.0, 1.0, 4) == 0
        assert quantize_value(1.0, 0.0, 1.0, 4) == 3
        assert quantize_value(0.49, 0.0, 1.0, 4) == 1


class TestProsodyAligner:
    def test_attaches_tokens_by_time_overlap(self) -> None:
        frames = [
            MetadataEnvelope(
                stream="prosody",
                kind=MetadataKind.PROSODY,
                sequence=0,
                start_ms=0.0,
                end_ms=100.0,
                prosody=ProsodyFrame(f0_hz=140.0, energy=0.3, is_pause=False),
            ),
            MetadataEnvelope(
                stream="prosody",
                kind=MetadataKind.PROSODY,
                sequence=1,
                start_ms=100.0,
                end_ms=200.0,
                prosody=ProsodyFrame(f0_hz=150.0, energy=0.35, is_pause=False),
            ),
            MetadataEnvelope(
                stream="prosody",
                kind=MetadataKind.PROSODY,
                sequence=2,
                start_ms=400.0,
                end_ms=500.0,
                prosody=ProsodyFrame(f0_hz=90.0, energy=0.1, is_pause=False),
            ),
        ]
        product = ListenProduct(
            sequence=0,
            utterance_id="utt-1",
            text="hello there",
            words=[
                WordSpan(text="hello", start_ms=0.0, end_ms=180.0),
                WordSpan(text="there", start_ms=250.0, end_ms=360.0),
            ],
        )

        aligned = ProsodyAligner().align(product, frames)
        assert aligned.words[0].prosody is not None
        assert aligned.words[0].prosody.f0_hz is not None
        assert aligned.words[0].prosody.energy_rms is not None
        assert 0 <= aligned.words[0].prosody.pitch_median < DEFAULT_N_BINS
        assert aligned.words[1].prosody is None
