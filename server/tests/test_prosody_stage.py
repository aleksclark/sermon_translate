from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import numpy as np
import pytest

from src.models import MetadataKind
from src.pipelines import (
    BaselineProsodyStage,
    PitchEstimate,
    PitchTracker,
    ProsodyStage,
    YinPitchTracker,
)

SEED = 20240607


def _pcm(samples: np.ndarray) -> bytes:
    return (samples * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


async def _one_chunk(chunk: bytes) -> AsyncIterator[bytes]:
    yield chunk


def _timebase(rate: int, seconds: float) -> np.ndarray:
    return np.arange(int(rate * seconds), dtype=np.float64) / rate


def _sine(freq: float, rate: int, seconds: float = 1.0, amplitude: float = 0.5) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * freq * _timebase(rate, seconds))


def _sawtooth(freq: float, rate: int, seconds: float = 1.0, amplitude: float = 0.4) -> np.ndarray:
    phase = _timebase(rate, seconds) * freq
    return amplitude * 2.0 * (phase - np.floor(0.5 + phase))


def _normalized(stacked: np.ndarray, amplitude: float = 0.5) -> np.ndarray:
    return amplitude * stacked / float(np.max(np.abs(stacked)))


def _harmonic_stack(freq: float, rate: int, seconds: float = 1.0, partials: int = 8) -> np.ndarray:
    t = _timebase(rate, seconds)
    stacked = np.stack(
        [(0.6**k) * np.sin(2 * np.pi * freq * (k + 1) * t + 0.3 * k) for k in range(partials)]
    )
    return _normalized(stacked.sum(axis=0))


def _missing_fundamental(freq: float, rate: int, seconds: float = 1.0) -> np.ndarray:
    t = _timebase(rate, seconds)
    stacked = np.stack([np.sin(2 * np.pi * freq * k * t + 0.1 * k) for k in (2, 3, 4, 5)])
    return _normalized(stacked.sum(axis=0))


def _white_noise(rate: int, seconds: float = 1.0, amplitude: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return amplitude * rng.standard_normal(int(rate * seconds))


class _ConstantTracker:
    def __init__(self, estimate: PitchEstimate) -> None:
        self.calls = 0
        self._estimate = estimate

    def estimate(self, frame: np.ndarray, sample_rate: int) -> PitchEstimate:
        self.calls += 1
        return self._estimate


class TestYinPitchTracker:
    def test_satisfies_pitch_tracker_protocol(self) -> None:
        assert isinstance(YinPitchTracker(), PitchTracker)

    @pytest.mark.parametrize("rate", [16000, 48000])
    @pytest.mark.parametrize("freq", [80.0, 100.0, 200.0, 330.0, 392.0])
    def test_recovers_pure_tone_fundamental(self, rate: int, freq: float) -> None:
        estimate = YinPitchTracker().estimate(_sine(freq, rate, 0.1), rate)

        assert estimate.voiced is True
        assert estimate.f0_hz is not None
        assert estimate.f0_hz == pytest.approx(freq, rel=0.01)
        assert estimate.confidence > 0.9

    @pytest.mark.parametrize("rate", [16000, 48000])
    @pytest.mark.parametrize("freq", [100.0, 200.0, 330.0])
    def test_recovers_harmonic_rich_fundamental_without_octave_error(
        self, rate: int, freq: float
    ) -> None:
        tracker = YinPitchTracker()

        for signal in (_sawtooth(freq, rate, 0.1), _harmonic_stack(freq, rate, 0.1)):
            estimate = tracker.estimate(signal, rate)
            assert estimate.voiced is True
            assert estimate.f0_hz is not None
            assert estimate.f0_hz == pytest.approx(freq, rel=0.01)
            assert estimate.confidence > 0.8

    def test_recovers_missing_fundamental(self) -> None:
        rate = 16000
        estimate = YinPitchTracker().estimate(_missing_fundamental(110.0, rate, 0.1), rate)

        assert estimate.voiced is True
        assert estimate.f0_hz is not None
        assert estimate.f0_hz == pytest.approx(110.0, rel=0.01)

    def test_survives_int16_quantization(self) -> None:
        rate = 16000
        raw = _harmonic_stack(155.0, rate, 0.1)
        quantized = np.frombuffer(_pcm(raw), dtype=np.int16).astype(np.float32) / 32768.0
        estimate = YinPitchTracker().estimate(quantized, rate)

        assert estimate.f0_hz is not None
        assert estimate.f0_hz == pytest.approx(155.0, rel=0.01)

    def test_tolerates_additive_noise(self) -> None:
        rate = 16000
        noisy = _sine(180.0, rate, 0.1) + _white_noise(rate, 0.1, amplitude=0.05)
        estimate = YinPitchTracker().estimate(noisy, rate)

        assert estimate.voiced is True
        assert estimate.f0_hz is not None
        assert estimate.f0_hz == pytest.approx(180.0, rel=0.02)

    def test_silence_is_unvoiced_with_zero_confidence(self) -> None:
        estimate = YinPitchTracker().estimate(np.zeros(1600, dtype=np.float32), 16000)

        assert estimate.voiced is False
        assert estimate.f0_hz is None
        assert estimate.confidence == 0.0

    def test_dc_offset_is_unvoiced(self) -> None:
        estimate = YinPitchTracker().estimate(np.full(1600, 0.4, dtype=np.float32), 16000)

        assert estimate.voiced is False
        assert estimate.f0_hz is None

    @pytest.mark.parametrize("rate", [16000, 48000])
    def test_white_noise_is_unvoiced_with_low_confidence(self, rate: int) -> None:
        estimate = YinPitchTracker().estimate(_white_noise(rate, 0.1), rate)

        assert estimate.voiced is False
        assert estimate.f0_hz is None
        assert estimate.confidence < 0.45

    def test_out_of_range_tone_is_rejected(self) -> None:
        rate = 16000
        estimate = YinPitchTracker().estimate(_sine(1000.0, rate, 0.1), rate)

        assert estimate.voiced is False
        assert estimate.f0_hz is None

    def test_search_range_is_configurable(self) -> None:
        rate = 16000
        signal = _sine(500.0, rate, 0.1)

        assert YinPitchTracker().estimate(signal, rate).f0_hz is None

        wide_tracker = YinPitchTracker(f0_max=800.0)
        assert wide_tracker.f0_range_hz == (60.0, 800.0)

        widened = wide_tracker.estimate(signal, rate)
        assert widened.f0_hz is not None
        assert widened.f0_hz == pytest.approx(500.0, rel=0.01)

    def test_voicing_threshold_is_configurable(self) -> None:
        rate = 16000
        noise = _white_noise(rate, 0.1)

        assert YinPitchTracker().estimate(noise, rate).voiced is False
        assert YinPitchTracker(voicing_confidence=0.0).estimate(noise, rate).voiced is True

    def test_sub_bin_interpolation_beats_integer_lag_resolution(self) -> None:
        rate = 16000
        freq = 337.0
        estimate = YinPitchTracker().estimate(_sine(freq, rate, 0.1), rate)

        assert estimate.f0_hz is not None
        nearest_integer_lag = round(rate / freq)
        assert abs(estimate.f0_hz - freq) < abs(rate / nearest_integer_lag - freq)

    def test_too_short_frame_is_unvoiced(self) -> None:
        estimate = YinPitchTracker().estimate(_sine(200.0, 16000, 0.002), 16000)

        assert estimate.voiced is False
        assert estimate.f0_hz is None

    def test_empty_frame_is_unvoiced(self) -> None:
        estimate = YinPitchTracker().estimate(np.zeros(0, dtype=np.float32), 16000)

        assert estimate.voiced is False
        assert estimate.f0_hz is None

    def test_is_deterministic(self) -> None:
        rate = 16000
        tracker = YinPitchTracker()
        signal = _harmonic_stack(212.5, rate, 0.1)

        assert tracker.estimate(signal, rate) == tracker.estimate(signal, rate)
        assert YinPitchTracker().estimate(signal, rate) == tracker.estimate(signal, rate)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"f0_min": 0.0},
            {"f0_min": 400.0, "f0_max": 400.0},
            {"aperiodicity_threshold": 0.0},
            {"aperiodicity_threshold": 1.0},
            {"voicing_confidence": 1.5},
            {"min_rms": -1.0},
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            YinPitchTracker(**kwargs)


class TestBaselineProsodyStage:
    def test_satisfies_prosody_stage_protocol(self) -> None:
        assert isinstance(BaselineProsodyStage(), ProsodyStage)

    async def test_silence_is_detected_as_pause(self) -> None:
        stage = BaselineProsodyStage(sample_rate=16000, frame_ms=100.0)
        silence = _pcm(np.zeros(16000, dtype=np.float32))
        frames = [f async for f in stage.analyze(_one_chunk(silence), "prosody")]

        assert len(frames) == 10
        for frame in frames:
            assert frame.kind == MetadataKind.PROSODY
            assert frame.prosody is not None
            assert frame.prosody.is_pause is True
            assert frame.prosody.energy == 0.0

    async def test_tone_has_energy_and_pitch(self) -> None:
        rate = 16000
        stage = BaselineProsodyStage(sample_rate=rate, frame_ms=100.0)
        tone = _pcm(_sine(220.0, rate))
        frames = [f async for f in stage.analyze(_one_chunk(tone), "prosody")]

        assert frames
        prosody = frames[0].prosody
        assert prosody is not None
        assert prosody.is_pause is False
        assert prosody.energy is not None and prosody.energy > 0.1
        assert prosody.f0_hz is not None and 180.0 < prosody.f0_hz < 260.0

    async def test_frames_are_sequential_and_ordered(self) -> None:
        rate = 16000
        stage = BaselineProsodyStage(sample_rate=rate, frame_ms=50.0)
        tone = _pcm(_sine(200.0, rate))
        frames = [f async for f in stage.analyze(_one_chunk(tone), "prosody")]

        assert [f.sequence for f in frames] == list(range(len(frames)))
        for prev, nxt in zip(frames, frames[1:], strict=False):
            assert prev.end_ms is not None and nxt.start_ms is not None
            assert nxt.start_ms >= prev.end_ms - 1e-6

    async def test_deterministic_features(self) -> None:
        stage = BaselineProsodyStage(sample_rate=16000, frame_ms=100.0)
        silence = _pcm(np.zeros(8000, dtype=np.float32))
        first = [f.prosody async for f in stage.analyze(_one_chunk(silence), "prosody")]
        second = [f.prosody async for f in stage.analyze(_one_chunk(silence), "prosody")]
        assert [p.model_dump() for p in first if p] == [p.model_dump() for p in second if p]

    async def test_deterministic_features_on_voiced_audio(self) -> None:
        rate = 16000
        stage = BaselineProsodyStage(sample_rate=rate, frame_ms=100.0)
        tone = _pcm(_harmonic_stack(143.0, rate, 0.5))
        first = [f.model_dump() async for f in stage.analyze(_one_chunk(tone), "prosody")]
        second = [f.model_dump() async for f in stage.analyze(_one_chunk(tone), "prosody")]

        assert first == second
        assert len(first) == 5

    @pytest.mark.parametrize("freq", [100.0, 200.0, 330.0])
    async def test_recovers_known_fundamental_from_pcm(self, freq: float) -> None:
        rate = 16000
        stage = BaselineProsodyStage(sample_rate=rate, frame_ms=100.0)
        tone = _pcm(_sine(freq, rate, 0.5))
        frames = [f async for f in stage.analyze(_one_chunk(tone), "prosody")]

        assert len(frames) == 5
        for frame in frames:
            assert frame.prosody is not None
            assert frame.prosody.f0_hz is not None
            assert frame.prosody.f0_hz == pytest.approx(freq, rel=0.01)
            assert frame.prosody.pitch_confidence is not None
            assert frame.prosody.pitch_confidence > 0.9
            assert frame.prosody.features["voiced"] == 1.0

    @pytest.mark.parametrize("freq", [100.0, 200.0, 330.0])
    async def test_recovers_harmonic_rich_fundamental_from_pcm(self, freq: float) -> None:
        rate = 48000
        stage = BaselineProsodyStage(sample_rate=rate, frame_ms=100.0)
        tone = _pcm(_sawtooth(freq, rate, 0.3))
        frames = [f async for f in stage.analyze(_one_chunk(tone), "prosody")]

        assert frames
        for frame in frames:
            assert frame.prosody is not None
            assert frame.prosody.f0_hz is not None
            assert frame.prosody.f0_hz == pytest.approx(freq, rel=0.01)

    async def test_white_noise_reports_no_pitch_with_low_confidence(self) -> None:
        rate = 16000
        stage = BaselineProsodyStage(sample_rate=rate, frame_ms=100.0)
        noise = _pcm(_white_noise(rate, 0.5))
        frames = [f async for f in stage.analyze(_one_chunk(noise), "prosody")]

        assert frames
        for frame in frames:
            assert frame.prosody is not None
            assert frame.prosody.is_pause is False
            assert frame.prosody.f0_hz is None
            assert frame.prosody.pitch_confidence is not None
            assert frame.prosody.pitch_confidence < 0.45
            assert frame.prosody.features["voiced"] == 0.0

    async def test_silence_reports_no_pitch(self) -> None:
        stage = BaselineProsodyStage(sample_rate=16000, frame_ms=100.0)
        silence = _pcm(np.zeros(4800, dtype=np.float32))
        frames = [f async for f in stage.analyze(_one_chunk(silence), "prosody")]

        assert frames
        for frame in frames:
            assert frame.prosody is not None
            assert frame.prosody.f0_hz is None
            assert frame.prosody.pitch_confidence == 0.0
            assert frame.prosody.boundary == "pause"

    async def test_pause_detection_tracks_speech_and_silence(self) -> None:
        rate = 16000
        stage = BaselineProsodyStage(sample_rate=rate, frame_ms=100.0)
        speech = _harmonic_stack(150.0, rate, 0.3)
        silence = np.zeros(int(rate * 0.3), dtype=np.float64)
        signal = _pcm(np.concatenate([speech, silence, speech]))
        frames = [f async for f in stage.analyze(_one_chunk(signal), "prosody")]

        pauses = [f.prosody.is_pause for f in frames if f.prosody is not None]
        boundaries = [f.prosody.boundary for f in frames if f.prosody is not None]

        assert pauses == [False] * 3 + [True] * 3 + [False] * 3
        assert boundaries == [None] * 3 + ["pause"] * 3 + [None] * 3

    async def test_timestamps_follow_the_audio_clock(self) -> None:
        rate = 16000
        stage = BaselineProsodyStage(sample_rate=rate, frame_ms=100.0)
        tone = _pcm(_sine(200.0, rate, 0.25))
        frames = [f async for f in stage.analyze(_one_chunk(tone), "prosody")]

        assert [f.start_ms for f in frames] == [0.0, 100.0, 200.0]
        assert [f.end_ms for f in frames] == [100.0, 200.0, 250.0]
        assert [f.sequence for f in frames] == [0, 1, 2]

    async def test_chunk_boundaries_do_not_shift_the_clock(self) -> None:
        rate = 16000
        stage = BaselineProsodyStage(sample_rate=rate, frame_ms=100.0)
        tone = _sine(200.0, rate, 0.3)

        async def chunks() -> AsyncIterator[bytes]:
            for piece in np.array_split(tone, 7):
                yield _pcm(piece)

        frames = [f async for f in stage.analyze(chunks(), "prosody")]

        assert [f.start_ms for f in frames] == [0.0, 100.0, 200.0]
        assert [f.end_ms for f in frames] == [100.0, 200.0, 300.0]

    async def test_features_dict_is_additive(self) -> None:
        rate = 16000
        stage = BaselineProsodyStage(sample_rate=rate, frame_ms=100.0)
        tone = _pcm(_sine(200.0, rate, 0.1))
        frames = [f async for f in stage.analyze(_one_chunk(tone), "prosody")]

        prosody = frames[0].prosody
        assert prosody is not None
        assert set(prosody.features) == {"zero_crossing_rate", "voiced"}
        assert prosody.features["zero_crossing_rate"] == pytest.approx(400.0, rel=0.05)
        assert prosody.energy is not None
        assert prosody.confidence == 1.0

    async def test_pitch_tracker_is_swappable(self) -> None:
        rate = 16000
        tracker = _ConstantTracker(PitchEstimate(f0_hz=123.5, confidence=0.75, voiced=True))
        stage = BaselineProsodyStage(sample_rate=rate, frame_ms=100.0, pitch_tracker=tracker)
        noise = _pcm(_white_noise(rate, 0.3))
        frames = [f async for f in stage.analyze(_one_chunk(noise), "prosody")]

        assert tracker.calls == 3
        for frame in frames:
            assert frame.prosody is not None
            assert frame.prosody.f0_hz == 123.5
            assert frame.prosody.pitch_confidence == 0.75

    async def test_f0_range_is_configurable_on_the_stage(self) -> None:
        rate = 16000
        signal = _pcm(_sine(520.0, rate, 0.2))

        default_stage = BaselineProsodyStage(sample_rate=rate, frame_ms=100.0)
        default_frames = [f async for f in default_stage.analyze(_one_chunk(signal), "prosody")]
        assert all(f.prosody is not None and f.prosody.f0_hz is None for f in default_frames)

        wide_stage = BaselineProsodyStage(sample_rate=rate, frame_ms=100.0, f0_max_hz=800.0)
        wide_frames = [f async for f in wide_stage.analyze(_one_chunk(signal), "prosody")]
        for frame in wide_frames:
            assert frame.prosody is not None
            assert frame.prosody.f0_hz == pytest.approx(520.0, rel=0.01)

    async def test_empty_stream_yields_nothing(self) -> None:
        stage = BaselineProsodyStage(sample_rate=16000, frame_ms=100.0)

        async def empty() -> AsyncIterator[bytes]:
            for chunk in ():
                yield chunk

        assert [f async for f in stage.analyze(empty(), "prosody")] == []

    @pytest.mark.parametrize(
        "kwargs", [{"sample_rate": 0}, {"frame_ms": 0.0}, {"silence_rms": -0.1}]
    )
    def test_rejects_invalid_parameters(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            BaselineProsodyStage(**kwargs)
