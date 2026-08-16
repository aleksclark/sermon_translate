from __future__ import annotations

from collections.abc import Sequence

from src.models import MetadataEnvelope, MetadataKind
from src.models.stage_messages import ListenProduct, ProsodyToken, WordSpan

# Documented default bin edges (inclusive low, exclusive high semantics via clamp).
# Tunable later without changing the ProsodyToken wire shape.
F0_BIN_LOW_HZ = 50.0
F0_BIN_HIGH_HZ = 400.0
F0_RANGE_LOW_HZ = 0.0
F0_RANGE_HIGH_HZ = 200.0
F0_SLOPE_LOW_HZ_PER_S = -200.0
F0_SLOPE_HIGH_HZ_PER_S = 200.0
DURATION_BIN_LOW_MS = 0.0
DURATION_BIN_HIGH_MS = 1000.0
ENERGY_BIN_LOW = 0.0
ENERGY_BIN_HIGH = 1.0
DEFAULT_N_BINS = 32
PAUSE_TOKEN = ProsodyToken(
    pitch_median=0,
    pitch_range=0,
    pitch_slope=DEFAULT_N_BINS // 2,
    duration=0,
    energy=0,
    f0_hz=None,
    energy_rms=0.0,
)


def quantize_value(value: float, low: float, high: float, n_bins: int) -> int:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    if high <= low:
        return 0
    if value <= low:
        return 0
    if value >= high:
        return n_bins - 1
    ratio = (value - low) / (high - low)
    return min(n_bins - 1, max(0, int(ratio * n_bins)))


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _slope_hz_per_s(f0_values: Sequence[float], duration_ms: float) -> float:
    if len(f0_values) < 2 or duration_ms <= 0.0:
        return 0.0
    n = len(f0_values)
    xs = [i * (duration_ms / 1000.0) / (n - 1) for i in range(n)]
    x_mean = _mean(xs)
    y_mean = _mean(f0_values)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0.0:
        return 0.0
    numer = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, f0_values, strict=True))
    return numer / denom


def quantize_prosody(
    *,
    f0_values: Sequence[float],
    energy_values: Sequence[float],
    duration_ms: float,
    n_bins: int = DEFAULT_N_BINS,
    start_ms: float | None = None,
    end_ms: float | None = None,
) -> ProsodyToken:
    """Map continuous prosody features into a stable 5-dim token.

    Bin edges (defaults):
      pitch_median: F0 Hz in [50, 400]
      pitch_range:  F0 range Hz in [0, 200]
      pitch_slope:  F0 slope Hz/s in [-200, 200]
      duration:     span ms in [0, 1000]
      energy:       RMS in [0, 1]
    """
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    voiced = [v for v in f0_values if v > 0.0]
    energies = list(energy_values)
    if not voiced and (not energies or all(e <= 0.0 for e in energies)):
        token = PAUSE_TOKEN.model_copy()
        token.duration = quantize_value(
            max(0.0, duration_ms), DURATION_BIN_LOW_MS, DURATION_BIN_HIGH_MS, n_bins
        )
        token.start_ms = start_ms
        token.end_ms = end_ms
        return token

    f0_median = _median(voiced) if voiced else 0.0
    f0_range = (max(voiced) - min(voiced)) if len(voiced) >= 2 else 0.0
    f0_slope = _slope_hz_per_s(voiced, duration_ms) if voiced else 0.0
    energy_rms = _mean(energies) if energies else 0.0

    return ProsodyToken(
        pitch_median=quantize_value(f0_median, F0_BIN_LOW_HZ, F0_BIN_HIGH_HZ, n_bins),
        pitch_range=quantize_value(f0_range, F0_RANGE_LOW_HZ, F0_RANGE_HIGH_HZ, n_bins),
        pitch_slope=quantize_value(
            f0_slope, F0_SLOPE_LOW_HZ_PER_S, F0_SLOPE_HIGH_HZ_PER_S, n_bins
        ),
        duration=quantize_value(
            max(0.0, duration_ms), DURATION_BIN_LOW_MS, DURATION_BIN_HIGH_MS, n_bins
        ),
        energy=quantize_value(energy_rms, ENERGY_BIN_LOW, ENERGY_BIN_HIGH, n_bins),
        f0_hz=f0_median if voiced else None,
        energy_rms=energy_rms,
        start_ms=start_ms,
        end_ms=end_ms,
    )


def _frame_overlaps_word(frame: MetadataEnvelope, word: WordSpan) -> bool:
    if word.start_ms is None or word.end_ms is None:
        return False
    if frame.start_ms is None or frame.end_ms is None:
        return False
    return frame.start_ms < word.end_ms and frame.end_ms > word.start_ms


class ProsodyAligner:
    """Attach quantized prosody tokens to listen words by frame time overlap."""

    def __init__(self, *, n_bins: int = DEFAULT_N_BINS) -> None:
        self._n_bins = n_bins

    def align_word(
        self, word: WordSpan, frames: Sequence[MetadataEnvelope]
    ) -> WordSpan:
        if word.start_ms is None or word.end_ms is None:
            return word

        overlapping = [
            frame
            for frame in frames
            if frame.kind == MetadataKind.PROSODY
            and frame.prosody is not None
            and _frame_overlaps_word(frame, word)
        ]
        if not overlapping:
            return word

        f0_values: list[float] = []
        energy_values: list[float] = []
        for frame in overlapping:
            prosody = frame.prosody
            assert prosody is not None
            if prosody.f0_hz is not None and prosody.f0_hz > 0.0:
                f0_values.append(prosody.f0_hz)
            if prosody.energy is not None:
                energy_values.append(prosody.energy)

        duration_ms = max(0.0, word.end_ms - word.start_ms)
        token = quantize_prosody(
            f0_values=f0_values,
            energy_values=energy_values,
            duration_ms=duration_ms,
            n_bins=self._n_bins,
            start_ms=word.start_ms,
            end_ms=word.end_ms,
        )
        return word.model_copy(update={"prosody": token})

    def align(
        self, product: ListenProduct, frames: Sequence[MetadataEnvelope]
    ) -> ListenProduct:
        if not product.words:
            return product
        words = [self.align_word(word, frames) for word in product.words]
        return product.model_copy(update={"words": words})
