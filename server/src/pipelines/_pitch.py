"""Numpy-only fundamental frequency estimation used by prosody stages.

The tracker is deliberately model-agnostic and dependency-free so prosody
analysis stays available in every deployment. :class:`PitchTracker` is the
seam: a stage holds one and never assumes which algorithm produced the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

_EPS = 1e-12

DEFAULT_F0_MIN_HZ = 60.0
DEFAULT_F0_MAX_HZ = 400.0
DEFAULT_APERIODICITY_THRESHOLD = 0.15
DEFAULT_VOICING_CONFIDENCE = 0.45
DEFAULT_MIN_RMS = 1e-4


@dataclass(frozen=True, slots=True)
class PitchEstimate:
    """Result of analysing a single analysis window.

    ``f0_hz`` is ``None`` whenever the window is judged unvoiced, so callers
    never see a fabricated pitch for silence or noise. ``confidence`` is the
    normalized periodicity of the window in ``[0, 1]`` and stays meaningful
    even when ``voiced`` is ``False``.
    """

    f0_hz: float | None
    confidence: float
    voiced: bool


UNVOICED = PitchEstimate(f0_hz=None, confidence=0.0, voiced=False)


@runtime_checkable
class PitchTracker(Protocol):
    """Estimates the fundamental frequency of one analysis window."""

    def estimate(self, frame: np.ndarray, sample_rate: int) -> PitchEstimate: ...


def zero_crossing_rate(frame: np.ndarray, sample_rate: int) -> float:
    """Sign changes per second, a cheap voicing/fricative hint."""
    if frame.size < 2 or sample_rate <= 0:
        return 0.0
    signs = np.signbit(frame)
    crossings = int(np.count_nonzero(signs[1:] != signs[:-1]))
    return crossings * sample_rate / (frame.size - 1)


def _difference_function(samples: np.ndarray, tau_max: int) -> np.ndarray:
    """YIN squared-difference ``d(tau)`` for ``tau`` in ``[0, tau_max]``.

    Expanding ``sum (x[j] - x[j + tau])**2`` into two power terms and a
    correlation term lets the whole lag axis be computed with one FFT pair
    instead of a Python loop over lags.
    """
    n = samples.size
    window = n - tau_max
    power = np.concatenate(([0.0], np.cumsum(np.square(samples))))
    head = power[window] - power[0]
    tail = power[window : window + tau_max + 1] - power[: tau_max + 1]
    size = 1 << int(n + window - 1).bit_length()
    spectrum = np.fft.rfft(samples, size)
    kernel = np.fft.rfft(samples[:window][::-1], size)
    correlation = np.fft.irfft(spectrum * kernel, size)[window - 1 : window + tau_max]
    return np.maximum(head + tail - 2.0 * correlation, 0.0)


def _cumulative_mean_normalized(difference: np.ndarray) -> np.ndarray:
    """Divide ``d(tau)`` by its running mean so lag 0 stops being the winner."""
    normalized = np.ones_like(difference)
    running = np.cumsum(difference[1:])
    lags = np.arange(1, difference.size, dtype=np.float64)
    normalized[1:] = np.where(
        running > _EPS, difference[1:] * lags / np.maximum(running, _EPS), 1.0
    )
    return normalized


def _select_lag(normalized: np.ndarray, tau_min: int, tau_max: int, threshold: float) -> int:
    """Pick the first dip below ``threshold``, falling back to the global minimum.

    Taking the *first* qualifying dip rather than the deepest one is what keeps
    harmonic-rich signals from locking onto an integer multiple of the true
    period, which is the classic sub-octave error of plain autocorrelation.
    """
    search = normalized[tau_min : tau_max + 1]
    below = np.flatnonzero(search < threshold)
    if below.size:
        index = int(below[0])
        while index + 1 < search.size and search[index + 1] < search[index]:
            index += 1
        return tau_min + index
    return tau_min + int(np.argmin(search))


def _has_shorter_period(normalized: np.ndarray, tau: int, threshold: float) -> bool:
    """True when the window already repeats at half the chosen lag or less.

    A tone whose fundamental sits above the configured range still looks
    perfectly periodic at integer multiples of its period, so a lag-domain
    search restricted to the range would happily report a sub-harmonic.
    Rejecting those windows turns a confidently wrong number into no number.
    """
    limit = tau // 2
    if limit < 2:
        return False
    return bool(np.min(normalized[2 : limit + 1]) < threshold)


def _refine_lag(normalized: np.ndarray, tau: int) -> tuple[float, float]:
    """Fit a parabola through the minimum and its neighbours for sub-bin accuracy."""
    if tau <= 0 or tau + 1 >= normalized.size:
        return float(tau), float(normalized[tau])
    left = float(normalized[tau - 1])
    center = float(normalized[tau])
    right = float(normalized[tau + 1])
    curvature = left + right - 2.0 * center
    if curvature <= _EPS:
        return float(tau), center
    shift = float(np.clip(0.5 * (left - right) / curvature, -1.0, 1.0))
    return float(tau) + shift, center - 0.25 * (left - right) * shift


class YinPitchTracker:
    """YIN cumulative-mean-normalized-difference pitch tracker.

    Every knob is a constructor argument so a stage can retune the search range
    or the voicing decision without subclassing, and an entirely different
    tracker can replace this one behind :class:`PitchTracker`.
    """

    def __init__(
        self,
        f0_min: float = DEFAULT_F0_MIN_HZ,
        f0_max: float = DEFAULT_F0_MAX_HZ,
        aperiodicity_threshold: float = DEFAULT_APERIODICITY_THRESHOLD,
        voicing_confidence: float = DEFAULT_VOICING_CONFIDENCE,
        min_rms: float = DEFAULT_MIN_RMS,
    ) -> None:
        if f0_min <= 0.0:
            raise ValueError("f0_min must be positive")
        if f0_max <= f0_min:
            raise ValueError("f0_max must be greater than f0_min")
        if not 0.0 < aperiodicity_threshold < 1.0:
            raise ValueError("aperiodicity_threshold must be in (0, 1)")
        if not 0.0 <= voicing_confidence <= 1.0:
            raise ValueError("voicing_confidence must be in [0, 1]")
        if min_rms < 0.0:
            raise ValueError("min_rms must be non-negative")
        self._f0_min = f0_min
        self._f0_max = f0_max
        self._aperiodicity_threshold = aperiodicity_threshold
        self._voicing_confidence = voicing_confidence
        self._min_rms = min_rms

    @property
    def f0_range_hz(self) -> tuple[float, float]:
        return self._f0_min, self._f0_max

    def estimate(self, frame: np.ndarray, sample_rate: int) -> PitchEstimate:
        if sample_rate <= 0 or frame.size == 0:
            return UNVOICED

        samples = np.asarray(frame, dtype=np.float64).ravel()
        samples = samples - samples.mean()
        if float(np.sqrt(np.mean(np.square(samples)))) < self._min_rms:
            return UNVOICED

        tau_min = max(2, int(np.ceil(sample_rate / self._f0_max)))
        tau_max = min(int(np.floor(sample_rate / self._f0_min)), samples.size // 2)
        if tau_max <= tau_min + 1:
            return UNVOICED

        normalized = _cumulative_mean_normalized(_difference_function(samples, tau_max))
        lag = _select_lag(normalized, tau_min, tau_max, self._aperiodicity_threshold)
        if _has_shorter_period(normalized, lag, self._aperiodicity_threshold):
            return UNVOICED

        tau, value = _refine_lag(normalized, lag)
        confidence = float(np.clip(1.0 - value, 0.0, 1.0))
        if tau <= 0.0:
            return UNVOICED

        f0 = sample_rate / tau
        voiced = confidence >= self._voicing_confidence and self._f0_min <= f0 <= self._f0_max
        if not voiced:
            return PitchEstimate(f0_hz=None, confidence=confidence, voiced=False)
        return PitchEstimate(f0_hz=float(f0), confidence=confidence, voiced=True)
