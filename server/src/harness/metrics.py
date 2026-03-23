"""Text evaluation metrics — no external dependencies."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


# ---------------------------------------------------------------------------
# Word Error Rate
# ---------------------------------------------------------------------------

def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute WER between a reference and hypothesis string.

    Returns a float in [0, ∞).  0.0 = perfect, 1.0 = 100 % errors.
    """
    ref = _tokenize(reference)
    hyp = _tokenize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,      # deletion
                d[i][j - 1] + 1,      # insertion
                d[i - 1][j - 1] + cost,  # substitution
            )

    return d[len(ref)][len(hyp)] / len(ref)


# ---------------------------------------------------------------------------
# BLEU (sentence-level, smoothed)
# ---------------------------------------------------------------------------

def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def bleu_score(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Sentence-level BLEU with +1 smoothing.

    Returns a float in [0, 1].  1.0 = perfect.
    """
    import math

    ref_tok = _tokenize(reference)
    hyp_tok = _tokenize(hypothesis)
    if not hyp_tok or not ref_tok:
        return 1.0 if (not hyp_tok and not ref_tok) else 0.0

    log_bleu = 0.0
    for n in range(1, max_n + 1):
        ref_ng = _ngrams(ref_tok, n)
        hyp_ng = _ngrams(hyp_tok, n)
        clipped = sum(min(hyp_ng[ng], ref_ng[ng]) for ng in hyp_ng)
        total = max(sum(hyp_ng.values()), 1)
        # +1 smoothing
        precision = (clipped + 1) / (total + 1)
        log_bleu += math.log(precision) / max_n

    # brevity penalty
    bp = 1.0
    if len(hyp_tok) < len(ref_tok):
        bp = math.exp(1 - len(ref_tok) / len(hyp_tok))

    return bp * math.exp(log_bleu)


# ---------------------------------------------------------------------------
# Duplicate / repetition detection
# ---------------------------------------------------------------------------

@dataclass
class DuplicateInfo:
    repeated_segments: list[str]
    total_segments: int
    duplicate_ratio: float


def detect_duplicates(segments: list[str], min_words: int = 3) -> DuplicateInfo:
    """Find segments that are exact or near-exact duplicates of a previous one."""
    seen: set[str] = set()
    repeated: list[str] = []
    for seg in segments:
        norm = _normalize(seg)
        if len(norm.split()) < min_words:
            continue
        if norm in seen:
            repeated.append(seg)
        else:
            seen.add(norm)

    total = len(segments)
    ratio = len(repeated) / total if total > 0 else 0.0
    return DuplicateInfo(
        repeated_segments=repeated,
        total_segments=total,
        duplicate_ratio=ratio,
    )


def detect_ngram_repetition(text: str, n: int = 4) -> float:
    """Fraction of n-grams that are repeated within a single text.

    High values indicate the model is in a repetition loop.
    Returns a float in [0, 1].
    """
    tokens = _tokenize(text)
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(grams)
