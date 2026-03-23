from __future__ import annotations

from src.harness.metrics import (
    bleu_score,
    detect_duplicates,
    detect_ngram_repetition,
    word_error_rate,
)


class TestWordErrorRate:
    def test_identical(self) -> None:
        assert word_error_rate("hello world", "hello world") == 0.0

    def test_completely_wrong(self) -> None:
        assert word_error_rate("hello world", "foo bar") == 1.0

    def test_insertion(self) -> None:
        wer = word_error_rate("the cat", "the big cat")
        assert 0 < wer < 1

    def test_deletion(self) -> None:
        wer = word_error_rate("the big cat", "the cat")
        assert 0 < wer < 1

    def test_empty_reference(self) -> None:
        assert word_error_rate("", "") == 0.0
        assert word_error_rate("", "hello") == 1.0

    def test_case_insensitive(self) -> None:
        assert word_error_rate("Hello World", "hello world") == 0.0

    def test_punctuation_ignored(self) -> None:
        assert word_error_rate("Hello, world!", "hello world") == 0.0


class TestBleuScore:
    def test_identical(self) -> None:
        score = bleu_score("the cat sat on the mat", "the cat sat on the mat")
        assert score > 0.9

    def test_completely_different(self) -> None:
        score = bleu_score("the cat sat on the mat", "dogs run through fields")
        assert score < 0.3

    def test_empty(self) -> None:
        assert bleu_score("", "") == 1.0
        assert bleu_score("hello world", "") == 0.0

    def test_partial_overlap(self) -> None:
        score = bleu_score("the cat sat on the mat", "the cat sat on a rug")
        assert 0.2 < score < 0.9


class TestDetectDuplicates:
    def test_no_duplicates(self) -> None:
        result = detect_duplicates(["hello world", "goodbye world"])
        assert len(result.repeated_segments) == 0
        assert result.duplicate_ratio == 0.0

    def test_with_duplicates(self) -> None:
        result = detect_duplicates(["hello world foo", "hello world foo", "goodbye world bar"])
        assert len(result.repeated_segments) == 1
        assert result.duplicate_ratio > 0

    def test_short_segments_skipped(self) -> None:
        result = detect_duplicates(["hi", "hi", "hello world foo"])
        assert len(result.repeated_segments) == 0

    def test_empty(self) -> None:
        result = detect_duplicates([])
        assert result.duplicate_ratio == 0.0


class TestNgramRepetition:
    def test_no_repetition(self) -> None:
        rep = detect_ngram_repetition("the cat sat on the mat today")
        assert rep < 0.3

    def test_high_repetition(self) -> None:
        rep = detect_ngram_repetition(
            "the cat sat the cat sat the cat sat the cat sat"
        )
        assert rep > 0.3

    def test_short_text(self) -> None:
        assert detect_ngram_repetition("hi") == 0.0

    def test_empty(self) -> None:
        assert detect_ngram_repetition("") == 0.0
