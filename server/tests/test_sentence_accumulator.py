from __future__ import annotations

from src.pipelines._audio import SentenceAccumulator


class TestSentenceAccumulator:
    def test_complete_sentences_emitted(self) -> None:
        acc = SentenceAccumulator()
        result = acc.push(["Hello world.", "How are you?"])
        assert result == ["Hello world.", "How are you?"]
        assert acc.flush() == []

    def test_incomplete_carried_over(self) -> None:
        acc = SentenceAccumulator()
        result = acc.push(["God does not intend"])
        assert result == []
        result = acc.push(["for the church to be a cult."])
        assert len(result) == 1
        assert "God does not intend" in result[0]
        assert "cult." in result[0]

    def test_mixed_complete_and_incomplete(self) -> None:
        acc = SentenceAccumulator()
        result = acc.push(["First sentence.", "Second part"])
        assert result == ["First sentence."]
        result = acc.push(["continues here."])
        assert len(result) == 1
        assert "Second part" in result[0]
        assert "continues here." in result[0]

    def test_flush_emits_remainder(self) -> None:
        acc = SentenceAccumulator()
        acc.push(["Trailing fragment"])
        result = acc.flush()
        assert result == ["Trailing fragment"]

    def test_flush_empty(self) -> None:
        acc = SentenceAccumulator()
        assert acc.flush() == []

    def test_question_mark_ends_sentence(self) -> None:
        acc = SentenceAccumulator()
        result = acc.push(["Is this real?", "Maybe not"])
        assert result == ["Is this real?"]
        assert acc.flush() == ["Maybe not"]

    def test_exclamation_ends_sentence(self) -> None:
        acc = SentenceAccumulator()
        result = acc.push(["Wow!", "So cool"])
        assert result == ["Wow!"]

    def test_empty_input(self) -> None:
        acc = SentenceAccumulator()
        assert acc.push([]) == []

    def test_comma_flushes_clause(self) -> None:
        acc = SentenceAccumulator()
        result = acc.push(["Hello world,", "and goodbye"])
        assert result == ["Hello world,"]
        assert acc.flush() == ["and goodbye"]

    def test_comma_disabled(self) -> None:
        acc = SentenceAccumulator(flush_on_comma=False)
        result = acc.push(["Hello world,", "and goodbye"])
        assert result == []
        assert acc.flush() == ["Hello world, and goodbye"]

    def test_comma_mid_text(self) -> None:
        acc = SentenceAccumulator()
        result = acc.push([
            "about 28% of the entire globe,",
            "it's a little bit harder.",
        ])
        assert len(result) == 2
        assert "globe," in result[0]
        assert "harder." in result[1]
