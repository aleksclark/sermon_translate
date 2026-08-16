"""G2: revision ledger + committed-delta routing tests."""

from __future__ import annotations

import pytest

from src.pipelines.commit_barrier import (
    CommittedDeltaRouter,
    RevisionLedger,
)
from src.stage_v1.models import DropReason, StageErrorCode, StageKind
from src.stage_v1.validation import DedupeResult, ValidationError


class TestRevisionLedgerMonotonicCommit:
    def test_partials_without_commit_produce_no_delta(self) -> None:
        ledger = RevisionLedger()
        r0 = ledger.observe(
            "utt-1",
            revision=0,
            text="Hel",
            committed_prefix_chars=0,
            stage_kind=StageKind.LISTEN,
            utterance_id="utt-1",
        )
        assert r0.dedupe is DedupeResult.NEW
        assert r0.delta is None

        r1 = ledger.observe(
            "utt-1",
            revision=1,
            text="Hello",
            committed_prefix_chars=0,
            utterance_id="utt-1",
        )
        assert r1.delta is None

    def test_commit_advance_emits_only_new_delta(self) -> None:
        ledger = RevisionLedger()
        ledger.observe("u", revision=0, text="Hello world", committed_prefix_chars=0)
        r = ledger.observe(
            "u",
            revision=1,
            text="Hello world",
            committed_prefix_chars=5,
        )
        assert r.delta is not None
        assert r.delta.text == "Hello"
        assert r.delta.char_start == 0
        assert r.delta.char_end == 5

        r2 = ledger.observe(
            "u",
            revision=2,
            text="Hello world!",
            committed_prefix_chars=11,
        )
        assert r2.delta is not None
        assert r2.delta.text == " world"
        assert r2.delta.char_start == 5
        assert r2.delta.char_end == 11

    def test_finality_requires_full_commit(self) -> None:
        ledger = RevisionLedger()
        with pytest.raises(ValidationError) as ei:
            # observe returns failed result rather than always raising —
            # StageSession raises; ledger returns failed.
            r = ledger.observe(
                "u",
                revision=0,
                text="Hi",
                committed_prefix_chars=1,
                is_final=True,
            )
            assert r.failed
            assert r.error is not None
            raise r.error
        assert ei.value.code == StageErrorCode.INVALID_ARGUMENT

    def test_final_commit_delta(self) -> None:
        ledger = RevisionLedger()
        r = ledger.observe(
            "u",
            revision=0,
            text="Done",
            committed_prefix_chars=4,
            is_final=True,
        )
        assert r.delta is not None
        assert r.delta.is_final is True
        assert r.delta.text == "Done"
        state = ledger.get_state("u")
        assert state is not None
        assert state.finalized is True

    def test_commit_retraction_fails_scope(self) -> None:
        ledger = RevisionLedger()
        ledger.observe("u", revision=0, text="Hello", committed_prefix_chars=5)
        r = ledger.observe(
            "u",
            revision=1,
            text="Yello",
            committed_prefix_chars=5,
        )
        assert r.failed
        assert r.error is not None
        assert r.error.code == StageErrorCode.COMMIT_RETRACTION
        assert ledger.is_failed("u")

        # Further observes on failed scope stay failed.
        r2 = ledger.observe("u", revision=2, text="Yello!", committed_prefix_chars=5)
        assert r2.failed

    def test_commit_prefix_regression_fails(self) -> None:
        ledger = RevisionLedger()
        ledger.observe("u", revision=0, text="Hello", committed_prefix_chars=5)
        r = ledger.observe(
            "u",
            revision=1,
            text="Hello!",
            committed_prefix_chars=3,
        )
        assert r.failed
        assert r.error is not None
        assert r.error.code == StageErrorCode.COMMIT_RETRACTION

    def test_sequence_gap_fails_scope(self) -> None:
        ledger = RevisionLedger()
        ledger.observe("u", revision=0, text="A", committed_prefix_chars=0)
        r = ledger.observe("u", revision=2, text="ABC", committed_prefix_chars=1)
        assert r.failed
        assert r.error is not None
        assert r.error.code == StageErrorCode.SEQUENCE_GAP

    def test_idempotent_revision(self) -> None:
        ledger = RevisionLedger()
        ledger.observe("u", revision=0, text="Hi", committed_prefix_chars=2, is_final=True)
        r = ledger.observe(
            "u", revision=0, text="Hi", committed_prefix_chars=2, is_final=True
        )
        assert r.dedupe is DedupeResult.IDEMPOTENT
        assert r.delta is None

    def test_duplicate_conflict_fails(self) -> None:
        ledger = RevisionLedger()
        ledger.observe("u", revision=0, text="Hi", committed_prefix_chars=0)
        r = ledger.observe("u", revision=0, text="Ho", committed_prefix_chars=0)
        assert r.failed
        assert r.error is not None
        assert r.error.code == StageErrorCode.DUPLICATE_CONFLICT


class TestCommittedDeltaRouter:
    def test_translate_receives_only_newly_committed_source_deltas(self) -> None:
        router = CommittedDeltaRouter()
        # Unstable partials — no route
        router.observe_listen_product(
            utterance_id="utt-a",
            revision=0,
            text="God",
            committed_prefix_chars=0,
        )
        router.observe_listen_product(
            utterance_id="utt-a",
            revision=1,
            text="God is",
            committed_prefix_chars=0,
        )
        assert router.translate_requests_from_listen() == []

        # Commit first word
        r = router.observe_listen_product(
            utterance_id="utt-a",
            revision=2,
            text="God is good",
            committed_prefix_chars=3,
        )
        assert r.delta is not None
        assert r.delta.text == "God"
        reqs = router.translate_requests_from_listen()
        assert len(reqs) == 1
        assert reqs[0].text == "God"
        assert reqs[0].source_span_id is not None

        # Another partial — still one request
        router.observe_listen_product(
            utterance_id="utt-a",
            revision=3,
            text="God is good today",
            committed_prefix_chars=3,
        )
        assert len(router.translate_requests_from_listen()) == 1

        # Commit more
        r2 = router.observe_listen_product(
            utterance_id="utt-a",
            revision=4,
            text="God is good today",
            committed_prefix_chars=11,
            is_final=False,
        )
        assert r2.delta is not None
        assert r2.delta.text == " is good"
        assert len(router.translate_requests_from_listen()) == 2
        # No duplicate spans
        spans = [d.source_span_id for d in router.translate_requests_from_listen()]
        assert len(spans) == len(set(spans))

    def test_speak_receives_only_committed_target_deltas(self) -> None:
        router = CommittedDeltaRouter()
        router.observe_translate_product(
            source_span_id="src-1",
            target_span_id="tgt-1",
            revision=0,
            text="Dios",
            committed_prefix_chars=0,
        )
        assert router.speak_requests_from_translate() == []

        r = router.observe_translate_product(
            source_span_id="src-1",
            target_span_id="tgt-1",
            revision=1,
            text="Dios es bueno",
            committed_prefix_chars=4,
        )
        assert r.delta is not None
        assert r.delta.text == "Dios"
        speak = router.speak_requests_from_translate()
        assert len(speak) == 1
        assert speak[0].target_span_id == "tgt-1"

    def test_coalesce_superseded_uncommitted_emits_dropped(self) -> None:
        router = CommittedDeltaRouter()
        router.observe_listen_product(
            utterance_id="u",
            revision=0,
            text="Aa",
            committed_prefix_chars=0,
        )
        router.observe_listen_product(
            utterance_id="u",
            revision=1,
            text="Aaa",
            committed_prefix_chars=0,
        )
        r = router.observe_listen_product(
            utterance_id="u",
            revision=2,
            text="Apple",
            committed_prefix_chars=5,
        )
        assert r.delta is not None
        assert r.dropped is not None
        assert r.dropped.reason is DropReason.SUPERSEDED_UNCOMMITTED
        assert r.dropped.revision_start == 0
        assert r.dropped.revision_end == 1
        assert len(router.dropped_events) == 1
        payload = r.dropped.to_payload()
        assert payload.reason is DropReason.SUPERSEDED_UNCOMMITTED

    def test_zero_duplicate_committed_spans_across_revisions(self) -> None:
        router = CommittedDeltaRouter()
        texts = [
            (0, "The", 0),
            (1, "The Lord", 0),
            (2, "The Lord is", 3),  # commit "The"
            (3, "The Lord is my", 3),
            (4, "The Lord is my shepherd", 12),  # commit " Lord is"
            (5, "The Lord is my shepherd", 23, True),
        ]
        for row in texts:
            rev, text, commit = row[0], row[1], row[2]
            is_final = row[3] if len(row) > 3 else False
            router.observe_listen_product(
                utterance_id="ps23",
                revision=rev,
                text=text,
                committed_prefix_chars=commit,
                is_final=bool(is_final),
            )
        deltas = router.translate_requests_from_listen()
        joined = "".join(d.text for d in deltas)
        assert joined == "The Lord is my shepherd"
        # Contiguous non-overlapping spans
        cursor = 0
        for d in deltas:
            assert d.char_start == cursor
            assert d.char_end == cursor + len(d.text)
            cursor = d.char_end
        assert len({d.source_span_id for d in deltas}) == len(deltas)
