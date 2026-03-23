"""Format RunResult + metrics into readable reports."""

from __future__ import annotations

from src.harness.metrics import (
    DuplicateInfo,
    bleu_score,
    detect_duplicates,
    detect_ngram_repetition,
    word_error_rate,
)
from src.harness.runner import RunResult


def _join_texts(result: RunResult, stream: str) -> str:
    entries = result.text_streams.get(stream, [])
    return " ".join(e.text for e in entries)


def evaluate(
    result: RunResult,
    *,
    en_reference: str | None = None,
    es_reference: str | None = None,
    audio_transcript: str | None = None,
) -> str:
    """Build a markdown report for a single pipeline run."""
    lines: list[str] = []
    lines.append(f"### {result.pipeline_id}")
    lines.append("")

    if result.error:
        lines.append(f"**ERROR:** {result.error}")
        lines.append("")
        return "\n".join(lines)

    # --- Latency --------------------------------------------------------
    lines.append("#### Latency")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Audio duration | {result.audio_duration_seconds:.2f}s |")
    lines.append(f"| Wall time | {result.wall_seconds:.2f}s |")
    dur = result.audio_duration_seconds
    rtf = result.wall_seconds / dur if dur else 0
    lines.append(f"| Real-time factor | {rtf:.2f}x |")
    if result.first_audio_seconds is not None:
        lines.append(f"| First audio out | {result.first_audio_seconds:.2f}s |")
    for stream, t in sorted(result.first_text_seconds.items()):
        lines.append(f"| First text ({stream}) | {t:.2f}s |")
    lines.append(f"| Audio chunks out | {result.audio_chunks_out} |")
    lines.append(f"| Audio bytes out | {result.audio_bytes_out:,} |")
    lines.append("")

    # --- Per-stream quality ---------------------------------------------
    for stream_name, entries in sorted(result.text_streams.items()):
        if not entries:
            continue

        lines.append(f"#### Stream: `{stream_name}`")
        lines.append("")

        texts = [e.text for e in entries]
        joined = " ".join(texts)

        lines.append(f"**Segments ({len(texts)}):**")
        lines.append("")
        for i, e in enumerate(entries):
            lines.append(f"{i + 1}. [{e.elapsed_seconds:.2f}s] {e.text}")
        lines.append("")

        # Pick the right reference
        ref: str | None = None
        if "en" in stream_name and en_reference:
            ref = en_reference
        elif "es" in stream_name and es_reference:
            ref = es_reference
        elif stream_name == "transcript" and en_reference:
            ref = en_reference

        lines.append("| Metric | Value |")
        lines.append("|---|---|")

        if ref:
            wer = word_error_rate(ref, joined)
            bleu = bleu_score(ref, joined)
            lines.append(f"| WER | {wer:.1%} |")
            lines.append(f"| BLEU | {bleu:.3f} |")

        # Duplicates
        dup_info: DuplicateInfo = detect_duplicates(texts)
        n_dup = len(dup_info.repeated_segments)
        n_tot = dup_info.total_segments
        pct = dup_info.duplicate_ratio
        lines.append(f"| Duplicate segments | {n_dup}/{n_tot} ({pct:.0%}) |")

        # Internal repetition
        rep = detect_ngram_repetition(joined)
        lines.append(f"| 4-gram repetition | {rep:.1%} |")

        lines.append("")

    # --- Audio output validation (Whisper-on-output) ---------------------
    if audio_transcript and es_reference:
        lines.append("#### Audio Output Validation (Whisper on output audio)")
        lines.append("")
        preview = audio_transcript[:300]
        if len(audio_transcript) > 300:
            preview += "..."
        lines.append(f"> {preview}")
        lines.append("")
        wer = word_error_rate(es_reference, audio_transcript)
        bleu = bleu_score(es_reference, audio_transcript)
        rep = detect_ngram_repetition(audio_transcript)
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| WER vs ES reference | {wer:.1%} |")
        lines.append(f"| BLEU vs ES reference | {bleu:.3f} |")
        lines.append(f"| 4-gram repetition | {rep:.1%} |")
        lines.append("")

    return "\n".join(lines)


def summary_table(results: list[tuple[RunResult, str]]) -> str:
    """Build a cross-pipeline comparison table.

    *results* is a list of (RunResult, evaluate_output) pairs.
    """
    lines: list[str] = []
    lines.append("## Summary")
    lines.append("")

    # Header
    cols = [
        "Pipeline",
        "Wall (s)",
        "RTF",
        "1st Audio (s)",
        "1st Text (s)",
        "Audio Out",
        "Errors",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join("---" for _ in cols) + " |")

    for result, _ in results:
        if result.error:
            lines.append(
                f"| {result.pipeline_id} | — | — | — | — | — | {result.error} |"
            )
            continue

        dur = result.audio_duration_seconds
        rtf = result.wall_seconds / dur if dur else 0
        fa = result.first_audio_seconds
        first_audio = f"{fa:.2f}" if fa is not None else "—"
        first_texts = sorted(result.first_text_seconds.items())
        first_text = f"{first_texts[0][1]:.2f}" if first_texts else "—"
        audio_out = f"{result.audio_bytes_out:,}B" if result.audio_bytes_out else "—"

        lines.append(
            f"| {result.pipeline_id} | {result.wall_seconds:.2f} | {rtf:.2f}x "
            f"| {first_audio} | {first_text} | {audio_out} | — |"
        )

    lines.append("")
    return "\n".join(lines)
