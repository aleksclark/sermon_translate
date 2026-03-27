"""Pipeline evaluation harness CLI.

Usage::

    uv run python -m src.harness                            # all pipelines, default fixture
    uv run python -m src.harness -p whisper-tts             # one pipeline
    uv run python -m src.harness -p spanish-translation     # one pipeline
    uv run python -m src.harness --audio file.mp3           # custom audio (no reference)
    uv run python -m src.harness --audio file.mp3 --en-ref "..." --es-ref "..."
    uv run python -m src.harness --fixture path/to/ref.json # custom fixture
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import sys
from pathlib import Path

from src.harness.report import evaluate, summary_table
from src.harness.runner import RunResult, load_audio, run_pipeline, transcribe_output_audio
from src.pipelines.base import BasePipeline

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
DEFAULT_FIXTURE = FIXTURES_DIR / "test-speech.json"

KNOWN_PIPELINES = [
    ("src.pipelines.echo", "EchoPipeline"),
    ("src.pipelines.whisper_tts", "WhisperTTSPipeline"),
    ("src.pipelines.spanish", "SpanishTranslationPipeline"),
    ("src.pipelines.spanish_direct", "SpanishDirectPipeline"),
    ("src.pipelines.spanish_fast", "SpanishFastPipeline"),
    ("src.pipelines.spanish_fast_v2", "SpanishFastV2Pipeline"),
    ("src.pipelines.gpu_pipelines", "GPUWhisperT2STPipeline"),
    ("src.pipelines.gpu_pipelines", "GPUS2STPipeline"),
    ("src.pipelines.gpu_pipelines", "GPUWhisperOpusPipeline"),
    ("src.pipelines.moonshine_pipeline", "MoonshineStreamingPipeline"),
    ("src.pipelines.nova_sonic", "NovaSonicPipeline"),
    ("src.pipelines.simul_streaming", "SimulStreamingPipeline"),
    ("src.pipelines.simul_streaming_vc", "SimulStreamingVoiceClonePipeline"),
    ("src.pipelines.seamless_streaming", "SeamlessStreamingPipeline"),
]


def _discover_pipelines(filter_ids: list[str] | None = None) -> list[BasePipeline]:
    pipelines: list[BasePipeline] = []
    for mod_path, cls_name in KNOWN_PIPELINES:
        try:
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            p = cls()
            if filter_ids is None or p.info.id in filter_ids:
                pipelines.append(p)
        except Exception:
            logging.getLogger(__name__).info("skipping %s.%s", mod_path, cls_name)
    return pipelines


def _load_fixture(path: Path) -> tuple[Path, str | None, str | None]:
    with open(path) as f:
        data = json.load(f)
    audio_path = Path(data["audio_file"])
    if not audio_path.is_absolute():
        audio_path = (path.parent / audio_path).resolve()
    return audio_path, data.get("en_reference"), data.get("es_reference")


async def _main(args: argparse.Namespace) -> None:
    # Resolve audio and references
    if args.audio:
        audio_path = Path(args.audio)
        en_ref = args.en_ref
        es_ref = args.es_ref
    else:
        fixture = Path(args.fixture) if args.fixture else DEFAULT_FIXTURE
        audio_path, en_ref, es_ref = _load_fixture(fixture)

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading audio: {audio_path}")
    chunks = load_audio(audio_path)
    audio_seconds = len(chunks) * 0.02
    print(f"Audio: {audio_seconds:.1f}s, {len(chunks)} chunks")
    if en_ref:
        print(f"EN reference: {en_ref}")
    if es_ref:
        print(f"ES reference: {es_ref}")
    print()

    # Resolve pipelines
    filter_ids = [p.strip() for p in args.pipeline.split(",")] if args.pipeline else None
    pipelines = _discover_pipelines(filter_ids)
    if not pipelines:
        print("No pipelines matched", file=sys.stderr)
        sys.exit(1)

    print(f"Pipelines: {', '.join(p.info.id for p in pipelines)}")
    print("=" * 60)
    print()

    results: list[tuple[RunResult, str]] = []

    for pipeline in pipelines:
        print(f"Running {pipeline.info.id}...")
        result = await run_pipeline(pipeline, chunks)

        audio_transcript: str | None = None
        if result.output_audio_raw and es_ref:
            print(f"  Transcribing {len(result.output_audio_raw):,} bytes of output audio...")
            audio_transcript = transcribe_output_audio(result, language="es")

        report = evaluate(
            result,
            en_reference=en_ref,
            es_reference=es_ref,
            audio_transcript=audio_transcript,
        )
        results.append((result, report))
        print(report)
        print()

    if len(results) > 1:
        print(summary_table(results))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline evaluation harness")
    parser.add_argument("-p", "--pipeline", help="Comma-separated pipeline IDs to run")
    parser.add_argument("--audio", help="Path to audio file (overrides fixture)")
    parser.add_argument("--en-ref", help="English reference transcript")
    parser.add_argument("--es-ref", help="Spanish reference transcript")
    parser.add_argument("--fixture", help="Path to JSON fixture file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
