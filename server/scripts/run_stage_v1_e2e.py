#!/usr/bin/env python3
"""stage.v1 real EN→ES pre-EOS E2E harness (Wave 4 / G4).

Example:
  WHISPER_MODEL_SIZE=base uv run python scripts/run_stage_v1_e2e.py \\
    --fixture tests/fixtures/audio/public-domain-en-01.wav \\
    --listen whisper-listen \\
    --translate opus-mt-en-es \\
    --speak edge-tts-es \\
    --require-first-audio-before-source-eos \\
    --runs 2 \\
    --output .evidence/stage-v1-real-e2e
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import subprocess
import sys
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Allow `uv run python scripts/run_stage_v1_e2e.py` from server/
SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from src.stage_v1.pipeline_e2e import (  # noqa: E402
    RealModelUnavailable,
    create_warm_bundle,
    pcm_to_wav_bytes,
    read_wav_pcm,
    run_pre_eos_pipeline,
    sha256_bytes,
    sha256_file,
    shutdown_bundle,
    write_wav,
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _git_sha(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_dirty(repo: Path) -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True


def _hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_device_count"] = torch.cuda.device_count()
    except Exception as exc:  # noqa: BLE001 — diagnostic only
        info["torch_error"] = str(exc)
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        info["nvidia_smi"] = out.strip().splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["nvidia_smi"] = None
    return info


def _load_fixture_manifest(fixture_path: Path) -> dict[str, Any]:
    manifest_path = fixture_path.parent / "MANIFEST.sha256.json"
    if not manifest_path.is_file():
        return {"path": str(manifest_path), "present": False}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["path"] = str(manifest_path)
    data["present"] = True
    return data


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_stage_v1_e2e",
        description="Real stage.v1 EN→ES pre-EOS E2E harness (G4)",
    )
    p.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/audio/public-domain-en-01.wav"),
        help="Path to WAV/PCM fixture",
    )
    p.add_argument("--listen", default="whisper-listen")
    p.add_argument("--translate", default="opus-mt-en-es")
    p.add_argument(
        "--speak",
        default="edge-tts-es",
        help="edge-tts-es (default) or pocket-tts-spanish-24l if installed",
    )
    p.add_argument(
        "--require-first-audio-before-source-eos",
        action="store_true",
        default=True,
        help="Anti-cheat: withhold source EOS until first speak.audio (default on)",
    )
    p.add_argument(
        "--no-require-first-audio-before-source-eos",
        action="store_false",
        dest="require_first_audio_before_source_eos",
    )
    p.add_argument("--runs", type=int, default=2, help="Sequential runs (warm reuse)")
    p.add_argument(
        "--output",
        type=Path,
        default=Path(".evidence/stage-v1-real-e2e"),
        help="Evidence root directory",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Evidence run id (default: utc timestamp + short uuid)",
    )
    p.add_argument(
        "--max-wait-s",
        type=float,
        default=180.0,
        help="Max seconds to wait at pre-EOS hold for first speak.audio",
    )
    p.add_argument(
        "--whisper-model-size",
        default=None,
        help="Override WHISPER_MODEL_SIZE (tiny/base/small/...)",
    )
    return p


async def _async_main(args: argparse.Namespace) -> int:
    repo_root = SERVER_ROOT.parent
    fixture_path = args.fixture
    if not fixture_path.is_file():
        # try relative to server root
        alt = SERVER_ROOT / fixture_path
        if alt.is_file():
            fixture_path = alt
        else:
            print(f"ERROR: fixture not found: {args.fixture}", file=sys.stderr)
            return 2

    run_id = args.run_id or (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    )
    out_dir = args.output
    if not out_dir.is_absolute():
        out_dir = (SERVER_ROOT / out_dir).resolve()
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    command = " ".join(sys.argv)
    started = _utc_now()
    code_sha = _git_sha(repo_root)
    dirty = _git_dirty(repo_root)
    fixture_sha = sha256_file(fixture_path)
    fixture_manifest = _load_fixture_manifest(fixture_path)

    # Verify fixture digest against committed manifest when present.
    fixture_name = fixture_path.name
    expected = None
    if fixture_manifest.get("present"):
        expected = fixture_manifest.get("files", {}).get(fixture_name, {}).get("sha256")
    if expected and expected != fixture_sha:
        print(
            f"ERROR: fixture sha mismatch for {fixture_name}: "
            f"got {fixture_sha}, expected {expected}",
            file=sys.stderr,
        )
        return 2

    pcm, sample_rate = read_wav_pcm(fixture_path)
    print(
        f"fixture={fixture_path} samples={len(pcm) // 2} rate={sample_rate} "
        f"sha256={fixture_sha[:16]}…",
        flush=True,
    )

    timeline_path = run_dir / "event-timeline.jsonl"
    results: list[dict[str, Any]] = []
    gate = "STOP"
    error: str | None = None
    bundle = None
    loader_final: dict[str, int] = {}

    try:
        if args.whisper_model_size:
            os.environ["WHISPER_MODEL_SIZE"] = args.whisper_model_size
        print(
            f"warming stages listen={args.listen} translate={args.translate} "
            f"speak={args.speak} whisper={os.environ.get('WHISPER_MODEL_SIZE', 'base')}",
            flush=True,
        )
        bundle = await create_warm_bundle(
            listen_id=args.listen,
            translate_id=args.translate,
            speak_id=args.speak,
            sample_rate=sample_rate,
            whisper_model_size=args.whisper_model_size,
        )
        print(f"warm boot_id={bundle.boot_id} loaders={bundle.counters.as_dict()}", flush=True)

        for i in range(args.runs):
            print(f"=== run {i + 1}/{args.runs} ===", flush=True)
            result = await run_pre_eos_pipeline(
                bundle,
                pcm,
                run_index=i,
                require_first_audio_before_source_eos=args.require_first_audio_before_source_eos,
                max_wait_s=args.max_wait_s,
            )
            results.append(result.to_dict())
            loader_final = result.loader_counts_after
            # Append timeline
            with timeline_path.open("a", encoding="utf-8") as tf:
                for ev in result.timeline:
                    row = ev.to_dict()
                    row["run_index"] = i
                    tf.write(json.dumps(row, ensure_ascii=False) + "\n")
            # Write per-run audio from last successful pcm (overwrite final)
            if result.output_pcm:
                write_wav(run_dir / "output-es.wav", result.output_pcm, sample_rate=sample_rate)
                (run_dir / "output-es.pcm").write_bytes(result.output_pcm)
                (run_dir / "output-es.sha256").write_text(
                    sha256_bytes(result.output_pcm) + "\n", encoding="utf-8"
                )
            print(
                f"run {i}: listen={result.listen_texts!r} "
                f"translate={result.translate_texts!r} "
                f"pcm_bytes={len(result.output_pcm)} rms={result.pcm_energy_rms:.5f} "
                f"pre_eos={result.first_speak_before_source_eos} err={result.error}",
                flush=True,
            )
            if result.error:
                error = result.error

        # Gate evaluation
        if not results:
            error = error or "no runs completed"
        else:
            loaders_ok = all(
                r["loader_counts_after"].get("whisper", 0) == 1
                and r["loader_counts_after"].get("opus_mt", 0) == 1
                for r in results
            )
            pre_eos_ok = all(r["first_speak_before_source_eos"] for r in results)
            audio_ok = all(
                r["pcm_sample_count"] > 0 and r["pcm_energy_rms"] > 1e-4 for r in results
            )
            lang_ok = all(
                r["target_language"] == "es" and not r["english_spoken_as_spanish"]
                for r in results
            )
            contig_ok = all(r["contiguous_frames"] for r in results)
            no_err = all(r.get("error") is None for r in results)
            if loaders_ok and pre_eos_ok and audio_ok and lang_ok and contig_ok and no_err:
                gate = "PROCEED"
            else:
                bits = {
                    "loaders_ok": loaders_ok,
                    "pre_eos_ok": pre_eos_ok,
                    "audio_ok": audio_ok,
                    "lang_ok": lang_ok,
                    "contig_ok": contig_ok,
                    "no_err": no_err,
                }
                error = error or f"gate checks failed: {bits}"

    except RealModelUnavailable as exc:
        error = f"RealModelUnavailable: {exc}"
        print(f"BLOCKED: {error}", file=sys.stderr)
        gate = "STOP"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        gate = "STOP"
    finally:
        if bundle is not None:
            try:
                await shutdown_bundle(bundle)
            except Exception:  # noqa: BLE001
                traceback.print_exc()

    ended = _utc_now()
    last_pcm = b""
    if (run_dir / "output-es.pcm").is_file():
        last_pcm = (run_dir / "output-es.pcm").read_bytes()

    provenance = {
        "boot_id": getattr(bundle, "boot_id", None) if bundle else None,
        "listen_id": args.listen,
        "translate_id": args.translate,
        "speak_id": args.speak,
        "whisper_model_size": os.environ.get("WHISPER_MODEL_SIZE", "base"),
        "translate_model_id": "Helsinki-NLP/opus-mt-en-es",
        "speak_voice": "es-ES-AlvaroNeural",
        "speak_provider": "edge-tts" if args.speak.startswith("edge") else args.speak,
        "model_loader_counts": loader_final,
        "code_git_sha": code_sha,
        "fixture_sha256": fixture_sha,
    }
    (run_dir / "provenance-list.json").write_text(
        json.dumps([provenance], indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "protocol-version.txt").write_text("stage.v1\n", encoding="utf-8")
    (run_dir / "fixture-manifest.json").write_text(
        json.dumps(fixture_manifest, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "test-commands.txt").write_text(command + "\n", encoding="utf-8")
    (run_dir / "listening-notes-template.md").write_text(
        "# Listening notes (calibration only)\n\n"
        "- Intelligibility:\n"
        "- Naturalness:\n"
        "- Theology/name fidelity:\n"
        "- Notes:\n",
        encoding="utf-8",
    )
    metrics = {
        "runs": len(results),
        "gate": gate,
        "loader_counts": loader_final,
        "output_pcm_bytes": len(last_pcm),
        "output_pcm_sha256": sha256_bytes(last_pcm) if last_pcm else None,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    # Optional empty stubs for bundle completeness
    (run_dir / "queue-capacity.json").write_text(
        json.dumps({"note": "in-process e2e; no remote queue"}, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "health-transitions.jsonl").write_text("", encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "schema_version": "stage.v1",
        "gate": gate,
        "result": "pass" if gate == "PROCEED" else "fail",
        "error": error,
        "started_at": started,
        "ended_at": ended,
        "command": command,
        "repository": str(repo_root),
        "code_git_sha": code_sha,
        "dirty": dirty,
        "python": sys.version,
        "platform": platform.platform(),
        "hardware": _hardware(),
        "fixture": {
            "path": str(fixture_path),
            "sha256": fixture_sha,
            "sample_rate_hz": sample_rate,
            "pcm_bytes": len(pcm),
        },
        "stages": {
            "listen": args.listen,
            "translate": args.translate,
            "speak": args.speak,
        },
        "require_first_audio_before_source_eos": args.require_first_audio_before_source_eos,
        "runs_requested": args.runs,
        "runs": results,
        "provenance": provenance,
        "evidence_dir": str(run_dir),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    # Ensure wav exists even if empty for tooling
    if not (run_dir / "output-es.wav").is_file() and last_pcm:
        (run_dir / "output-es.wav").write_bytes(
            pcm_to_wav_bytes(last_pcm, sample_rate=sample_rate)
        )

    print(json.dumps({"gate": gate, "run_id": run_id, "evidence": str(run_dir), "error": error}))
    return 0 if gate == "PROCEED" else 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
