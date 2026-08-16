"""Real-model EN→ES E2E orchestration glue for stage.v1 Wave 4 / G4.

Warm model hosts come from ``stage_v1.adapters`` factories (Lane E). Session
stages bind via ``open_*_session_stage`` so weights load once and outlive runs.
Legacy Counting* wrappers remain only as a fallback if adapters are unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import struct
import time
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16_000
FRAME_MS = 20
BYTES_PER_SAMPLE = 2

# Common English words that should not dominate Spanish TTS text.
_ENGLISH_MARKERS = frozenset(
    {
        "the",
        "and",
        "god",
        "created",
        "heaven",
        "earth",
        "beginning",
        "lord",
        "shepherd",
        "blessed",
        "meek",
        "inherit",
        "shall",
        "want",
        "maketh",
        "green",
        "pastures",
    }
)


class RealModelUnavailable(RuntimeError):
    """Raised when real models/network required for G4 are missing."""


@dataclass
class LoaderCounters:
    """Tracks warm-model load counts across sequential runs."""

    whisper: int = 0
    opus_mt: int = 0
    edge_tts: int = 0  # network speak; start is no-op but counted for symmetry

    def as_dict(self) -> dict[str, int]:
        return {
            "whisper": self.whisper,
            "opus_mt": self.opus_mt,
            "edge_tts": self.edge_tts,
        }

    @property
    def total_model_loads(self) -> int:
        # Edge TTS has no resident weights; Whisper + Opus are the warm models.
        return self.whisper + self.opus_mt


@dataclass
class TimelineEvent:
    t_rel_ms: float
    event_type: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "t_rel_ms": round(self.t_rel_ms, 3),
            "event_type": self.event_type,
            "detail": self.detail,
        }


@dataclass
class E2ERunResult:
    run_index: int
    listen_texts: list[str]
    translate_texts: list[str]
    output_pcm: bytes
    sample_rate_hz: int
    first_speak_before_source_eos: bool
    source_eos_released: bool
    timeline: list[TimelineEvent]
    loader_counts_after: dict[str, int]
    pcm_energy_rms: float
    pcm_sample_count: int
    target_language: str
    english_spoken_as_spanish: bool
    contiguous_frames: bool
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_index": self.run_index,
            "listen_texts": self.listen_texts,
            "translate_texts": self.translate_texts,
            "sample_rate_hz": self.sample_rate_hz,
            "first_speak_before_source_eos": self.first_speak_before_source_eos,
            "source_eos_released": self.source_eos_released,
            "loader_counts_after": self.loader_counts_after,
            "pcm_energy_rms": self.pcm_energy_rms,
            "pcm_sample_count": self.pcm_sample_count,
            "pcm_bytes": len(self.output_pcm),
            "target_language": self.target_language,
            "english_spoken_as_spanish": self.english_spoken_as_spanish,
            "contiguous_frames": self.contiguous_frames,
            "error": self.error,
            "timeline": [e.to_dict() for e in self.timeline],
        }


@dataclass
class WarmStageBundle:
    """Warm-resident stages reused across sequential E2E runs."""

    listen: Any
    translate: Any
    speak: Any
    counters: LoaderCounters
    sample_rate: int
    listen_id: str
    translate_id: str
    speak_id: str
    boot_id: str = field(default_factory=lambda: str(uuid4()))


def pcm_rms(pcm: bytes) -> float:
    if len(pcm) < 2:
        return 0.0
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples / 32768.0))))


def write_wav(path: Path, pcm: bytes, *, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


def read_wav_pcm(path: Path) -> tuple[bytes, int]:
    """Return (pcm_s16le_bytes, sample_rate_hz) from a mono WAV or raw PCM."""
    data = path.read_bytes()
    if path.suffix.lower() == ".pcm":
        return data, DEFAULT_SAMPLE_RATE
    with wave.open(io.BytesIO(data), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if sampwidth != 2:
        raise ValueError(f"expected 16-bit PCM, got sampwidth={sampwidth}")
    if channels == 1:
        return frames, rate
    # Downmix interleaved multi-channel to mono
    samples = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels)
    mono = samples.mean(axis=1).astype(np.int16)
    return mono.tobytes(), rate


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def looks_like_english_not_spanish(text: str) -> bool:
    """Heuristic fail-closed check: Spanish product should not be source English."""
    tokens = [t.strip(".,;:!?\"'").lower() for t in text.split() if t.strip()]
    if not tokens:
        return False
    # Spanish diacritics / common function words → treat as Spanish
    if any(ch in text for ch in "áéíóúñüÁÉÍÓÚÑÜ"):
        return False
    spanishish = {
        "el",
        "la",
        "los",
        "las",
        "de",
        "en",
        "y",
        "que",
        "por",
        "dios",
        "cielos",
        "tierra",
        "principio",
        "creó",
        "creo",
        "señor",
        "pastor",
        "bienaventurados",
        "mansos",
        "heredarán",
        "heredad",
    }
    if any(t in spanishish for t in tokens):
        return False
    hits = sum(1 for t in tokens if t in _ENGLISH_MARKERS)
    return hits >= max(2, len(tokens) // 3)


def _try_import_adapters() -> Any | None:
    """Return stage_v1.adapters via submodule import (avoid package __init__ cycle)."""
    try:
        import importlib

        return importlib.import_module("src.stage_v1.adapters")
    except Exception:
        return None


class CountingWhisperListen:
    """Legacy fallback: wraps WhisperListenStage and counts model loads."""

    def __init__(self, *, sample_rate: int, counters: LoaderCounters, model_size: str) -> None:
        from src.pipelines.stages_listen.whisper import WhisperListenStage

        self._inner = WhisperListenStage(sample_rate=sample_rate, model_size=model_size)
        self._counters = counters
        self.info = self._inner.info

    async def start(self) -> None:
        before = self._inner._model  # noqa: SLF001
        await self._inner.start()
        if before is None and self._inner._model is not None:  # noqa: SLF001
            self._counters.whisper += 1

    async def stop(self) -> None:
        # Warm reuse: do NOT unload resident model between runs.
        return

    def transcribe(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[Any]:
        return self._inner.transcribe(audio_stream)

    async def shutdown(self) -> None:
        await self._inner.stop()


class CountingOpusMT:
    def __init__(self, *, sample_rate: int, counters: LoaderCounters) -> None:
        from src.pipelines.stages_translate.opus_mt import OpusMTTranslateStage

        self._inner = OpusMTTranslateStage(sample_rate=sample_rate)
        self._counters = counters
        self.info = self._inner.info

    async def start(self) -> None:
        before = self._inner._translator  # noqa: SLF001
        await self._inner.start()
        if before is None and self._inner._translator is not None:  # noqa: SLF001
            self._counters.opus_mt += 1

    async def stop(self) -> None:
        return

    def translate(
        self,
        text_stream: AsyncIterator[Any],
        *,
        prosody: AsyncIterator[Any] | None = None,
    ) -> AsyncIterator[Any]:
        return self._inner.translate(text_stream, prosody=prosody)

    async def shutdown(self) -> None:
        await self._inner.stop()


class CountingEdgeTTS:
    def __init__(self, *, sample_rate: int, counters: LoaderCounters) -> None:
        from src.pipelines.stages_speak.edge_tts import EdgeTTSSpeakStage

        self._inner = EdgeTTSSpeakStage(sample_rate=sample_rate)
        self._counters = counters
        self.info = self._inner.info
        self._started = False

    async def start(self) -> None:
        if not self._started:
            await self._inner.start()
            self._counters.edge_tts += 1
            self._started = True

    async def stop(self) -> None:
        return

    def synthesize(self, text_stream: AsyncIterator[Any]) -> AsyncIterator[bytes]:
        return self._inner.synthesize(text_stream)

    async def shutdown(self) -> None:
        await self._inner.stop()
        self._started = False


async def create_warm_bundle(
    *,
    listen_id: str = "whisper-listen",
    translate_id: str = "opus-mt-en-es",
    speak_id: str = "edge-tts-es",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    whisper_model_size: str | None = None,
) -> WarmStageBundle:
    """Create and warm-start real stages via adapters factories when present."""
    counters = LoaderCounters()
    model_size = whisper_model_size or os.environ.get("WHISPER_MODEL_SIZE", "base")

    # Fail-fast import checks for honest skip/block messages.
    try:
        import faster_whisper  # noqa: F401
    except ImportError as exc:
        raise RealModelUnavailable("faster-whisper not installed") from exc
    try:
        import ctranslate2  # noqa: F401
        import sentencepiece  # noqa: F401
    except ImportError as exc:
        raise RealModelUnavailable("ctranslate2/sentencepiece not installed") from exc
    try:
        import edge_tts  # noqa: F401
    except ImportError as exc:
        raise RealModelUnavailable("edge-tts not installed") from exc

    adapters = _try_import_adapters()
    if (
        adapters is not None
        and hasattr(adapters, "build_whisper_listen_host")
        and hasattr(adapters, "open_whisper_session_stage")
    ):
        boot_id = str(uuid4())

        if listen_id != "whisper-listen":
            raise RealModelUnavailable(f"unsupported listen stage: {listen_id}")
        if translate_id != "opus-mt-en-es":
            raise RealModelUnavailable(f"unsupported translate stage: {translate_id}")
        if speak_id not in {"edge-tts-es", "pocket-tts-spanish-24l"}:
            raise RealModelUnavailable(f"unsupported speak stage: {speak_id}")

        from src.pipelines.stages_listen.whisper import load_whisper_model
        from src.pipelines.stages_speak.edge_tts import load_edge_tts_model
        from src.pipelines.stages_translate.opus_mt import load_opus_mt_model

        def whisper_loader() -> Any:
            counters.whisper += 1
            return load_whisper_model(model_size=model_size)

        def opus_loader() -> Any:
            counters.opus_mt += 1
            return load_opus_mt_model()

        def edge_loader() -> Any:
            counters.edge_tts += 1
            return load_edge_tts_model()

        listen_host = adapters.build_whisper_listen_host(
            model_size=model_size,
            sample_rate=sample_rate,
            max_sessions=4,
            boot_id=boot_id,
            model_loader=whisper_loader,
        )
        translate_host = adapters.build_opus_mt_host(
            max_sessions=4,
            boot_id=boot_id,
            model_loader=opus_loader,
        )

        if speak_id == "edge-tts-es":
            speak_host = adapters.build_edge_tts_host(
                sample_rate=sample_rate,
                max_sessions=4,
                boot_id=boot_id,
                model_loader=edge_loader,
            )
        else:
            if not getattr(adapters, "POCKET_TTS_AVAILABLE", False):
                raise RealModelUnavailable(
                    "pocket-tts not installed; use --speak edge-tts-es"
                )

            def pocket_loader() -> Any:
                counters.edge_tts += 1
                from src.pipelines.stages_speak.pocket_tts import load_pocket_tts_model

                return load_pocket_tts_model()

            speak_host = adapters.build_pocket_tts_host(
                sample_rate=sample_rate,
                max_sessions=4,
                boot_id=boot_id,
                model_loader=pocket_loader,
            )

        await listen_host.warmup()
        await translate_host.warmup()
        await speak_host.warmup()

        listen_session = await listen_host.open_session(attempt_id="e2e-listen")
        translate_session = await translate_host.open_session(attempt_id="e2e-translate")
        speak_session = await speak_host.open_session(attempt_id="e2e-speak")

        listen = adapters.open_whisper_session_stage(
            listen_host, listen_session, sample_rate=sample_rate
        )
        translate = adapters.open_opus_mt_session_stage(translate_host, translate_session)
        if speak_id == "edge-tts-es":
            speak = adapters.open_edge_tts_session_stage(
                speak_host, speak_session, sample_rate=sample_rate
            )
        else:
            speak = adapters.open_pocket_tts_session_stage(
                speak_host, speak_session, sample_rate=sample_rate
            )

        await listen.start()
        await translate.start()
        await speak.start()

        bundle = WarmStageBundle(
            listen=listen,
            translate=translate,
            speak=speak,
            counters=counters,
            sample_rate=sample_rate,
            listen_id=listen_id,
            translate_id=translate_id,
            speak_id=speak_id,
            boot_id=boot_id,
        )
        # Attach host/session refs for orderly shutdown (not part of public dataclass).
        bundle._hosts = (listen_host, translate_host, speak_host)  # type: ignore[attr-defined]
        bundle._sessions = (listen_session, translate_session, speak_session)  # type: ignore[attr-defined]
        return bundle

    # Legacy fallback path (no adapters module).
    if listen_id != "whisper-listen":
        raise RealModelUnavailable(f"unsupported listen stage: {listen_id}")
    if translate_id != "opus-mt-en-es":
        raise RealModelUnavailable(f"unsupported translate stage: {translate_id}")
    if speak_id not in {"edge-tts-es", "pocket-tts-spanish-24l"}:
        raise RealModelUnavailable(f"unsupported speak stage: {speak_id}")

    if speak_id == "pocket-tts-spanish-24l":
        raise RealModelUnavailable(
            "pocket-tts requires stage_v1.adapters; use --speak edge-tts-es"
        )

    listen = CountingWhisperListen(
        sample_rate=sample_rate, counters=counters, model_size=model_size
    )
    translate = CountingOpusMT(sample_rate=sample_rate, counters=counters)
    speak = CountingEdgeTTS(sample_rate=sample_rate, counters=counters)

    await listen.start()
    await translate.start()
    await speak.start()

    return WarmStageBundle(
        listen=listen,
        translate=translate,
        speak=speak,
        counters=counters,
        sample_rate=sample_rate,
        listen_id=listen_id,
        translate_id=translate_id,
        speak_id=speak_id,
    )


async def shutdown_bundle(bundle: WarmStageBundle) -> None:
    """Stop session stages and close adapter hosts when present."""
    for stage in (bundle.speak, bundle.translate, bundle.listen):
        shutdown = getattr(stage, "shutdown", None)
        if shutdown is not None:
            await shutdown()
        else:
            stop = getattr(stage, "stop", None)
            if stop is not None:
                await stop()

    sessions = getattr(bundle, "_sessions", None)
    hosts = getattr(bundle, "_hosts", None)
    if sessions is not None and hosts is not None:
        for host, session in zip(hosts, sessions, strict=False):
            close = getattr(host, "close_session", None)
            if close is not None:
                try:
                    await close(session.session_state_id)
                except Exception:
                    logger.exception(
                        "failed to close e2e session on %s",
                        getattr(host, "stage_id", host),
                    )
            host_shutdown = getattr(host, "shutdown", None)
            if host_shutdown is not None:
                try:
                    await host_shutdown()
                except Exception:
                    logger.exception(
                        "failed to shutdown e2e host %s",
                        getattr(host, "stage_id", host),
                    )

def chunk_pcm(pcm: bytes, *, sample_rate: int, frame_ms: int = FRAME_MS) -> list[bytes]:
    frame_bytes = int(sample_rate * frame_ms / 1000) * BYTES_PER_SAMPLE
    if frame_bytes <= 0:
        raise ValueError("invalid frame size")
    # Align to sample boundary
    if len(pcm) % BYTES_PER_SAMPLE:
        pcm = pcm[: len(pcm) - (len(pcm) % BYTES_PER_SAMPLE)]
    return [pcm[i : i + frame_bytes] for i in range(0, len(pcm), frame_bytes) if i < len(pcm)]


async def run_pre_eos_pipeline(
    bundle: WarmStageBundle,
    pcm: bytes,
    *,
    run_index: int = 0,
    require_first_audio_before_source_eos: bool = True,
    hold_tail_seconds: float = 0.5,
    max_wait_s: float = 180.0,
    frame_ms: int = FRAME_MS,
) -> E2ERunResult:
    """Run Listen→Translate→Speak with anti-cheat EOS withhold.

    Source audio frames are fed until a hold point near the end. EOS is only
    released after the first non-silent speak.audio chunk is observed (or the
    full listen+translate+speak chain products if speak is delayed).
    """
    t0 = time.perf_counter()
    timeline: list[TimelineEvent] = []

    def mark(event_type: str, **detail: Any) -> None:
        timeline.append(
            TimelineEvent(
                t_rel_ms=(time.perf_counter() - t0) * 1000.0,
                event_type=event_type,
                detail=detail,
            )
        )

    first_speak = asyncio.Event()
    source_eos_released = asyncio.Event()
    listen_texts: list[str] = []
    translate_texts: list[str] = []
    out_chunks: list[bytes] = []
    error: str | None = None
    first_speak_before_eos = False

    frames = chunk_pcm(pcm, sample_rate=bundle.sample_rate, frame_ms=frame_ms)
    if not frames:
        raise ValueError("fixture produced zero audio frames")

    hold_frames = max(1, int(hold_tail_seconds * 1000 / frame_ms))
    # Keep at least ~3.5s of audio before the hold so Whisper BUFFER_SECONDS can fire.
    min_pre_hold = int(3.5 * 1000 / frame_ms)
    if len(frames) <= hold_frames + min_pre_hold:
        # Short fixture: hold only the final frame.
        release_at = max(0, len(frames) - 1)
    else:
        release_at = len(frames) - hold_frames

    mark(
        "fixture.ready",
        frames=len(frames),
        release_at=release_at,
        sample_rate=bundle.sample_rate,
        pcm_bytes=len(pcm),
    )

    audio_q: asyncio.Queue[bytes | None] = asyncio.Queue()
    listen_q: asyncio.Queue[Any | None] = asyncio.Queue()
    translate_q: asyncio.Queue[Any | None] = asyncio.Queue()

    async def source_feeder() -> None:
        for idx, frame in enumerate(frames):
            if idx == release_at and require_first_audio_before_source_eos:
                mark("source.hold_before_eos", frame_index=idx)
                try:
                    await asyncio.wait_for(first_speak.wait(), timeout=max_wait_s)
                except TimeoutError:
                    mark("source.hold_timeout", frame_index=idx)
                    # Fail closed: still release so the pipeline can unwind.
                else:
                    mark("source.hold_released_after_first_speak", frame_index=idx)
            await audio_q.put(frame)
            if idx == 0:
                mark("source.first_frame")
            if idx % 50 == 0:
                mark("source.frame", index=idx)
        mark("source.eos")
        source_eos_released.set()
        await audio_q.put(None)

    async def audio_iter() -> AsyncIterator[bytes]:
        while True:
            item = await audio_q.get()
            if item is None:
                return
            yield item

    async def listen_pump() -> None:
        try:
            async for product in bundle.listen.transcribe(audio_iter()):
                text = getattr(product, "text", "") or ""
                listen_texts.append(text)
                mark(
                    "listen.product",
                    text=text[:200],
                    is_final=bool(getattr(product, "is_final", False)),
                    sequence=getattr(product, "sequence", None),
                )
                await listen_q.put(product)
        except Exception as exc:
            mark("listen.error", error=str(exc))
            raise
        finally:
            await listen_q.put(None)

    async def listen_iter() -> AsyncIterator[Any]:
        while True:
            item = await listen_q.get()
            if item is None:
                return
            yield item

    async def translate_pump() -> None:
        try:
            async for product in bundle.translate.translate(listen_iter()):
                text = getattr(product, "text", "") or ""
                translate_texts.append(text)
                mark(
                    "translate.product",
                    text=text[:200],
                    is_final=bool(getattr(product, "is_final", False)),
                    sequence=getattr(product, "sequence", None),
                    language="es",
                )
                await translate_q.put(product)
        except Exception as exc:
            mark("translate.error", error=str(exc))
            raise
        finally:
            await translate_q.put(None)

    async def translate_iter() -> AsyncIterator[Any]:
        while True:
            item = await translate_q.get()
            if item is None:
                return
            yield item

    async def speak_pump() -> None:
        nonlocal first_speak_before_eos
        try:
            async for chunk in bundle.speak.synthesize(translate_iter()):
                if not chunk:
                    continue
                energy = pcm_rms(chunk)
                out_chunks.append(chunk)
                mark(
                    "speak.audio",
                    bytes=len(chunk),
                    rms=energy,
                    media_sequence=len(out_chunks) - 1,
                )
                if energy > 1e-4 and not first_speak.is_set():
                    first_speak_before_eos = not source_eos_released.is_set()
                    mark(
                        "speak.first_nonsilent",
                        before_source_eos=first_speak_before_eos,
                        rms=energy,
                    )
                    first_speak.set()
        except Exception as exc:
            mark("speak.error", error=str(exc))
            raise
        finally:
            # If speak never fired but we got translate, still unblock feeder.
            if not first_speak.is_set() and translate_texts:
                mark("speak.fallback_unblock_after_translate")
                first_speak.set()
            mark("speak.complete", chunks=len(out_chunks))

    feeder_task = asyncio.create_task(source_feeder(), name="e2e-source")
    listen_task = asyncio.create_task(listen_pump(), name="e2e-listen")
    translate_task = asyncio.create_task(translate_pump(), name="e2e-translate")
    speak_task = asyncio.create_task(speak_pump(), name="e2e-speak")

    try:
        await asyncio.wait_for(
            asyncio.gather(feeder_task, listen_task, translate_task, speak_task),
            timeout=max_wait_s + 60.0,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        mark("pipeline.error", error=error)
        for task in (feeder_task, listen_task, translate_task, speak_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(
            feeder_task, listen_task, translate_task, speak_task, return_exceptions=True
        )
        # Ensure EOS latch is not left hanging for callers.
        first_speak.set()

    output_pcm = b"".join(out_chunks)
    energy = pcm_rms(output_pcm)
    sample_count = len(output_pcm) // BYTES_PER_SAMPLE
    joined_es = " ".join(translate_texts)
    english_as_es = looks_like_english_not_spanish(joined_es) if joined_es else False
    contiguous = _frames_contiguous(out_chunks, sample_rate=bundle.sample_rate, frame_ms=frame_ms)

    if require_first_audio_before_source_eos and not first_speak_before_eos and error is None:
        error = "first speak.audio was not observed before source EOS (anti-cheat fail)"

    if (not output_pcm or energy <= 1e-4) and error is None:
        error = "output PCM missing or silent"

    if english_as_es and error is None:
        error = "target text appears to be English spoken as Spanish"

    mark(
        "run.complete",
        pcm_bytes=len(output_pcm),
        rms=energy,
        first_speak_before_source_eos=first_speak_before_eos,
    )

    return E2ERunResult(
        run_index=run_index,
        listen_texts=listen_texts,
        translate_texts=translate_texts,
        output_pcm=output_pcm,
        sample_rate_hz=bundle.sample_rate,
        first_speak_before_source_eos=first_speak_before_eos,
        source_eos_released=source_eos_released.is_set(),
        timeline=timeline,
        loader_counts_after=bundle.counters.as_dict(),
        pcm_energy_rms=energy,
        pcm_sample_count=sample_count,
        target_language="es",
        english_spoken_as_spanish=english_as_es,
        contiguous_frames=contiguous,
        error=error,
    )


def _frames_contiguous(
    chunks: list[bytes], *, sample_rate: int, frame_ms: int
) -> bool:
    """Structural check: chunks are non-empty PCM with even byte length."""
    if not chunks:
        return False
    return all(len(chunk) > 0 and len(chunk) % BYTES_PER_SAMPLE == 0 for chunk in chunks)


def probe_real_stack_available() -> tuple[bool, str]:
    """Return (ok, reason) for pytest skip decisions."""
    missing: list[str] = []
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        missing.append("faster-whisper")
    try:
        import ctranslate2  # noqa: F401
    except ImportError:
        missing.append("ctranslate2")
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        missing.append("edge-tts")
    try:
        import sentencepiece  # noqa: F401
    except ImportError:
        missing.append("sentencepiece")
    if missing:
        return False, f"missing packages: {', '.join(missing)}"
    # Optional network probe for edge-tts is deferred to runtime; offline envs skip later.
    return True, "ok"


def pcm_to_wav_bytes(pcm: bytes, *, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def wav_header_struct(sample_rate: int, data_bytes: int) -> bytes:
    """Minimal RIFF header helper (unused by wave module path; kept for tests)."""
    byte_rate = sample_rate * 1 * 2
    block_align = 2
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_bytes,
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        sample_rate,
        byte_rate,
        block_align,
        16,
        b"data",
        data_bytes,
    )


__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "E2ERunResult",
    "LoaderCounters",
    "RealModelUnavailable",
    "WarmStageBundle",
    "chunk_pcm",
    "create_warm_bundle",
    "looks_like_english_not_spanish",
    "pcm_rms",
    "pcm_to_wav_bytes",
    "probe_real_stack_available",
    "read_wav_pcm",
    "run_pre_eos_pipeline",
    "sha256_bytes",
    "sha256_file",
    "shutdown_bundle",
    "write_wav",
]
