"""Warm StageHost binders and stage.v1 product mapping for real adapters.

D6: models load once via StageHost.model_loader and outlive sessions.
Session close only clears decoder/stream/context/voice/utterance state.

Commit policy (listen)
----------------------
On each successful buffered decode frame that yields non-empty text, the full
text of that frame is committed (``committed_prefix_chars == len(text)``).
Mid-stream frames set ``is_final=False``; only the true audio-stream EOS flush
sets ``is_final=True``. This guarantees at least one non-empty committed
prefix before EOS whenever the decoder produces text from a full buffer
window (pre-EOS path for Wave 4).

Translate only committed source deltas (here: each committed listen product).
Speak synthesizes committed Spanish; edge-tts is utterance-buffered
(``streams_pcm=False``), pocket-tts may stream when the backend supports it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from src.models import ListenProduct, TranslateProduct, WordSpan
from src.pipelines._audio import EDGE_TTS_VOICE
from src.pipelines.stages_listen.whisper import (
    WhisperListenStage,
    WhisperLoadedModel,
    load_whisper_model,
)
from src.pipelines.stages_speak.edge_tts import (
    EdgeTTSLoadedModel,
    EdgeTTSSpeakStage,
    load_edge_tts_model,
)
from src.pipelines.stages_translate.opus_mt import (
    OpusMTLoadedModel,
    OpusMTTranslateStage,
    load_opus_mt_model,
)
from src.stage_v1.host import SessionState, StageHost
from src.stage_v1.models import (
    BASELINE_SAMPLE_RATE_HZ,
    AlignmentKind,
    ArtifactDigestStatus,
    ListenProductPayload,
    StageKind,
    TimingKind,
    TranslateProductPayload,
    WordTiming,
)

logger = logging.getLogger(__name__)

STAGE_VERSION = "1.0.0"

# Optional pocket-tts (extra tts-pocket). Absent install is CONDITIONAL.
# Detect the optional package itself — stage module imports cleanly without it.
POCKET_TTS_AVAILABLE = False
try:
    import pocket_tts as _pocket_tts_pkg  # type: ignore[import-not-found]  # noqa: F401

    POCKET_TTS_AVAILABLE = True
except ImportError:
    pass


def _require_pocket() -> tuple[Callable[..., Any], type[Any]]:
    if not POCKET_TTS_AVAILABLE:
        raise ImportError(
            "pocket-tts is not installed; install with "
            "`uv sync --extra tts-pocket` or use build_edge_tts_host()"
        )
    from src.pipelines.stages_speak.pocket_tts import (
        PocketTTSSpeakStage,
        load_pocket_tts_model,
    )

    return load_pocket_tts_model, PocketTTSSpeakStage


def _ms_to_sample(ms: float | None, sample_rate: int) -> int:
    if ms is None:
        return 0
    return max(0, int(round((ms / 1000.0) * sample_rate)))


def word_spans_to_timings(
    words: list[WordSpan],
    *,
    sample_rate: int = BASELINE_SAMPLE_RATE_HZ,
) -> list[WordTiming]:
    timings: list[WordTiming] = []
    for word in words:
        start = _ms_to_sample(word.start_ms, sample_rate)
        end = _ms_to_sample(word.end_ms, sample_rate)
        if end < start:
            end = start
        timings.append(
            WordTiming(
                text=word.text,
                start_sample=start,
                end_sample=end,
                confidence=word.conf,
                timing_kind=(
                    TimingKind.CHUNK if word.start_ms is not None else TimingKind.UNAVAILABLE
                ),
            )
        )
    return timings


def listen_product_to_payload(
    product: ListenProduct,
    *,
    revision: int,
    sample_rate: int = BASELINE_SAMPLE_RATE_HZ,
    commit_full_text: bool = True,
) -> ListenProductPayload:
    """Map pipeline ListenProduct → stage.v1 ListenProductPayload.

    Commit policy: commit full frame text on each successful decode
    (``commit_full_text=True``). ``is_final`` is only True on true stream EOS.
    """
    text = product.text
    committed = len(text) if commit_full_text else 0
    if product.is_final:
        committed = len(text)
    words = word_spans_to_timings(product.words, sample_rate=sample_rate)
    start_sample = words[0].start_sample if words else 0
    end_sample = words[-1].end_sample if words else start_sample
    return ListenProductPayload(
        revision=revision,
        text=text,
        committed_prefix_chars=committed,
        is_final=product.is_final,
        language=product.language or "en",
        source_start_sample=start_sample,
        source_end_sample=end_sample,
        words=words,
    )


def translate_product_to_payload(
    product: TranslateProduct,
    *,
    revision: int,
    source_char_start: int = 0,
    source_char_end: int | None = None,
    source_span_id: str | None = None,
) -> TranslateProductPayload:
    """Map pipeline TranslateProduct → stage.v1 TranslateProductPayload.

    Only committed Spanish is emitted; full text is committed per product.
    """
    text = product.text
    end = source_char_end if source_char_end is not None else source_char_start + len(text)
    return TranslateProductPayload(
        source_span_id=source_span_id or product.source_utterance_id,
        target_span_id=product.target_utterance_id,
        revision=revision,
        text=text,
        committed_prefix_chars=len(text),
        is_final=product.is_final,
        source_char_start=source_char_start,
        source_char_end=end,
        target_language="es",
        alignment_kind=AlignmentKind.UNAVAILABLE,
    )


@dataclass
class SessionRuntime:
    """Per-session stage handles bound to a warm loaded model."""

    stage: Any
    sample_rate: int = BASELINE_SAMPLE_RATE_HZ
    listen_revision: int = 0
    translate_revision: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


def bind_stage_to_session(
    host: StageHost,
    session: SessionState,
    *,
    stage_factory: Callable[[Any], Any],
    sample_rate: int = BASELINE_SAMPLE_RATE_HZ,
) -> SessionRuntime:
    """Create per-session runtime from the host's already-loaded model."""
    model = host.model
    if model is None:
        raise RuntimeError("host model is not loaded")
    stage = stage_factory(model)
    runtime = SessionRuntime(stage=stage, sample_rate=sample_rate)
    session.data["runtime"] = runtime
    return runtime


async def run_listen_session(
    runtime: SessionRuntime,
    audio_stream: AsyncIterator[bytes],
) -> AsyncIterator[ListenProductPayload]:
    """Stream PCM → listen.product payloads with revision/commit semantics."""
    stage: WhisperListenStage = runtime.stage
    await stage.start()
    try:
        async for product in stage.transcribe(audio_stream):
            if not product.text and not product.is_final:
                continue
            payload = listen_product_to_payload(
                product,
                revision=runtime.listen_revision,
                sample_rate=runtime.sample_rate,
                commit_full_text=True,
            )
            runtime.listen_revision += 1
            # Skip empty non-useful finals with zero commit history if desired;
            # still emit finals so consumers see EOS.
            if payload.text or payload.is_final:
                yield payload
    finally:
        await stage.stop()


async def run_translate_session(
    runtime: SessionRuntime,
    listen_products: AsyncIterator[ListenProduct],
) -> AsyncIterator[TranslateProductPayload]:
    """Translate committed listen products → translate.product payloads."""
    stage: OpusMTTranslateStage = runtime.stage
    await stage.start()
    try:
        async for product in stage.translate(listen_products):
            if not product.text.strip() and not product.is_final:
                continue
            payload = translate_product_to_payload(
                product,
                revision=runtime.translate_revision,
            )
            runtime.translate_revision += 1
            yield payload
    finally:
        await stage.stop()


async def run_speak_session(
    runtime: SessionRuntime,
    translate_products: AsyncIterator[TranslateProduct],
) -> AsyncIterator[bytes]:
    """Speak committed Spanish → PCM chunks (utterance-buffered for edge-tts)."""
    stage = runtime.stage
    await stage.start()
    try:
        async for pcm in stage.synthesize(translate_products):
            if pcm:
                yield pcm
    finally:
        await stage.stop()


def _host_kwargs(
    *,
    stage_kind: StageKind,
    stage_id: str,
    model_loader: Callable[[], Any | Awaitable[Any]],
    model_revision: str,
    model_provider_id: str,
    model_artifact_digest: str = "unavailable",
    model_artifact_status: ArtifactDigestStatus | str = ArtifactDigestStatus.UNAVAILABLE,
    max_sessions: int = 1,
    canary: Callable[[Any], bool | Awaitable[bool]] | None = None,
    code_git_sha: str = "unknown",
    boot_id: str | None = None,
    local_dev: bool = True,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "stage_kind": stage_kind,
        "stage_id": stage_id,
        "stage_version": STAGE_VERSION,
        "model_loader": model_loader,
        "canary": canary,
        "max_sessions": max_sessions,
        "code_git_sha": code_git_sha,
        "model_provider_id": model_provider_id,
        "model_revision": model_revision,
        "model_artifact_digest": model_artifact_digest,
        "model_artifact_status": model_artifact_status,
        "boot_id": boot_id,
        "local_dev": local_dev,
        **extra,
    }


def build_whisper_listen_host(
    *,
    model_size: str | None = None,
    cache: Any = None,
    sample_rate: int = BASELINE_SAMPLE_RATE_HZ,
    max_sessions: int = 1,
    code_git_sha: str = "unknown",
    boot_id: str | None = None,
    model_loader: Callable[[], WhisperLoadedModel | Awaitable[WhisperLoadedModel]] | None = None,
    canary: Callable[[WhisperLoadedModel], bool | Awaitable[bool]] | None = None,
    local_dev: bool = True,
    **host_extra: Any,
) -> StageHost:
    """Build a StageHost that loads Whisper once and isolates session state."""
    size = model_size or os.environ.get("WHISPER_MODEL_SIZE", "base")

    def _loader() -> WhisperLoadedModel:
        return load_whisper_model(model_size=size, cache=cache)

    def _default_canary(loaded: WhisperLoadedModel) -> bool:
        return loaded.model is not None

    host = StageHost(
        **_host_kwargs(
            stage_kind=StageKind.LISTEN,
            stage_id="whisper-listen",
            model_loader=model_loader or _loader,
            model_revision=str(size),
            model_provider_id="faster-whisper",
            max_sessions=max_sessions,
            canary=canary or _default_canary,
            code_git_sha=code_git_sha,
            boot_id=boot_id,
            local_dev=local_dev,
            **host_extra,
        )
    )
    # Stash sample_rate for session binders.
    host.runtime_versions = {**host.runtime_versions, "sample_rate_hz": str(sample_rate)}
    return host


def build_opus_mt_host(
    *,
    model_id: str | None = None,
    cache: Any = None,
    max_sessions: int = 1,
    code_git_sha: str = "unknown",
    boot_id: str | None = None,
    model_loader: Callable[[], OpusMTLoadedModel | Awaitable[OpusMTLoadedModel]] | None = None,
    canary: Callable[[OpusMTLoadedModel], bool | Awaitable[bool]] | None = None,
    local_dev: bool = True,
    **host_extra: Any,
) -> StageHost:
    """Build a StageHost that loads Opus-MT once and isolates session state."""
    mid = model_id or os.environ.get("TRANSLATE_MODEL_ID", "Helsinki-NLP/opus-mt-en-es")

    def _loader() -> OpusMTLoadedModel:
        return load_opus_mt_model(model_id=mid, cache=cache)

    def _default_canary(loaded: OpusMTLoadedModel) -> bool:
        return loaded.translator is not None

    return StageHost(
        **_host_kwargs(
            stage_kind=StageKind.TRANSLATE,
            stage_id="opus-mt-en-es",
            model_loader=model_loader or _loader,
            model_revision=mid,
            model_provider_id="helsinki-nlp",
            max_sessions=max_sessions,
            canary=canary or _default_canary,
            code_git_sha=code_git_sha,
            boot_id=boot_id,
            local_dev=local_dev,
            **host_extra,
        )
    )


def build_edge_tts_host(
    *,
    voice_id: str | None = None,
    sample_rate: int = BASELINE_SAMPLE_RATE_HZ,
    max_sessions: int = 4,
    code_git_sha: str = "unknown",
    boot_id: str | None = None,
    model_loader: Callable[[], EdgeTTSLoadedModel | Awaitable[EdgeTTSLoadedModel]] | None = None,
    canary: Callable[[EdgeTTSLoadedModel], bool | Awaitable[bool]] | None = None,
    local_dev: bool = True,
    **host_extra: Any,
) -> StageHost:
    """Build a StageHost for edge-tts (network voice; utterance-buffered PCM)."""
    voice = voice_id or EDGE_TTS_VOICE

    def _loader() -> EdgeTTSLoadedModel:
        return load_edge_tts_model(voice_id=voice)

    def _default_canary(loaded: EdgeTTSLoadedModel) -> bool:
        return bool(loaded.voice_id) and loaded.streams_pcm is False

    host = StageHost(
        **_host_kwargs(
            stage_kind=StageKind.SPEAK,
            stage_id="edge-tts-es",
            model_loader=model_loader or _loader,
            model_revision=f"edge-tts:{voice}",
            model_provider_id="edge-tts",
            max_sessions=max_sessions,
            canary=canary or _default_canary,
            code_git_sha=code_git_sha,
            boot_id=boot_id,
            local_dev=local_dev,
            voice_digest=f"voice:{voice}",
            **host_extra,
        )
    )
    host.runtime_versions = {
        **host.runtime_versions,
        "sample_rate_hz": str(sample_rate),
        "streams_pcm": "false",
    }
    return host


def build_pocket_tts_host(
    *,
    language: str | None = None,
    voice: str | None = None,
    cache: Any = None,
    sample_rate: int = BASELINE_SAMPLE_RATE_HZ,
    max_sessions: int = 1,
    code_git_sha: str = "unknown",
    boot_id: str | None = None,
    model_loader: Callable[[], Any | Awaitable[Any]] | None = None,
    canary: Callable[[Any], bool | Awaitable[bool]] | None = None,
    local_dev: bool = True,
    **host_extra: Any,
) -> StageHost:
    """Build a StageHost for pocket-tts when the optional extra is installed.

    Raises ImportError if ``pocket_tts`` is not available (CONDITIONAL path).
    """
    load_fn, _stage_cls = _require_pocket()

    def _loader() -> Any:
        return load_fn(language=language, voice=voice, cache=cache)

    def _default_canary(loaded: Any) -> bool:
        return getattr(loaded, "model", None) is not None

    host = StageHost(
        **_host_kwargs(
            stage_kind=StageKind.SPEAK,
            stage_id="pocket-tts-spanish-24l",
            model_loader=model_loader or _loader,
            model_revision=f"pocket-tts:{language or 'spanish_24l'}:{voice or 'lola'}",
            model_provider_id="pocket-tts",
            max_sessions=max_sessions,
            canary=canary or _default_canary,
            code_git_sha=code_git_sha,
            boot_id=boot_id,
            local_dev=local_dev,
            voice_digest=f"voice:{voice or 'lola'}",
            **host_extra,
        )
    )
    host.runtime_versions = {
        **host.runtime_versions,
        "sample_rate_hz": str(sample_rate),
    }
    return host


def open_whisper_session_stage(
    host: StageHost,
    session: SessionState,
    *,
    sample_rate: int = BASELINE_SAMPLE_RATE_HZ,
) -> WhisperListenStage:
    """Bind a warm WhisperLoadedModel to a per-session WhisperListenStage."""
    loaded = host.model
    if not isinstance(loaded, WhisperLoadedModel):
        raise TypeError(f"expected WhisperLoadedModel, got {type(loaded)!r}")
    stage = WhisperListenStage(sample_rate=sample_rate, loaded_model=loaded)
    session.data["stage"] = stage
    session.data["loaded_model"] = loaded
    return stage


def open_opus_mt_session_stage(
    host: StageHost,
    session: SessionState,
) -> OpusMTTranslateStage:
    """Bind a warm OpusMTLoadedModel to a per-session OpusMTTranslateStage."""
    loaded = host.model
    if not isinstance(loaded, OpusMTLoadedModel):
        raise TypeError(f"expected OpusMTLoadedModel, got {type(loaded)!r}")
    stage = OpusMTTranslateStage(loaded_model=loaded)
    session.data["stage"] = stage
    session.data["loaded_model"] = loaded
    return stage


def open_edge_tts_session_stage(
    host: StageHost,
    session: SessionState,
    *,
    sample_rate: int = BASELINE_SAMPLE_RATE_HZ,
) -> EdgeTTSSpeakStage:
    """Bind a warm EdgeTTSLoadedModel to a per-session EdgeTTSSpeakStage."""
    loaded = host.model
    if not isinstance(loaded, EdgeTTSLoadedModel):
        raise TypeError(f"expected EdgeTTSLoadedModel, got {type(loaded)!r}")
    stage = EdgeTTSSpeakStage(sample_rate=sample_rate, loaded_model=loaded)
    session.data["stage"] = stage
    session.data["loaded_model"] = loaded
    return stage


def open_pocket_tts_session_stage(
    host: StageHost,
    session: SessionState,
    *,
    sample_rate: int = BASELINE_SAMPLE_RATE_HZ,
) -> Any:
    """Bind a warm PocketTTSLoadedModel to a per-session stage (if available)."""
    _load_fn, stage_cls = _require_pocket()
    loaded = host.model
    stage = stage_cls(sample_rate=sample_rate, loaded_model=loaded)
    session.data["stage"] = stage
    session.data["loaded_model"] = loaded
    return stage
