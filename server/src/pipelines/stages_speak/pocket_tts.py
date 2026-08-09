from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from src.models import StageInfo, StageKind, TranslateProduct
from src.pipelines._audio import downsample
from src.runtime.nvidia_libs import ensure_nvidia_library_path

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "spanish_24l"
DEFAULT_VOICE = "lola"


@dataclass(frozen=True, slots=True)
class PocketTTSLoadedModel:
    """Immutable resident Pocket TTS weights + shared voice prompt state (D6)."""

    model: Any
    voice_state: Any
    language: str
    voice: str
    revision: str = "unknown"
    streams_pcm: bool = True


def load_pocket_tts_model(
    *,
    language: str | None = None,
    voice: str | None = None,
    cache: Any = None,
) -> PocketTTSLoadedModel:
    """Load Pocket TTS once (sync). Used by StageHost model_loader when extra is installed."""
    ensure_nvidia_library_path()
    if cache is not None:
        os.environ.setdefault("HF_HOME", str(cache.path_for("huggingface")))
        os.environ.setdefault("TORCH_HOME", str(cache.path_for("torch")))
    os.environ.setdefault("NO_CUDA_GRAPH", "1")
    from pocket_tts import TTSModel  # type: ignore[import-not-found]

    from src.runtime.gpu_lock import gpu_model_load_lock

    lang = language or os.environ.get("POCKET_TTS_LANGUAGE", DEFAULT_LANGUAGE)
    voice_name = voice or os.environ.get("POCKET_TTS_VOICE", DEFAULT_VOICE)
    with gpu_model_load_lock():
        model = TTSModel.load_model(language=lang)
        voice_state = model.get_state_for_audio_prompt(voice_name)
    return PocketTTSLoadedModel(
        model=model,
        voice_state=voice_state,
        language=lang,
        voice=voice_name,
        revision=f"pocket-tts:{lang}:{voice_name}",
        streams_pcm=callable(getattr(model, "generate_audio_stream", None)),
    )


class PocketTTSSpeakStage:
    """Kyutai Pocket TTS spanish_24l CPU fallback.

    Weights may be injected via ``loaded_model`` so ``stop()`` never unloads them.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        cache: Any = None,
        language: str | None = None,
        voice: str | None = None,
        loaded_model: PocketTTSLoadedModel | None = None,
        **_: object,
    ) -> None:
        ensure_nvidia_library_path()
        self._sample_rate = sample_rate
        self._cache = cache
        self._language = language or os.environ.get("POCKET_TTS_LANGUAGE", DEFAULT_LANGUAGE)
        self._voice = voice or os.environ.get("POCKET_TTS_VOICE", DEFAULT_VOICE)
        self._loaded: PocketTTSLoadedModel | None = loaded_model
        self._owns_model = loaded_model is None
        self._session_active = False
        self._utterance_count = 0
        self.info = StageInfo(
            id="pocket-tts-spanish-24l",
            kind=StageKind.SPEAK,
            name="Pocket TTS Spanish",
            description="Kyutai Pocket TTS spanish_24l CPU fallback.",
            requires_gpu=False,
            default_for_kind=False,
        )

    @property
    def loaded_model(self) -> PocketTTSLoadedModel | None:
        return self._loaded

    @property
    def _model(self) -> Any | None:
        return None if self._loaded is None else self._loaded.model

    @_model.setter
    def _model(self, value: Any | None) -> None:
        if value is None:
            if self._owns_model:
                self._loaded = None
            return
        if self._loaded is not None and not self._owns_model:
            return
        voice_state = self._loaded.voice_state if self._loaded is not None else None
        self._loaded = PocketTTSLoadedModel(
            model=value,
            voice_state=voice_state,
            language=self._language,
            voice=self._voice,
            revision=f"pocket-tts:{self._language}:{self._voice}",
        )

    @property
    def _voice_state(self) -> Any | None:
        return None if self._loaded is None else self._loaded.voice_state

    @_voice_state.setter
    def _voice_state(self, value: Any | None) -> None:
        if self._loaded is None:
            if value is None:
                return
            self._loaded = PocketTTSLoadedModel(
                model=object(),
                voice_state=value,
                language=self._language,
                voice=self._voice,
            )
            return
        if not self._owns_model:
            return
        self._loaded = PocketTTSLoadedModel(
            model=self._loaded.model,
            voice_state=value,
            language=self._loaded.language,
            voice=self._loaded.voice,
            revision=self._loaded.revision,
            streams_pcm=self._loaded.streams_pcm,
        )

    async def start(self) -> None:
        self._session_active = True
        self._utterance_count = 0
        if self._loaded is not None:
            return
        if self._cache is not None:
            os.environ.setdefault("HF_HOME", str(self._cache.path_for("huggingface")))
            os.environ.setdefault("TORCH_HOME", str(self._cache.path_for("torch")))
        loop = asyncio.get_running_loop()
        loaded = await loop.run_in_executor(
            None,
            partial(
                load_pocket_tts_model,
                language=self._language,
                voice=self._voice,
                cache=self._cache,
            ),
        )
        self._loaded = loaded
        self._owns_model = True
        logger.info(
            "pocket-tts loaded language=%s voice=%s sample_rate=%s",
            loaded.language,
            loaded.voice,
            getattr(loaded.model, "sample_rate", "?"),
        )

    def _load(self) -> None:
        """Legacy sync loader used by older call sites."""
        self._loaded = load_pocket_tts_model(
            language=self._language,
            voice=self._voice,
            cache=self._cache,
        )
        self._owns_model = True

    async def stop(self) -> None:
        """Clear per-session utterance state. Never unloads preloaded weights."""
        self._session_active = False
        self._utterance_count = 0
        if self._owns_model:
            self._loaded = None

    async def synthesize(
        self, text_stream: AsyncIterator[TranslateProduct]
    ) -> AsyncIterator[bytes]:
        if self._loaded is None:
            await self.start()
        assert self._loaded is not None

        loop = asyncio.get_running_loop()
        async for product in text_stream:
            text = product.text.strip()
            if not text:
                continue
            pcm = await loop.run_in_executor(None, self._synth_sync, text)
            self._utterance_count += 1
            if pcm:
                yield pcm

    def _synth_sync(self, text: str) -> bytes:
        assert self._loaded is not None
        model = self._loaded.model
        voice_state = self._loaded.voice_state
        chunks: list[np.ndarray] = []
        stream = getattr(model, "generate_audio_stream", None)
        if callable(stream):
            stream_iter = stream(voice_state, text)
            for chunk in stream_iter:  # type: ignore[attr-defined]
                arr = _to_float_numpy(chunk)
                if arr.size:
                    chunks.append(arr)
        else:
            audio = model.generate_audio(voice_state, text)
            arr = _to_float_numpy(audio)
            if arr.size:
                chunks.append(arr)
        if not chunks:
            return b""
        audio_f = np.concatenate(chunks)
        model_rate = int(getattr(model, "sample_rate", 24000))
        if model_rate != self._sample_rate:
            audio_f = downsample(audio_f.astype(np.float32), model_rate, self._sample_rate)
        pcm = (audio_f * 32767.0).clip(-32768, 32767).astype(np.int16)
        return pcm.tobytes()


def _to_float_numpy(audio: Any) -> np.ndarray:
    if audio is None:
        return np.zeros(0, dtype=np.float32)
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    return arr


class PocketTTSFactory:
    @property
    def info(self) -> StageInfo:
        return StageInfo(
            id="pocket-tts-spanish-24l",
            kind=StageKind.SPEAK,
            name="Pocket TTS Spanish",
            description="Kyutai Pocket TTS spanish_24l CPU fallback.",
            requires_gpu=False,
            default_for_kind=False,
        )

    def create(self, **kwargs: Any) -> PocketTTSSpeakStage:
        return PocketTTSSpeakStage(**kwargs)
