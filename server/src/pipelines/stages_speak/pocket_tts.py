from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from src.models import StageInfo, StageKind, TranslateProduct
from src.pipelines._audio import downsample
from src.runtime.nvidia_libs import ensure_nvidia_library_path

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "spanish_24l"
DEFAULT_VOICE = "lola"


class PocketTTSSpeakStage:
    """Kyutai Pocket TTS spanish_24l CPU fallback."""

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        cache: Any = None,
        language: str | None = None,
        voice: str | None = None,
        **_: object,
    ) -> None:
        ensure_nvidia_library_path()
        self._sample_rate = sample_rate
        self._cache = cache
        self._language = language or os.environ.get("POCKET_TTS_LANGUAGE", DEFAULT_LANGUAGE)
        self._voice = voice or os.environ.get("POCKET_TTS_VOICE", DEFAULT_VOICE)
        self._model: Any = None
        self._voice_state: Any = None
        self.info = StageInfo(
            id="pocket-tts-spanish-24l",
            kind=StageKind.SPEAK,
            name="Pocket TTS Spanish",
            description="Kyutai Pocket TTS spanish_24l CPU fallback.",
            requires_gpu=False,
            default_for_kind=False,
        )

    async def start(self) -> None:
        if self._model is not None:
            return
        if self._cache is not None:
            os.environ.setdefault("HF_HOME", str(self._cache.path_for("huggingface")))
            os.environ.setdefault("TORCH_HOME", str(self._cache.path_for("torch")))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load)
        logger.info(
            "pocket-tts loaded language=%s voice=%s sample_rate=%s",
            self._language,
            self._voice,
            getattr(self._model, "sample_rate", "?"),
        )

    def _load(self) -> None:
        ensure_nvidia_library_path()
        os.environ.setdefault("NO_CUDA_GRAPH", "1")
        from pocket_tts import TTSModel  # type: ignore[import-not-found]

        from src.runtime.gpu_lock import gpu_model_load_lock

        with gpu_model_load_lock():
            self._model = TTSModel.load_model(language=self._language)
            self._voice_state = self._model.get_state_for_audio_prompt(self._voice)

    async def stop(self) -> None:
        self._model = None
        self._voice_state = None

    async def synthesize(
        self, text_stream: AsyncIterator[TranslateProduct]
    ) -> AsyncIterator[bytes]:
        if self._model is None:
            await self.start()
        assert self._model is not None
        assert self._voice_state is not None

        loop = asyncio.get_running_loop()
        async for product in text_stream:
            text = product.text.strip()
            if not text:
                continue
            pcm = await loop.run_in_executor(None, self._synth_sync, text)
            if pcm:
                yield pcm

    def _synth_sync(self, text: str) -> bytes:
        assert self._model is not None
        assert self._voice_state is not None
        chunks: list[np.ndarray] = []
        stream = getattr(self._model, "generate_audio_stream", None)
        if callable(stream):
            stream_iter = stream(self._voice_state, text)
            for chunk in stream_iter:  # type: ignore[attr-defined]
                arr = _to_float_numpy(chunk)
                if arr.size:
                    chunks.append(arr)
        else:
            audio = self._model.generate_audio(self._voice_state, text)
            arr = _to_float_numpy(audio)
            if arr.size:
                chunks.append(arr)
        if not chunks:
            return b""
        audio_f = np.concatenate(chunks)
        model_rate = int(getattr(self._model, "sample_rate", 24000))
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
