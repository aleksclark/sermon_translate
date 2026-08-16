"""Kyutai STT-1B listen backend via the moshi package."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from src.models import ListenProduct, StageInfo, StageKind, WordSpan
from src.runtime.nvidia_libs import ensure_nvidia_library_path

logger = logging.getLogger(__name__)

HF_REPO_DEFAULT = "kyutai/stt-1b-en_fr"
MIMI_SAMPLE_RATE = 24000
EMIT_EVERY_TOKENS = 2


class KyutaiListenStage:
    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        cache: Any = None,
        hf_repo: str | None = None,
        device: str | None = None,
        **_: object,
    ) -> None:
        ensure_nvidia_library_path()
        import moshi  # type: ignore[import-not-found]  # noqa: F401

        self._sample_rate = sample_rate
        self._cache = cache
        self._hf_repo = hf_repo or os.environ.get("KYUTAI_STT_HF_REPO", HF_REPO_DEFAULT)
        self._device_override = device or os.environ.get("KYUTAI_STT_DEVICE", "").strip()
        self._mimi: Any = None
        self._lm_gen: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._julius: Any = None
        self._device = "cpu"
        self._prefix_seconds = 0.0
        self._delay_seconds = 0.5
        self._padding_token_id = 3
        self.info = StageInfo(
            id="kyutai-stt-1b",
            kind=StageKind.LISTEN,
            name="Kyutai STT-1B",
            description="Kyutai STT-1B streaming ASR via moshi (GPU recommended).",
            requires_gpu=True,
            default_for_kind=False,
        )

    async def start(self) -> None:
        if self._lm_gen is not None:
            return
        if self._cache is not None:
            os.environ.setdefault("HF_HOME", str(self._cache.path_for("huggingface")))
            os.environ.setdefault("TORCH_HOME", str(self._cache.path_for("torch")))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_models)
        logger.info("kyutai-stt-1b loaded repo=%s device=%s", self._hf_repo, self._device)

    def _load_models(self) -> None:
        ensure_nvidia_library_path()
        # Moshi CUDA graphs break under concurrent session/model loads.
        os.environ.setdefault("NO_CUDA_GRAPH", "1")
        import julius
        import torch
        from moshi.models.lm import LMGen
        from moshi.models.loaders import CheckpointInfo
        from moshi.utils.compile import no_cuda_graph

        from src.runtime.gpu_lock import gpu_model_load_lock

        self._torch = torch
        self._julius = julius
        if self._device_override:
            self._device = self._device_override
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        with gpu_model_load_lock(), no_cuda_graph():
            if self._device.startswith("cuda"):
                torch.cuda.synchronize()
            info = CheckpointInfo.from_hf_repo(self._hf_repo)
            self._mimi = info.get_mimi(device=self._device)
            self._tokenizer = info.get_text_tokenizer()
            dtype = torch.bfloat16 if self._device.startswith("cuda") else torch.float32
            lm = info.get_moshi(device=self._device, dtype=dtype)
            self._lm_gen = LMGen(lm, temp=0, temp_text=0.0)
            if self._device.startswith("cuda"):
                torch.cuda.synchronize()
        stt_config = info.stt_config or {}
        raw_config = info.raw_config or {}
        self._prefix_seconds = float(stt_config.get("audio_silence_prefix_seconds", 0.0))
        self._delay_seconds = float(stt_config.get("audio_delay_seconds", 0.5))
        self._padding_token_id = int(raw_config.get("text_padding_token_id", 3))

    async def stop(self) -> None:
        self._mimi = None
        self._lm_gen = None
        self._tokenizer = None
        self._torch = None
        self._julius = None

    def _decode_token(self, text_tokens: Any) -> str | None:
        if text_tokens is None:
            return None
        token = int(text_tokens.reshape(-1)[0].detach().cpu().item())
        if token in (0, self._padding_token_id):
            return None
        piece = self._tokenizer.id_to_piece(token)
        text = str(piece).replace("\u2581", " ")
        return text if text else None

    def _product(self, *, sequence: int, text: str, is_final: bool) -> ListenProduct:
        words = [
            WordSpan(text=word, start_ms=None, end_ms=None, conf=1.0)
            for word in text.split()
            if word
        ]
        return ListenProduct(
            sequence=sequence,
            utterance_id=f"kyutai-{sequence}",
            text=text,
            is_final=is_final,
            words=words,
            language="en",
        )

    def _resample_chunk(self, pcm_f32: np.ndarray) -> np.ndarray:
        assert self._torch is not None and self._julius is not None
        if pcm_f32.size == 0:
            return pcm_f32
        if self._sample_rate == MIMI_SAMPLE_RATE:
            return pcm_f32.astype(np.float32, copy=False)
        tensor = self._torch.from_numpy(np.ascontiguousarray(pcm_f32, dtype=np.float32))
        if tensor.dim() == 1:
            tensor = tensor[None, :]
        resampled = self._julius.resample_frac(tensor, self._sample_rate, MIMI_SAMPLE_RATE)
        return resampled.reshape(-1).detach().cpu().numpy().astype(np.float32, copy=False)

    async def transcribe(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[ListenProduct]:
        if self._lm_gen is None:
            await self.start()
        assert self._mimi is not None
        assert self._lm_gen is not None
        assert self._tokenizer is not None
        assert self._torch is not None

        torch = self._torch
        frame_size = int(self._mimi.frame_size)
        mimi_rate = int(getattr(self._mimi, "sample_rate", MIMI_SAMPLE_RATE))
        buffer = np.zeros(0, dtype=np.float32)
        sequence = 0
        text_parts: list[str] = []
        last_emitted = ""
        tokens_since_emit = 0
        silence = torch.zeros((1, 1, frame_size), dtype=torch.float32, device=self._device)
        n_prefix = max(0, int(round(self._prefix_seconds * mimi_rate / frame_size)))
        n_suffix = max(1, int(round(self._delay_seconds * mimi_rate / frame_size)))

        def _step(chunk: Any) -> str | None:
            audio_tokens = self._mimi.encode(chunk)
            text_tokens = self._lm_gen.step(audio_tokens)
            return self._decode_token(text_tokens)

        loop = asyncio.get_running_loop()

        def _maybe_product(*, force: bool = False, final: bool = False) -> ListenProduct | None:
            nonlocal sequence, last_emitted, tokens_since_emit
            text = "".join(text_parts).strip()
            if not text or text == last_emitted:
                return None
            if not force and tokens_since_emit < EMIT_EVERY_TOKENS and not final:
                return None
            product = self._product(sequence=sequence, text=text, is_final=final)
            last_emitted = text
            sequence += 1
            tokens_since_emit = 0
            return product

        with self._mimi.streaming(1), self._lm_gen.streaming(1):
            for _ in range(n_prefix):
                await loop.run_in_executor(None, _step, silence)

            async for raw in audio_stream:
                if not raw:
                    continue
                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if pcm.size == 0:
                    continue
                pcm = await loop.run_in_executor(None, self._resample_chunk, pcm)
                buffer = np.concatenate([buffer, pcm])

                while buffer.size >= frame_size:
                    frame, buffer = buffer[:frame_size], buffer[frame_size:]
                    tensor = (
                        torch.from_numpy(np.ascontiguousarray(frame))
                        .to(self._device)
                        .view(1, 1, -1)
                    )
                    piece = await loop.run_in_executor(None, _step, tensor)
                    if piece:
                        text_parts.append(piece)
                        tokens_since_emit += 1
                        boundary = piece.startswith(" ") or piece.endswith(" ") or " " in piece
                        product = _maybe_product(force=boundary)
                        if product is not None:
                            yield product

                # Live delay flush: tokens lag audio by ~delay_seconds.
                if text_parts:
                    for _ in range(min(2, n_suffix)):
                        piece = await loop.run_in_executor(None, _step, silence)
                        if piece:
                            text_parts.append(piece)
                            tokens_since_emit += 1
                            product = _maybe_product(force=True)
                            if product is not None:
                                yield product

            if buffer.size > 0:
                pad = frame_size - (buffer.size % frame_size)
                if pad != frame_size:
                    buffer = np.concatenate([buffer, np.zeros(pad, dtype=np.float32)])
                for i in range(0, buffer.size, frame_size):
                    frame = buffer[i : i + frame_size]
                    tensor = (
                        torch.from_numpy(np.ascontiguousarray(frame))
                        .to(self._device)
                        .view(1, 1, -1)
                    )
                    piece = await loop.run_in_executor(None, _step, tensor)
                    if piece:
                        text_parts.append(piece)
                        tokens_since_emit += 1

            for _ in range(n_suffix * 2):
                piece = await loop.run_in_executor(None, _step, silence)
                if piece:
                    text_parts.append(piece)
                    tokens_since_emit += 1

        product = _maybe_product(force=True, final=True)
        if product is not None:
            yield product
        elif last_emitted:
            yield self._product(sequence=sequence, text=last_emitted, is_final=True)


class KyutaiListenFactory:
    @property
    def info(self) -> StageInfo:
        return StageInfo(
            id="kyutai-stt-1b",
            kind=StageKind.LISTEN,
            name="Kyutai STT-1B",
            description="Kyutai STT-1B streaming ASR via moshi (GPU recommended).",
            requires_gpu=True,
            default_for_kind=False,
        )

    def create(self, **kwargs: Any) -> KyutaiListenStage:
        return KyutaiListenStage(**kwargs)
