"""Kyutai STT-1B listen backend via the moshi package."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from src.models import ListenProduct, StageInfo, StageKind, WordSpan
from src.pipelines._audio import downsample
from src.runtime.nvidia_libs import ensure_nvidia_library_path

logger = logging.getLogger(__name__)

HF_REPO_DEFAULT = "kyutai/stt-1b-en_fr"
MIMI_SAMPLE_RATE = 24000
# Emit a partial product at least this often while tokens accumulate.
EMIT_EVERY_FRAMES = 5


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
        self._device = "cpu"
        self._prefix_seconds = 1.0
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
        import torch
        from moshi.models.lm import LMGen
        from moshi.models.loaders import CheckpointInfo

        self._torch = torch
        if self._device_override:
            self._device = self._device_override
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        info = CheckpointInfo.from_hf_repo(self._hf_repo)
        self._mimi = info.get_mimi(device=self._device)
        self._tokenizer = info.get_text_tokenizer()
        dtype = torch.bfloat16 if self._device.startswith("cuda") else torch.float32
        lm = info.get_moshi(device=self._device, dtype=dtype)
        self._lm_gen = LMGen(lm, temp=0, temp_text=0.0)
        stt_config = info.stt_config or {}
        raw_config = info.raw_config or {}
        self._prefix_seconds = float(stt_config.get("audio_silence_prefix_seconds", 1.0))
        self._delay_seconds = float(stt_config.get("audio_delay_seconds", 0.5))
        self._padding_token_id = int(raw_config.get("text_padding_token_id", 3))

    async def stop(self) -> None:
        self._mimi = None
        self._lm_gen = None
        self._tokenizer = None
        self._torch = None

    def _product(
        self, *, sequence: int, text: str, is_final: bool, start_ms: float, end_ms: float
    ) -> ListenProduct:
        words = [
            WordSpan(text=word, start_ms=None, end_ms=None, conf=1.0)
            for word in text.split()
            if word
        ]
        if not words and text:
            words = [WordSpan(text=text, start_ms=start_ms, end_ms=end_ms, conf=1.0)]
        return ListenProduct(
            sequence=sequence,
            utterance_id=f"kyutai-{sequence}",
            text=text,
            is_final=is_final,
            words=words,
            language="en",
        )

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
        frame_ms = frame_size / mimi_rate * 1000.0
        buffer = np.zeros(0, dtype=np.float32)
        sequence = 0
        text_parts: list[str] = []
        last_emitted = ""
        frames_since_emit = 0
        emitted_ms = 0.0
        utterance_start_ms = 0.0

        silence = torch.zeros((1, 1, frame_size), dtype=torch.float32, device=self._device)
        n_prefix = max(0, int(round(self._prefix_seconds * mimi_rate / frame_size)))
        n_suffix = max(1, int(round(self._delay_seconds * mimi_rate / frame_size)))

        def _step(chunk: Any) -> str | None:
            audio_tokens = self._mimi.encode(chunk)
            text_tokens = self._lm_gen.step(audio_tokens)
            # LMGen may return [B, 1, 1] or similar; take first scalar token.
            token_tensor = text_tokens.reshape(-1)[0]
            token = int(token_tensor.detach().cpu().item())
            if token in (0, self._padding_token_id):
                return None
            piece = self._tokenizer.id_to_piece(token)
            text = str(piece).replace("▁", " ")
            return text if text else None

        def _should_emit(piece: str | None, force: bool = False) -> bool:
            nonlocal frames_since_emit
            if force:
                return True
            if piece is None:
                return False
            # Word boundary or enough frames of new content.
            if piece.startswith(" ") or piece.endswith(" ") or " " in piece.strip():
                return True
            return frames_since_emit >= EMIT_EVERY_FRAMES

        loop = asyncio.get_running_loop()
        with self._mimi.streaming(1), self._lm_gen.streaming(1):
            for _ in range(n_prefix):
                await loop.run_in_executor(None, _step, silence)

            async for raw in audio_stream:
                if not raw:
                    continue
                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if pcm.size == 0:
                    continue
                # Average multi-channel down to mono if needed.
                if pcm.size % 1 == 0 and self._sample_rate > 0:
                    pass
                if self._sample_rate != mimi_rate:
                    pcm = downsample(pcm, self._sample_rate, mimi_rate)
                buffer = np.concatenate([buffer, pcm.astype(np.float32, copy=False)])

                while buffer.size >= frame_size:
                    frame = buffer[:frame_size]
                    buffer = buffer[frame_size:]
                    tensor = (
                        torch.from_numpy(np.ascontiguousarray(frame))
                        .to(self._device)
                        .view(1, 1, -1)
                    )
                    piece = await loop.run_in_executor(None, _step, tensor)
                    emitted_ms += frame_ms
                    frames_since_emit += 1
                    if piece:
                        if not text_parts:
                            utterance_start_ms = max(0.0, emitted_ms - frame_ms)
                        text_parts.append(piece)

                    text = "".join(text_parts).strip()
                    if text and text != last_emitted and _should_emit(piece):
                        yield self._product(
                            sequence=sequence,
                            text=text,
                            is_final=False,
                            start_ms=utterance_start_ms,
                            end_ms=emitted_ms,
                        )
                        last_emitted = text
                        sequence += 1
                        frames_since_emit = 0

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
                    emitted_ms += frame_ms
                    if piece:
                        text_parts.append(piece)

            # Audio delay: tokens lag real speech; flush with silence after input.
            for _ in range(n_suffix):
                piece = await loop.run_in_executor(None, _step, silence)
                emitted_ms += frame_ms
                if piece:
                    text_parts.append(piece)

        text = "".join(text_parts).strip()
        if text and text != last_emitted or text:
            yield self._product(
                sequence=sequence,
                text=text,
                is_final=True,
                start_ms=utterance_start_ms,
                end_ms=emitted_ms,
            )


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
