"""Direct pipelines using SeamlessM4T for single-model inference.

- S2TT: audio → Spanish text (no intermediate English)
- T2ST: English text → Spanish audio (Whisper ASR + SeamlessM4T text-to-speech)
- S2ST: audio → Spanish audio (fully end-to-end)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

import numpy as np

from src.models import PipelineInfo, Session
from src.pipelines._audio import downsample
from src.pipelines.base import BasePipeline, OutputStreamDescriptor, OutputStreamKind

logger = logging.getLogger(__name__)

SEAMLESS_SAMPLE_RATE = 16000
SEAMLESS_OUTPUT_RATE = 16000
BUFFER_SECONDS = 5
MIN_BUFFER_SECONDS = 1.5


def _load_seamless() -> tuple[Any, Any]:
    import json

    from huggingface_hub import try_to_load_from_cache
    from transformers import (
        SeamlessM4TFeatureExtractor,
        SeamlessM4TProcessor,
        SeamlessM4TTokenizer,
        SeamlessM4Tv2Model,
    )

    model_id = "facebook/seamless-m4t-v2-large"
    cached = try_to_load_from_cache(model_id, "config.json")
    local_only = isinstance(cached, str)

    sp_path = try_to_load_from_cache(model_id, "tokenizer.model")
    if not isinstance(sp_path, str):
        from huggingface_hub import hf_hub_download

        sp_path = hf_hub_download(model_id, "tokenizer.model")

    cfg_path = try_to_load_from_cache(model_id, "tokenizer_config.json")
    if not isinstance(cfg_path, str):
        from huggingface_hub import hf_hub_download

        cfg_path = hf_hub_download(model_id, "tokenizer_config.json")

    with open(cfg_path) as f:
        additional = json.load(f).get("additional_special_tokens", [])

    tokenizer = SeamlessM4TTokenizer(
        vocab_file=sp_path,
        src_lang="eng",
        tgt_lang="spa",
        additional_special_tokens=additional,
    )
    feat_ext = SeamlessM4TFeatureExtractor.from_pretrained(
        model_id, local_files_only=local_only,
    )
    processor = SeamlessM4TProcessor(
        feature_extractor=feat_ext, tokenizer=tokenizer,
    )
    model = SeamlessM4Tv2Model.from_pretrained(
        model_id, local_files_only=local_only,
    )
    model.eval()
    return processor, model


def _s2tt_sync(processor: Any, model: Any, audio: np.ndarray) -> str:
    import torch

    inputs = processor(
        audio=audio, src_lang="eng", return_tensors="pt",
        sampling_rate=SEAMLESS_SAMPLE_RATE,
    )
    with torch.no_grad():
        out = model.generate(**inputs, tgt_lang="spa", generate_speech=False)
    tokens = out[0][0].tolist()
    return processor.decode(tokens, skip_special_tokens=True).strip()


def _s2st_sync(
    processor: Any, model: Any, audio: np.ndarray,
) -> tuple[str, np.ndarray]:
    """Return (text, waveform_float32) for audio → audio translation."""
    import torch

    inputs = processor(
        audio=audio, src_lang="eng", return_tensors="pt",
        sampling_rate=SEAMLESS_SAMPLE_RATE,
    )
    with torch.no_grad():
        out = model.generate(**inputs, tgt_lang="spa", generate_speech=True)

    waveform = out[0]
    if waveform.dim() == 0 or waveform.numel() == 0:
        return "", np.array([], dtype=np.float32)

    wav_np = waveform.squeeze().cpu().numpy().astype(np.float32)

    text_tokens = model.generate(
        **inputs, tgt_lang="spa", generate_speech=False,
    )
    text = processor.decode(text_tokens[0][0].tolist(), skip_special_tokens=True).strip()

    return text, wav_np


def _t2st_sync(
    processor: Any, model: Any, text: str,
) -> tuple[str, np.ndarray]:
    """Return (es_text, waveform_float32) for text → audio translation."""
    import torch

    inputs = processor(text=text, src_lang="eng", return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, tgt_lang="spa", generate_speech=True)

    waveform = out[0]
    if waveform.dim() == 0 or waveform.numel() == 0:
        return "", np.array([], dtype=np.float32)

    wav_np = waveform.squeeze().cpu().numpy().astype(np.float32)

    text_out = model.generate(**inputs, tgt_lang="spa", generate_speech=False)
    es_text = processor.decode(
        text_out[0][0].tolist(), skip_special_tokens=True,
    ).strip()

    return es_text, wav_np


def _wav_to_pcm_s16(wav: np.ndarray, src_rate: int, target_rate: int) -> bytes:
    resampled = downsample(wav, src_rate, target_rate)
    pcm = (resampled * 32767).clip(-32768, 32767).astype(np.int16)
    return pcm.tobytes()


# ---------------------------------------------------------------------------
# S2TT Pipeline: audio → Spanish text
# ---------------------------------------------------------------------------


class SeamlessS2TTPipeline(BasePipeline):
    """Direct audio → Spanish text via SeamlessM4T (no intermediate English)."""

    def __init__(self, sample_rate: int = 48000) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._processor: Any = None
        self._model: Any = None
        self._es_queue: asyncio.Queue[str | None] = asyncio.Queue()

    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            id="seamless-s2tt",
            name="SeamlessM4T S2TT (audio→Spanish text)",
            description="Direct speech-to-text translation via SeamlessM4T.",
            output_streams=self._build_output_stream_info(),
        )

    @property
    def output_streams(self) -> list[OutputStreamDescriptor]:
        return [
            OutputStreamDescriptor(
                name="es-transcript", kind=OutputStreamKind.TEXT, label="Spanish",
            ),
        ]

    async def _do_start(self) -> None:
        if self._model is not None:
            return
        loop = asyncio.get_running_loop()
        self._processor, self._model = await loop.run_in_executor(
            None, _load_seamless,
        )
        logger.info("SeamlessM4T loaded for S2TT")

    async def _do_stop(self) -> None:
        self._processor = None
        self._model = None
        await self._es_queue.put(None)

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None,
    ) -> AsyncIterator[bytes]:
        return
        yield  # noqa: F841

    def iter_stream(
        self, name: str, audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        if name == "es-transcript":
            return self._process_text(audio_stream)
        return None

    async def _process_text(
        self, audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str]:
        if self._model is None:
            await self.start()

        loop = asyncio.get_running_loop()
        buffer = np.array([], dtype=np.float32)
        samples_needed = int(SEAMLESS_SAMPLE_RATE * BUFFER_SECONDS)
        min_samples = int(SEAMLESS_SAMPLE_RATE * MIN_BUFFER_SECONDS)

        async for chunk in audio_stream:
            pcm_int16 = np.frombuffer(chunk, dtype=np.int16)
            pcm_float = pcm_int16.astype(np.float32) / 32768.0
            downsampled = downsample(pcm_float, self._sample_rate, SEAMLESS_SAMPLE_RATE)
            buffer = np.concatenate([buffer, downsampled])

            if len(buffer) >= samples_needed:
                segment = buffer.copy()
                buffer = np.array([], dtype=np.float32)
                text = await loop.run_in_executor(
                    None,
                    partial(_s2tt_sync, self._processor, self._model, segment),
                )
                if text:
                    await self._es_queue.put(text)
                    yield text

        if len(buffer) >= min_samples:
            text = await loop.run_in_executor(
                None,
                partial(_s2tt_sync, self._processor, self._model, buffer),
            )
            if text:
                await self._es_queue.put(text)
                yield text

        await self._es_queue.put(None)


# ---------------------------------------------------------------------------
# T2ST Pipeline: English text → Spanish audio (Whisper + SeamlessM4T T2ST)
# ---------------------------------------------------------------------------


class SeamlessT2STPipeline(BasePipeline):
    """English audio → Spanish audio via Whisper ASR + SeamlessM4T T2ST.

    Uses Whisper for accurate English transcription, then SeamlessM4T
    to generate Spanish speech directly from English text.
    """

    def __init__(
        self,
        whisper_model_size: str = "small",
        sample_rate: int = 48000,
    ) -> None:
        super().__init__()
        self._whisper_model_size = whisper_model_size
        self._sample_rate = sample_rate
        self._whisper_model: Any = None
        self._processor: Any = None
        self._model: Any = None
        self._en_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._es_queue: asyncio.Queue[str | None] = asyncio.Queue()

    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            id="seamless-t2st",
            name="SeamlessM4T T2ST (text→Spanish audio)",
            description=(
                "Whisper ASR → SeamlessM4T text-to-speech translation. "
                "Single model handles translation + voice synthesis."
            ),
            output_streams=self._build_output_stream_info(),
        )

    @property
    def output_streams(self) -> list[OutputStreamDescriptor]:
        return [
            OutputStreamDescriptor(
                name="audio", kind=OutputStreamKind.AUDIO, label="Spanish Audio",
            ),
            OutputStreamDescriptor(
                name="en-transcript", kind=OutputStreamKind.TEXT, label="English",
            ),
            OutputStreamDescriptor(
                name="es-transcript", kind=OutputStreamKind.TEXT, label="Spanish",
            ),
        ]

    async def _do_start(self) -> None:
        loop = asyncio.get_running_loop()
        if self._whisper_model is None:
            from faster_whisper import WhisperModel

            self._whisper_model = await loop.run_in_executor(
                None,
                lambda: WhisperModel(
                    self._whisper_model_size, device="cpu", compute_type="int8",
                ),
            )
        if self._model is None:
            self._processor, self._model = await loop.run_in_executor(
                None, _load_seamless,
            )
        logger.info("SeamlessM4T T2ST loaded")

    async def _do_stop(self) -> None:
        self._whisper_model = None
        self._processor = None
        self._model = None
        await self._en_queue.put(None)
        await self._es_queue.put(None)

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None,
    ) -> AsyncIterator[bytes]:
        if self._model is None:
            await self.start()

        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def ingest_and_process() -> None:
            buffer = np.array([], dtype=np.float32)
            samples_needed = int(WHISPER_SAMPLE_RATE * BUFFER_SECONDS)
            min_samples = int(WHISPER_SAMPLE_RATE * MIN_BUFFER_SECONDS)

            async for chunk in audio_stream:
                pcm_int16 = np.frombuffer(chunk, dtype=np.int16)
                pcm_float = pcm_int16.astype(np.float32) / 32768.0
                ds = downsample(pcm_float, self._sample_rate, WHISPER_SAMPLE_RATE)
                buffer = np.concatenate([buffer, ds])

                if len(buffer) >= samples_needed:
                    segment = buffer.copy()
                    buffer = np.array([], dtype=np.float32)
                    await self._process_buffer(loop, segment, audio_queue)

            if len(buffer) >= min_samples:
                await self._process_buffer(loop, buffer, audio_queue)

            await audio_queue.put(None)

        ingest_task = asyncio.create_task(ingest_and_process())

        while True:
            data = await audio_queue.get()
            if data is None:
                break
            yield data

        await ingest_task
        await self._en_queue.put(None)
        await self._es_queue.put(None)

    async def _process_buffer(
        self,
        loop: asyncio.AbstractEventLoop,
        segment: np.ndarray,
        audio_queue: asyncio.Queue[bytes | None],
    ) -> None:
        from src.pipelines.spanish_fast import _transcribe_sync

        en_texts = await loop.run_in_executor(
            None,
            partial(_transcribe_sync, self._whisper_model, segment),
        )
        if not en_texts:
            return

        joined_en = " ".join(en_texts)
        for en in en_texts:
            await self._en_queue.put(en)

        es_text, wav = await loop.run_in_executor(
            None,
            partial(_t2st_sync, self._processor, self._model, joined_en),
        )

        if es_text:
            await self._es_queue.put(es_text)

        if wav.size > 0:
            pcm = _wav_to_pcm_s16(wav, SEAMLESS_OUTPUT_RATE, self._sample_rate)
            if pcm:
                await audio_queue.put(pcm)

    def iter_stream(
        self, name: str, audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        if name == "en-transcript":
            return self._drain_queue(self._en_queue)
        if name == "es-transcript":
            return self._drain_queue(self._es_queue)
        return None


# ---------------------------------------------------------------------------
# S2ST Pipeline: audio → Spanish audio (fully end-to-end)
# ---------------------------------------------------------------------------


class SeamlessS2STPipeline(BasePipeline):
    """Direct audio → Spanish audio via SeamlessM4T S2ST.

    Single model, no ASR or TTS — the model translates and synthesises
    speech in one forward pass.
    """

    def __init__(self, sample_rate: int = 48000) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._processor: Any = None
        self._model: Any = None
        self._es_queue: asyncio.Queue[str | None] = asyncio.Queue()

    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            id="seamless-s2st",
            name="SeamlessM4T S2ST (audio→Spanish audio)",
            description=(
                "Fully end-to-end speech-to-speech translation. "
                "Single SeamlessM4T forward pass, no intermediate text."
            ),
            output_streams=self._build_output_stream_info(),
        )

    @property
    def output_streams(self) -> list[OutputStreamDescriptor]:
        return [
            OutputStreamDescriptor(
                name="audio", kind=OutputStreamKind.AUDIO, label="Spanish Audio",
            ),
            OutputStreamDescriptor(
                name="es-transcript", kind=OutputStreamKind.TEXT, label="Spanish",
            ),
        ]

    async def _do_start(self) -> None:
        if self._model is not None:
            return
        loop = asyncio.get_running_loop()
        self._processor, self._model = await loop.run_in_executor(
            None, _load_seamless,
        )
        logger.info("SeamlessM4T loaded for S2ST")

    async def _do_stop(self) -> None:
        self._processor = None
        self._model = None
        await self._es_queue.put(None)

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None,
    ) -> AsyncIterator[bytes]:
        if self._model is None:
            await self.start()

        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def ingest_and_process() -> None:
            buffer = np.array([], dtype=np.float32)
            samples_needed = int(SEAMLESS_SAMPLE_RATE * BUFFER_SECONDS)
            min_samples = int(SEAMLESS_SAMPLE_RATE * MIN_BUFFER_SECONDS)

            async for chunk in audio_stream:
                pcm_int16 = np.frombuffer(chunk, dtype=np.int16)
                pcm_float = pcm_int16.astype(np.float32) / 32768.0
                ds = downsample(pcm_float, self._sample_rate, SEAMLESS_SAMPLE_RATE)
                buffer = np.concatenate([buffer, ds])

                if len(buffer) >= samples_needed:
                    segment = buffer.copy()
                    buffer = np.array([], dtype=np.float32)
                    text, wav = await loop.run_in_executor(
                        None,
                        partial(
                            _s2st_sync, self._processor, self._model, segment,
                        ),
                    )
                    if text:
                        await self._es_queue.put(text)
                    if wav.size > 0:
                        pcm = _wav_to_pcm_s16(
                            wav, SEAMLESS_OUTPUT_RATE, self._sample_rate,
                        )
                        if pcm:
                            await audio_queue.put(pcm)

            if len(buffer) >= min_samples:
                text, wav = await loop.run_in_executor(
                    None,
                    partial(
                        _s2st_sync, self._processor, self._model, buffer,
                    ),
                )
                if text:
                    await self._es_queue.put(text)
                if wav.size > 0:
                    pcm = _wav_to_pcm_s16(
                        wav, SEAMLESS_OUTPUT_RATE, self._sample_rate,
                    )
                    if pcm:
                        await audio_queue.put(pcm)

            await audio_queue.put(None)

        ingest_task = asyncio.create_task(ingest_and_process())

        while True:
            data = await audio_queue.get()
            if data is None:
                break
            yield data

        await ingest_task
        await self._es_queue.put(None)

    def iter_stream(
        self, name: str, audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        if name == "es-transcript":
            return self._drain_queue(self._es_queue)
        return None
