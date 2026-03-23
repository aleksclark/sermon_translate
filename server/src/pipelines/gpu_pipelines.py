"""GPU-accelerated pipelines using SeamlessM4T and Whisper on CUDA.

Requires an NVIDIA GPU with ≥6 GB VRAM.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

import numpy as np

from src.models import PipelineInfo, Session
from src.pipelines._audio import EDGE_TTS_VOICE, SentenceAccumulator, downsample
from src.pipelines.base import BasePipeline, OutputStreamDescriptor, OutputStreamKind

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000
SEAMLESS_SAMPLE_RATE = 16000
SEAMLESS_OUTPUT_RATE = 16000
BUFFER_SECONDS = 5
MIN_BUFFER_SECONDS = 1.5
TARGET_RATIO = 1.05
MAX_RATE_BOOST = 40
EDGE_TTS_SAMPLE_RATE = 24000


def _has_cuda() -> bool:
    import torch

    return torch.cuda.is_available()


def _load_seamless_gpu() -> tuple[Any, Any]:
    import json

    import torch
    from huggingface_hub import try_to_load_from_cache
    from transformers import (
        SeamlessM4TFeatureExtractor,
        SeamlessM4TProcessor,
        SeamlessM4TTokenizer,
        SeamlessM4Tv2Model,
    )

    model_id = "facebook/seamless-m4t-v2-large"
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
        vocab_file=sp_path, src_lang="eng", tgt_lang="spa",
        additional_special_tokens=additional,
    )
    feat_ext = SeamlessM4TFeatureExtractor.from_pretrained(
        model_id, local_files_only=True,
    )
    processor = SeamlessM4TProcessor(
        feature_extractor=feat_ext, tokenizer=tokenizer,
    )
    model = SeamlessM4Tv2Model.from_pretrained(
        model_id, local_files_only=True, dtype=torch.float16,
    )
    model = model.to("cuda")
    model.eval()
    return processor, model


def _wav_to_pcm_s16(wav: np.ndarray, src_rate: int, tgt_rate: int) -> bytes:
    resampled = downsample(wav, src_rate, tgt_rate)
    pcm = (resampled * 32767).clip(-32768, 32767).astype(np.int16)
    return pcm.tobytes()


def _synthesize_edge_tts_sync(
    text: str, target_rate: int, rate_pct: int,
) -> bytes:
    """Synchronous wrapper around edge-tts for use in executor."""
    import asyncio as _aio

    return _aio.run(_synthesize_edge_tts(text, target_rate, rate_pct))


async def _synthesize_edge_tts(
    text: str, target_rate: int, rate_pct: int = 0,
) -> bytes:
    import io

    import av
    import edge_tts

    if not text or not text.strip():
        return b""

    rate_str = f"+{rate_pct}%" if rate_pct >= 0 else f"{rate_pct}%"
    try:
        communicate = edge_tts.Communicate(text, voice=EDGE_TTS_VOICE, rate=rate_str)
        mp3_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_data += chunk.get("data", b"")
    except Exception:
        logger.exception("edge-tts failed")
        return b""

    if not mp3_data:
        return b""

    container = av.open(io.BytesIO(mp3_data), format="mp3")
    frames: list[np.ndarray] = []
    for frame in container.decode(audio=0):  # type: ignore[attr-defined]
        frames.append(frame.to_ndarray())
    if not frames:
        return b""
    audio = np.concatenate(frames, axis=1)
    mono = audio[0] if audio.shape[0] > 1 else audio.flatten()
    resampled = downsample(mono.astype(np.float32), EDGE_TTS_SAMPLE_RATE, target_rate)
    pcm = (resampled * 32767).clip(-32768, 32767).astype(np.int16)
    return pcm.tobytes()


# ---------------------------------------------------------------------------
# Shared GPU buffer processing mixin
# ---------------------------------------------------------------------------


class _BufferedGPUMixin:
    """Common audio ingest loop for GPU pipelines."""

    _sample_rate: int

    async def _run_buffered(
        self,
        audio_stream: AsyncIterator[bytes],
        process_fn: Any,
        audio_queue: asyncio.Queue[bytes | None],
        *,
        flush_fn: Any | None = None,
    ) -> None:
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
                await process_fn(segment, BUFFER_SECONDS)

        if len(buffer) >= min_samples:
            await process_fn(buffer, len(buffer) / SEAMLESS_SAMPLE_RATE)

        if flush_fn is not None:
            await flush_fn()

        await audio_queue.put(None)


# ---------------------------------------------------------------------------
# 1. Whisper GPU + SeamlessM4T T2ST: text→Spanish audio (best quality)
# ---------------------------------------------------------------------------


class GPUWhisperT2STPipeline(_BufferedGPUMixin, BasePipeline):
    """Whisper small GPU → SeamlessM4T T2ST GPU.

    Best-quality path: accurate English ASR on GPU, then SeamlessM4T
    translates text and synthesises Spanish speech in one forward pass.
    ~0.8s total processing per 5s buffer on RTX 3060.
    """

    def __init__(self, sample_rate: int = 48000) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._whisper: Any = None
        self._processor: Any = None
        self._model: Any = None
        self._en_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._es_queue: asyncio.Queue[str | None] = asyncio.Queue()

    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            id="gpu-whisper-t2st",
            name="GPU Whisper + SeamlessM4T T2ST",
            description=(
                "Whisper small (GPU) ASR → SeamlessM4T T2ST (GPU) "
                "for translation + voice synthesis. ~0.8s per 5s buffer."
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
        if self._whisper is None:
            from faster_whisper import WhisperModel

            self._whisper = await loop.run_in_executor(
                None,
                lambda: WhisperModel("small", device="cuda", compute_type="float16"),
            )
        if self._model is None:
            self._processor, self._model = await loop.run_in_executor(
                None, _load_seamless_gpu,
            )
        logger.info("GPU Whisper + SeamlessM4T loaded")

    async def _do_stop(self) -> None:
        self._whisper = None
        self._processor = None
        self._model = None
        await self._en_queue.put(None)
        await self._es_queue.put(None)

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None,
    ) -> AsyncIterator[bytes]:
        if self._model is None:
            await self.start()

        import torch

        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        acc = SentenceAccumulator()

        async def _emit_sentences(sentences: list[str]) -> None:
            if not sentences:
                return
            joined_en = " ".join(sentences)
            for en in sentences:
                await self._en_queue.put(en)

            def _t2st() -> tuple[str, np.ndarray]:
                inputs = self._processor(
                    text=joined_en, src_lang="eng", return_tensors="pt",
                )
                inputs = {
                    k: v.to("cuda") if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }
                with torch.no_grad():
                    out = self._model.generate(
                        **inputs, tgt_lang="spa", generate_speech=True,
                    )
                wav = out[0].squeeze().cpu().numpy().astype(np.float32)
                txt = self._model.generate(
                    **inputs, tgt_lang="spa", generate_speech=False,
                )
                es = self._processor.decode(
                    txt[0][0].tolist(), skip_special_tokens=True,
                ).strip()
                return es, wav

            es_text, wav = await loop.run_in_executor(None, _t2st)
            if es_text:
                await self._es_queue.put(es_text)
            if wav.size > 0:
                pcm = _wav_to_pcm_s16(wav, SEAMLESS_OUTPUT_RATE, self._sample_rate)
                if pcm:
                    await audio_queue.put(pcm)

        async def process_buffer(segment: np.ndarray, input_s: float) -> None:
            en_texts = await loop.run_in_executor(
                None,
                partial(self._transcribe, segment),
            )
            sentences = acc.push(en_texts)
            await _emit_sentences(sentences)

        async def flush() -> None:
            sentences = acc.flush()
            await _emit_sentences(sentences)

        ingest = asyncio.create_task(
            self._run_buffered(
                audio_stream, process_buffer, audio_queue, flush_fn=flush,
            ),
        )

        while True:
            data = await audio_queue.get()
            if data is None:
                break
            yield data

        await ingest
        await self._en_queue.put(None)
        await self._es_queue.put(None)

    def _transcribe(self, audio: np.ndarray) -> list[str]:
        segs, _ = self._whisper.transcribe(
            audio, beam_size=5, language="en", vad_filter=True,
        )
        return [s.text.strip() for s in segs if s.text.strip()]

    def iter_stream(
        self, name: str, audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        if name == "en-transcript":
            return self._drain_queue(self._en_queue)
        if name == "es-transcript":
            return self._drain_queue(self._es_queue)
        return None


# ---------------------------------------------------------------------------
# 2. SeamlessM4T S2ST: direct audio→audio (simplest, single model)
# ---------------------------------------------------------------------------


class GPUS2STPipeline(_BufferedGPUMixin, BasePipeline):
    """SeamlessM4T S2ST on GPU — direct audio→Spanish audio.

    Single model, no Whisper, no Edge TTS. The model translates
    and synthesises speech in one forward pass.
    ~0.5s per 5s buffer on RTX 3060.
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
            id="gpu-s2st",
            name="GPU SeamlessM4T S2ST (audio→audio)",
            description=(
                "Direct speech-to-speech on GPU via SeamlessM4T. "
                "Single forward pass, ~0.5s per 5s buffer."
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
            None, _load_seamless_gpu,
        )
        logger.info("GPU SeamlessM4T S2ST loaded")

    async def _do_stop(self) -> None:
        self._processor = None
        self._model = None
        await self._es_queue.put(None)

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None,
    ) -> AsyncIterator[bytes]:
        if self._model is None:
            await self.start()

        import torch

        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def process_buffer(segment: np.ndarray, input_s: float) -> None:
            def _s2st() -> tuple[str, np.ndarray]:
                inputs = self._processor(
                    audio=segment, src_lang="eng", return_tensors="pt",
                    sampling_rate=SEAMLESS_SAMPLE_RATE,
                )
                inputs = {
                    k: v.to("cuda") if hasattr(v, "to") else v
                    for k, v in inputs.items()
                }
                with torch.no_grad():
                    out = self._model.generate(
                        **inputs, tgt_lang="spa", generate_speech=True,
                    )
                wav = out[0]
                if wav.dim() == 0 or wav.numel() == 0:
                    wav_np = np.array([], dtype=np.float32)
                else:
                    wav_np = wav.squeeze().cpu().numpy().astype(np.float32)

                txt = self._model.generate(
                    **inputs, tgt_lang="spa", generate_speech=False,
                )
                es = self._processor.decode(
                    txt[0][0].tolist(), skip_special_tokens=True,
                ).strip()
                return es, wav_np

            es_text, wav = await loop.run_in_executor(None, _s2st)
            if es_text:
                await self._es_queue.put(es_text)
            if wav.size > 0:
                pcm = _wav_to_pcm_s16(wav, SEAMLESS_OUTPUT_RATE, self._sample_rate)
                if pcm:
                    await audio_queue.put(pcm)

        ingest = asyncio.create_task(
            self._run_buffered(audio_stream, process_buffer, audio_queue),
        )

        while True:
            data = await audio_queue.get()
            if data is None:
                break
            yield data

        await ingest
        await self._es_queue.put(None)

    def iter_stream(
        self, name: str, audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        if name == "es-transcript":
            return self._drain_queue(self._es_queue)
        return None


# ---------------------------------------------------------------------------
# 3. Whisper GPU + Opus-MT + Edge TTS (adaptive rate) — best text quality
# ---------------------------------------------------------------------------


class GPUWhisperOpusPipeline(_BufferedGPUMixin, BasePipeline):
    """Whisper small GPU → batched Opus-MT → Edge TTS with adaptive rate.

    Highest text accuracy (Opus-MT produces excellent translations),
    with adaptive TTS speed to stay synchronised with the speaker.
    """

    def __init__(self, sample_rate: int = 48000) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._whisper: Any = None
        self._translator: Any = None
        self._sp_source: Any = None
        self._sp_target: Any = None
        self._en_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._es_queue: asyncio.Queue[str | None] = asyncio.Queue()

    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            id="gpu-whisper-opus",
            name="GPU Whisper + Opus-MT + Edge TTS",
            description=(
                "Whisper small (GPU) → batched Opus-MT (CPU) → "
                "Edge TTS with adaptive rate. Best text quality."
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
        if self._whisper is None:
            from faster_whisper import WhisperModel

            self._whisper = await loop.run_in_executor(
                None,
                lambda: WhisperModel("small", device="cuda", compute_type="float16"),
            )
        if self._translator is None:
            self._translator, self._sp_source, self._sp_target = (
                await loop.run_in_executor(None, self._load_translation)
            )
        logger.info("GPU Whisper + Opus-MT loaded")

    def _load_translation(self) -> tuple[Any, Any, Any]:
        import ctranslate2
        import sentencepiece as spm
        from huggingface_hub import snapshot_download

        from src.pipelines.spanish_fast import SpanishFastPipeline

        ct2_dir = SpanishFastPipeline._get_ct2_model_dir()
        hf_dir = snapshot_download("Helsinki-NLP/opus-mt-en-es")
        translator = ctranslate2.Translator(ct2_dir, device="cpu", compute_type="int8")
        sp_src = spm.SentencePieceProcessor()
        sp_src.load(f"{hf_dir}/source.spm")  # type: ignore[attr-defined]
        sp_tgt = spm.SentencePieceProcessor()
        sp_tgt.load(f"{hf_dir}/target.spm")  # type: ignore[attr-defined]
        return translator, sp_src, sp_tgt

    async def _do_stop(self) -> None:
        self._whisper = None
        self._translator = None
        self._sp_source = None
        self._sp_target = None
        await self._en_queue.put(None)
        await self._es_queue.put(None)

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None,
    ) -> AsyncIterator[bytes]:
        if self._whisper is None:
            await self.start()

        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        acc = SentenceAccumulator()
        cumul_in = 0.0
        cumul_out = 0.0

        def _compute_rate() -> int:
            if cumul_in < 1.0:
                return 0
            ratio = cumul_out / cumul_in
            if ratio <= TARGET_RATIO:
                return 0
            return min(int((ratio - TARGET_RATIO) * 100), MAX_RATE_BOOST)

        async def _emit_sentences(sentences: list[str]) -> None:
            nonlocal cumul_out
            if not sentences:
                return

            es_texts = await loop.run_in_executor(
                None, partial(self._translate_batch, sentences),
            )
            for en, es in zip(sentences, es_texts, strict=True):
                await self._en_queue.put(en)
                await self._es_queue.put(es)

            joined_es = " ".join(es_texts)
            rate_pct = _compute_rate()
            pcm = await _synthesize_edge_tts(joined_es, self._sample_rate, rate_pct)
            if pcm:
                cumul_out += len(pcm) / (self._sample_rate * 2)
                await audio_queue.put(pcm)

        async def process_buffer(segment: np.ndarray, input_s: float) -> None:
            nonlocal cumul_in
            cumul_in += input_s

            en_texts = await loop.run_in_executor(
                None, partial(self._transcribe, segment),
            )
            sentences = acc.push(en_texts)
            await _emit_sentences(sentences)

        async def flush() -> None:
            sentences = acc.flush()
            await _emit_sentences(sentences)

        ingest = asyncio.create_task(
            self._run_buffered(
                audio_stream, process_buffer, audio_queue, flush_fn=flush,
            ),
        )

        while True:
            data = await audio_queue.get()
            if data is None:
                break
            yield data

        await ingest
        await self._en_queue.put(None)
        await self._es_queue.put(None)

    def _transcribe(self, audio: np.ndarray) -> list[str]:
        segs, _ = self._whisper.transcribe(
            audio, beam_size=5, language="en", vad_filter=True,
        )
        return [s.text.strip() for s in segs if s.text.strip()]

    def _translate_batch(self, texts: list[str]) -> list[str]:
        all_tokens = [
            self._sp_source.encode(t, out_type=str) + ["</s>"] for t in texts
        ]
        results = self._translator.translate_batch(all_tokens)
        return [self._sp_target.decode(r.hypotheses[0]) for r in results]

    def iter_stream(
        self, name: str, audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        if name == "en-transcript":
            return self._drain_queue(self._en_queue)
        if name == "es-transcript":
            return self._drain_queue(self._es_queue)
        return None
