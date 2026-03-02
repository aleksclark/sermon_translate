from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from src.models import OutputStreamInfo, PipelineInfo, Session
from src.pipelines.base import (
    BasePipeline,
    OutputStreamDescriptor,
    OutputStreamKind,
)
from src.pipelines.spanish import (
    EDGE_TTS_VOICE,
    _decode_mp3_to_pcm,
)
from src.pipelines.whisper_tts import _downsample

logger = logging.getLogger(__name__)

SEAMLESS_SAMPLE_RATE = 16000
BUFFER_SECONDS = 3
MIN_BUFFER_SECONDS = 1.0


async def _synthesize_spanish(text: str, target_rate: int) -> bytes:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice=EDGE_TTS_VOICE)
    mp3_data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_data += chunk.get("data", b"")

    if not mp3_data:
        return b""

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _decode_mp3_to_pcm, mp3_data, target_rate
    )


def _translate_audio_sync(
    processor: Any,
    model: Any,
    audio: np.ndarray,
) -> str:
    import torch

    inputs = processor(
        audio=audio,
        src_lang="eng",
        return_tensors="pt",
        sampling_rate=SEAMLESS_SAMPLE_RATE,
    )
    with torch.no_grad():
        output = model.generate(
            **inputs,
            tgt_lang="spa",
            generate_speech=False,
        )
    sequences = output[0] if isinstance(output, (tuple, list)) else output
    if hasattr(sequences, "sequences"):
        sequences = sequences.sequences
    token_list = sequences[0].tolist()
    logger.debug("generate tokens: %s", token_list)

    tok = processor.tokenizer
    special_ids = set(tok.all_special_ids)
    sp_limit = tok.sp_model.get_piece_size() + tok.fairseq_offset
    filtered = [
        t for t in token_list
        if t not in special_ids and t < sp_limit
    ]
    text: str = tok.decode(filtered, skip_special_tokens=False)
    logger.debug("decoded: %r", text[:200])
    return text.strip()


def _translate_with_context_sync(
    processor: Any,
    model: Any,
    context_audio: np.ndarray,
    full_audio: np.ndarray,
) -> str:
    if len(context_audio) == 0:
        return _translate_audio_sync(processor, model, full_audio)

    full_text = _translate_audio_sync(processor, model, full_audio)

    context_text = _translate_audio_sync(processor, model, context_audio)

    if full_text.startswith(context_text):
        new_text = full_text[len(context_text):].strip()
        if new_text:
            return new_text

    if context_text and context_text in full_text:
        idx = full_text.index(context_text) + len(context_text)
        new_text = full_text[idx:].strip()
        if new_text:
            return new_text

    return full_text


class SpanishDirectPipeline(BasePipeline):
    """English audio in → Spanish audio + transcript via SeamlessM4T."""

    def __init__(self, sample_rate: int = 48000) -> None:
        self._sample_rate = sample_rate
        self._processor: Any = None
        self._model: Any = None
        self._audio_context_seconds: float = 0.0
        self._es_queue: asyncio.Queue[str | None] = asyncio.Queue()

    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            id="spanish-direct",
            name="Spanish Direct (SeamlessM4T)",
            description=(
                "Direct English speech to Spanish translation "
                "using SeamlessM4T. Supports audio context."
            ),
            output_streams=[
                OutputStreamInfo(
                    name=s.name, kind=s.kind.value, label=s.label
                )
                for s in self.output_streams
            ],
        )

    @property
    def output_streams(self) -> list[OutputStreamDescriptor]:
        return [
            OutputStreamDescriptor(
                name="audio",
                kind=OutputStreamKind.AUDIO,
                label="Spanish Audio",
            ),
            OutputStreamDescriptor(
                name="es-transcript",
                kind=OutputStreamKind.TEXT,
                label="Spanish",
            ),
        ]

    def configure_session(self, session: Session) -> None:
        self._audio_context_seconds = session.audio_context_seconds

    async def start(self) -> None:
        if self._model is not None:
            return
        loop = asyncio.get_running_loop()
        self._processor, self._model = await loop.run_in_executor(
            None, self._load_model
        )
        logger.info("SeamlessM4T model loaded")

    @staticmethod
    def _load_model() -> tuple[Any, Any]:
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
        if local_only:
            logger.info("Loading SeamlessM4T from cache")
        else:
            logger.info("Downloading SeamlessM4T from HuggingFace")

        sp_path = try_to_load_from_cache(model_id, "tokenizer.model")
        if not isinstance(sp_path, str):
            from huggingface_hub import hf_hub_download

            sp_path = hf_hub_download(model_id, "tokenizer.model")

        cfg_path = try_to_load_from_cache(
            model_id, "tokenizer_config.json"
        )
        if not isinstance(cfg_path, str):
            from huggingface_hub import hf_hub_download

            cfg_path = hf_hub_download(
                model_id, "tokenizer_config.json"
            )
        with open(cfg_path) as f:
            additional = json.load(f).get(
                "additional_special_tokens", []
            )

        tokenizer = SeamlessM4TTokenizer(
            vocab_file=sp_path,
            src_lang="eng",
            tgt_lang="spa",
            additional_special_tokens=additional,
        )
        feat_ext = SeamlessM4TFeatureExtractor.from_pretrained(
            model_id, local_files_only=local_only
        )
        processor = SeamlessM4TProcessor(
            feature_extractor=feat_ext, tokenizer=tokenizer
        )

        model = SeamlessM4Tv2Model.from_pretrained(
            model_id, local_files_only=local_only
        )
        model.eval()
        return processor, model

    async def stop(self) -> None:
        self._processor = None
        self._model = None
        await self._es_queue.put(None)

    async def process(
        self, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[bytes]:
        if self._model is None:
            await self.start()

        buffer = np.array([], dtype=np.float32)
        context = np.array([], dtype=np.float32)
        samples_needed = int(SEAMLESS_SAMPLE_RATE * BUFFER_SECONDS)
        min_samples = int(SEAMLESS_SAMPLE_RATE * MIN_BUFFER_SECONDS)
        context_samples = int(
            SEAMLESS_SAMPLE_RATE * self._audio_context_seconds
        )
        loop = asyncio.get_running_loop()

        logger.info(
            "process() started: sample_rate=%d, buffer_s=%d, need=%d",
            self._sample_rate, BUFFER_SECONDS, samples_needed,
        )

        async for chunk in audio_stream:
            pcm_int16 = np.frombuffer(chunk, dtype=np.int16)
            pcm_float = pcm_int16.astype(np.float32) / 32768.0
            downsampled = _downsample(
                pcm_float, self._sample_rate, SEAMLESS_SAMPLE_RATE
            )
            buffer = np.concatenate([buffer, downsampled])
            logger.debug(
                "chunk %d bytes -> %d ds samples, buf=%d/%d",
                len(chunk), len(downsampled), len(buffer),
                samples_needed,
            )

            if len(buffer) >= samples_needed:
                segment = buffer.copy()
                buffer = np.array([], dtype=np.float32)
                async for out in self._process_segment(
                    segment, context, context_samples, loop
                ):
                    yield out
                context = self._update_context(
                    context, segment, context_samples
                )

        logger.info(
            "stream ended, remaining buf=%d, min=%d",
            len(buffer), min_samples,
        )
        if len(buffer) >= min_samples:
            async for out in self._process_segment(
                buffer, context, context_samples, loop
            ):
                yield out

        await self._es_queue.put(None)

    async def _process_segment(
        self,
        segment: np.ndarray,
        context: np.ndarray,
        context_samples: int,
        loop: asyncio.AbstractEventLoop,
    ) -> AsyncIterator[bytes]:
        if context_samples > 0 and len(context) > 0:
            full_audio = np.concatenate([context, segment])
            es_text = await loop.run_in_executor(
                None,
                _translate_with_context_sync,
                self._processor,
                self._model,
                context,
                full_audio,
            )
        else:
            es_text = await loop.run_in_executor(
                None,
                _translate_audio_sync,
                self._processor,
                self._model,
                segment,
            )

        logger.info("segment translated: %r", es_text[:120] if es_text else "")
        if es_text:
            await self._es_queue.put(es_text)
            pcm_bytes = await _synthesize_spanish(
                es_text, self._sample_rate
            )
            logger.info("TTS produced %d bytes", len(pcm_bytes))
            if pcm_bytes:
                yield pcm_bytes

    @staticmethod
    def _update_context(
        context: np.ndarray,
        segment: np.ndarray,
        context_samples: int,
    ) -> np.ndarray:
        if context_samples <= 0:
            return np.array([], dtype=np.float32)
        combined = np.concatenate([context, segment])
        if len(combined) > context_samples:
            return combined[-context_samples:]
        return combined

    def iter_stream(
        self, name: str, audio_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        if name == "es-transcript":
            return self._drain_queue(self._es_queue)
        return None

    @staticmethod
    async def _drain_queue(
        q: asyncio.Queue[str | None],
    ) -> AsyncIterator[str]:
        while True:
            text = await q.get()
            if text is None:
                return
            yield text
