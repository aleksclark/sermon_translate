from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol

import numpy as np

from src.models import (
    ListenProduct,
    MetadataEnvelope,
    StageInfo,
    StageKind,
    SynthesisInstructions,
    TranslateProduct,
    WordSpan,
)
from src.pipelines.stage_registry import StageFactory
from src.pipelines.stages import BaselineProsodyStage

SILENCE_RMS = 0.01
SPEAK_CHUNK_SECONDS = 0.05
DEFAULT_WORD_MS = 200.0


class StageRegistryLike(Protocol):
    def register(self, stage_factory: StageFactory) -> None: ...


def _split_words(text: str, *, start_ms: float = 0.0) -> list[WordSpan]:
    tokens = [part for part in text.split() if part]
    if not tokens:
        return []
    words: list[WordSpan] = []
    cursor = start_ms
    for token in tokens:
        end_ms = cursor + DEFAULT_WORD_MS
        words.append(WordSpan(text=token, start_ms=cursor, end_ms=end_ms, conf=1.0))
        cursor = end_ms
    return words


def _map_target_words(source_words: list[WordSpan], target_text: str) -> list[WordSpan]:
    target_tokens = [part for part in target_text.split() if part]
    if not target_tokens:
        return []
    mapped: list[WordSpan] = []
    for index, token in enumerate(target_tokens):
        source = source_words[index] if index < len(source_words) else None
        prosody = None
        if source is not None and source.prosody is not None:
            prosody = source.prosody.model_copy()
        mapped.append(
            WordSpan(
                text=token,
                start_ms=source.start_ms if source is not None else None,
                end_ms=source.end_ms if source is not None else None,
                conf=source.conf if source is not None else 1.0,
                prosody=prosody,
            )
        )
    return mapped


def _instructions_from_words(words: list[WordSpan]) -> SynthesisInstructions:
    markers: list[dict[str, object]] = []
    for word in words:
        marker: dict[str, object] = {"word": word.text}
        if word.start_ms is not None:
            marker["start_ms"] = word.start_ms
        if word.end_ms is not None:
            marker["end_ms"] = word.end_ms
        if word.prosody is not None:
            marker["prosody"] = word.prosody.model_dump(exclude_none=True)
        markers.append(marker)
    return SynthesisInstructions(markers=markers)


async def _drain_prosody(prosody: AsyncIterator[MetadataEnvelope]) -> None:
    async for _ in prosody:
        pass



class PassthroughListenStage:
    def __init__(self, *, sample_rate: int = 48000, silence_rms: float = SILENCE_RMS) -> None:
        self._sample_rate = sample_rate
        self._silence_rms = silence_rms
        self.info = StageInfo(
            id="passthrough-listen",
            kind=StageKind.LISTEN,
            name="Passthrough Listen",
            description="Emits placeholder transcript markers for non-silent audio.",
            default_for_kind=True,
        )

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def transcribe(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[ListenProduct]:
        index = 0
        cursor_ms = 0.0
        async for chunk in audio_stream:
            if not chunk:
                continue
            pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            energy = float(np.sqrt(np.mean(np.square(pcm)))) if pcm.size else 0.0
            duration_ms = (pcm.size / self._sample_rate) * 1000.0 if self._sample_rate else 0.0
            if energy < self._silence_rms:
                cursor_ms += duration_ms
                continue
            index += 1
            text = f"speech-{index}"
            words = _split_words(text, start_ms=cursor_ms)
            yield ListenProduct(
                sequence=index - 1,
                utterance_id=f"utt-{index}",
                text=text,
                is_final=True,
                words=words,
                language="en",
            )
            cursor_ms += duration_ms
        if index == 0:
            yield ListenProduct(
                sequence=0,
                utterance_id="utt-silence",
                text="silence",
                is_final=True,
                words=_split_words("silence"),
                language="en",
            )


class PassthroughTranslateStage:
    def __init__(self, *, sample_rate: int = 48000) -> None:
        self._sample_rate = sample_rate
        self.info = StageInfo(
            id="passthrough-translate",
            kind=StageKind.TRANSLATE,
            name="Passthrough Translate",
            description="Identity translation stub for composition tests.",
            default_for_kind=True,
        )

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def translate(
        self,
        text_stream: AsyncIterator[ListenProduct],
        *,
        prosody: AsyncIterator[MetadataEnvelope] | None = None,
    ) -> AsyncIterator[TranslateProduct]:
        drain_task: asyncio.Task[None] | None = None
        if prosody is not None:
            drain_task = asyncio.create_task(_drain_prosody(prosody))
        try:
            async for product in text_stream:
                words = _map_target_words(product.words, product.text)
                yield TranslateProduct(
                    sequence=product.sequence,
                    source_utterance_id=product.utterance_id,
                    target_utterance_id=f"tgt-{product.utterance_id}",
                    text=product.text,
                    is_final=product.is_final,
                    words=words,
                    instructions=_instructions_from_words(words),
                )
        finally:
            if drain_task is not None:
                await drain_task



class PassthroughSpeakStage:
    def __init__(self, *, sample_rate: int = 48000) -> None:
        self._sample_rate = sample_rate
        self.info = StageInfo(
            id="passthrough-speak",
            kind=StageKind.SPEAK,
            name="Passthrough Speak",
            description="Emits short silence for each text chunk.",
            default_for_kind=True,
        )

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def synthesize(
        self, text_stream: AsyncIterator[TranslateProduct]
    ) -> AsyncIterator[bytes]:
        samples = max(1, int(self._sample_rate * SPEAK_CHUNK_SECONDS))
        silence = np.zeros(samples, dtype=np.int16).tobytes()
        async for product in text_stream:
            if product.text:
                yield silence


class _Factory:
    def __init__(self, info: StageInfo, builder: type) -> None:
        self._info = info
        self._builder = builder

    @property
    def info(self) -> StageInfo:
        return self._info

    def create(self, *, sample_rate: int = 48000) -> object:
        return self._builder(sample_rate=sample_rate)


class BaselineProsodyFactory:
    @property
    def info(self) -> StageInfo:
        return StageInfo(
            id="baseline-prosody",
            kind=StageKind.PROSODY,
            name="Baseline Prosody",
            description="YIN F0 + energy + pause detector.",
            default_for_kind=True,
        )

    def create(self, *, sample_rate: int = 48000) -> BaselineProsodyStage:
        return BaselineProsodyStage(sample_rate=sample_rate)


def register_stub_stages(registry: StageRegistryLike) -> None:
    registry.register(
        _Factory(
            StageInfo(
                id="passthrough-listen",
                kind=StageKind.LISTEN,
                name="Passthrough Listen",
                description="Emits placeholder transcript markers for non-silent audio.",
                default_for_kind=True,
            ),
            PassthroughListenStage,
        )
    )
    registry.register(
        _Factory(
            StageInfo(
                id="passthrough-translate",
                kind=StageKind.TRANSLATE,
                name="Passthrough Translate",
                description="Identity translation stub for composition tests.",
                default_for_kind=True,
            ),
            PassthroughTranslateStage,
        )
    )
    registry.register(
        _Factory(
            StageInfo(
                id="passthrough-speak",
                kind=StageKind.SPEAK,
                name="Passthrough Speak",
                description="Emits short silence for each text chunk.",
                default_for_kind=True,
            ),
            PassthroughSpeakStage,
        )
    )
    registry.register(BaselineProsodyFactory())
