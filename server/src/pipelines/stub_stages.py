from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

import numpy as np

from src.models import StageInfo, StageKind
from src.pipelines.stage_registry import StageFactory
from src.pipelines.stages import BaselineProsodyStage

SILENCE_RMS = 0.01
SPEAK_CHUNK_SECONDS = 0.05


class StageRegistryLike(Protocol):
    def register(self, stage_factory: StageFactory) -> None: ...


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

    async def transcribe(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
        index = 0
        async for chunk in audio_stream:
            if not chunk:
                continue
            pcm = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            energy = float(np.sqrt(np.mean(np.square(pcm)))) if pcm.size else 0.0
            if energy < self._silence_rms:
                continue
            index += 1
            yield f"speech-{index}"
        if index == 0:
            yield "silence"


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

    async def translate(self, text_stream: AsyncIterator[str]) -> AsyncIterator[str]:
        async for text in text_stream:
            yield text


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

    async def synthesize(self, text_stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
        samples = max(1, int(self._sample_rate * SPEAK_CHUNK_SECONDS))
        silence = np.zeros(samples, dtype=np.int16).tobytes()
        async for text in text_stream:
            if text:
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
