from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from functools import partial
from typing import Any

from src.models import (
    ListenProduct,
    MetadataEnvelope,
    StageInfo,
    StageKind,
    SynthesisInstructions,
    TranslateProduct,
    WordSpan,
)
from src.pipelines.spanish import TRANSLATION_MODEL_ID, SpanishTranslationPipeline, _translate_sync

logger = logging.getLogger(__name__)


class OpusMTTranslateStage:
    """Incremental EN→ES translation via Helsinki-NLP Opus-MT."""

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        model_id: str | None = None,
        cache: Any = None,
        **_: object,
    ) -> None:
        self._sample_rate = sample_rate
        self._model_id = model_id or os.environ.get("TRANSLATE_MODEL_ID", TRANSLATION_MODEL_ID)
        self._cache = cache
        self._translator: Any = None
        self._sp_source: Any = None
        self._sp_target: Any = None
        self.info = StageInfo(
            id="opus-mt-en-es",
            kind=StageKind.TRANSLATE,
            name="Opus-MT EN→ES",
            description="Helsinki-NLP Opus-MT English to Spanish.",
            requires_gpu=False,
            default_for_kind=True,
        )

    async def start(self) -> None:
        if self._translator is not None:
            return
        if self._cache is not None:
            os.environ.setdefault("MODEL_CACHE_DIR", str(self._cache.root))
        loop = asyncio.get_running_loop()
        helper = SpanishTranslationPipeline()
        self._translator, self._sp_source, self._sp_target = await loop.run_in_executor(
            None, helper._load_translation
        )
        logger.info("opus-mt-en-es model loaded: %s", self._model_id)

    async def stop(self) -> None:
        self._translator = None
        self._sp_source = None
        self._sp_target = None

    async def translate(
        self,
        text_stream: AsyncIterator[ListenProduct],
        *,
        prosody: AsyncIterator[MetadataEnvelope] | None = None,
    ) -> AsyncIterator[TranslateProduct]:
        if self._translator is None:
            await self.start()
        assert self._translator is not None
        assert self._sp_source is not None
        assert self._sp_target is not None

        drain_task: asyncio.Task[None] | None = None
        if prosody is not None:

            async def _drain() -> None:
                async for _ in prosody:
                    pass

            drain_task = asyncio.create_task(_drain())

        loop = asyncio.get_running_loop()
        try:
            async for product in text_stream:
                if not product.text.strip():
                    continue
                spanish = await loop.run_in_executor(
                    None,
                    partial(
                        _translate_sync,
                        self._translator,
                        self._sp_source,
                        self._sp_target,
                        product.text,
                    ),
                )
                words = _map_target_words(product.words, spanish)
                yield TranslateProduct(
                    sequence=product.sequence,
                    source_utterance_id=product.utterance_id,
                    target_utterance_id=f"tgt-{product.utterance_id}",
                    text=spanish,
                    is_final=product.is_final,
                    words=words,
                    instructions=_instructions_from_words(words),
                )
        finally:
            if drain_task is not None:
                await drain_task


def _map_target_words(source_words: list[WordSpan], target_text: str) -> list[WordSpan]:
    target_tokens = [part for part in target_text.split() if part]
    if not target_tokens:
        return []
    mapped: list[WordSpan] = []
    for index, token in enumerate(target_tokens):
        source = source_words[index] if index < len(source_words) else None
        prosody = source.prosody.model_copy() if source is not None and source.prosody else None
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


class OpusMTTranslateFactory:
    @property
    def info(self) -> StageInfo:
        return StageInfo(
            id="opus-mt-en-es",
            kind=StageKind.TRANSLATE,
            name="Opus-MT EN→ES",
            description="Helsinki-NLP Opus-MT English to Spanish.",
            requires_gpu=False,
            default_for_kind=True,
        )

    def create(self, **kwargs: Any) -> OpusMTTranslateStage:
        return OpusMTTranslateStage(**kwargs)
