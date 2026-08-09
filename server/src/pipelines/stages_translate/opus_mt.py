from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class OpusMTLoadedModel:
    """Immutable resident Opus-MT translator + sentencepiece processors (D6)."""

    translator: Any
    sp_source: Any
    sp_target: Any
    model_id: str
    revision: str = "unknown"


def load_opus_mt_model(
    *,
    model_id: str | None = None,
    cache: Any = None,
) -> OpusMTLoadedModel:
    """Load Helsinki-NLP Opus-MT EN→ES once (sync). Used by StageHost model_loader."""
    mid = model_id or os.environ.get("TRANSLATE_MODEL_ID", TRANSLATION_MODEL_ID)
    if cache is not None:
        os.environ.setdefault("MODEL_CACHE_DIR", str(cache.root))
    helper = SpanishTranslationPipeline()
    translator, sp_source, sp_target = helper._load_translation()
    return OpusMTLoadedModel(
        translator=translator,
        sp_source=sp_source,
        sp_target=sp_target,
        model_id=mid,
        revision=mid,
    )


class OpusMTTranslateStage:
    """Incremental EN→ES translation via Helsinki-NLP Opus-MT.

    Weights may be injected via ``loaded_model`` so ``stop()`` never unloads them.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        model_id: str | None = None,
        cache: Any = None,
        loaded_model: OpusMTLoadedModel | None = None,
        **_: object,
    ) -> None:
        self._sample_rate = sample_rate
        self._model_id = model_id or os.environ.get("TRANSLATE_MODEL_ID", TRANSLATION_MODEL_ID)
        self._cache = cache
        self._loaded: OpusMTLoadedModel | None = loaded_model
        self._owns_model = loaded_model is None
        self._session_active = False
        self.info = StageInfo(
            id="opus-mt-en-es",
            kind=StageKind.TRANSLATE,
            name="Opus-MT EN→ES",
            description="Helsinki-NLP Opus-MT English to Spanish.",
            requires_gpu=False,
            default_for_kind=True,
        )

    @property
    def loaded_model(self) -> OpusMTLoadedModel | None:
        return self._loaded

    # Compatibility attributes used by older tests / call sites.
    @property
    def _translator(self) -> Any | None:
        return None if self._loaded is None else self._loaded.translator

    @_translator.setter
    def _translator(self, value: Any | None) -> None:
        self._set_component("translator", value)

    @property
    def _sp_source(self) -> Any | None:
        return None if self._loaded is None else self._loaded.sp_source

    @_sp_source.setter
    def _sp_source(self, value: Any | None) -> None:
        self._set_component("sp_source", value)

    @property
    def _sp_target(self) -> Any | None:
        return None if self._loaded is None else self._loaded.sp_target

    @_sp_target.setter
    def _sp_target(self, value: Any | None) -> None:
        self._set_component("sp_target", value)

    def _set_component(self, name: str, value: Any | None) -> None:
        if value is None:
            if self._owns_model and name == "translator":
                self._loaded = None
            return
        if self._loaded is None:
            self._loaded = OpusMTLoadedModel(
                translator=value if name == "translator" else object(),
                sp_source=value if name == "sp_source" else object(),
                sp_target=value if name == "sp_target" else object(),
                model_id=self._model_id,
                revision=self._model_id,
            )
            return
        if not self._owns_model:
            return
        data = {
            "translator": self._loaded.translator,
            "sp_source": self._loaded.sp_source,
            "sp_target": self._loaded.sp_target,
            "model_id": self._loaded.model_id,
            "revision": self._loaded.revision,
        }
        data[name] = value
        self._loaded = OpusMTLoadedModel(**data)

    async def start(self) -> None:
        """Ensure weights are available; open per-session runtime."""
        self._session_active = True
        if self._loaded is not None:
            return
        if self._cache is not None:
            os.environ.setdefault("MODEL_CACHE_DIR", str(self._cache.root))
        loop = asyncio.get_running_loop()
        loaded = await loop.run_in_executor(
            None,
            partial(load_opus_mt_model, model_id=self._model_id, cache=self._cache),
        )
        self._loaded = loaded
        self._owns_model = True
        logger.info("opus-mt-en-es model loaded: %s", loaded.model_id)

    async def stop(self) -> None:
        """Clear per-session state. Never unloads preloaded weights."""
        self._session_active = False
        if self._owns_model:
            self._loaded = None

    async def translate(
        self,
        text_stream: AsyncIterator[ListenProduct],
        *,
        prosody: AsyncIterator[MetadataEnvelope] | None = None,
    ) -> AsyncIterator[TranslateProduct]:
        if self._loaded is None:
            await self.start()
        assert self._loaded is not None

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
                        self._loaded.translator,
                        self._loaded.sp_source,
                        self._loaded.sp_target,
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
