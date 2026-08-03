"""Optional Kyutai STT-1B listen backend.

Importing this module succeeds only when the kyutai extra is installed.
The factory is registered by stages_listen.register_listen_stages.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from src.models import ListenProduct, StageInfo, StageKind


class KyutaiListenStage:
    def __init__(self, *, sample_rate: int = 48000, cache: Any = None, **_: object) -> None:
        import kyutai  # type: ignore[import-not-found]  # noqa: F401

        self._sample_rate = sample_rate
        self._cache = cache
        self.info = StageInfo(
            id="kyutai-stt-1b",
            kind=StageKind.LISTEN,
            name="Kyutai STT-1B",
            description="Kyutai STT-1B true streaming ASR (optional extra).",
            requires_gpu=True,
            default_for_kind=False,
        )

    async def start(self) -> None:
        raise RuntimeError("Kyutai STT backend wiring is pending package API integration")

    async def stop(self) -> None: ...

    async def transcribe(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[ListenProduct]:
        if False:
            yield ListenProduct(sequence=0, utterance_id="x", text="")
        raise RuntimeError("Kyutai STT backend wiring is pending package API integration")


class KyutaiListenFactory:
    @property
    def info(self) -> StageInfo:
        return StageInfo(
            id="kyutai-stt-1b",
            kind=StageKind.LISTEN,
            name="Kyutai STT-1B",
            description="Kyutai STT-1B true streaming ASR (optional extra).",
            requires_gpu=True,
            default_for_kind=False,
        )

    def create(self, **kwargs: Any) -> KyutaiListenStage:
        return KyutaiListenStage(**kwargs)
