from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from src.models import StageInfo, StageKind, TranslateProduct

logger = logging.getLogger(__name__)


class Qwen3TTSClientStage:
    """Client for Qwen3-TTS-12Hz via vLLM-Omni WebSocket endpoint.

    Requires QWEN3_TTS_WS_URL. Without it, start() raises so the stage is only
    selected when an operator has deployed the remote service.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 48000,
        ws_url: str | None = None,
        cache: Any = None,
        **_: object,
    ) -> None:
        self._sample_rate = sample_rate
        self._ws_url = ws_url or os.environ.get("QWEN3_TTS_WS_URL", "").strip()
        self._cache = cache
        self.info = StageInfo(
            id="qwen3-tts-0.6b",
            kind=StageKind.SPEAK,
            name="Qwen3-TTS 0.6B",
            description="Qwen3-TTS-12Hz via vLLM-Omni WebSocket (set QWEN3_TTS_WS_URL).",
            requires_gpu=True,
            default_for_kind=False,
        )

    async def start(self) -> None:
        if not self._ws_url:
            raise RuntimeError("QWEN3_TTS_WS_URL is required for qwen3-tts-0.6b")

    async def stop(self) -> None: ...

    async def synthesize(
        self, text_stream: AsyncIterator[TranslateProduct]
    ) -> AsyncIterator[bytes]:
        from websockets.asyncio.client import connect

        assert self._ws_url
        async with connect(self._ws_url) as ws:
            async for product in text_stream:
                payload = {
                    "text": product.text,
                    "sample_rate": self._sample_rate,
                    "instructions": (
                        product.instructions.model_dump(mode="json")
                        if product.instructions is not None
                        else None
                    ),
                }
                await ws.send(__import__("json").dumps(payload))
                while True:
                    raw = await ws.recv()
                    if isinstance(raw, str):
                        data = __import__("json").loads(raw)
                        if data.get("type") == "eos":
                            break
                        if data.get("type") == "error":
                            raise RuntimeError(data.get("message", "qwen3 tts error"))
                        continue
                    if isinstance(raw, (bytes, bytearray)) and raw:
                        yield bytes(raw)


class Qwen3TTSFactory:
    @property
    def info(self) -> StageInfo:
        return StageInfo(
            id="qwen3-tts-0.6b",
            kind=StageKind.SPEAK,
            name="Qwen3-TTS 0.6B",
            description="Qwen3-TTS-12Hz via vLLM-Omni WebSocket (set QWEN3_TTS_WS_URL).",
            requires_gpu=True,
            default_for_kind=False,
        )

    def create(self, **kwargs: Any) -> Qwen3TTSClientStage:
        return Qwen3TTSClientStage(**kwargs)
