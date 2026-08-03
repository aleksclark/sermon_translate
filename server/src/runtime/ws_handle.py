from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from typing import Any

from websockets.asyncio.client import ClientConnection, connect

from src.models import (
    ListenProduct,
    MetadataEnvelope,
    Session,
    StageInfo,
    StageKind,
    TranslateProduct,
)
from src.runtime.protocol import (
    WorkerMessage,
    WorkerMessageType,
    b64_to_pcm,
    pcm_to_b64,
)


class RemoteStageHandle:
    """WebSocket-backed stage handle used by subprocess and remote runtimes."""

    def __init__(
        self,
        *,
        info: StageInfo,
        url: str,
        session: Session,
        start_timeout: float = 60.0,
    ) -> None:
        self.info = info
        self._url = url
        self._session = session
        self._start_timeout = start_timeout
        self._ws: ClientConnection | None = None
        self._started = False

    @property
    def stage(self) -> RemoteStageHandle:
        return self

    async def start(self) -> None:
        if self._started:
            return
        self._ws = await connect(self._url, open_timeout=self._start_timeout)
        await self._send(
            WorkerMessage(
                type=WorkerMessageType.HELLO,
                stage_id=self.info.id,
                session_id=self._session.id,
                config={
                    "sample_rate": self._session.sample_rate,
                    "channels": self._session.channels,
                },
            )
        )
        await self._send(WorkerMessage(type=WorkerMessageType.START))
        ready = await asyncio.wait_for(self._recv(), timeout=self._start_timeout)
        if ready.type == WorkerMessageType.ERROR:
            raise RuntimeError(ready.message or "stage worker error")
        if ready.type != WorkerMessageType.READY:
            raise RuntimeError(f"expected ready, got {ready.type}")
        self._started = True

    async def stop(self) -> None:
        if self._ws is None:
            return
        try:
            if self._started:
                with contextlib.suppress(Exception):
                    await self._send(WorkerMessage(type=WorkerMessageType.STOP))
        finally:
            with contextlib.suppress(Exception):
                await self._ws.close()
            self._ws = None
            self._started = False

    async def transcribe(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[ListenProduct]:
        self._require_kind(StageKind.LISTEN)
        await self._pump_audio(audio_stream)
        async for message in self._iter_until_eos():
            if message.type == WorkerMessageType.LISTEN_PRODUCT and message.product is not None:
                yield ListenProduct.model_validate(message.product)
            elif message.type == WorkerMessageType.ERROR:
                raise RuntimeError(message.message or "listen worker error")

    async def translate(
        self,
        text_stream: AsyncIterator[ListenProduct],
        *,
        prosody: AsyncIterator[MetadataEnvelope] | None = None,
    ) -> AsyncIterator[TranslateProduct]:
        self._require_kind(StageKind.TRANSLATE)
        if prosody is not None:
            async def _drain() -> None:
                async for _ in prosody:
                    pass

            drain_task = asyncio.create_task(_drain())
        else:
            drain_task = None
        try:
            seq = 0
            async for product in text_stream:
                await self._send(
                    WorkerMessage(
                        type=WorkerMessageType.LISTEN_PRODUCT,
                        seq=seq,
                        product=product.model_dump(mode="json"),
                    )
                )
                seq += 1
            await self._send(WorkerMessage(type=WorkerMessageType.EOS))
            async for message in self._iter_until_eos():
                if (
                    message.type == WorkerMessageType.TRANSLATE_PRODUCT
                    and message.product is not None
                ):
                    yield TranslateProduct.model_validate(message.product)
                elif message.type == WorkerMessageType.ERROR:
                    raise RuntimeError(message.message or "translate worker error")
        finally:
            if drain_task is not None:
                await drain_task

    async def synthesize(
        self, text_stream: AsyncIterator[TranslateProduct]
    ) -> AsyncIterator[bytes]:
        self._require_kind(StageKind.SPEAK)
        seq = 0
        async for product in text_stream:
            await self._send(
                WorkerMessage(
                    type=WorkerMessageType.TRANSLATE_PRODUCT,
                    seq=seq,
                    product=product.model_dump(mode="json"),
                )
            )
            seq += 1
        await self._send(WorkerMessage(type=WorkerMessageType.EOS))
        async for message in self._iter_until_eos():
            if message.type == WorkerMessageType.AUDIO_OUT and message.pcm_b64:
                yield b64_to_pcm(message.pcm_b64)
            elif message.type == WorkerMessageType.ERROR:
                raise RuntimeError(message.message or "speak worker error")

    async def analyze(
        self, audio_stream: AsyncIterator[bytes], stream_name: str
    ) -> AsyncIterator[MetadataEnvelope]:
        self._require_kind(StageKind.PROSODY)
        await self._pump_audio(audio_stream, extra={"stream_name": stream_name})
        async for message in self._iter_until_eos():
            if message.type == WorkerMessageType.METADATA and message.envelope is not None:
                yield MetadataEnvelope.model_validate(message.envelope)
            elif message.type == WorkerMessageType.ERROR:
                raise RuntimeError(message.message or "prosody worker error")

    async def _pump_audio(
        self,
        audio_stream: AsyncIterator[bytes],
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        seq = 0
        async for chunk in audio_stream:
            msg = WorkerMessage(
                type=WorkerMessageType.AUDIO_IN,
                seq=seq,
                sample_rate=self._session.sample_rate,
                channels=self._session.channels,
                pcm_b64=pcm_to_b64(chunk),
                config=extra or {},
            )
            await self._send(msg)
            seq += 1
        await self._send(WorkerMessage(type=WorkerMessageType.EOS))

    async def _iter_until_eos(self) -> AsyncIterator[WorkerMessage]:
        while True:
            message = await self._recv()
            if message.type == WorkerMessageType.EOS:
                return
            yield message

    async def _send(self, message: WorkerMessage) -> None:
        if self._ws is None:
            raise RuntimeError("stage handle is not started")
        await self._ws.send(message.encode())

    async def _recv(self) -> WorkerMessage:
        if self._ws is None:
            raise RuntimeError("stage handle is not started")
        raw = await self._ws.recv()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return WorkerMessage.decode(raw)

    def _require_kind(self, kind: StageKind) -> None:
        if self.info.kind != kind:
            raise ValueError(f"handle kind is {self.info.kind.value}, expected {kind.value}")
