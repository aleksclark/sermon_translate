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
    """WebSocket-backed stage handle used by subprocess and remote runtimes.

    Uses a full-duplex pump: concurrent sender and receiver tasks with bounded
    queues so products can arrive before source/request EOS (stage.v1 Wave 1).
    Legacy worker JSON protocol is retained for existing subprocess workers;
    stage.v1 binary path lives in ``src.stage_v1.client.StageV1Client``.
    """

    def __init__(
        self,
        *,
        info: StageInfo,
        url: str,
        session: Session,
        start_timeout: float = 60.0,
        outbound_queue_size: int = 64,
        inbound_queue_size: int = 64,
    ) -> None:
        self.info = info
        self._url = url
        self._session = session
        self._start_timeout = start_timeout
        self._ws: ClientConnection | None = None
        self._started = False
        self._outbound_queue_size = max(1, outbound_queue_size)
        self._inbound_queue_size = max(1, inbound_queue_size)
        self._outbound: asyncio.Queue[WorkerMessage | None] | None = None
        self._inbound: asyncio.Queue[WorkerMessage | None] | None = None
        self._sender_task: asyncio.Task[None] | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._pump_error: BaseException | None = None
        self._outbound_high_water = 0
        self._inbound_high_water = 0

    @property
    def stage(self) -> RemoteStageHandle:
        return self

    @property
    def sender_task(self) -> asyncio.Task[None] | None:
        return self._sender_task

    @property
    def receiver_task(self) -> asyncio.Task[None] | None:
        return self._receiver_task

    @property
    def outbound_high_water(self) -> int:
        return self._outbound_high_water

    @property
    def inbound_high_water(self) -> int:
        return self._inbound_high_water

    def duplex_tasks_live(self) -> bool:
        return (
            self._sender_task is not None
            and self._receiver_task is not None
            and not self._sender_task.done()
            and not self._receiver_task.done()
        )

    async def start(self) -> None:
        if self._started:
            return
        self._ws = await connect(self._url, open_timeout=self._start_timeout)
        self._outbound = asyncio.Queue(maxsize=self._outbound_queue_size)
        self._inbound = asyncio.Queue(maxsize=self._inbound_queue_size)
        self._pump_error = None
        self._sender_task = asyncio.create_task(self._sender_loop(), name="remote-stage-sender")
        self._receiver_task = asyncio.create_task(
            self._receiver_loop(), name="remote-stage-receiver"
        )
        await self._send_q(
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
        await self._send_q(WorkerMessage(type=WorkerMessageType.START))
        ready = await asyncio.wait_for(self._recv_q(), timeout=self._start_timeout)
        if ready.type == WorkerMessageType.ERROR:
            raise RuntimeError(ready.message or "stage worker error")
        if ready.type != WorkerMessageType.READY:
            raise RuntimeError(f"expected ready, got {ready.type}")
        self._started = True

    async def stop(self) -> None:
        try:
            if self._started and self._ws is not None:
                with contextlib.suppress(Exception):
                    await self._send_q(WorkerMessage(type=WorkerMessageType.STOP))
        finally:
            await self._shutdown_pump()
            if self._ws is not None:
                with contextlib.suppress(Exception):
                    await self._ws.close()
            self._ws = None
            self._started = False

    async def transcribe(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[ListenProduct]:
        self._require_kind(StageKind.LISTEN)
        self._require_started()

        async def _send_audio() -> None:
            seq = 0
            async for chunk in audio_stream:
                await self._send_q(
                    WorkerMessage(
                        type=WorkerMessageType.AUDIO_IN,
                        seq=seq,
                        sample_rate=self._session.sample_rate,
                        channels=self._session.channels,
                        pcm_b64=pcm_to_b64(chunk),
                    )
                )
                seq += 1
            await self._send_q(WorkerMessage(type=WorkerMessageType.EOS))

        send_task = asyncio.create_task(_send_audio(), name="remote-listen-send")
        try:
            async for message in self._iter_until_eos():
                if message.type == WorkerMessageType.LISTEN_PRODUCT and message.product is not None:
                    yield ListenProduct.model_validate(message.product)
                elif message.type == WorkerMessageType.ERROR:
                    raise RuntimeError(message.message or "listen worker error")
        finally:
            await self._stop_task(send_task)

    async def translate(
        self,
        text_stream: AsyncIterator[ListenProduct],
        *,
        prosody: AsyncIterator[MetadataEnvelope] | None = None,
    ) -> AsyncIterator[TranslateProduct]:
        self._require_kind(StageKind.TRANSLATE)
        self._require_started()
        if prosody is not None:

            async def _drain() -> None:
                async for _ in prosody:
                    pass

            drain_task: asyncio.Task[None] | None = asyncio.create_task(_drain())
        else:
            drain_task = None

        async def _send_text() -> None:
            seq = 0
            async for product in text_stream:
                await self._send_q(
                    WorkerMessage(
                        type=WorkerMessageType.LISTEN_PRODUCT,
                        seq=seq,
                        product=product.model_dump(mode="json"),
                    )
                )
                seq += 1
            await self._send_q(WorkerMessage(type=WorkerMessageType.EOS))

        send_task = asyncio.create_task(_send_text(), name="remote-translate-send")
        try:
            async for message in self._iter_until_eos():
                if (
                    message.type == WorkerMessageType.TRANSLATE_PRODUCT
                    and message.product is not None
                ):
                    yield TranslateProduct.model_validate(message.product)
                elif message.type == WorkerMessageType.ERROR:
                    raise RuntimeError(message.message or "translate worker error")
        finally:
            await self._stop_task(send_task)
            if drain_task is not None:
                await drain_task

    async def synthesize(
        self, text_stream: AsyncIterator[TranslateProduct]
    ) -> AsyncIterator[bytes]:
        self._require_kind(StageKind.SPEAK)
        self._require_started()

        async def _send_text() -> None:
            seq = 0
            async for product in text_stream:
                await self._send_q(
                    WorkerMessage(
                        type=WorkerMessageType.TRANSLATE_PRODUCT,
                        seq=seq,
                        product=product.model_dump(mode="json"),
                    )
                )
                seq += 1
            await self._send_q(WorkerMessage(type=WorkerMessageType.EOS))

        send_task = asyncio.create_task(_send_text(), name="remote-speak-send")
        try:
            async for message in self._iter_until_eos():
                if message.type == WorkerMessageType.AUDIO_OUT and message.pcm_b64:
                    yield b64_to_pcm(message.pcm_b64)
                elif message.type == WorkerMessageType.ERROR:
                    raise RuntimeError(message.message or "speak worker error")
        finally:
            await self._stop_task(send_task)

    async def analyze(
        self, audio_stream: AsyncIterator[bytes], stream_name: str
    ) -> AsyncIterator[MetadataEnvelope]:
        self._require_kind(StageKind.PROSODY)
        self._require_started()

        async def _send_audio() -> None:
            seq = 0
            async for chunk in audio_stream:
                await self._send_q(
                    WorkerMessage(
                        type=WorkerMessageType.AUDIO_IN,
                        seq=seq,
                        sample_rate=self._session.sample_rate,
                        channels=self._session.channels,
                        pcm_b64=pcm_to_b64(chunk),
                        config={"stream_name": stream_name},
                    )
                )
                seq += 1
            await self._send_q(WorkerMessage(type=WorkerMessageType.EOS))

        send_task = asyncio.create_task(_send_audio(), name="remote-prosody-send")
        try:
            async for message in self._iter_until_eos():
                if message.type == WorkerMessageType.METADATA and message.envelope is not None:
                    yield MetadataEnvelope.model_validate(message.envelope)
                elif message.type == WorkerMessageType.ERROR:
                    raise RuntimeError(message.message or "prosody worker error")
        finally:
            await self._stop_task(send_task)

    async def _sender_loop(self) -> None:
        assert self._outbound is not None
        try:
            while True:
                message = await self._outbound.get()
                if message is None:
                    return
                if self._ws is None:
                    raise RuntimeError("stage handle is not started")
                await self._ws.send(message.encode())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_pump(exc)

    async def _receiver_loop(self) -> None:
        assert self._inbound is not None
        try:
            while True:
                if self._ws is None:
                    raise RuntimeError("stage handle is not started")
                raw = await self._ws.recv()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                message = WorkerMessage.decode(raw)
                await self._inbound.put(message)
                self._inbound_high_water = max(self._inbound_high_water, self._inbound.qsize())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_pump(exc)
        finally:
            if self._inbound is not None:
                with contextlib.suppress(asyncio.QueueFull):
                    self._inbound.put_nowait(None)

    async def _send_q(self, message: WorkerMessage) -> None:
        self._raise_if_failed()
        if self._outbound is None:
            raise RuntimeError("stage handle is not started")
        await self._outbound.put(message)
        self._outbound_high_water = max(self._outbound_high_water, self._outbound.qsize())

    async def _recv_q(self) -> WorkerMessage:
        self._raise_if_failed()
        if self._inbound is None:
            raise RuntimeError("stage handle is not started")
        message = await self._inbound.get()
        if message is None:
            self._raise_if_failed()
            raise RuntimeError("stage connection closed")
        return message

    async def _iter_until_eos(self) -> AsyncIterator[WorkerMessage]:
        while True:
            message = await self._recv_q()
            if message.type == WorkerMessageType.EOS:
                return
            yield message

    def _fail_pump(self, exc: BaseException) -> None:
        if self._pump_error is None:
            self._pump_error = exc
        if self._inbound is not None:
            with contextlib.suppress(asyncio.QueueFull):
                self._inbound.put_nowait(None)

    def _raise_if_failed(self) -> None:
        if self._pump_error is not None:
            raise RuntimeError(str(self._pump_error)) from self._pump_error

    def _require_started(self) -> None:
        if not self._started or self._ws is None:
            raise RuntimeError("stage handle is not started")
        self._raise_if_failed()

    def _require_kind(self, kind: StageKind) -> None:
        if self.info.kind != kind:
            raise ValueError(f"handle kind is {self.info.kind.value}, expected {kind.value}")

    async def _stop_task(self, task: asyncio.Task[Any]) -> None:
        if task.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=self._start_timeout)
        except TimeoutError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        except Exception:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _shutdown_pump(self) -> None:
        if self._outbound is not None:
            with contextlib.suppress(asyncio.QueueFull):
                self._outbound.put_nowait(None)
        for task in (self._sender_task, self._receiver_task):
            if task is not None and not task.done():
                task.cancel()
        tasks = [t for t in (self._sender_task, self._receiver_task) if t is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._sender_task = None
        self._receiver_task = None
        self._outbound = None
        self._inbound = None
