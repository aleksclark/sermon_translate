from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse

from src.config import get_settings
from src.models import ListenProduct, StageKind, TranslateProduct
from src.pipelines.stage_registry import create_default_stage_registry
from src.pipelines.stages import ASRStage, ProsodyStage, TranslationStage, TTSStage
from src.runtime.model_cache import ModelCache
from src.runtime.protocol import (
    WorkerMessage,
    WorkerMessageType,
    b64_to_pcm,
    pcm_to_b64,
)

logger = logging.getLogger(__name__)


def create_worker_app(stage_id: str) -> FastAPI:
    settings = get_settings()
    cache = ModelCache(settings.model_cache_dir)
    try:
        cache.ensure_root()
    except OSError:
        logger.warning("model cache unavailable: %s", cache.root)

    registry = create_default_stage_registry()
    factory = registry.get(stage_id)
    if factory is None:
        raise SystemExit(f"unknown stage id: {stage_id}")

    app = FastAPI(title=f"stage-worker:{stage_id}", version="0.1.0")
    lock = asyncio.Lock()

    @app.get("/healthz")
    async def healthz() -> PlainTextResponse:
        return PlainTextResponse("ok")

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        async with lock:
            try:
                await _serve_session(ws, factory, cache)
            except WebSocketDisconnect:
                logger.info("worker client disconnected stage=%s", stage_id)
            except Exception as exc:
                logger.exception("worker session failed stage=%s", stage_id)
                with contextlib.suppress(Exception):
                    await ws.send_text(
                        WorkerMessage(
                            type=WorkerMessageType.ERROR,
                            message=f"{type(exc).__name__}: {exc}",
                        ).encode()
                    )

    return app


async def _recv(ws: WebSocket) -> WorkerMessage:
    raw = await ws.receive_text()
    return WorkerMessage.decode(raw)


async def _send(ws: WebSocket, message: WorkerMessage) -> None:
    await ws.send_text(message.encode())


async def _audio_from_ws(ws: WebSocket) -> AsyncIterator[bytes]:
    while True:
        message = await _recv(ws)
        if message.type == WorkerMessageType.EOS:
            return
        if message.type == WorkerMessageType.STOP:
            return
        if message.type == WorkerMessageType.AUDIO_IN and message.pcm_b64:
            yield b64_to_pcm(message.pcm_b64)


async def _listen_from_ws(ws: WebSocket) -> AsyncIterator[ListenProduct]:
    while True:
        message = await _recv(ws)
        if message.type == WorkerMessageType.EOS:
            return
        if message.type == WorkerMessageType.STOP:
            return
        if message.type == WorkerMessageType.LISTEN_PRODUCT and message.product is not None:
            yield ListenProduct.model_validate(message.product)


async def _translate_from_ws(ws: WebSocket) -> AsyncIterator[TranslateProduct]:
    while True:
        message = await _recv(ws)
        if message.type == WorkerMessageType.EOS:
            return
        if message.type == WorkerMessageType.STOP:
            return
        if message.type == WorkerMessageType.TRANSLATE_PRODUCT and message.product is not None:
            yield TranslateProduct.model_validate(message.product)


async def _serve_session(ws: WebSocket, factory: Any, cache: ModelCache) -> None:
    hello = await _recv(ws)
    if hello.type != WorkerMessageType.HELLO:
        await _send(
            ws,
            WorkerMessage(type=WorkerMessageType.ERROR, message="expected hello"),
        )
        return
    start = await _recv(ws)
    if start.type != WorkerMessageType.START:
        await _send(
            ws,
            WorkerMessage(type=WorkerMessageType.ERROR, message="expected start"),
        )
        return

    sample_rate = int(hello.config.get("sample_rate", 48000))
    stage = factory.create(sample_rate=sample_rate, cache=cache)
    await stage.start()
    await _send(ws, WorkerMessage(type=WorkerMessageType.READY, stage_id=factory.info.id))

    try:
        kind = factory.info.kind
        if kind == StageKind.LISTEN:
            assert isinstance(stage, ASRStage)
            seq = 0
            async for product in stage.transcribe(_audio_from_ws(ws)):
                await _send(
                    ws,
                    WorkerMessage(
                        type=WorkerMessageType.LISTEN_PRODUCT,
                        seq=seq,
                        product=product.model_dump(mode="json"),
                    ),
                )
                seq += 1
            await _send(ws, WorkerMessage(type=WorkerMessageType.EOS))
        elif kind == StageKind.TRANSLATE:
            assert isinstance(stage, TranslationStage)
            seq = 0
            async for product in stage.translate(_listen_from_ws(ws)):
                await _send(
                    ws,
                    WorkerMessage(
                        type=WorkerMessageType.TRANSLATE_PRODUCT,
                        seq=seq,
                        product=product.model_dump(mode="json"),
                    ),
                )
                seq += 1
            await _send(ws, WorkerMessage(type=WorkerMessageType.EOS))
        elif kind == StageKind.SPEAK:
            assert isinstance(stage, TTSStage)
            seq = 0
            async for chunk in stage.synthesize(_translate_from_ws(ws)):
                await _send(
                    ws,
                    WorkerMessage(
                        type=WorkerMessageType.AUDIO_OUT,
                        seq=seq,
                        pcm_b64=pcm_to_b64(chunk),
                        sample_rate=sample_rate,
                    ),
                )
                seq += 1
            await _send(ws, WorkerMessage(type=WorkerMessageType.EOS))
        elif kind == StageKind.PROSODY:
            assert isinstance(stage, ProsodyStage)
            stream_name = str(hello.config.get("stream_name", "prosody"))
            seq = 0
            async for envelope in stage.analyze(_audio_from_ws(ws), stream_name):
                await _send(
                    ws,
                    WorkerMessage(
                        type=WorkerMessageType.METADATA,
                        seq=seq,
                        envelope=envelope.model_dump(mode="json"),
                    ),
                )
                seq += 1
            await _send(ws, WorkerMessage(type=WorkerMessageType.EOS))
        else:
            await _send(
                ws,
                WorkerMessage(
                    type=WorkerMessageType.ERROR,
                    message=f"unsupported stage kind: {kind}",
                ),
            )
    finally:
        await stage.stop()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Sermon translate stage worker")
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    app = create_worker_app(args.stage_id)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
