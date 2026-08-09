from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator, Callable
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.config import get_settings
from src.models import ListenProduct, StageKind, TranslateProduct
from src.pipelines.stage_registry import create_default_stage_registry
from src.pipelines.stages import ASRStage, ProsodyStage, TranslationStage, TTSStage
from src.runtime.model_cache import ModelCache
from src.runtime.nvidia_libs import ensure_nvidia_library_path
from src.runtime.protocol import (
    WorkerMessage,
    WorkerMessageType,
    b64_to_pcm,
    pcm_to_b64,
)
from src.stage_v1.health import mount_health_routes
from src.stage_v1.host import StageHost, StageHostError
from src.stage_v1.models import (
    ArtifactDigestStatus,
    LimitsAdvertised,
    StageErrorCode,
)
from src.stage_v1.models import (
    StageKind as V1StageKind,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_SESSIONS = int(os.environ.get("STAGE_MAX_SESSIONS", "1"))


def create_worker_app(
    stage_id: str,
    *,
    max_sessions: int | None = None,
    host: StageHost | None = None,
) -> FastAPI:
    ensure_nvidia_library_path()
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

    capacity = max_sessions if max_sessions is not None else DEFAULT_MAX_SESSIONS
    if capacity < 1:
        capacity = 1

    app = FastAPI(title=f"stage-worker:{stage_id}", version="0.1.0")

    # Truthful admission counter — no global lock that queues sessions.
    admission = _AdmissionController(max_sessions=capacity)

    stage_host = host
    if stage_host is None:
        # Optional warm host for stage.v1 health; legacy path still creates
        # per-session stages until adapters move model load fully into the host.
        stage_host = StageHost(
            stage_kind=_map_stage_kind(factory.info.kind),
            stage_id=factory.info.id,
            stage_version="0.1.0",
            model_loader=lambda: {"stage_id": factory.info.id},
            max_sessions=capacity,
            limits=LimitsAdvertised(max_sessions=capacity),
            model_provider_id="legacy-worker",
            model_revision="legacy",
            model_artifact_digest="provider_managed:legacy:legacy",
            model_artifact_status=ArtifactDigestStatus.PROVIDER_MANAGED,
            local_dev=True,
        )

    app.state.stage_host = stage_host
    app.state.admission = admission
    app.state.factory = factory
    app.state.cache = cache

    @app.on_event("startup")
    async def _startup() -> None:
        host_obj: StageHost = app.state.stage_host
        if not host_obj.model_loaded:
            with contextlib.suppress(StageHostError):
                await host_obj.load()
                await host_obj.warmup()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        host_obj: StageHost = app.state.stage_host
        await host_obj.shutdown()

    mount_health_routes(app, stage_host)

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        admitted = await admission.try_acquire()
        if not admitted:
            with contextlib.suppress(Exception):
                await ws.send_text(
                    WorkerMessage(
                        type=WorkerMessageType.ERROR,
                        message=f"{StageErrorCode.RESOURCE_EXHAUSTED}: no admission capacity",
                        config={
                            "code": StageErrorCode.RESOURCE_EXHAUSTED.value,
                            "retryable": True,
                        },
                    ).encode()
                )
            with contextlib.suppress(Exception):
                await ws.close(code=1013)
            return

        # Mirror admission into StageHost when available (capacity-aligned).
        session_state_id: str | None = None
        host_obj: StageHost = app.state.stage_host
        try:
            if host_obj.model_warm and not host_obj.draining:
                try:
                    state = await host_obj.open_session()
                    session_state_id = state.session_state_id
                except StageHostError as exc:
                    if exc.payload.code == StageErrorCode.RESOURCE_EXHAUSTED:
                        with contextlib.suppress(Exception):
                            await ws.send_text(
                                WorkerMessage(
                                    type=WorkerMessageType.ERROR,
                                    message=(
                                        f"{StageErrorCode.RESOURCE_EXHAUSTED}: "
                                        f"{exc.payload.message}"
                                    ),
                                    config={
                                        "code": StageErrorCode.RESOURCE_EXHAUSTED.value,
                                        "retryable": True,
                                    },
                                ).encode()
                            )
                        with contextlib.suppress(Exception):
                            await ws.close(code=1013)
                        return
                    # Host not ready for session tracking — continue legacy path.
                    logger.debug("stage host open_session skipped: %s", exc.payload.message)

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
        finally:
            if session_state_id is not None:
                with contextlib.suppress(Exception):
                    await host_obj.close_session(session_state_id)
            await admission.release()

    return app


class _AdmissionController:
    """Immediate capacity gate — never waits on a lock for admission."""

    def __init__(self, *, max_sessions: int) -> None:
        self.max_sessions = max_sessions
        self._active = 0
        self._lock = asyncio.Lock()

    @property
    def active(self) -> int:
        return self._active

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._active >= self.max_sessions:
                return False
            self._active += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            if self._active > 0:
                self._active -= 1


def _map_stage_kind(kind: StageKind) -> V1StageKind:
    mapping = {
        StageKind.LISTEN: V1StageKind.LISTEN,
        StageKind.TRANSLATE: V1StageKind.TRANSLATE,
        StageKind.SPEAK: V1StageKind.SPEAK,
        StageKind.PROSODY: V1StageKind.PROSODY,
    }
    return mapping.get(kind, V1StageKind.LISTEN)


async def _recv(ws: WebSocket) -> WorkerMessage:
    raw = await ws.receive_text()
    return WorkerMessage.decode(raw)


async def _send(ws: WebSocket, message: WorkerMessage) -> None:
    await ws.send_text(message.encode())


async def _duplex_product_stream(
    ws: WebSocket,
    *,
    accept: Callable[[WorkerMessage], Any | None],
    run: Callable[[AsyncIterator[Any]], AsyncIterator[Any]],
    encode_product: Callable[[int, Any], WorkerMessage],
) -> None:
    """Concurrently consume input and emit products (true duplex).

    A background feeder task fills a bounded queue while the stage iterator
    yields products that are sent immediately — never wait for source EOS first.
    """
    queue: asyncio.Queue[Any | None] = asyncio.Queue(maxsize=32)
    stop = asyncio.Event()
    feed_error: list[BaseException] = []

    async def _feeder() -> None:
        try:
            while not stop.is_set():
                message = await _recv(ws)
                if message.type in {WorkerMessageType.EOS, WorkerMessageType.STOP}:
                    await queue.put(None)
                    return
                item = accept(message)
                if item is not None:
                    await queue.put(item)
        except WebSocketDisconnect:
            await queue.put(None)
        except Exception as exc:
            feed_error.append(exc)
            await queue.put(None)

    async def _inputs() -> AsyncIterator[Any]:
        while True:
            item = await queue.get()
            if item is None:
                return
            yield item

    feeder = asyncio.create_task(_feeder())
    seq = 0
    try:
        async for product in run(_inputs()):
            await _send(ws, encode_product(seq, product))
            seq += 1
        if feed_error:
            raise feed_error[0]
        await _send(ws, WorkerMessage(type=WorkerMessageType.EOS))
    finally:
        stop.set()
        if not feeder.done():
            feeder.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await feeder


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

            def accept_audio(message: WorkerMessage) -> bytes | None:
                if message.type == WorkerMessageType.AUDIO_IN and message.pcm_b64:
                    return b64_to_pcm(message.pcm_b64)
                return None

            async def run_listen(inputs: AsyncIterator[bytes]) -> AsyncIterator[ListenProduct]:
                async for product in stage.transcribe(inputs):
                    yield product

            def encode_listen(seq: int, product: ListenProduct) -> WorkerMessage:
                return WorkerMessage(
                    type=WorkerMessageType.LISTEN_PRODUCT,
                    seq=seq,
                    product=product.model_dump(mode="json"),
                )

            await _duplex_product_stream(
                ws,
                accept=accept_audio,
                run=run_listen,
                encode_product=encode_listen,
            )
        elif kind == StageKind.TRANSLATE:
            assert isinstance(stage, TranslationStage)

            def accept_listen(message: WorkerMessage) -> ListenProduct | None:
                if (
                    message.type == WorkerMessageType.LISTEN_PRODUCT
                    and message.product is not None
                ):
                    return ListenProduct.model_validate(message.product)
                return None

            async def run_translate(
                inputs: AsyncIterator[ListenProduct],
            ) -> AsyncIterator[TranslateProduct]:
                async for product in stage.translate(inputs):
                    yield product

            def encode_translate(seq: int, product: TranslateProduct) -> WorkerMessage:
                return WorkerMessage(
                    type=WorkerMessageType.TRANSLATE_PRODUCT,
                    seq=seq,
                    product=product.model_dump(mode="json"),
                )

            await _duplex_product_stream(
                ws,
                accept=accept_listen,
                run=run_translate,
                encode_product=encode_translate,
            )
        elif kind == StageKind.SPEAK:
            assert isinstance(stage, TTSStage)

            def accept_translate(message: WorkerMessage) -> TranslateProduct | None:
                if (
                    message.type == WorkerMessageType.TRANSLATE_PRODUCT
                    and message.product is not None
                ):
                    return TranslateProduct.model_validate(message.product)
                return None

            async def run_speak(inputs: AsyncIterator[TranslateProduct]) -> AsyncIterator[bytes]:
                async for chunk in stage.synthesize(inputs):
                    yield chunk

            def encode_audio(seq: int, chunk: bytes) -> WorkerMessage:
                return WorkerMessage(
                    type=WorkerMessageType.AUDIO_OUT,
                    seq=seq,
                    pcm_b64=pcm_to_b64(chunk),
                    sample_rate=sample_rate,
                )

            await _duplex_product_stream(
                ws,
                accept=accept_translate,
                run=run_speak,
                encode_product=encode_audio,
            )
        elif kind == StageKind.PROSODY:
            assert isinstance(stage, ProsodyStage)
            stream_name = str(hello.config.get("stream_name", "prosody"))

            def accept_audio_p(message: WorkerMessage) -> bytes | None:
                if message.type == WorkerMessageType.AUDIO_IN and message.pcm_b64:
                    return b64_to_pcm(message.pcm_b64)
                return None

            async def run_prosody(inputs: AsyncIterator[bytes]) -> AsyncIterator[Any]:
                async for envelope in stage.analyze(inputs, stream_name):
                    yield envelope

            def encode_meta(seq: int, envelope: Any) -> WorkerMessage:
                return WorkerMessage(
                    type=WorkerMessageType.METADATA,
                    seq=seq,
                    envelope=envelope.model_dump(mode="json"),
                )

            await _duplex_product_stream(
                ws,
                accept=accept_audio_p,
                run=run_prosody,
                encode_product=encode_meta,
            )
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
    ensure_nvidia_library_path()
    parser = argparse.ArgumentParser(description="Sermon translate stage worker")
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Truthful admission capacity (default STAGE_MAX_SESSIONS or 1)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)
    app = create_worker_app(args.stage_id, max_sessions=args.max_sessions)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
