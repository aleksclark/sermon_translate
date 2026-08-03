from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import SessionStore, crosstalk_router, init_deps
from src.api import router as api_router
from src.config import get_settings
from src.models import ServerStatsTracker
from src.pipelines import create_default_registry, create_default_stage_registry
from src.runtime.local import LocalStageRuntime
from src.runtime.model_cache import ModelCache
from src.runtime.nvidia_libs import ensure_nvidia_library_path
from src.runtime.remote_runtime import RemoteStageRuntime
from src.runtime.subprocess_runtime import SubprocessStageRuntime
from src.transport.crosstalk_service import CrosstalkService

LOG_FILE = Path(__file__).resolve().parent.parent.parent / "server.log"
logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(LOG_FILE, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)


def create_app() -> FastAPI:
    ensure_nvidia_library_path()
    _configure_logging()
    app = FastAPI(title="Sermon Translate", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    settings = get_settings()
    cache = ModelCache(settings.model_cache_dir)
    try:
        cache.ensure_root()
        logger.info("model cache directory: %s", cache.root)
    except OSError:
        logger.warning("model cache directory unavailable: %s", cache.root)

    store = SessionStore()
    stage_registry = create_default_stage_registry()
    if settings.stage_runtime == "subprocess":
        runtime = SubprocessStageRuntime(
            stage_registry,
            cache,
            python=settings.stage_worker_python or None,
            start_timeout=settings.stage_worker_start_timeout,
        )
        logger.info("stage runtime: subprocess")
    elif settings.stage_runtime == "remote":
        runtime = RemoteStageRuntime(
            stage_registry,
            settings.stage_remote_urls,
            start_timeout=settings.stage_worker_start_timeout,
        )
        logger.info("stage runtime: remote (%d urls)", len(settings.stage_remote_urls))
    else:
        runtime = LocalStageRuntime(stage_registry, cache)
        logger.info("stage runtime: local")
    registry = create_default_registry(
        stage_registry=stage_registry,
        cache=cache,
        runtime=runtime,
    )
    stats = ServerStatsTracker()
    crosstalk_service = CrosstalkService()
    init_deps(
        store,
        registry,
        stats,
        crosstalk_service,
        stage_registry=registry.stage_registry,
    )

    app.include_router(api_router)
    app.include_router(crosstalk_router)

    return app


app = create_app()
