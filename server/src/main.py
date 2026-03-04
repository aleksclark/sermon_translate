from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import SessionStore, init_deps
from src.api import router as api_router
from src.models import ServerStatsTracker
from src.pipelines import create_default_registry

LOG_FILE = Path(__file__).resolve().parent.parent.parent / "server.log"


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
    _configure_logging()
    app = FastAPI(title="Sermon Translate", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    store = SessionStore()
    registry = create_default_registry()
    stats = ServerStatsTracker()
    init_deps(store, registry, stats)

    app.include_router(api_router)

    return app


app = create_app()
