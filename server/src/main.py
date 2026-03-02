from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocket

from src.api import SessionStore, init_deps
from src.api import router as api_router
from src.models import ServerStatsTracker
from src.pipelines import create_default_registry
from src.transport import handle_stream


def create_app() -> FastAPI:
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

    @app.websocket("/ws/stream/{session_id}")
    async def ws_stream(ws: WebSocket, session_id: str) -> None:
        await handle_stream(ws, session_id)

    return app


app = create_app()
