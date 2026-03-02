from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.models import (
    PipelineInfo,
    ServerStats,
    Session,
    SessionCreate,
    SessionUpdate,
)

from .deps import get_pipeline_registry, get_server_stats, get_session_store

router = APIRouter(prefix="/api")


@router.get("/stats", response_model=ServerStats)
async def server_stats() -> ServerStats:
    tracker = get_server_stats()
    store = get_session_store()
    registry = get_pipeline_registry()
    return tracker.snapshot(store.active_count(), len(registry))


@router.get("/pipelines", response_model=list[PipelineInfo])
async def list_pipelines() -> list[PipelineInfo]:
    return get_pipeline_registry().list_all()


@router.post("/sessions", response_model=Session, status_code=201)
async def create_session(req: SessionCreate) -> Session:
    registry = get_pipeline_registry()
    if registry.get(req.pipeline_id) is None:
        raise HTTPException(status_code=400, detail=f"Unknown pipeline: {req.pipeline_id}")
    store = get_session_store()
    session = store.create(req)
    tracker = get_server_stats()
    tracker.total_sessions += 1
    return session


@router.get("/sessions", response_model=list[Session])
async def list_sessions() -> list[Session]:
    return get_session_store().list_all()


@router.get("/sessions/{session_id}", response_model=Session)
async def get_session(session_id: str) -> Session:
    session = get_session_store().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.patch("/sessions/{session_id}", response_model=Session)
async def update_session(session_id: str, req: SessionUpdate) -> Session:
    session = get_session_store().update(session_id, req)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str) -> None:
    if not get_session_store().delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
