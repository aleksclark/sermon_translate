from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from src.models import (
    PipelineInfo,
    RTCOffer,
    ServerStats,
    Session,
    SessionCreate,
    SessionUpdate,
)

from .deps import get_pipeline_registry, get_server_stats, get_session_store

logger = logging.getLogger(__name__)

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


@router.post("/sessions/{session_id}/offer")
async def webrtc_offer(session_id: str, offer: RTCOffer) -> dict:
    """Exchange SDP offer/answer to establish a WebRTC connection."""
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription  # noqa: F811

    from src.transport.handler import run_session
    from src.transport.rtc import WebRTCTransport

    store = get_session_store()
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    registry = get_pipeline_registry()
    if registry.get(session.pipeline_id) is None:
        raise HTTPException(status_code=400, detail="Pipeline not found")

    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    transport = WebRTCTransport(pc, sample_rate=session.sample_rate)

    pc.addTrack(transport.output_track)

    @pc.on("datachannel")
    def on_datachannel(channel) -> None:  # type: ignore[no-untyped-def]
        transport.setup_data_channel(channel)

    @pc.on("track")
    def on_track(track) -> None:  # type: ignore[no-untyped-def]
        if track.kind == "audio":
            transport.setup_incoming_track(track)

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=offer.sdp, type=offer.type)
    )
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    asyncio.create_task(run_session(transport, session_id))

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
