from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.models import CrosstalkChannelInfo, CrosstalkSessionInfo
from src.transport.crosstalk_client import CrosstalkError, CrosstalkSSRFError

from .deps import get_crosstalk_service, get_session_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/crosstalk")


def _require_configured() -> None:
    if not get_crosstalk_service().configured():
        raise HTTPException(status_code=503, detail="Crosstalk is not configured")


@router.get("/sessions", response_model=list[CrosstalkSessionInfo])
async def list_crosstalk_sessions() -> list[CrosstalkSessionInfo]:
    _require_configured()
    try:
        sessions = await get_crosstalk_service().client().list_sessions()
    except CrosstalkSSRFError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CrosstalkError as exc:
        raise HTTPException(status_code=502, detail="Crosstalk request failed") from exc
    return [
        CrosstalkSessionInfo(id=s.id, name=s.name, description=s.description) for s in sessions
    ]


@router.get("/sessions/{crosstalk_session_id}/channels", response_model=list[CrosstalkChannelInfo])
async def list_crosstalk_channels(crosstalk_session_id: str) -> list[CrosstalkChannelInfo]:
    _require_configured()
    try:
        channels = await get_crosstalk_service().client().list_channels(crosstalk_session_id)
    except CrosstalkSSRFError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CrosstalkError as exc:
        raise HTTPException(status_code=502, detail="Crosstalk request failed") from exc
    return [
        CrosstalkChannelInfo(id=c.id, name=c.name, type=c.type, session_id=c.session_id)
        for c in channels
    ]


@router.post("/translations/{session_id}/start", status_code=202)
async def start_crosstalk_translation(session_id: str) -> dict[str, str]:
    _require_configured()
    session = get_session_store().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.crosstalk_session_id:
        raise HTTPException(status_code=400, detail="Session has no Crosstalk source")
    try:
        await get_crosstalk_service().start_translation(
            session_id,
            session.crosstalk_session_id,
            sample_rate=session.sample_rate,
        )
    except CrosstalkSSRFError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except CrosstalkError as exc:
        raise HTTPException(status_code=502, detail="Crosstalk request failed") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"session_id": session_id, "status": "starting"}


@router.post("/translations/{session_id}/stop", status_code=200)
async def stop_crosstalk_translation(session_id: str) -> dict[str, str]:
    stopped = await get_crosstalk_service().stop_translation(session_id)
    if not stopped:
        raise HTTPException(status_code=404, detail="No running Crosstalk translation")
    return {"session_id": session_id, "status": "stopped"}
