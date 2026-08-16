"""HTTP health endpoints for stage.v1 workers: live / startup / ready."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from src.stage_v1.host import StageHost


def build_health_router(host: StageHost) -> APIRouter:
    """Mountable router exposing /health/live, /health/startup, /health/ready."""
    router = APIRouter(tags=["health"])

    @router.get("/health/live")
    async def health_live() -> Response:
        # Process is responsive if we can handle the request.
        if host._shutdown:  # noqa: SLF001 — intentional internal lifecycle check
            return JSONResponse(
                status_code=503,
                content={"status": "shutdown", "detail": "host shut down"},
            )
        return JSONResponse(status_code=200, content={"status": "live"})

    @router.get("/health/startup")
    async def health_startup() -> Response:
        if host.is_startup_complete():
            return JSONResponse(
                status_code=200,
                content={
                    "status": "started",
                    "model_loaded": True,
                    "stage_id": host.stage_id,
                    "boot_id": host.boot_id,
                },
            )
        detail: dict[str, Any] = {
            "status": "starting",
            "model_loaded": host.model_loaded,
            "digest_mismatch": host.digest_mismatch,
            "stage_id": host.stage_id,
            "boot_id": host.boot_id,
        }
        if host.digest_mismatch:
            detail["status"] = "failed"
            detail["reason"] = "digest_mismatch"
        elif host._load_failed:  # noqa: SLF001
            detail["status"] = "failed"
            detail["reason"] = "load_failed"
        return JSONResponse(status_code=503, content=detail)

    @router.get("/health/ready")
    async def health_ready() -> Response:
        detail = host.readiness_detail()
        if host.is_ready():
            return JSONResponse(status_code=200, content=detail)
        return JSONResponse(status_code=503, content=detail)

    return router


def mount_health_routes(app: FastAPI, host: StageHost) -> None:
    """Attach stage.v1 health routes and keep /healthz as a liveness alias only."""
    app.include_router(build_health_router(host))

    @app.get("/healthz")
    async def healthz_compat() -> PlainTextResponse:
        """Compatibility alias — not for scheduling readiness."""
        if host._shutdown:  # noqa: SLF001
            return PlainTextResponse("unavailable", status_code=503)
        return PlainTextResponse("ok")


def health_status_code(host: StageHost, kind: str) -> int:
    """Pure helper for tests: return 200 or 503 for live/startup/ready."""
    if kind == "live":
        return 503 if host._shutdown else 200  # noqa: SLF001
    if kind == "startup":
        return 200 if host.is_startup_complete() else 503
    if kind == "ready":
        return 200 if host.is_ready() else 503
    raise ValueError(f"unknown health kind: {kind}")
