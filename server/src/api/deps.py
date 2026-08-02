from __future__ import annotations

from typing import TYPE_CHECKING

from src.models import ServerStatsTracker
from src.pipelines import PipelineRegistry

from .store import SessionStore

if TYPE_CHECKING:
    from src.transport.crosstalk_service import CrosstalkService

_session_store: SessionStore | None = None
_pipeline_registry: PipelineRegistry | None = None
_server_stats: ServerStatsTracker | None = None
_crosstalk_service: CrosstalkService | None = None


def init_deps(
    store: SessionStore,
    registry: PipelineRegistry,
    stats: ServerStatsTracker,
    crosstalk_service: CrosstalkService | None = None,
) -> None:
    global _session_store, _pipeline_registry, _server_stats, _crosstalk_service
    _session_store = store
    _pipeline_registry = registry
    _server_stats = stats
    _crosstalk_service = crosstalk_service


def get_session_store() -> SessionStore:
    assert _session_store is not None
    return _session_store


def get_pipeline_registry() -> PipelineRegistry:
    assert _pipeline_registry is not None
    return _pipeline_registry


def get_server_stats() -> ServerStatsTracker:
    assert _server_stats is not None
    return _server_stats


def get_crosstalk_service() -> CrosstalkService:
    assert _crosstalk_service is not None
    return _crosstalk_service
