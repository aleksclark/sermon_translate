from __future__ import annotations

from src.models import ServerStatsTracker
from src.pipelines import PipelineRegistry

from .store import SessionStore

_session_store: SessionStore | None = None
_pipeline_registry: PipelineRegistry | None = None
_server_stats: ServerStatsTracker | None = None


def init_deps(
    store: SessionStore,
    registry: PipelineRegistry,
    stats: ServerStatsTracker,
) -> None:
    global _session_store, _pipeline_registry, _server_stats
    _session_store = store
    _pipeline_registry = registry
    _server_stats = stats


def get_session_store() -> SessionStore:
    assert _session_store is not None
    return _session_store


def get_pipeline_registry() -> PipelineRegistry:
    assert _pipeline_registry is not None
    return _pipeline_registry


def get_server_stats() -> ServerStatsTracker:
    assert _server_stats is not None
    return _server_stats
