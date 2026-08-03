from __future__ import annotations

from typing import TYPE_CHECKING

from src.models import Session, StageKind
from src.runtime.ws_handle import RemoteStageHandle

if TYPE_CHECKING:
    from src.pipelines.stage_registry import StageRegistry


class RemoteStageRuntime:
    """Connect to pre-started stage workers by URL map."""

    def __init__(
        self,
        stage_registry: StageRegistry,
        urls: dict[str, str],
        *,
        start_timeout: float = 60.0,
    ) -> None:
        self._registry = stage_registry
        self._urls = urls
        self._start_timeout = start_timeout

    async def spawn(
        self,
        stage_id: str,
        session: Session,
        *,
        kind: StageKind | None = None,
    ) -> RemoteStageHandle:
        factory = self._registry.get(stage_id)
        if factory is None:
            raise ValueError(f"Unknown stage: {stage_id}")
        if kind is not None and factory.info.kind != kind:
            raise ValueError(
                f"Stage {stage_id} has kind {factory.info.kind.value}, expected {kind.value}"
            )
        url = self._urls.get(stage_id)
        if not url:
            raise ValueError(f"No remote URL configured for stage: {stage_id}")
        return RemoteStageHandle(
            info=factory.info,
            url=url,
            session=session,
            start_timeout=self._start_timeout,
        )
