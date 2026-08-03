from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from src.models import Session, StageInfo, StageKind
from src.runtime.model_cache import ModelCache

if TYPE_CHECKING:
    from src.pipelines.stage_registry import StageRegistry


class LocalStageHandle:
    def __init__(self, info: StageInfo, stage: Any) -> None:
        self.info = info
        self._stage = stage

    @property
    def stage(self) -> Any:
        return self._stage

    async def start(self) -> None:
        await self._stage.start()

    async def stop(self) -> None:
        await self._stage.stop()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stage, name)


class LocalStageRuntime:
    """In-process stage runtime backed by StageRegistry factories."""

    def __init__(self, stage_registry: StageRegistry, cache: ModelCache) -> None:
        self._registry = stage_registry
        self._cache = cache

    async def spawn(
        self,
        stage_id: str,
        session: Session,
        *,
        kind: StageKind | None = None,
    ) -> LocalStageHandle:
        factory = self._registry.get(stage_id)
        if factory is None:
            raise ValueError(f"Unknown stage: {stage_id}")
        if kind is not None and factory.info.kind != kind:
            raise ValueError(
                f"Stage {stage_id} has kind {factory.info.kind.value}, expected {kind.value}"
            )

        kwargs: dict[str, Any] = {
            "sample_rate": session.sample_rate,
            "cache": self._cache,
            "session": session,
        }
        create = factory.create
        try:
            signature = inspect.signature(create)
        except (TypeError, ValueError):
            signature = None

        if signature is not None:
            accepts_var_kw = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            )
            if not accepts_var_kw:
                kwargs = {
                    key: value
                    for key, value in kwargs.items()
                    if key in signature.parameters
                }

        stage = create(**kwargs)
        return LocalStageHandle(factory.info, stage)
