from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.models import StageInfo, StageKind


@runtime_checkable
class StageFactory(Protocol):
    @property
    def info(self) -> StageInfo: ...

    def create(self, *, sample_rate: int = 48000) -> object: ...


class StageRegistry:
    """Central registry of selectable pipeline stages."""

    def __init__(self) -> None:
        self._factories: dict[str, StageFactory] = {}

    def register(self, stage_factory: StageFactory) -> None:
        self._factories[stage_factory.info.id] = stage_factory

    def get(self, stage_id: str) -> StageFactory | None:
        return self._factories.get(stage_id)

    def list_all(self, kind: StageKind | None = None) -> list[StageInfo]:
        infos = [factory.info for factory in self._factories.values()]
        if kind is not None:
            infos = [info for info in infos if info.kind == kind]
        return infos

    def __len__(self) -> int:
        return len(self._factories)


def create_default_stage_registry() -> StageRegistry:
    from src.pipelines.stub_stages import register_stub_stages

    registry = StageRegistry()
    register_stub_stages(registry)
    return registry
