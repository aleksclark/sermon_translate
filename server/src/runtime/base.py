from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.models import Session, StageInfo, StageKind


@runtime_checkable
class StageHandle(Protocol):
    info: StageInfo

    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    @property
    def stage(self) -> Any:
        """Underlying stage implementation (ASR/Translation/TTS/Prosody)."""
        ...


@runtime_checkable
class StageRuntime(Protocol):
    async def spawn(
        self,
        stage_id: str,
        session: Session,
        *,
        kind: StageKind | None = None,
    ) -> StageHandle: ...
