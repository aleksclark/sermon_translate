"""Warm model StageHost: load once, isolate per-session state, admit by capacity."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.stage_v1.models import (
    ArtifactDigestStatus,
    DrainingPayload,
    ErrorPayload,
    ErrorScope,
    HealthPayload,
    LimitsAdvertised,
    ProvenanceBlock,
    StageErrorCode,
    StageKind,
)
from src.stage_v1.provenance import provenance_id_from_block

ModelLoader = Callable[[], Any | Awaitable[Any]]
CanaryFn = Callable[[Any], bool | Awaitable[bool]]
UnloadFn = Callable[[Any], None | Awaitable[None]]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class StageHostError(Exception):
    """Host-level failure with a stage.v1 error payload."""

    def __init__(self, payload: ErrorPayload) -> None:
        super().__init__(payload.message)
        self.payload = payload


@dataclass
class SessionState:
    """Isolated per-attempt session state. Never holds resident model weights."""

    session_state_id: str
    attempt_id: str | None = None
    cancel_id: str | None = None
    session_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    opened_at: str = field(default_factory=_utc_now_iso)

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def cancel(self) -> None:
        self.cancel_event.set()


class StageHost:
    """Process-scoped host: one model load, many isolated sessions, truthful capacity."""

    def __init__(
        self,
        *,
        stage_kind: StageKind | str,
        stage_id: str,
        stage_version: str,
        model_loader: ModelLoader,
        canary: CanaryFn | None = None,
        model_unloader: UnloadFn | None = None,
        max_sessions: int = 1,
        limits: LimitsAdvertised | None = None,
        code_git_sha: str = "unknown",
        container_image_digest: str | None = None,
        model_provider_id: str = "local",
        model_revision: str = "unknown",
        model_artifact_digest: str = "unavailable",
        model_artifact_status: ArtifactDigestStatus | str = ArtifactDigestStatus.UNAVAILABLE,
        expected_artifact_digest: str | None = None,
        runtime_versions: dict[str, str] | None = None,
        prompt_digest: str | None = None,
        glossary_digest: str | None = None,
        voice_digest: str | None = None,
        stage_config_digest: str | None = None,
        hardware_class: str | None = None,
        boot_id: str | None = None,
        stage_instance_id: str | None = None,
        local_dev: bool = True,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be >= 1")

        if isinstance(stage_kind, StageKind):
            self.stage_kind = stage_kind
        else:
            self.stage_kind = StageKind(stage_kind)
        self.stage_id = stage_id
        self.stage_version = stage_version
        self._model_loader = model_loader
        self._canary = canary
        self._model_unloader = model_unloader
        self.max_sessions = max_sessions
        self.limits = limits or LimitsAdvertised(max_sessions=max_sessions)
        if self.limits.max_sessions != max_sessions:
            self.limits = self.limits.model_copy(update={"max_sessions": max_sessions})

        self.code_git_sha = code_git_sha
        self.container_image_digest = container_image_digest
        self.model_provider_id = model_provider_id
        self.model_revision = model_revision
        self.model_artifact_digest = model_artifact_digest
        self.model_artifact_status = (
            ArtifactDigestStatus(model_artifact_status)
            if not isinstance(model_artifact_status, ArtifactDigestStatus)
            else model_artifact_status
        )
        self.expected_artifact_digest = expected_artifact_digest
        self.runtime_versions = runtime_versions or {}
        self.prompt_digest = prompt_digest
        self.glossary_digest = glossary_digest
        self.voice_digest = voice_digest
        self.stage_config_digest = stage_config_digest
        self.hardware_class = hardware_class
        self.local_dev = local_dev

        self.boot_id = boot_id or str(uuid.uuid4())
        self.stage_instance_id = stage_instance_id or str(uuid.uuid4())

        self._lock = asyncio.Lock()
        self._sessions: dict[str, SessionState] = {}
        self._model: Any | None = None
        self.loader_invocation_count = 0
        self.model_loaded = False
        self.model_warm = False
        self.draining = False
        self.drain_reason: str | None = None
        self.digest_mismatch = False
        self.last_canary_at: str | None = None
        self.last_canary_ok: bool | None = None
        self.provenance: ProvenanceBlock | None = None
        self.provenance_id: str | None = None
        self._shutdown = False
        self._load_failed = False

    @property
    def model(self) -> Any | None:
        return self._model

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    @property
    def available_capacity(self) -> int:
        if self.draining or self._shutdown:
            return 0
        return max(0, self.max_sessions - self.active_sessions)

    def get_session(self, session_state_id: str) -> SessionState | None:
        return self._sessions.get(session_state_id)

    def list_session_ids(self) -> list[str]:
        return list(self._sessions.keys())

    async def load(self) -> None:
        """Load model artifacts once for this process boot."""
        async with self._lock:
            if self._shutdown:
                raise StageHostError(
                    ErrorPayload(
                        code=StageErrorCode.MODEL_UNAVAILABLE,
                        message="host is shut down",
                        retryable=False,
                        scope=ErrorScope.CONNECTION,
                    )
                )
            if self.model_loaded and self._model is not None:
                return
            if (
                self.expected_artifact_digest is not None
                and self.model_artifact_digest != self.expected_artifact_digest
            ):
                self.digest_mismatch = True
                self._load_failed = True
                raise StageHostError(
                    ErrorPayload(
                        code=StageErrorCode.MODEL_UNAVAILABLE,
                        message=(
                            "model artifact digest mismatch: "
                            f"got {self.model_artifact_digest!r}, "
                            f"expected {self.expected_artifact_digest!r}"
                        ),
                        retryable=False,
                        scope=ErrorScope.CONNECTION,
                    )
                )
            self.digest_mismatch = False
            try:
                model = await self._invoke_loader()
            except StageHostError:
                self._load_failed = True
                raise
            except Exception as exc:
                self._load_failed = True
                raise StageHostError(
                    ErrorPayload(
                        code=StageErrorCode.MODEL_UNAVAILABLE,
                        message=f"model load failed: {type(exc).__name__}: {exc}",
                        retryable=True,
                        scope=ErrorScope.CONNECTION,
                    )
                ) from exc
            self._model = model
            self.model_loaded = True
            self._load_failed = False
            self._rebuild_provenance()

    async def warmup(self) -> None:
        """Run optional canary after load; marks model warm on success."""
        if not self.model_loaded or self._model is None:
            await self.load()
        assert self._model is not None

        async with self._lock:
            if self.model_warm and self.last_canary_ok is True:
                return
            if self._canary is None:
                self.model_warm = True
                self.last_canary_ok = True
                self.last_canary_at = _utc_now_iso()
                self._rebuild_provenance()
                return
            try:
                result = self._canary(self._model)
                if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                    ok = bool(await result)  # type: ignore[arg-type]
                else:
                    ok = bool(result)
            except Exception as exc:
                self.model_warm = False
                self.last_canary_ok = False
                self.last_canary_at = _utc_now_iso()
                raise StageHostError(
                    ErrorPayload(
                        code=StageErrorCode.MODEL_UNAVAILABLE,
                        message=f"warmup canary failed: {type(exc).__name__}: {exc}",
                        retryable=True,
                        scope=ErrorScope.CONNECTION,
                    )
                ) from exc
            self.last_canary_at = _utc_now_iso()
            self.last_canary_ok = ok
            if not ok:
                self.model_warm = False
                raise StageHostError(
                    ErrorPayload(
                        code=StageErrorCode.MODEL_UNAVAILABLE,
                        message="warmup canary returned false",
                        retryable=True,
                        scope=ErrorScope.CONNECTION,
                    )
                )
            self.model_warm = True
            self._rebuild_provenance()

    async def open_session(
        self,
        *,
        attempt_id: str | None = None,
        cancel_id: str | None = None,
        session_id: str | None = None,
        initial_data: dict[str, Any] | None = None,
    ) -> SessionState:
        """Admit a new isolated session. Never loads/unloads the model."""
        async with self._lock:
            if self._shutdown:
                raise StageHostError(
                    ErrorPayload(
                        code=StageErrorCode.MODEL_UNAVAILABLE,
                        message="host is shut down",
                        retryable=False,
                        scope=ErrorScope.CONNECTION,
                    )
                )
            if self.draining:
                raise StageHostError(
                    ErrorPayload(
                        code=StageErrorCode.RESOURCE_EXHAUSTED,
                        message="worker is draining; not admitting new sessions",
                        retryable=True,
                        scope=ErrorScope.ATTEMPT,
                        retry_after_ms=1_000,
                    )
                )
            if not self.model_loaded or self._model is None:
                raise StageHostError(
                    ErrorPayload(
                        code=StageErrorCode.MODEL_UNAVAILABLE,
                        message="model is not loaded",
                        retryable=True,
                        scope=ErrorScope.ATTEMPT,
                    )
                )
            if not self.model_warm:
                raise StageHostError(
                    ErrorPayload(
                        code=StageErrorCode.MODEL_UNAVAILABLE,
                        message="model is not warm",
                        retryable=True,
                        scope=ErrorScope.ATTEMPT,
                    )
                )
            if self.active_sessions >= self.max_sessions:
                raise StageHostError(
                    ErrorPayload(
                        code=StageErrorCode.RESOURCE_EXHAUSTED,
                        message=(
                            f"no admission capacity: active={self.active_sessions} "
                            f"max={self.max_sessions}"
                        ),
                        retryable=True,
                        scope=ErrorScope.ATTEMPT,
                        retry_after_ms=500,
                    )
                )
            state = SessionState(
                session_state_id=str(uuid.uuid4()),
                attempt_id=attempt_id,
                cancel_id=cancel_id,
                session_id=session_id,
                data=dict(initial_data or {}),
            )
            self._sessions[state.session_state_id] = state
            return state

    async def close_session(self, session_state_id: str) -> None:
        """Dispose per-attempt state. MUST NOT unload resident model weights."""
        async with self._lock:
            state = self._sessions.pop(session_state_id, None)
            if state is not None:
                state.cancel()
                state.data.clear()
            # Intentionally leave self._model loaded.

    async def cancel_session(self, session_state_id: str) -> SessionState | None:
        """Mark session cancelled and dispose state without unloading the model."""
        async with self._lock:
            state = self._sessions.get(session_state_id)
            if state is None:
                return None
            state.cancel()
            self._sessions.pop(session_state_id, None)
            state.data.clear()
            return state

    def begin_drain(
        self, *, reason: str = "planned_shutdown", grace_ms: int = 0
    ) -> DrainingPayload:
        """Reject new opens; existing sessions may complete within grace."""
        self.draining = True
        self.drain_reason = reason
        return DrainingPayload(
            reason=reason,
            grace_ms=grace_ms,
            active_sessions=self.active_sessions,
        )

    async def cancel_remaining(self) -> int:
        """Cancel and dispose all active sessions; model stays loaded until shutdown."""
        async with self._lock:
            ids = list(self._sessions.keys())
            for sid in ids:
                state = self._sessions.pop(sid)
                state.cancel()
                state.data.clear()
            return len(ids)

    async def shutdown(self) -> None:
        """Drain, cancel sessions, and unload the model (process exit path)."""
        self.begin_drain(reason="shutdown")
        await self.cancel_remaining()
        async with self._lock:
            self._shutdown = True
            model = self._model
            self._model = None
            self.model_loaded = False
            self.model_warm = False
            if model is not None and self._model_unloader is not None:
                result = self._model_unloader(model)
                if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
                    await result  # type: ignore[arg-type]
            self._model = None

    def is_startup_complete(self) -> bool:
        return (
            self.model_loaded
            and not self._load_failed
            and not self.digest_mismatch
            and not self._shutdown
        )

    def is_ready(self) -> bool:
        return (
            self.is_startup_complete()
            and self.model_warm
            and self.last_canary_ok is True
            and not self.draining
            and not self.digest_mismatch
            and self.available_capacity > 0
        )

    def health_payload(self, *, status: str | None = None) -> HealthPayload:
        if status is None:
            if self._shutdown:
                status = "shutdown"
            elif self.draining:
                status = "draining"
            elif self.is_ready():
                status = "ready"
            elif self.model_warm:
                status = "warm"
            elif self.model_loaded:
                status = "loaded"
            elif self._load_failed or self.digest_mismatch:
                status = "failed"
            else:
                status = "loading"
        return HealthPayload(
            status=status,
            stage_kind=self.stage_kind,
            stage_id=self.stage_id,
            stage_version=self.stage_version,
            stage_instance_id=self.stage_instance_id,
            boot_id=self.boot_id,
            active_sessions=self.active_sessions,
            max_sessions=self.max_sessions,
            model_loaded=self.model_loaded,
            model_warm=self.model_warm,
            draining=self.draining,
            last_canary_at=self.last_canary_at,
            last_canary_ok=self.last_canary_ok,
            provenance_id=self.provenance_id,
            limits=self.limits,
        )

    def readiness_detail(self) -> dict[str, Any]:
        payload = self.health_payload()
        detail = payload.model_dump(mode="json")
        detail["digest_mismatch"] = self.digest_mismatch
        detail["available_capacity"] = self.available_capacity
        detail["loader_invocation_count"] = self.loader_invocation_count
        if self.provenance is not None:
            detail["provenance"] = self.provenance.model_dump(mode="json", exclude_none=True)
        return detail

    def _rebuild_provenance(self) -> None:
        container_digest = self.container_image_digest
        if container_digest is None and not self.local_dev:
            container_digest = "unavailable"
        block = ProvenanceBlock(
            stage_id=self.stage_id,
            stage_version=self.stage_version,
            code_git_sha=self.code_git_sha,
            container_image_digest=container_digest,
            model_provider_id=self.model_provider_id,
            model_revision=self.model_revision,
            model_artifact_digest=self.model_artifact_digest,
            model_artifact_status=self.model_artifact_status,
            runtime_versions=dict(self.runtime_versions),
            prompt_digest=self.prompt_digest,
            glossary_digest=self.glossary_digest,
            voice_digest=self.voice_digest,
            stage_config_digest=self.stage_config_digest,
            hardware_class=self.hardware_class,
            boot_id=self.boot_id,
        )
        self.provenance = block
        self.provenance_id = provenance_id_from_block(block)

    async def _invoke_loader(self) -> Any:
        self.loader_invocation_count += 1
        result = self._model_loader()
        if asyncio.iscoroutine(result) or isinstance(result, Awaitable):
            return await result  # type: ignore[arg-type]
        return result
