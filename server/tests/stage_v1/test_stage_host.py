"""StageHost warm lifecycle, capacity admission, and session isolation."""

from __future__ import annotations

import asyncio

import pytest

from src.stage_v1.host import SessionState, StageHost, StageHostError
from src.stage_v1.models import ArtifactDigestStatus, StageErrorCode, StageKind
from src.stage_v1.provenance import provenance_id_from_block


def _make_host(
    *,
    max_sessions: int = 2,
    loader=None,
    canary=None,
    unloader=None,
    expected_digest: str | None = None,
    artifact_digest: str = "sha256:" + "ab" * 32,
    artifact_status: ArtifactDigestStatus = ArtifactDigestStatus.VERIFIED,
) -> StageHost:
    load_count = {"n": 0}

    def default_loader() -> dict[str, object]:
        load_count["n"] += 1
        return {"weights": "resident-model", "load_n": load_count["n"]}

    return StageHost(
        stage_kind=StageKind.LISTEN,
        stage_id="test-listen",
        stage_version="1.0.0",
        model_loader=loader or default_loader,
        canary=canary,
        model_unloader=unloader,
        max_sessions=max_sessions,
        code_git_sha="deadbeef",
        model_provider_id="test",
        model_revision="rev-1",
        model_artifact_digest=artifact_digest,
        model_artifact_status=artifact_status,
        expected_artifact_digest=expected_digest,
        boot_id="boot-test-1",
        local_dev=True,
    )


@pytest.mark.asyncio
async def test_loader_invoked_once_across_two_sequential_sessions() -> None:
    calls = {"n": 0}

    def loader() -> str:
        calls["n"] += 1
        return "MODEL"

    host = _make_host(loader=loader, max_sessions=2)
    await host.load()
    await host.warmup()
    assert host.loader_invocation_count == 1
    assert calls["n"] == 1

    s1 = await host.open_session(attempt_id="a1", session_id="sess-1")
    s1.data["marker"] = "A"
    await host.close_session(s1.session_state_id)

    s2 = await host.open_session(attempt_id="a2", session_id="sess-2")
    assert "marker" not in s2.data
    await host.close_session(s2.session_state_id)

    assert host.loader_invocation_count == 1
    assert calls["n"] == 1
    assert host.model_loaded is True
    assert host.model == "MODEL"
    assert host.boot_id == "boot-test-1"


@pytest.mark.asyncio
async def test_session_state_ids_are_unique_and_isolated() -> None:
    host = _make_host(max_sessions=2)
    await host.load()
    await host.warmup()

    a = await host.open_session(attempt_id="att-a")
    b = await host.open_session(attempt_id="att-b")
    assert a.session_state_id != b.session_state_id

    a.data["decoder"] = {"ctx": "session-a"}
    b.data["decoder"] = {"ctx": "session-b"}
    assert a.data["decoder"]["ctx"] == "session-a"
    assert b.data["decoder"]["ctx"] == "session-b"
    assert host.get_session(a.session_state_id) is a
    assert host.get_session(b.session_state_id) is b

    await host.close_session(a.session_state_id)
    assert host.get_session(a.session_state_id) is None
    assert host.get_session(b.session_state_id) is b
    # B must not see A's state
    assert b.data["decoder"]["ctx"] == "session-b"
    assert "session-a" not in str(b.data)

    await host.close_session(b.session_state_id)
    assert host.model_loaded is True


@pytest.mark.asyncio
async def test_over_capacity_returns_resource_exhausted_immediately() -> None:
    host = _make_host(max_sessions=1)
    await host.load()
    await host.warmup()

    s1 = await host.open_session()
    assert host.active_sessions == 1
    assert host.available_capacity == 0

    with pytest.raises(StageHostError) as ei:
        await host.open_session()
    assert ei.value.payload.code == StageErrorCode.RESOURCE_EXHAUSTED
    assert ei.value.payload.retryable is True

    await host.close_session(s1.session_state_id)
    s2 = await host.open_session()
    await host.close_session(s2.session_state_id)


@pytest.mark.asyncio
async def test_concurrent_second_session_isolated_or_exhausted() -> None:
    host = _make_host(max_sessions=1)
    await host.load()
    await host.warmup()

    async def open_one(tag: str) -> SessionState | StageErrorCode:
        try:
            state = await host.open_session(session_id=tag)
            state.data["tag"] = tag
            return state
        except StageHostError as exc:
            return exc.payload.code

    results = await asyncio.gather(open_one("A"), open_one("B"))
    states = [r for r in results if isinstance(r, SessionState)]
    codes = [r for r in results if isinstance(r, StageErrorCode)]

    assert len(states) == 1
    assert len(codes) == 1
    assert codes[0] == StageErrorCode.RESOURCE_EXHAUSTED
    assert host.loader_invocation_count == 1
    assert host.model_loaded is True

    await host.close_session(states[0].session_state_id)


@pytest.mark.asyncio
async def test_cancel_does_not_unload_model() -> None:
    unloaded = {"n": 0}

    def unloader(_model: object) -> None:
        unloaded["n"] += 1

    host = _make_host(unloader=unloader, max_sessions=2)
    await host.load()
    await host.warmup()
    s = await host.open_session()
    s.data["stream"] = object()

    cancelled = await host.cancel_session(s.session_state_id)
    assert cancelled is not None
    assert cancelled.cancelled is True
    assert host.get_session(s.session_state_id) is None
    assert host.model_loaded is True
    assert host.model is not None
    assert unloaded["n"] == 0
    assert host.loader_invocation_count == 1

    # New session still works on same warm model
    s2 = await host.open_session()
    assert host.loader_invocation_count == 1
    await host.close_session(s2.session_state_id)


@pytest.mark.asyncio
async def test_close_session_never_unloads_model() -> None:
    unloaded = {"n": 0}

    async def unloader(_model: object) -> None:
        unloaded["n"] += 1

    host = _make_host(unloader=unloader)
    await host.load()
    await host.warmup()
    s = await host.open_session()
    await host.close_session(s.session_state_id)
    assert unloaded["n"] == 0
    assert host.model_loaded is True

    await host.shutdown()
    assert unloaded["n"] == 1
    assert host.model_loaded is False


@pytest.mark.asyncio
async def test_draining_rejects_new_opens_allows_close() -> None:
    host = _make_host(max_sessions=2)
    await host.load()
    await host.warmup()
    s = await host.open_session()

    payload = host.begin_drain(reason="deploy", grace_ms=100)
    assert payload.active_sessions == 1
    assert host.draining is True
    assert host.is_ready() is False

    with pytest.raises(StageHostError) as ei:
        await host.open_session()
    assert ei.value.payload.code == StageErrorCode.RESOURCE_EXHAUSTED

    await host.close_session(s.session_state_id)
    assert host.active_sessions == 0
    assert host.model_loaded is True


@pytest.mark.asyncio
async def test_draining_cancel_remaining() -> None:
    host = _make_host(max_sessions=3)
    await host.load()
    await host.warmup()
    a = await host.open_session()
    b = await host.open_session()
    host.begin_drain()
    n = await host.cancel_remaining()
    assert n == 2
    assert host.active_sessions == 0
    assert a.cancelled and b.cancelled
    assert host.model_loaded is True


@pytest.mark.asyncio
async def test_open_before_warmup_fails() -> None:
    host = _make_host()
    await host.load()
    assert host.model_loaded is True
    assert host.model_warm is False
    with pytest.raises(StageHostError) as ei:
        await host.open_session()
    assert ei.value.payload.code == StageErrorCode.MODEL_UNAVAILABLE


@pytest.mark.asyncio
async def test_async_loader_and_canary() -> None:
    async def loader() -> dict[str, str]:
        await asyncio.sleep(0)
        return {"m": "1"}

    async def canary(model: dict[str, str]) -> bool:
        await asyncio.sleep(0)
        return model["m"] == "1"

    host = _make_host(loader=loader, canary=canary)
    await host.load()
    await host.warmup()
    assert host.model_warm is True
    assert host.last_canary_ok is True
    assert host.provenance_id is not None
    assert host.provenance is not None
    assert provenance_id_from_block(host.provenance) == host.provenance_id
    s = await host.open_session()
    await host.close_session(s.session_state_id)


@pytest.mark.asyncio
async def test_failed_canary_blocks_ready() -> None:
    host = _make_host(canary=lambda _m: False)
    await host.load()
    with pytest.raises(StageHostError):
        await host.warmup()
    assert host.model_warm is False
    assert host.last_canary_ok is False
    assert host.is_ready() is False
