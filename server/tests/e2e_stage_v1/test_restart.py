"""Wave 5 lite: restart/reconnect skeleton for stage.v1.

Full G5 is out of Lane F scope; this module reserves the path and documents
expected fence/gap behavior so Wave 5 can fill RED→GREEN without renames.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.stage_v1_real


@pytest.mark.skip(reason="Wave 5 restart harness not implemented in Lane F (skeleton only)")
@pytest.mark.asyncio
async def test_restart_rejects_stale_fence() -> None:
    """After worker restart, old attempt fence products must be rejected."""
    raise NotImplementedError


@pytest.mark.skip(reason="Wave 5 restart harness not implemented in Lane F (skeleton only)")
@pytest.mark.asyncio
async def test_restart_emits_gap_for_unresumable_listen_audio() -> None:
    """Listen audio mid-stream cannot transparently resume; expect explicit gap."""
    raise NotImplementedError


@pytest.mark.skip(reason="Wave 5 restart harness not implemented in Lane F (skeleton only)")
@pytest.mark.asyncio
async def test_replay_unpublished_translate_speak_exactly_once() -> None:
    """Replay of unpublished committed unit publishes exactly once."""
    raise NotImplementedError
