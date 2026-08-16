from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from contextlib import contextmanager

_thread_lock = threading.Lock()
_async_lock: asyncio.Lock | None = None


def _get_async_lock() -> asyncio.Lock:
    global _async_lock
    if _async_lock is None:
        _async_lock = asyncio.Lock()
    return _async_lock


@contextmanager
def gpu_model_load_lock() -> Iterator[None]:
    """Serialize heavy CUDA model construction across threads."""
    with _thread_lock:
        yield


async def gpu_model_load_async() -> asyncio.Lock:
    """Async lock for coordinating stage start() coroutines."""
    return _get_async_lock()
