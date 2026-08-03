from __future__ import annotations

import asyncio
import os
import signal
import socket
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from src.models import Session, StageKind
from src.runtime.model_cache import ModelCache
from src.runtime.ws_handle import RemoteStageHandle

if TYPE_CHECKING:
    from src.pipelines.stage_registry import StageRegistry


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class SubprocessStageRuntime:
    """Spawn a local stage worker process and connect over WebSocket."""

    def __init__(
        self,
        stage_registry: StageRegistry,
        cache: ModelCache,
        *,
        python: str | None = None,
        start_timeout: float = 60.0,
    ) -> None:
        self._registry = stage_registry
        self._cache = cache
        self._python = python or sys.executable
        self._start_timeout = start_timeout
        self._processes: list[asyncio.subprocess.Process] = []

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

        env = os.environ.copy()
        env.update(self._cache.environ())
        env.setdefault("STAGE_RUNTIME", "local")
        try:
            from src.runtime.nvidia_libs import ensure_nvidia_library_path

            env["LD_LIBRARY_PATH"] = ensure_nvidia_library_path()
        except Exception:
            pass
        port = _free_port()

        process = await asyncio.create_subprocess_exec(
            self._python,
            "-m",
            "src.runtime.worker",
            "--stage-id",
            stage_id,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=str(Path(__file__).resolve().parents[2]),
            start_new_session=True,
        )
        self._processes.append(process)
        await asyncio.wait_for(self._wait_ready(process, port), timeout=self._start_timeout)
        handle = _ManagedRemoteHandle(
            info=factory.info,
            url=f"ws://127.0.0.1:{port}/ws",
            session=session,
            start_timeout=self._start_timeout,
            process=process,
            runtime=self,
        )
        return handle

    async def _wait_ready(self, process: asyncio.subprocess.Process, port: int) -> None:
        deadline = asyncio.get_running_loop().time() + self._start_timeout
        url = f"http://127.0.0.1:{port}/healthz"
        async with httpx.AsyncClient() as client:
            while True:
                if process.returncode is not None:
                    err = b""
                    if process.stderr is not None:
                        err = await process.stderr.read()
                    raise RuntimeError(
                        f"stage worker exited early ({process.returncode}): "
                        f"{err.decode('utf-8', errors='replace')}"
                    )
                try:
                    response = await client.get(url, timeout=0.5)
                    if response.status_code == 200:
                        return
                except (httpx.HTTPError, OSError):
                    pass
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError(f"stage worker did not become ready on port {port}")
                await asyncio.sleep(0.05)

    async def stop_all(self) -> None:
        for process in list(self._processes):
            await self._kill(process)
        self._processes.clear()

    async def _kill(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            if process.pid:
                os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            try:
                if process.pid:
                    os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                return
            await process.wait()


class _ManagedRemoteHandle(RemoteStageHandle):
    def __init__(
        self,
        *,
        info,
        url: str,
        session: Session,
        start_timeout: float,
        process: asyncio.subprocess.Process,
        runtime: SubprocessStageRuntime,
    ) -> None:
        super().__init__(info=info, url=url, session=session, start_timeout=start_timeout)
        self._process = process
        self._runtime = runtime

    async def stop(self) -> None:
        try:
            await super().stop()
        finally:
            await self._runtime._kill(self._process)
            if self._process in self._runtime._processes:
                self._runtime._processes.remove(self._process)
