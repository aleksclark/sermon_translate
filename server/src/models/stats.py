from __future__ import annotations

import time

from pydantic import BaseModel


class ServerStats(BaseModel):
    uptime_seconds: float = 0.0
    active_sessions: int = 0
    total_sessions: int = 0
    total_bytes_processed: int = 0
    available_pipelines: int = 0


class ServerStatsTracker:
    def __init__(self) -> None:
        self._start_time = time.time()
        self.total_sessions = 0
        self.total_bytes_processed = 0

    def snapshot(self, active_sessions: int, pipeline_count: int) -> ServerStats:
        return ServerStats(
            uptime_seconds=round(time.time() - self._start_time, 1),
            active_sessions=active_sessions,
            total_sessions=self.total_sessions,
            total_bytes_processed=self.total_bytes_processed,
            available_pipelines=pipeline_count,
        )
