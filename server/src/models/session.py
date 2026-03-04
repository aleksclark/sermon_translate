from __future__ import annotations

import time
import uuid
from enum import StrEnum

from pydantic import BaseModel, Field


class SessionStatus(StrEnum):
    CREATED = "created"
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"


class OutputStreamInfo(BaseModel):
    name: str
    kind: str
    label: str = ""


class PipelineInfo(BaseModel):
    id: str
    name: str
    description: str
    output_streams: list[OutputStreamInfo] = Field(default_factory=list)


class SessionCreate(BaseModel):
    pipeline_id: str
    sample_rate: int = 48000
    channels: int = 1
    label: str = ""
    audio_context_seconds: float = 0.0


class SessionUpdate(BaseModel):
    label: str | None = None
    status: SessionStatus | None = None


class RTCOffer(BaseModel):
    sdp: str
    type: str


class SessionStats(BaseModel):
    bytes_received: int = 0
    bytes_sent: int = 0
    chunks_received: int = 0
    chunks_sent: int = 0
    duration_seconds: float = 0.0
    pipeline_latency_ms: float = 0.0
    audio_delay_seconds: float = 0.0


class Session(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    pipeline_id: str
    label: str = ""
    status: SessionStatus = SessionStatus.CREATED
    sample_rate: int = 48000
    channels: int = 1
    audio_context_seconds: float = 0.0
    created_at: float = Field(default_factory=time.time)
    stats: SessionStats = Field(default_factory=SessionStats)
