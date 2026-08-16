from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

METADATA_SCHEMA_VERSION = 1


class MetadataKind(StrEnum):
    PROSODY = "prosody"
    INSTRUCTIONS = "instructions"


class ProsodyFrame(BaseModel):
    f0_hz: float | None = None
    pitch_confidence: float | None = None
    energy: float | None = None
    speaking_rate: float | None = None
    is_pause: bool | None = None
    boundary: str | None = None
    emphasis: float | None = None
    confidence: float | None = None
    features: dict[str, float] = Field(default_factory=dict)


class SynthesisInstructions(BaseModel):
    hints: dict[str, Any] = Field(default_factory=dict)
    markers: list[dict[str, Any]] = Field(default_factory=list)


class MetadataEnvelope(BaseModel):
    schema_version: int = METADATA_SCHEMA_VERSION
    stream: str
    kind: MetadataKind
    sequence: int
    source_utterance_id: str | None = None
    target_utterance_id: str | None = None
    start_ms: float | None = None
    end_ms: float | None = None
    prosody: ProsodyFrame | None = None
    instructions: SynthesisInstructions | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
