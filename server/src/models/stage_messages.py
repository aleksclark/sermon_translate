from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.metadata import SynthesisInstructions


class ProsodyToken(BaseModel):
    """5-dim quantized prosody token aligned to a word/span."""

    pitch_median: int
    pitch_range: int
    pitch_slope: int
    duration: int
    energy: int
    f0_hz: float | None = None
    energy_rms: float | None = None
    start_ms: float | None = None
    end_ms: float | None = None


class WordSpan(BaseModel):
    text: str
    start_ms: float | None = None
    end_ms: float | None = None
    conf: float | None = None
    prosody: ProsodyToken | None = None


class ListenProduct(BaseModel):
    sequence: int
    utterance_id: str
    text: str
    is_final: bool = False
    words: list[WordSpan] = Field(default_factory=list)
    language: str = "en"


class TranslateProduct(BaseModel):
    sequence: int
    source_utterance_id: str
    target_utterance_id: str
    text: str
    is_final: bool = False
    words: list[WordSpan] = Field(default_factory=list)
    instructions: SynthesisInstructions | None = None


class SpeakProduct(BaseModel):
    sequence: int
    target_utterance_id: str
    pcm: bytes
    sample_rate: int
    start_ms: float | None = None
    end_ms: float | None = None
