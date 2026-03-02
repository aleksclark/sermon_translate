from .base import BasePipeline
from .echo import EchoPipeline
from .registry import PipelineRegistry, create_default_registry
from .spanish import SpanishTranslationPipeline
from .whisper_tts import WhisperTTSPipeline

__all__ = [
    "BasePipeline",
    "EchoPipeline",
    "PipelineRegistry",
    "SpanishTranslationPipeline",
    "WhisperTTSPipeline",
    "create_default_registry",
]
