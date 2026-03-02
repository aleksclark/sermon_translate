from .base import BasePipeline, OutputStreamDescriptor, OutputStreamKind
from .echo import EchoPipeline
from .registry import PipelineRegistry, create_default_registry
from .spanish import SpanishTranslationPipeline
from .spanish_direct import SpanishDirectPipeline
from .whisper_tts import WhisperTTSPipeline

__all__ = [
    "BasePipeline",
    "EchoPipeline",
    "OutputStreamDescriptor",
    "OutputStreamKind",
    "PipelineRegistry",
    "SpanishDirectPipeline",
    "SpanishTranslationPipeline",
    "WhisperTTSPipeline",
    "create_default_registry",
]
