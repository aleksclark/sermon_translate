"""F5-TTS voice-cloned synthesis for the SimulStreaming pipeline.

Uses F5-TTS fine-tuned on real Spanish sermon audio to synthesise
Spanish speech.  The fine-tuned model has learned Spanish phonetics
from 44 minutes of training data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from src.pipelines._audio import downsample

logger = logging.getLogger(__name__)

F5_SAMPLE_RATE = 24000
_FIXTURES = Path(__file__).resolve().parent.parent / "harness" / "fixtures"
_CKPT = str(_FIXTURES / "f5_spanish_finetuned.pt")
_VOCAB = ""  # will be resolved at load time
_REF_AUDIO = str(_FIXTURES / "speaker-ref-es-24k.wav")
_REF_TEXT = (
    "que eso le iba a costar la entrada a la tierra prometida,"
)

_f5_model: Any = None


def _patch_torchaudio() -> None:
    """Monkey-patch torchaudio.load to use soundfile (torchcodec compat)."""
    import torch
    import torchaudio

    if getattr(torchaudio, "_patched_for_f5", False):
        return

    def _load(path: str, **_kwargs: Any) -> tuple[Any, int]:
        data, sr = sf.read(str(path))
        data = data[np.newaxis, :] if data.ndim == 1 else data.T
        return torch.tensor(data, dtype=torch.float32), sr

    torchaudio.load = _load  # type: ignore[assignment]
    torchaudio._patched_for_f5 = True  # type: ignore[attr-defined]


def _get_f5_model() -> Any:
    global _f5_model  # noqa: PLW0603
    if _f5_model is not None:
        return _f5_model

    _patch_torchaudio()
    import importlib.resources

    from f5_tts.api import F5TTS

    vocab = str(
        Path(importlib.import_module("f5_tts.infer.examples").__path__[0]) / "vocab.txt"
    )

    ckpt = _CKPT
    if not Path(ckpt).exists():
        logger.warning("F5-TTS fine-tuned checkpoint not found at %s, using base model", ckpt)
        _f5_model = F5TTS(device="cuda")
    else:
        _f5_model = F5TTS(ckpt_file=ckpt, vocab_file=vocab, device="cuda")
        logger.info("F5-TTS fine-tuned model loaded from %s", ckpt)

    return _f5_model


def release_f5_model() -> None:
    """Free the F5-TTS model from GPU memory."""
    global _f5_model  # noqa: PLW0603
    _f5_model = None


def synthesize_f5(
    text: str,
    ref_audio_path: str | None = None,
    ref_text: str | None = None,
    target_rate: int = 48000,
) -> bytes:
    """Synthesise Spanish text using the fine-tuned F5-TTS model.

    Uses the default Spanish reference clip if none is provided.
    Returns s16le PCM bytes at *target_rate*.
    """
    if not text or not text.strip():
        return b""

    if ref_audio_path is None:
        ref_audio_path = _REF_AUDIO
    if ref_text is None:
        ref_text = _REF_TEXT

    model = _get_f5_model()
    try:
        wav, sr, _ = model.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text,
            gen_text=text,
        )
    except Exception:
        logger.exception("F5-TTS synthesis failed")
        return b""

    if len(wav) == 0:
        return b""

    audio = wav.astype(np.float32)
    if sr != target_rate:
        audio = downsample(audio, sr, target_rate)
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    return pcm.tobytes()
