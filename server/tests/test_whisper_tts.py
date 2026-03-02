from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import MagicMock, patch

import numpy as np

from src.pipelines.whisper_tts import WhisperTTSPipeline, _downsample


def _make_audio_chunk(n_samples: int = 4096, sample_rate: int = 48000) -> bytes:
    t = np.linspace(0, n_samples / sample_rate, n_samples, dtype=np.float32)
    tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    return tone.tobytes()


def _make_mock_model(texts: list[str] | None = None):
    if texts is None:
        texts = ["Hello world."]
    model = MagicMock()
    segments = []
    for t in texts:
        seg = MagicMock()
        seg.text = t
        segments.append(seg)
    model.transcribe.return_value = (iter(segments), None)
    return model


class TestWhisperTTSPipeline:
    def test_info(self) -> None:
        p = WhisperTTSPipeline()
        assert p.info.id == "whisper-tts"
        assert p.info.name == "Whisper TTS"
        assert p.info.description

    async def test_process_yields_no_audio(self) -> None:
        p = WhisperTTSPipeline()

        async def input_stream() -> AsyncIterator[bytes]:
            yield b"chunk"

        output: list[bytes] = []
        async for out in p.process(input_stream()):
            output.append(out)

        assert output == []

    async def test_process_text_yields_transcription(self) -> None:
        p = WhisperTTSPipeline(sample_rate=16000)
        p._model = _make_mock_model(["Hello world.", "Testing."])

        chunk = _make_audio_chunk(n_samples=16000 * 4, sample_rate=16000)

        async def input_stream() -> AsyncIterator[bytes]:
            yield chunk

        text_iter = p.process_text(input_stream())
        assert text_iter is not None

        lines: list[str] = []
        async for line in text_iter:
            lines.append(line)

        assert lines == ["Hello world.", "Testing."]
        p._model.transcribe.assert_called_once()

    async def test_process_text_empty_stream(self) -> None:
        p = WhisperTTSPipeline()
        p._model = _make_mock_model()

        async def input_stream() -> AsyncIterator[bytes]:
            return
            yield  # noqa: F841

        text_iter = p.process_text(input_stream())
        assert text_iter is not None

        lines: list[str] = []
        async for line in text_iter:
            lines.append(line)

        assert lines == []

    async def test_process_text_flushes_remaining_buffer(self) -> None:
        p = WhisperTTSPipeline(sample_rate=16000)

        call_count = 0

        def mock_transcribe(audio, **kwargs):
            nonlocal call_count
            call_count += 1
            seg = MagicMock()
            seg.text = f"Segment {call_count}."
            return iter([seg]), None

        model = MagicMock()
        model.transcribe = mock_transcribe
        p._model = model

        chunk1 = _make_audio_chunk(n_samples=16000 * 4, sample_rate=16000)
        chunk2 = _make_audio_chunk(n_samples=16000 * 2, sample_rate=16000)

        async def input_stream() -> AsyncIterator[bytes]:
            yield chunk1
            yield chunk2

        lines: list[str] = []
        text_iter = p.process_text(input_stream())
        assert text_iter is not None
        async for line in text_iter:
            lines.append(line)

        assert len(lines) == 2
        assert lines[0] == "Segment 1."
        assert lines[1] == "Segment 2."

    async def test_process_text_skips_empty_text(self) -> None:
        p = WhisperTTSPipeline(sample_rate=16000)
        p._model = _make_mock_model(["  ", "Real text."])

        chunk = _make_audio_chunk(n_samples=16000 * 4, sample_rate=16000)

        async def input_stream() -> AsyncIterator[bytes]:
            yield chunk

        lines: list[str] = []
        text_iter = p.process_text(input_stream())
        assert text_iter is not None
        async for line in text_iter:
            lines.append(line)

        assert lines == ["Real text."]

    async def test_start_loads_model(self) -> None:
        p = WhisperTTSPipeline()
        mock_model = MagicMock()
        with patch.object(p, "_load_model", return_value=mock_model):
            await p.start()
        assert p._model is mock_model

    async def test_stop_releases_model(self) -> None:
        p = WhisperTTSPipeline()
        p._model = MagicMock()
        await p.stop()
        assert p._model is None

    def test_base_pipeline_process_text_returns_none(self) -> None:
        from src.pipelines.echo import EchoPipeline

        p = EchoPipeline()

        async def input_stream() -> AsyncIterator[bytes]:
            return
            yield  # noqa: F841

        assert p.process_text(input_stream()) is None

    def test_downsample(self) -> None:
        audio = np.ones(48000, dtype=np.float32)
        result = _downsample(audio, 48000, 16000)
        assert len(result) == 16000

    def test_downsample_noop(self) -> None:
        audio = np.ones(16000, dtype=np.float32)
        result = _downsample(audio, 16000, 16000)
        assert len(result) == 16000
        assert result is audio
