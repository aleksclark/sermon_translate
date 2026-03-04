from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from src.models import Session
from src.pipelines.spanish_direct import (
    SpanishDirectPipeline,
    _decode_tokens,
)


def _make_audio_chunk(n_samples: int = 4096, sample_rate: int = 48000) -> bytes:
    t = np.linspace(0, n_samples / sample_rate, n_samples, dtype=np.float32)
    tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    return tone.tobytes()


def _make_mock_processor_and_model(text: str = "Hola mundo."):
    processor = MagicMock()
    processor.return_value = {"input_features": MagicMock()}
    processor.decode.return_value = text

    model = MagicMock()
    output_tensor = MagicMock()
    output_tensor.tolist.return_value = [[1, 2, 3]]
    model.generate.return_value = [output_tensor]
    model.eval.return_value = model

    return processor, model


class TestSpanishDirectPipeline:
    def test_info(self) -> None:
        p = SpanishDirectPipeline()
        assert p.info.id == "spanish-direct"
        assert p.info.name == "Spanish Direct (SeamlessM4T)"
        assert p.info.description
        assert len(p.info.output_streams) == 2

    def test_output_streams(self) -> None:
        p = SpanishDirectPipeline()
        names = [s.name for s in p.output_streams]
        assert "audio" in names
        assert "es-transcript" in names

    def test_configure_session(self) -> None:
        p = SpanishDirectPipeline()
        session = Session(
            pipeline_id="spanish-direct",
            audio_context_seconds=10.0,
        )
        p.configure_session(session)
        assert p._audio_context_seconds == 10.0

    def test_configure_session_default(self) -> None:
        p = SpanishDirectPipeline()
        session = Session(pipeline_id="spanish-direct")
        p.configure_session(session)
        assert p._audio_context_seconds == 0.0

    async def test_process_produces_audio(self) -> None:
        p = SpanishDirectPipeline(sample_rate=16000)
        processor, model = _make_mock_processor_and_model()
        p._processor = processor
        p._model = model

        fake_pcm = b"\x00\x01" * 1000

        async def input_stream() -> AsyncIterator[bytes]:
            yield _make_audio_chunk(n_samples=16000 * 4, sample_rate=16000)

        with (
            patch(
                "src.pipelines.spanish_direct._synthesize_spanish",
                new=AsyncMock(return_value=fake_pcm),
            ),
            patch(
                "src.pipelines.spanish_direct._translate_audio_sync",
                return_value="Hola mundo.",
            ),
        ):
            output: list[bytes] = []
            async for chunk in p.process(input_stream()):
                output.append(chunk)

        assert len(output) >= 1
        assert all(isinstance(c, bytes) and len(c) > 0 for c in output)

    async def test_iter_stream_yields_transcript(self) -> None:
        p = SpanishDirectPipeline(sample_rate=16000)
        processor, model = _make_mock_processor_and_model("Hola.")
        p._processor = processor
        p._model = model

        fake_pcm = b"\x00\x01" * 100

        async def input_stream() -> AsyncIterator[bytes]:
            yield _make_audio_chunk(n_samples=16000 * 4, sample_rate=16000)

        with (
            patch(
                "src.pipelines.spanish_direct._synthesize_spanish",
                new=AsyncMock(return_value=fake_pcm),
            ),
            patch(
                "src.pipelines.spanish_direct._translate_audio_sync",
                return_value="Hola.",
            ),
        ):
            es_iter = p.iter_stream("es-transcript", input_stream())
            assert es_iter is not None

            audio_task = asyncio.create_task(
                _drain_async_iter(p.process(input_stream()))
            )

            lines: list[str] = []
            async for line in es_iter:
                lines.append(line)

            await audio_task

        assert len(lines) >= 1
        assert all(isinstance(line, str) for line in lines)

    async def test_iter_stream_unknown_returns_none(self) -> None:
        p = SpanishDirectPipeline()

        async def input_stream() -> AsyncIterator[bytes]:
            return
            yield  # noqa: F841

        assert p.iter_stream("unknown", input_stream()) is None

    async def test_empty_stream_yields_nothing(self) -> None:
        p = SpanishDirectPipeline()
        processor, model = _make_mock_processor_and_model()
        p._processor = processor
        p._model = model

        async def input_stream() -> AsyncIterator[bytes]:
            return
            yield  # noqa: F841

        with patch(
            "src.pipelines.spanish_direct._translate_audio_sync",
            return_value="Hola mundo.",
        ):
            output: list[bytes] = []
            async for chunk in p.process(input_stream()):
                output.append(chunk)

        assert output == []

    async def test_start_loads_model(self) -> None:
        p = SpanishDirectPipeline()
        mock_processor = MagicMock()
        mock_model = MagicMock()
        with patch.object(
            p, "_load_model", return_value=(mock_processor, mock_model)
        ):
            await p.start()
        assert p._processor is mock_processor
        assert p._model is mock_model

    async def test_stop_releases_model(self) -> None:
        p = SpanishDirectPipeline()
        p._processor = MagicMock()
        p._model = MagicMock()
        await p.stop()
        assert p._processor is None
        assert p._model is None

    def test_update_context_empty(self) -> None:
        context = np.array([], dtype=np.float32)
        segment = np.ones(1000, dtype=np.float32)
        result = SpanishDirectPipeline._update_context(context, segment, 500)
        assert len(result) == 500

    def test_update_context_trims(self) -> None:
        context = np.ones(300, dtype=np.float32)
        segment = np.ones(400, dtype=np.float32)
        result = SpanishDirectPipeline._update_context(
            context, segment, 500
        )
        assert len(result) == 500

    def test_update_context_zero_disabled(self) -> None:
        context = np.ones(300, dtype=np.float32)
        segment = np.ones(400, dtype=np.float32)
        result = SpanishDirectPipeline._update_context(context, segment, 0)
        assert len(result) == 0

    async def test_context_audio_fed_to_model(self) -> None:
        """When context is enabled, context+segment audio is concatenated for the model."""
        p = SpanishDirectPipeline(sample_rate=16000)
        processor, model = _make_mock_processor_and_model()
        p._processor = processor
        p._model = model
        p._audio_context_seconds = 3.0

        fake_pcm = b"\x00\x01" * 100
        captured_audio: list[np.ndarray] = []

        def mock_translate(proc, mdl, audio):
            captured_audio.append(audio)
            return "Hola."

        async def input_stream() -> AsyncIterator[bytes]:
            # Two segments worth of data
            yield _make_audio_chunk(n_samples=16000 * 4, sample_rate=16000)
            yield _make_audio_chunk(n_samples=16000 * 4, sample_rate=16000)

        with (
            patch(
                "src.pipelines.spanish_direct._synthesize_spanish",
                new=AsyncMock(return_value=fake_pcm),
            ),
            patch(
                "src.pipelines.spanish_direct._translate_audio_sync",
                side_effect=mock_translate,
            ),
        ):
            async for _ in p.process(input_stream()):
                pass

        # First call: segment only (no context yet)
        # Second call: context + segment (longer audio)
        assert len(captured_audio) == 2
        assert len(captured_audio[1]) > len(captured_audio[0])



class TestDecodeTokens:
    def test_filters_special_and_oov(self) -> None:
        processor = MagicMock()
        tok = MagicMock()
        processor.tokenizer = tok
        tok.all_special_ids = [0, 1, 2]
        tok.sp_model.get_piece_size.return_value = 100
        tok.fairseq_offset = 4
        tok.decode.return_value = "  hello world  "

        result = _decode_tokens(processor, [0, 1, 50, 105, 60])
        # Filters: 0,1 (special), 105 (>=104 sp_limit) → keeps [50, 60]
        tok.decode.assert_called_once_with([50, 60], skip_special_tokens=False)
        assert result == "hello world"


class TestSynthesizeSpanish:
    async def test_empty_text_returns_empty(self) -> None:
        from src.pipelines.spanish_direct import _synthesize_spanish

        assert await _synthesize_spanish("", 48000) == b""
        assert await _synthesize_spanish("   ", 48000) == b""

    async def test_edge_tts_exception_returns_empty(self) -> None:
        from src.pipelines.spanish_direct import _synthesize_spanish

        mock_communicate = MagicMock()
        mock_communicate.stream.side_effect = RuntimeError("boom")
        with patch("edge_tts.Communicate", return_value=mock_communicate):
            result = await _synthesize_spanish("Hola", 48000)

        assert result == b""


async def _drain_async_iter(it: AsyncIterator[bytes]) -> list[bytes]:
    result: list[bytes] = []
    async for item in it:
        result.append(item)
    return result
