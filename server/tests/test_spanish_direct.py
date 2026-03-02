from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from src.models import Session
from src.pipelines.spanish_direct import (
    SpanishDirectPipeline,
    _translate_with_context_sync,
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


class TestTranslateWithContext:
    def test_strips_context_prefix(self) -> None:
        processor = MagicMock()
        model = MagicMock()

        call_count = 0

        def mock_translate(p, m, audio):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Buenos días mundo"
            return "Buenos días"

        with patch(
            "src.pipelines.spanish_direct._translate_audio_sync",
            side_effect=mock_translate,
        ):
            result = _translate_with_context_sync(
                processor,
                model,
                np.ones(100, dtype=np.float32),
                np.ones(200, dtype=np.float32),
            )

        assert result == "mundo"

    def test_no_context_delegates(self) -> None:
        processor = MagicMock()
        model = MagicMock()

        with patch(
            "src.pipelines.spanish_direct._translate_audio_sync",
            return_value="Hola mundo",
        ) as mock:
            result = _translate_with_context_sync(
                processor,
                model,
                np.array([], dtype=np.float32),
                np.ones(200, dtype=np.float32),
            )

        assert result == "Hola mundo"
        mock.assert_called_once()

    def test_fallback_when_no_overlap(self) -> None:
        processor = MagicMock()
        model = MagicMock()

        call_count = 0

        def mock_translate(p, m, audio):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Completely different"
            return "Something else"

        with patch(
            "src.pipelines.spanish_direct._translate_audio_sync",
            side_effect=mock_translate,
        ):
            result = _translate_with_context_sync(
                processor,
                model,
                np.ones(100, dtype=np.float32),
                np.ones(200, dtype=np.float32),
            )

        assert result == "Completely different"


async def _drain_async_iter(it: AsyncIterator[bytes]) -> list[bytes]:
    result: list[bytes] = []
    async for item in it:
        result.append(item)
    return result
