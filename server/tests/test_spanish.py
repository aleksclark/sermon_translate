from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from src.pipelines.spanish import (
    SpanishTranslationPipeline,
    _translate_sync,
)


def _make_audio_chunk(n_samples: int = 4096, sample_rate: int = 48000) -> bytes:
    t = np.linspace(0, n_samples / sample_rate, n_samples, dtype=np.float32)
    tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    return tone.tobytes()


def _make_mock_whisper(texts: list[str] | None = None):
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


def _make_mock_translator(mapping: dict[str, str] | None = None):
    if mapping is None:
        mapping = {"Hello world.": "Hola mundo."}

    def translate_batch(token_lists):
        result = MagicMock()
        result.hypotheses = [token_lists[0]]
        return [result]

    translator = MagicMock()
    translator.translate_batch = translate_batch
    return translator, mapping


def _make_mock_sp(mapping: dict[str, str] | None = None):
    if mapping is None:
        mapping = {}
    sp = MagicMock()
    sp.encode.return_value = ["mock_token"]
    sp.decode.return_value = "Hola mundo."
    return sp


class TestSpanishTranslationPipeline:
    def test_info(self) -> None:
        p = SpanishTranslationPipeline()
        assert p.info.id == "spanish-translation"
        assert p.info.name == "Spanish Translation"
        assert p.info.description

    async def test_process_produces_audio(self) -> None:
        p = SpanishTranslationPipeline(sample_rate=16000)
        p._whisper_model = _make_mock_whisper(["Hello world."])
        p._translator = MagicMock()
        p._sp_source = _make_mock_sp()
        p._sp_target = _make_mock_sp()

        fake_pcm = b"\x00\x01" * 1000

        async def input_stream() -> AsyncIterator[bytes]:
            yield _make_audio_chunk(n_samples=16000 * 4, sample_rate=16000)

        with patch(
            "src.pipelines.spanish._synthesize_spanish",
            new=AsyncMock(return_value=fake_pcm),
        ):
            output: list[bytes] = []
            async for chunk in p.process(input_stream()):
                output.append(chunk)

        assert len(output) >= 1
        assert all(isinstance(c, bytes) and len(c) > 0 for c in output)

    async def test_iter_stream_yields_transcripts(self) -> None:
        p = SpanishTranslationPipeline(sample_rate=16000)
        p._whisper_model = _make_mock_whisper(["Hello."])
        p._translator = MagicMock()
        p._sp_source = _make_mock_sp()
        p._sp_target = _make_mock_sp()

        fake_pcm = b"\x00\x01" * 100

        async def input_stream() -> AsyncIterator[bytes]:
            yield _make_audio_chunk(n_samples=16000 * 4, sample_rate=16000)

        with patch(
            "src.pipelines.spanish._synthesize_spanish",
            new=AsyncMock(return_value=fake_pcm),
        ):
            en_iter = p.iter_stream("en-transcript", input_stream())
            es_iter = p.iter_stream("es-transcript", input_stream())
            assert en_iter is not None
            assert es_iter is not None

            audio_task = asyncio.create_task(_drain_async_iter(p.process(input_stream())))

            en_lines: list[str] = []
            async for line in en_iter:
                en_lines.append(line)

            es_lines: list[str] = []
            async for line in es_iter:
                es_lines.append(line)

            await audio_task

        assert len(en_lines) >= 1
        assert len(es_lines) >= 1
        assert all("[EN]" not in line for line in en_lines)
        assert all("[ES]" not in line for line in es_lines)

    async def test_empty_stream_yields_nothing(self) -> None:
        p = SpanishTranslationPipeline()
        p._whisper_model = _make_mock_whisper()
        p._translator = MagicMock()
        p._sp_source = _make_mock_sp()
        p._sp_target = _make_mock_sp()

        async def input_stream() -> AsyncIterator[bytes]:
            return
            yield  # noqa: F841

        output: list[bytes] = []
        async for chunk in p.process(input_stream()):
            output.append(chunk)

        assert output == []

    async def test_start_loads_models(self) -> None:
        p = SpanishTranslationPipeline()
        mock_whisper = MagicMock()
        mock_translator = MagicMock()
        mock_sp_src = MagicMock()
        mock_sp_tgt = MagicMock()

        with (
            patch.object(p, "_load_whisper", return_value=mock_whisper),
            patch.object(
                p, "_load_translation", return_value=(mock_translator, mock_sp_src, mock_sp_tgt)
            ),
        ):
            await p.start()

        assert p._whisper_model is mock_whisper
        assert p._translator is mock_translator

    async def test_stop_releases_models(self) -> None:
        p = SpanishTranslationPipeline()
        p._whisper_model = MagicMock()
        p._translator = MagicMock()
        p._sp_source = MagicMock()
        p._sp_target = MagicMock()
        await p.stop()
        assert p._whisper_model is None
        assert p._translator is None
        assert p._sp_source is None
        assert p._sp_target is None

    def test_translate_sync(self) -> None:
        translator = MagicMock()
        sp_source = MagicMock()
        sp_target = MagicMock()

        sp_source.encode.return_value = ["▁Hello", "▁world", "."]
        result_mock = MagicMock()
        result_mock.hypotheses = [["▁Hola", "▁mundo", "."]]
        translator.translate_batch.return_value = [result_mock]
        sp_target.decode.return_value = "Hola mundo."

        result = _translate_sync(translator, sp_source, sp_target, "Hello world.")
        assert result == "Hola mundo."
        sp_source.encode.assert_called_once()
        translator.translate_batch.assert_called_once()


async def _drain_async_iter(it: AsyncIterator[bytes]) -> list[bytes]:
    result: list[bytes] = []
    async for item in it:
        result.append(item)
    return result
