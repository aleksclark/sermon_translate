from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

SEAMLESS_STREAMING_MODULE = "src.pipelines.seamless_streaming"


def _make_audio_chunk(n_samples: int = 4096, sample_rate: int = 48000) -> bytes:
    t = np.linspace(0, n_samples / sample_rate, n_samples, dtype=np.float32)
    tone = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    return tone.tobytes()


@pytest.fixture()
def _mock_seamless(monkeypatch: pytest.MonkeyPatch):
    """Patch the heavy seamless_communication / simuleval imports."""
    mock_agent = MagicMock()
    mock_states = [MagicMock()]
    mock_agent.build_states.return_value = mock_states

    mock_output = MagicMock()
    mock_output.is_empty = False
    mock_output.content = "Hola"
    mock_output.finished = False
    mock_agent.pushpop.return_value = mock_output

    monkeypatch.setattr(
        f"{SEAMLESS_STREAMING_MODULE}._load_agent",
        MagicMock(return_value=mock_agent),
    )
    return mock_agent, mock_states


class TestSeamlessStreamingPipeline:
    def test_info(self, _mock_seamless) -> None:
        from src.pipelines.seamless_streaming import SeamlessStreamingPipeline

        p = SeamlessStreamingPipeline()
        assert p.info.id == "seamless-streaming"
        assert "SeamlessStreaming" in p.info.name
        assert len(p.info.output_streams) == 2

    def test_output_streams(self, _mock_seamless) -> None:
        from src.pipelines.seamless_streaming import SeamlessStreamingPipeline

        p = SeamlessStreamingPipeline()
        names = [s.name for s in p.output_streams]
        assert "audio" in names
        assert "es-transcript" in names

    async def test_start_loads_agent(self, _mock_seamless) -> None:
        from src.pipelines.seamless_streaming import SeamlessStreamingPipeline

        mock_agent, mock_states = _mock_seamless
        p = SeamlessStreamingPipeline()
        await p.start()
        assert p._agent is mock_agent

    async def test_start_idempotent(self, _mock_seamless) -> None:
        from src.pipelines.seamless_streaming import SeamlessStreamingPipeline

        mock_agent, _ = _mock_seamless
        p = SeamlessStreamingPipeline()
        await p.start()
        await p.start()
        from src.pipelines.seamless_streaming import _load_agent

        _load_agent.assert_called_once()

    async def test_stop_resets_state(self, _mock_seamless) -> None:
        from src.pipelines.seamless_streaming import SeamlessStreamingPipeline

        p = SeamlessStreamingPipeline()
        await p.start()
        await p.stop()
        assert p._agent is None

    async def test_process_produces_audio(self, _mock_seamless) -> None:
        from src.pipelines.seamless_streaming import SeamlessStreamingPipeline

        p = SeamlessStreamingPipeline(sample_rate=16000)
        await p.start()

        fake_pcm = b"\x00\x01" * 1000

        async def input_stream() -> AsyncIterator[bytes]:
            yield _make_audio_chunk(n_samples=16000 * 1, sample_rate=16000)

        with (
            patch(
                f"{SEAMLESS_STREAMING_MODULE}._synthesize_spanish",
                new=AsyncMock(return_value=fake_pcm),
            ),
            patch(
                f"{SEAMLESS_STREAMING_MODULE}._push_chunk_sync",
                return_value="Hola mundo",
            ),
        ):
            output: list[bytes] = []
            async for chunk in p.process(input_stream()):
                output.append(chunk)

        assert len(output) >= 1
        assert all(isinstance(c, bytes) and len(c) > 0 for c in output)

    async def test_iter_stream_yields_transcript(self, _mock_seamless) -> None:
        from src.pipelines.seamless_streaming import SeamlessStreamingPipeline

        p = SeamlessStreamingPipeline(sample_rate=16000)
        await p.start()

        fake_pcm = b"\x00\x01" * 100

        async def input_stream() -> AsyncIterator[bytes]:
            yield _make_audio_chunk(n_samples=16000 * 1, sample_rate=16000)

        with (
            patch(
                f"{SEAMLESS_STREAMING_MODULE}._synthesize_spanish",
                new=AsyncMock(return_value=fake_pcm),
            ),
            patch(
                f"{SEAMLESS_STREAMING_MODULE}._push_chunk_sync",
                return_value="Hola mundo",
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

    async def test_iter_stream_unknown_returns_none(self, _mock_seamless) -> None:
        from src.pipelines.seamless_streaming import SeamlessStreamingPipeline

        p = SeamlessStreamingPipeline()

        async def input_stream() -> AsyncIterator[bytes]:
            return
            yield  # noqa: F841

        assert p.iter_stream("unknown", input_stream()) is None

    async def test_empty_stream_yields_nothing(self, _mock_seamless) -> None:
        from src.pipelines.seamless_streaming import SeamlessStreamingPipeline

        p = SeamlessStreamingPipeline()
        await p.start()

        mock_agent, _ = _mock_seamless
        mock_output = MagicMock()
        mock_output.is_empty = True
        mock_agent.pushpop.return_value = mock_output

        async def input_stream() -> AsyncIterator[bytes]:
            return
            yield  # noqa: F841

        output: list[bytes] = []
        async for chunk in p.process(input_stream()):
            output.append(chunk)

        assert output == []

    async def test_states_reset_after_stream_ends(self, _mock_seamless) -> None:
        from src.pipelines.seamless_streaming import SeamlessStreamingPipeline

        mock_agent, mock_states = _mock_seamless
        p = SeamlessStreamingPipeline(sample_rate=16000)
        await p.start()

        async def input_stream() -> AsyncIterator[bytes]:
            yield _make_audio_chunk(n_samples=16000, sample_rate=16000)

        with (
            patch(
                f"{SEAMLESS_STREAMING_MODULE}._synthesize_spanish",
                new=AsyncMock(return_value=b"\x00\x01" * 100),
            ),
            patch(
                f"{SEAMLESS_STREAMING_MODULE}._push_chunk_sync",
                return_value="Hola",
            ),
        ):
            async for _ in p.process(input_stream()):
                pass

        mock_agent.build_states.assert_called()


class TestPushChunkSync:
    def _install_fake_simuleval(self) -> MagicMock:
        """Inject fake simuleval into sys.modules so lazy imports resolve."""
        mock_speech_segment_cls = MagicMock()
        segments_mod = MagicMock(SpeechSegment=mock_speech_segment_cls)
        data_mod = MagicMock(segments=segments_mod)
        simuleval_mod = MagicMock(data=data_mod)
        sys.modules.setdefault("simuleval", simuleval_mod)
        sys.modules.setdefault("simuleval.data", data_mod)
        sys.modules.setdefault("simuleval.data.segments", segments_mod)
        return mock_speech_segment_cls

    def _cleanup_fake_simuleval(self) -> None:
        for key in ["simuleval", "simuleval.data", "simuleval.data.segments"]:
            sys.modules.pop(key, None)

    def test_returns_text_on_non_empty(self, _mock_seamless) -> None:
        mock_agent, mock_states = _mock_seamless
        mock_torch = MagicMock()

        self._install_fake_simuleval()
        try:
            with patch(f"{SEAMLESS_STREAMING_MODULE}.torch", mock_torch, create=True):
                from src.pipelines.seamless_streaming import _push_chunk_sync

                result = _push_chunk_sync(
                    mock_agent, mock_states, np.zeros(100, dtype=np.float32), "spa",
                )
        finally:
            self._cleanup_fake_simuleval()

        assert result == "Hola"

    def test_returns_none_on_empty(self, _mock_seamless) -> None:
        mock_agent, mock_states = _mock_seamless
        empty_output = MagicMock()
        empty_output.is_empty = True
        mock_agent.pushpop.return_value = empty_output
        mock_torch = MagicMock()

        self._install_fake_simuleval()
        try:
            with patch(f"{SEAMLESS_STREAMING_MODULE}.torch", mock_torch, create=True):
                from src.pipelines.seamless_streaming import _push_chunk_sync

                result = _push_chunk_sync(
                    mock_agent, mock_states, np.zeros(100, dtype=np.float32), "spa",
                )
        finally:
            self._cleanup_fake_simuleval()

        assert result is None


class TestBuildAgentArgs:
    def test_cpu_args(self) -> None:
        from src.pipelines.seamless_streaming import _build_agent_args

        args = _build_agent_args("spa", "cpu")
        assert args.tgt_lang == "spa"
        assert args.device == "cpu"
        assert args.dtype == "fp32"
        assert args.fp16 is False
        assert args.task == "s2tt"

    def test_cuda_args(self) -> None:
        from src.pipelines.seamless_streaming import _build_agent_args

        args = _build_agent_args("fra", "cuda:0")
        assert args.tgt_lang == "fra"
        assert args.device == "cuda:0"
        assert args.dtype == "fp16"
        assert args.fp16 is True


class TestSynthesizeSpanish:
    async def test_empty_text_returns_empty(self) -> None:
        from src.pipelines.seamless_streaming import _synthesize_spanish

        assert await _synthesize_spanish("", 48000) == b""
        assert await _synthesize_spanish("   ", 48000) == b""

    async def test_edge_tts_exception_returns_empty(self) -> None:
        from src.pipelines.seamless_streaming import _synthesize_spanish

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
