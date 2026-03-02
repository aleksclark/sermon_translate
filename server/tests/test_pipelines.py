from __future__ import annotations

from src.pipelines import EchoPipeline, PipelineRegistry, create_default_registry


class TestEchoPipeline:
    def test_info(self) -> None:
        p = EchoPipeline()
        assert p.info.id == "echo"
        assert p.info.name
        assert p.info.description


class TestPipelineRegistry:
    def test_register_and_get(self) -> None:
        reg = PipelineRegistry()
        pipeline = EchoPipeline()
        reg.register(pipeline)
        assert reg.get("echo") is pipeline
        assert reg.get("missing") is None

    def test_list_all(self) -> None:
        reg = PipelineRegistry()
        reg.register(EchoPipeline())
        infos = reg.list_all()
        assert len(infos) == 1
        assert infos[0].id == "echo"

    def test_len(self) -> None:
        reg = PipelineRegistry()
        assert len(reg) == 0
        reg.register(EchoPipeline())
        assert len(reg) == 1

    def test_default_registry(self) -> None:
        reg = create_default_registry()
        assert len(reg) >= 3
        assert reg.get("echo") is not None
        assert reg.get("whisper-tts") is not None
        assert reg.get("spanish-translation") is not None
