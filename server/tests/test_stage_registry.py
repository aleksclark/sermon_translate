from __future__ import annotations

from src.models import StageKind
from src.pipelines import create_default_stage_registry
from src.pipelines.stage_registry import StageRegistry


class TestStageRegistry:
    def test_register_list_and_get(self) -> None:
        registry = create_default_stage_registry()
        assert len(registry) >= 4
        listen = registry.get("passthrough-listen")
        assert listen is not None
        assert listen.info.kind == StageKind.LISTEN

    def test_list_filter_by_kind(self) -> None:
        registry = create_default_stage_registry()
        listen_stages = registry.list_all(StageKind.LISTEN)
        assert listen_stages
        assert all(s.kind == StageKind.LISTEN for s in listen_stages)
        ids = {s.id for s in listen_stages}
        assert "passthrough-listen" in ids
        assert "whisper-listen" in ids

    def test_missing_stage(self) -> None:
        registry = StageRegistry()
        assert registry.get("missing") is None
        assert registry.list_all() == []

    def test_default_factories_create_instances(self) -> None:
        registry = create_default_stage_registry()
        for stage_id in (
            "passthrough-listen",
            "passthrough-translate",
            "passthrough-speak",
            "baseline-prosody",
        ):
            factory = registry.get(stage_id)
            assert factory is not None
            instance = factory.create(sample_rate=16000)
            assert instance is not None
