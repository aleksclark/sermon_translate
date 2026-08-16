from __future__ import annotations

from pathlib import Path

import pytest

from src.models import Session, StageKind, StageSelection
from src.pipelines import create_default_stage_registry
from src.runtime.local import LocalStageRuntime
from src.runtime.model_cache import ModelCache


class TestModelCache:
    def test_path_for_creates_parents(self, tmp_path: Path) -> None:
        cache = ModelCache(tmp_path / "models")
        target = cache.path_for("custom", "passthrough-listen", "weights.bin")
        assert target.parent.is_dir()
        assert target.parent == (tmp_path / "models" / "custom" / "passthrough-listen").resolve()
        assert target.name == "weights.bin"

    def test_rejects_path_escape(self, tmp_path: Path) -> None:
        cache = ModelCache(tmp_path / "models")
        with pytest.raises(ValueError, match="invalid path part"):
            cache.path_for("..", "etc")
        with pytest.raises(ValueError, match="invalid path part"):
            cache.path_for("foo/bar")

    def test_environ_points_under_root(self, tmp_path: Path) -> None:
        cache = ModelCache(tmp_path / "models")
        env = cache.environ()
        assert env["MODEL_CACHE_DIR"] == str(cache.root)
        assert env["HF_HOME"].startswith(str(cache.root))
        assert env["TORCH_HOME"].startswith(str(cache.root))

    def test_probe_round_trip(self, tmp_path: Path) -> None:
        cache = ModelCache(tmp_path / "models")
        marker = cache.path_for("custom", "probe", "marker.txt")
        marker.write_text("ok", encoding="utf-8")
        assert marker.read_text(encoding="utf-8") == "ok"


class TestLocalStageRuntime:
    async def test_spawn_passthrough_listen(self, tmp_path: Path) -> None:
        stages = create_default_stage_registry()
        cache = ModelCache(tmp_path / "models")
        runtime = LocalStageRuntime(stages, cache)
        session = Session(
            pipeline_id="composed",
            sample_rate=16000,
            stages=StageSelection(
                listen="passthrough-listen",
                translate="passthrough-translate",
                speak="passthrough-speak",
            ),
        )
        handle = await runtime.spawn("passthrough-listen", session, kind=StageKind.LISTEN)
        assert handle.info.id == "passthrough-listen"
        await handle.start()
        await handle.stop()
