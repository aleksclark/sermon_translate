from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Settings, default_model_cache_dir, load_settings


class TestComputeDevice:
    def test_defaults_to_cpu_int8(self) -> None:
        settings = Settings()
        assert settings.compute_device == "cpu"
        assert settings.resolved_compute_type() == "int8"

    def test_cuda_defaults_to_float16(self) -> None:
        settings = Settings(compute_device="cuda")
        assert settings.resolved_compute_type() == "float16"

    def test_explicit_compute_type_wins(self) -> None:
        settings = Settings(compute_device="cuda", compute_type="int8_float16")
        assert settings.resolved_compute_type() == "int8_float16"

    def test_load_settings_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COMPUTE_DEVICE", "cuda:1")
        monkeypatch.setenv("COMPUTE_TYPE", "float16")
        settings = load_settings()
        assert settings.compute_device == "cuda:1"
        assert settings.compute_type == "float16"
        assert settings.resolved_compute_type() == "float16"

    def test_load_settings_blank_env_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("COMPUTE_DEVICE", "")
        monkeypatch.delenv("COMPUTE_TYPE", raising=False)
        settings = load_settings()
        assert settings.compute_device == "cpu"
        assert settings.resolved_compute_type() == "int8"

    def test_model_cache_dir_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MODEL_CACHE_DIR", raising=False)
        settings = load_settings()
        assert settings.model_cache_dir == default_model_cache_dir()
        assert settings.stage_runtime == "local"

    def test_model_cache_dir_from_env(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "shared-models"))
        monkeypatch.setenv("STAGE_RUNTIME", "subprocess")
        settings = load_settings()
        assert settings.model_cache_dir == tmp_path / "shared-models"
        assert settings.stage_runtime == "subprocess"
