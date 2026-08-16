from __future__ import annotations

import pytest

from src.config import load_settings
from src.runtime.protocol import parse_remote_urls


def test_parse_remote_urls_object() -> None:
    urls = parse_remote_urls('{"a":"ws://x/ws","b":"ws://y/ws"}')
    assert urls == {"a": "ws://x/ws", "b": "ws://y/ws"}


def test_load_settings_remote_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAGE_RUNTIME", "remote")
    monkeypatch.setenv(
        "STAGE_REMOTE_URLS",
        '{"passthrough-listen":"ws://127.0.0.1:8101/ws"}',
    )
    settings = load_settings()
    assert settings.stage_runtime == "remote"
    assert settings.stage_remote_urls["passthrough-listen"] == "ws://127.0.0.1:8101/ws"
