from __future__ import annotations

from src.config import IceServerConfig, Settings
from src.transport.ice import build_ice_servers, build_rtc_configuration


class TestSettingsIce:
    def test_ice_servers_include_stun_and_turn(self) -> None:
        settings = Settings(
            ice_stun_urls=["stun:stun.example:3478"],
            turn_urls=["turn:turn.example:3478"],
            turn_username="u",
            turn_credential="c",
        )
        servers = settings.ice_servers()
        assert servers[0] == IceServerConfig(urls=["stun:stun.example:3478"])
        assert servers[1].username == "u"
        assert servers[1].credential == "c"

    def test_empty_when_unconfigured(self) -> None:
        assert Settings().ice_servers() == []

    def test_build_rtc_configuration_not_empty(self) -> None:
        settings = Settings(ice_stun_urls=["stun:stun.example:3478"])
        config = build_rtc_configuration(settings)
        assert config.iceServers
        assert config.iceServers[0].urls == ["stun:stun.example:3478"]

    def test_build_ice_servers_maps_credentials(self) -> None:
        servers = build_ice_servers(
            [IceServerConfig(urls=["turn:t:3478"], username="u", credential="c")]
        )
        assert servers[0].username == "u"
        assert servers[0].credential == "c"
