from __future__ import annotations

from aiortc import RTCConfiguration, RTCIceServer

from src.config import IceServerConfig, Settings, get_settings


def build_ice_servers(configs: list[IceServerConfig]) -> list[RTCIceServer]:
    servers: list[RTCIceServer] = []
    for cfg in configs:
        servers.append(
            RTCIceServer(
                urls=list(cfg.urls),
                username=cfg.username,
                credential=cfg.credential,
            )
        )
    return servers


def build_rtc_configuration(settings: Settings | None = None) -> RTCConfiguration:
    settings = settings or get_settings()
    return RTCConfiguration(iceServers=build_ice_servers(settings.ice_servers()))
