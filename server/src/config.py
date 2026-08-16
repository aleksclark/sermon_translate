from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class IceServerConfig:
    urls: list[str]
    username: str | None = None
    credential: str | None = None


@dataclass(frozen=True)
class Settings:
    ice_stun_urls: list[str] = field(default_factory=list)
    turn_urls: list[str] = field(default_factory=list)
    turn_username: str | None = None
    turn_credential: str | None = None

    crosstalk_base_url: str = ""
    crosstalk_username: str = ""
    crosstalk_password: str = ""
    crosstalk_allow_private_hosts: bool = False
    crosstalk_request_timeout: float = 10.0

    compute_device: str = "cpu"
    compute_type: str = ""

    def resolved_compute_type(self) -> str:
        if self.compute_type:
            return self.compute_type
        return "float16" if self.compute_device.startswith("cuda") else "int8"

    def ice_servers(self) -> list[IceServerConfig]:
        servers: list[IceServerConfig] = []
        if self.ice_stun_urls:
            servers.append(IceServerConfig(urls=list(self.ice_stun_urls)))
        if self.turn_urls:
            servers.append(
                IceServerConfig(
                    urls=list(self.turn_urls),
                    username=self.turn_username,
                    credential=self.turn_credential,
                )
            )
        return servers


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def load_settings() -> Settings:
    stun = os.environ.get("ICE_STUN_URLS", "stun:stun.l.google.com:19302")
    return Settings(
        ice_stun_urls=_split_csv(stun),
        turn_urls=_split_csv(os.environ.get("TURN_URLS", "")),
        turn_username=os.environ.get("TURN_USERNAME") or None,
        turn_credential=os.environ.get("TURN_CREDENTIAL") or None,
        crosstalk_base_url=os.environ.get("CROSSTALK_BASE_URL", ""),
        crosstalk_username=os.environ.get("CROSSTALK_USERNAME", ""),
        crosstalk_password=os.environ.get("CROSSTALK_PASSWORD", ""),
        crosstalk_allow_private_hosts=_bool_env("CROSSTALK_ALLOW_PRIVATE_HOSTS", False),
        crosstalk_request_timeout=_float_env("CROSSTALK_REQUEST_TIMEOUT", 10.0),
        compute_device=os.environ.get("COMPUTE_DEVICE", "cpu").strip() or "cpu",
        compute_type=os.environ.get("COMPUTE_TYPE", "").strip(),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()
