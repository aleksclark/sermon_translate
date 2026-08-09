"""stage.v1 connection auth and transport policy (D11 fail-closed).

Production mode requires:
- secure transport (wss / TLS termination via trusted reverse-proxy headers)
- bearer/workload token from HTTP upgrade headers before session allocation

Dev/test mode may allow loopback credential-free upgrades.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from src.stage_v1.models import StageErrorCode

STAGE_V1_SUBPROTOCOL = "stage.v1"


class StageV1Mode(StrEnum):
    PRODUCTION = "production"
    DEV = "dev"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class StageV1AuthConfig:
    """Resolved auth/transport policy for a worker process."""

    mode: StageV1Mode
    auth_token: str
    trust_proxy: bool = False
    allow_loopback_without_auth: bool = True

    @property
    def is_production(self) -> bool:
        return self.mode == StageV1Mode.PRODUCTION

    def validate_boot(self) -> None:
        """Fail closed at process start when production config is incomplete."""
        if self.is_production and not self.auth_token:
            raise RuntimeError(
                "STAGE_V1_MODE=production requires non-empty STAGE_AUTH_TOKEN "
                "(fail-closed: refuse to start without workload credentials)"
            )


@dataclass(frozen=True, slots=True)
class AuthDecision:
    ok: bool
    code: StageErrorCode | None = None
    message: str = ""
    close_code: int = 1008


def parse_stage_v1_mode(raw: str | None) -> StageV1Mode:
    value = (raw or "dev").strip().lower()
    if value in {"production", "prod"}:
        return StageV1Mode.PRODUCTION
    if value in {"test", "testing"}:
        return StageV1Mode.TEST
    if value in {"dev", "development", "local"}:
        return StageV1Mode.DEV
    # Unknown values fail closed to production semantics only when explicitly
    # requested; otherwise treat as dev for local ergonomics.
    if value in {"", "auto"}:
        return StageV1Mode.DEV
    # Explicit unknown → production-safe refusal at boot if token missing.
    return StageV1Mode.PRODUCTION


def load_stage_v1_auth_config(
    *,
    environ: Mapping[str, str] | None = None,
) -> StageV1AuthConfig:
    env = environ if environ is not None else os.environ
    mode = parse_stage_v1_mode(env.get("STAGE_V1_MODE"))
    token = (env.get("STAGE_AUTH_TOKEN") or "").strip()
    trust_proxy = _truthy(env.get("STAGE_TRUST_PROXY"), default=False)
    allow_loopback = _truthy(
        env.get("STAGE_ALLOW_LOOPBACK_NO_AUTH"),
        default=mode != StageV1Mode.PRODUCTION,
    )
    if mode == StageV1Mode.PRODUCTION:
        allow_loopback = False
    return StageV1AuthConfig(
        mode=mode,
        auth_token=token,
        trust_proxy=trust_proxy,
        allow_loopback_without_auth=allow_loopback,
    )


def extract_bearer_token(headers: Mapping[str, str]) -> str | None:
    """Extract credential from Authorization: Bearer or X-Stage-Auth."""
    normalized = {_norm_header(k): v for k, v in headers.items()}
    auth = normalized.get("authorization")
    if auth:
        parts = auth.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
            return parts[1].strip()
        # Accept raw token in Authorization only when not a scheme-bearing value.
        if len(parts) == 1 and parts[0] and ":" not in parts[0]:
            return parts[0].strip()
    stage_auth = normalized.get("x-stage-auth")
    if stage_auth and stage_auth.strip():
        return stage_auth.strip()
    return None


def is_loopback_client(client_host: str | None) -> bool:
    if not client_host:
        return False
    host = client_host.strip().lower()
    return host in {"127.0.0.1", "::1", "localhost", "testclient"}


def is_secure_transport(
    *,
    url_scheme: str | None,
    headers: Mapping[str, str],
    trust_proxy: bool,
) -> bool:
    """Return True when the connection is wss/TLS or trusted proxy TLS."""
    scheme = (url_scheme or "").strip().lower()
    if scheme in {"wss", "https"}:
        return True
    if not trust_proxy:
        return False
    normalized = {_norm_header(k): v for k, v in headers.items()}
    forwarded = (normalized.get("x-forwarded-proto") or "").strip().lower()
    if forwarded in {"https", "wss"}:
        return True
    # Some proxies set multiple values: "https,http"
    return any(part.strip() in {"https", "wss"} for part in forwarded.split(","))


def authorize_stage_upgrade(
    *,
    config: StageV1AuthConfig,
    headers: Mapping[str, str],
    url_scheme: str | None,
    client_host: str | None,
) -> AuthDecision:
    """Authorize WebSocket upgrade before accept/session/model allocation.

    Production:
      - require secure transport (wss or trusted X-Forwarded-Proto)
      - require matching bearer/workload token
      - empty configured token → AUTHENTICATION_FAILED (fail closed)
    Dev/test:
      - loopback without credentials allowed when configured
      - non-loopback still requires token when one is configured
    """
    if config.is_production:
        if not config.auth_token:
            return AuthDecision(
                ok=False,
                code=StageErrorCode.AUTHENTICATION_FAILED,
                message="production mode requires STAGE_AUTH_TOKEN",
            )
        if not is_secure_transport(
            url_scheme=url_scheme,
            headers=headers,
            trust_proxy=config.trust_proxy,
        ):
            return AuthDecision(
                ok=False,
                code=StageErrorCode.AUTHENTICATION_FAILED,
                message=(
                    "production requires wss/TLS "
                    "(or trusted reverse-proxy X-Forwarded-Proto=https)"
                ),
            )
        provided = extract_bearer_token(headers)
        if provided is None:
            return AuthDecision(
                ok=False,
                code=StageErrorCode.AUTHENTICATION_FAILED,
                message="missing Authorization Bearer or X-Stage-Auth credential",
            )
        if not hmac.compare_digest(provided, config.auth_token):
            return AuthDecision(
                ok=False,
                code=StageErrorCode.AUTHENTICATION_FAILED,
                message="invalid stage credential",
            )
        return AuthDecision(ok=True)

    # Dev / test
    provided = extract_bearer_token(headers)
    if config.auth_token:
        if provided is None:
            # Credential-free only on loopback when allowed.
            if config.allow_loopback_without_auth and is_loopback_client(client_host):
                return AuthDecision(ok=True)
            return AuthDecision(
                ok=False,
                code=StageErrorCode.AUTHENTICATION_FAILED,
                message="missing Authorization Bearer or X-Stage-Auth credential",
            )
        if not hmac.compare_digest(provided, config.auth_token):
            return AuthDecision(
                ok=False,
                code=StageErrorCode.AUTHENTICATION_FAILED,
                message="invalid stage credential",
            )
        return AuthDecision(ok=True)

    # No token configured: loopback-only when allow_loopback_without_auth.
    if config.allow_loopback_without_auth and is_loopback_client(client_host):
        return AuthDecision(ok=True)
    if is_loopback_client(client_host):
        return AuthDecision(ok=True)
    return AuthDecision(
        ok=False,
        code=StageErrorCode.AUTHENTICATION_FAILED,
        message="credential-free stage upgrades are loopback-only outside production",
    )


def headers_from_scope(scope: Mapping[str, Any]) -> dict[str, str]:
    raw_headers = scope.get("headers") or []
    out: dict[str, str] = {}
    for key, value in raw_headers:
        try:
            k = key.decode("latin-1") if isinstance(key, (bytes, bytearray)) else str(key)
            v = value.decode("latin-1") if isinstance(value, (bytes, bytearray)) else str(value)
        except Exception:
            continue
        out[k] = v
    return out


def client_host_from_scope(scope: Mapping[str, Any]) -> str | None:
    client = scope.get("client")
    if isinstance(client, (list, tuple)) and client:
        host = client[0]
        return str(host) if host is not None else None
    return None


def url_scheme_from_scope(scope: Mapping[str, Any]) -> str | None:
    scheme = scope.get("scheme")
    return str(scheme) if scheme is not None else None


def _norm_header(name: str) -> str:
    return name.strip().lower()


def _truthy(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
