from __future__ import annotations

import base64
import json
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class WorkerMessageType(StrEnum):
    HELLO = "hello"
    START = "start"
    STOP = "stop"
    READY = "ready"
    ERROR = "error"
    AUDIO_IN = "audio_in"
    AUDIO_OUT = "audio_out"
    LISTEN_PRODUCT = "listen_product"
    TRANSLATE_PRODUCT = "translate_product"
    METADATA = "metadata"
    EOS = "eos"


class WorkerMessage(BaseModel):
    type: WorkerMessageType
    stage_id: str | None = None
    session_id: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    message: str | None = None
    seq: int | None = None
    sample_rate: int | None = None
    channels: int | None = None
    pcm_b64: str | None = None
    product: dict[str, Any] | None = None
    envelope: dict[str, Any] | None = None

    def encode(self) -> str:
        return self.model_dump_json(exclude_none=True)

    @classmethod
    def decode(cls, raw: str | bytes) -> WorkerMessage:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return cls.model_validate_json(raw)


def pcm_to_b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


def b64_to_pcm(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def dumps_message(message: WorkerMessage) -> str:
    return message.encode()


def loads_message(raw: str | bytes) -> WorkerMessage:
    return WorkerMessage.decode(raw)


def parse_remote_urls(raw: str) -> dict[str, str]:
    text = raw.strip()
    if not text:
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("STAGE_REMOTE_URLS must be a JSON object")
    return {str(key): str(value) for key, value in data.items()}
