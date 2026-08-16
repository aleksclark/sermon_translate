from __future__ import annotations

from src.runtime.protocol import (
    WorkerMessage,
    WorkerMessageType,
    b64_to_pcm,
    parse_remote_urls,
    pcm_to_b64,
)


class TestWorkerProtocol:
    def test_round_trip_control(self) -> None:
        message = WorkerMessage(
            type=WorkerMessageType.HELLO,
            stage_id="passthrough-listen",
            session_id="abc",
            config={"sample_rate": 16000},
        )
        restored = WorkerMessage.decode(message.encode())
        assert restored == message

    def test_pcm_base64_round_trip(self) -> None:
        pcm = b"\x00\x01\x02\xff"
        assert b64_to_pcm(pcm_to_b64(pcm)) == pcm

    def test_parse_remote_urls(self) -> None:
        urls = parse_remote_urls(
            '{"passthrough-listen":"ws://127.0.0.1:9001/ws","passthrough-speak":"ws://host/ws"}'
        )
        assert urls["passthrough-listen"] == "ws://127.0.0.1:9001/ws"
        assert parse_remote_urls("") == {}
