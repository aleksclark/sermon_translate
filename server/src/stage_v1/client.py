"""stage.v1 full-duplex client transport.

Concurrent sender/receiver tasks, bounded queues, application window/ack
credits, per-event deadline checks, structured cancel, and STG1 binary frames
for listen.audio / speak.audio (no base64 on the stage.v1 path).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator, Awaitable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import uuid4

from src.stage_v1.framing import (
    DecodedBinaryFrame,
    FramingError,
    decode_binary_frame,
    encode_binary_frame,
    validate_text_frame_size,
)
from src.stage_v1.models import (
    BASELINE_CHANNELS,
    BASELINE_SAMPLE_RATE_HZ,
    DEFAULT_MAX_FRAME_BYTES,
    SCHEMA_VERSION,
    AcceptedPayload,
    AudioFormat,
    BinaryAudioPayload,
    CancelPayload,
    CancelScope,
    EosPayload,
    ErrorPayload,
    EventEnvelope,
    EventType,
    HelloPayload,
    LimitsRequested,
    ListenProductPayload,
    OpenPayload,
    SpeakCompletePayload,
    SpeakRequestPayload,
    StageErrorCode,
    StageKind,
    TranslateProductPayload,
    TranslateRequestPayload,
    WindowPayload,
    parse_event,
    parse_event_json,
)
from src.stage_v1.peer import CreditWindow, ScriptedStagePeer
from src.stage_v1.validation import (
    EventSequenceTracker,
    Fence,
    ValidationError,
    check_deadline,
    check_fence,
)


class StageTransport(Protocol):
    """Minimal duplex byte/text transport used by StageV1Client."""

    async def send_text(self, data: str) -> None: ...

    async def send_bytes(self, data: bytes) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


class StageClientError(Exception):
    def __init__(self, code: StageErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _utc_now_rfc3339() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _deadline_after(seconds: float) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%S.%f")[
        :-3
    ] + "Z"


def _deadline_timeout_s(deadline_at: str | None, *, fallback: float) -> float:
    if deadline_at is None:
        return fallback
    remaining = (parse_rfc3339_local(deadline_at) - datetime.now(UTC)).total_seconds()
    if remaining <= 0:
        raise StageClientError(StageErrorCode.DEADLINE_EXCEEDED, f"deadline_at={deadline_at}")
    return remaining


def parse_rfc3339_local(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass
class OutboundItem:
    envelope: EventEnvelope
    pcm: bytes | None = None
    enqueued_at: float = field(default_factory=lambda: asyncio.get_running_loop().time())


@dataclass
class InboundItem:
    envelope: EventEnvelope
    pcm: bytes | None = None


class PeerTransport:
    """Bridge ScriptedStagePeer into StageTransport for in-process tests."""

    def __init__(self, peer: ScriptedStagePeer) -> None:
        self._peer = peer
        self._closed = False

    @property
    def peer(self) -> ScriptedStagePeer:
        return self._peer

    async def send_text(self, data: str) -> None:
        if self._closed:
            raise StageClientError(StageErrorCode.INTERNAL, "transport closed")
        validate_text_frame_size(data)
        envelope = parse_event_json(data)
        await self._peer.send(envelope)

    async def send_bytes(self, data: bytes) -> None:
        if self._closed:
            raise StageClientError(StageErrorCode.INTERNAL, "transport closed")
        frame = decode_binary_frame(data, max_frame_bytes=self._peer.max_frame_bytes)
        await self._peer.send(frame.envelope, pcm=frame.pcm)

    async def recv(self) -> str | bytes:
        if self._closed:
            raise StageClientError(StageErrorCode.INTERNAL, "transport closed")
        item = await self._peer.recv(timeout=None)
        if isinstance(item, tuple):
            envelope, pcm = item
            return encode_binary_frame(envelope, pcm, max_frame_bytes=self._peer.max_frame_bytes)
        validate_text_frame_size(envelope_json := item.model_dump_json(exclude_none=True))
        return envelope_json

    async def close(self) -> None:
        self._closed = True


class WebSocketTransport:
    """Adapter around websockets ClientConnection."""

    def __init__(self, ws: Any) -> None:
        self._ws = ws

    async def send_text(self, data: str) -> None:
        validate_text_frame_size(data)
        await self._ws.send(data)

    async def send_bytes(self, data: bytes) -> None:
        await self._ws.send(data)

    async def recv(self) -> str | bytes:
        raw = await self._ws.recv()
        if isinstance(raw, (str, bytes)):
            return raw
        raise StageClientError(
            StageErrorCode.INVALID_ARGUMENT,
            f"unexpected ws payload {type(raw)}",
        )

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self._ws.close()


class StageV1Client:
    """Full-duplex stage.v1 orchestrator-side client."""

    def __init__(
        self,
        transport: StageTransport,
        *,
        session_id: str,
        stage_kind: StageKind,
        stage_id: str,
        correlation_id: str | None = None,
        owner_generation: int = 0,
        attempt_id: str | None = None,
        cancel_id: str | None = None,
        outbound_queue_size: int = 32,
        inbound_queue_size: int = 32,
        default_deadline_s: float = 30.0,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        initial_event_credits: int = 32,
        initial_byte_credits: int | None = None,
        sample_rate_hz: int = BASELINE_SAMPLE_RATE_HZ,
        channels: int = BASELINE_CHANNELS,
    ) -> None:
        self._transport = transport
        self.session_id = session_id
        self.stage_kind = stage_kind
        self.stage_id = stage_id
        self.correlation_id = correlation_id or f"corr-{uuid4()}"
        self.owner_generation = owner_generation
        self.attempt_id = attempt_id or str(uuid4())
        self.cancel_id = cancel_id or str(uuid4())
        self.default_deadline_s = default_deadline_s
        self.max_frame_bytes = max_frame_bytes
        self.sample_rate_hz = sample_rate_hz
        self.channels = channels

        self._outbound: asyncio.Queue[OutboundItem | None] = asyncio.Queue(
            maxsize=max(1, outbound_queue_size)
        )
        self._inbound: asyncio.Queue[InboundItem | None] = asyncio.Queue(
            maxsize=max(1, inbound_queue_size)
        )
        self._credit_event = asyncio.Event()
        self._credit_event.set()
        byte_credits = (
            initial_byte_credits
            if initial_byte_credits is not None
            else max_frame_bytes * initial_event_credits
        )
        self._windows: dict[str, CreditWindow] = {
            "default": CreditWindow(
                stream_id="default",
                available_events=initial_event_credits,
                available_bytes=byte_credits,
            )
        }
        self._out_seq = EventSequenceTracker()
        self._in_seq = EventSequenceTracker()
        self._next_out_seq = 0
        self._sender_task: asyncio.Task[None] | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._pump_error: BaseException | None = None
        self._closed = False
        self._started = False
        self._accepted: AcceptedPayload | None = None
        self._fence: Fence | None = None
        self._stage_version: str | None = None
        self._model_revision: str | None = None
        self._model_artifact_digest: str | None = None
        self._stage_instance_id: str | None = None
        self._provenance_id: str | None = None
        self._cancel_requested = False
        self._outbound_high_water = 0
        self._inbound_high_water = 0
        self._lock = asyncio.Lock()

    @property
    def accepted(self) -> AcceptedPayload | None:
        return self._accepted

    @property
    def fence(self) -> Fence | None:
        return self._fence

    @property
    def sender_task(self) -> asyncio.Task[None] | None:
        return self._sender_task

    @property
    def receiver_task(self) -> asyncio.Task[None] | None:
        return self._receiver_task

    @property
    def outbound_high_water(self) -> int:
        return self._outbound_high_water

    @property
    def inbound_high_water(self) -> int:
        return self._inbound_high_water

    def window(self, stream_id: str = "default") -> CreditWindow:
        if stream_id not in self._windows:
            self._windows[stream_id] = CreditWindow(
                stream_id=stream_id,
                available_events=0,
                available_bytes=0,
            )
        return self._windows[stream_id]

    async def start(self) -> AcceptedPayload:
        if self._started:
            assert self._accepted is not None
            return self._accepted
        self._sender_task = asyncio.create_task(self._sender_loop(), name="stage-v1-sender")
        self._receiver_task = asyncio.create_task(self._receiver_loop(), name="stage-v1-receiver")
        self._started = True
        hello = self._make_envelope(
            EventType.HELLO,
            HelloPayload(
                audio_formats=[
                    AudioFormat(
                        sample_rate_hz=self.sample_rate_hz,
                        channels=self.channels,
                    )
                ],
                limits_requested=LimitsRequested(
                    max_frame_bytes=self.max_frame_bytes,
                    max_inflight_events=self._outbound.maxsize,
                ),
            ).model_dump(mode="json"),
            deadline_at=_deadline_after(self.default_deadline_s),
            include_worker_identity=False,
        )
        await self._enqueue_outbound(hello)
        accepted_item = await self._recv_control(
            expected={EventType.ACCEPTED},
            timeout=self.default_deadline_s,
        )
        accepted = AcceptedPayload.model_validate(accepted_item.envelope.payload)
        self._accepted = accepted
        self._stage_version = accepted.stage_version
        self._model_revision = accepted.model_revision
        self._model_artifact_digest = accepted.model_artifact_digest
        self._stage_instance_id = accepted.stage_instance_id
        self._provenance_id = accepted.provenance_id
        self.max_frame_bytes = min(self.max_frame_bytes, accepted.limits.max_frame_bytes)
        self._fence = Fence.from_envelope(accepted_item.envelope)
        # Drain optional initial window/health that may already be queued.
        return accepted

    async def open(
        self,
        *,
        utterance_id: str | None = None,
        utterance_sequence: int | None = None,
    ) -> None:
        payload = OpenPayload(
            utterance_id=utterance_id,
            utterance_sequence=utterance_sequence,
        ).model_dump(mode="json", exclude_none=True)
        env = self._make_envelope(
            EventType.OPEN,
            payload,
            deadline_at=_deadline_after(self.default_deadline_s),
            utterance_id=utterance_id,
            utterance_sequence=utterance_sequence,
        )
        await self._enqueue_outbound(env)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(asyncio.QueueFull):
            self._outbound.put_nowait(None)
        if self._sender_task is not None:
            self._sender_task.cancel()
        if self._receiver_task is not None:
            self._receiver_task.cancel()
        tasks = [t for t in (self._sender_task, self._receiver_task) if t is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        with contextlib.suppress(Exception):
            await self._transport.close()

    async def cancel(
        self,
        *,
        scope: CancelScope = CancelScope.ATTEMPT,
        reason: str = "client_cancel",
        utterance_id: str | None = None,
    ) -> EventEnvelope:
        self._cancel_requested = True
        env = self._make_envelope(
            EventType.CANCEL,
            CancelPayload(
                scope=scope,
                reason=reason,
                utterance_id=utterance_id,
                attempt_id=self.attempt_id,
                session_id=self.session_id,
            ).model_dump(mode="json", exclude_none=True),
            deadline_at=_deadline_after(self.default_deadline_s),
            utterance_id=utterance_id,
        )
        await self._enqueue_outbound(env, require_credits=False)
        item = await self._recv_control(
            expected={EventType.CANCELLED, EventType.ERROR},
            timeout=self.default_deadline_s,
        )
        return item.envelope

    async def listen(
        self,
        audio_stream: AsyncIterator[bytes],
        *,
        stream_id: str = "source:main",
        utterance_id: str | None = None,
        utterance_sequence: int | None = 0,
        deadline_at: str | None = None,
    ) -> AsyncIterator[ListenProductPayload]:
        self._require_kind(StageKind.LISTEN)
        await self._ensure_started()
        utt = utterance_id or str(uuid4())
        utt_seq = 0 if utterance_sequence is None else utterance_sequence
        dl = deadline_at or _deadline_after(self.default_deadline_s)

        async def _produce() -> None:
            media_seq = 0
            start_sample = 0
            last_media_seq: int | None = None
            last_sample_end: int | None = None
            try:
                async for chunk in audio_stream:
                    if not chunk:
                        continue
                    check_deadline(dl)
                    sample_count = len(chunk) // (2 * self.channels)
                    payload = BinaryAudioPayload(
                        stream_id=stream_id,
                        media_sequence=media_seq,
                        start_sample=start_sample,
                        sample_count=sample_count,
                        payload_bytes=len(chunk),
                        format=AudioFormat(
                            sample_rate_hz=self.sample_rate_hz,
                            channels=self.channels,
                        ),
                    )
                    env = self._make_envelope(
                        EventType.LISTEN_AUDIO,
                        payload.model_dump(mode="json"),
                        deadline_at=dl,
                        utterance_id=utt,
                        utterance_sequence=utt_seq,
                    )
                    await self._enqueue_outbound(env, pcm=chunk, stream_id=stream_id)
                    last_media_seq = media_seq
                    last_sample_end = start_sample + sample_count
                    media_seq += 1
                    start_sample += sample_count
                eos = self._make_envelope(
                    EventType.EOS,
                    EosPayload(
                        stream_id=stream_id,
                        last_media_sequence=last_media_seq,
                        last_sample_end=last_sample_end,
                        utterance_id=utt,
                    ).model_dump(mode="json", exclude_none=True),
                    deadline_at=dl,
                    utterance_id=utt,
                    utterance_sequence=utt_seq,
                )
                await self._enqueue_outbound(eos, stream_id=stream_id, require_credits=False)
            except Exception as exc:
                self._fail_pump(exc)
                raise

        produce_task = asyncio.create_task(_produce(), name="stage-v1-listen-produce")
        try:
            async for item in self._iter_products(
                accept={EventType.LISTEN_PRODUCT},
                terminal_on_final=True,
                deadline_at=dl,
            ):
                payload = ListenProductPayload.model_validate(item.envelope.payload)
                yield payload
                if payload.is_final:
                    break
        finally:
            await self._stop_task(produce_task)

    async def translate(
        self,
        requests: AsyncIterator[TranslateRequestPayload | dict[str, Any]],
        *,
        stream_id: str = "default",
        utterance_id: str | None = None,
        utterance_sequence: int | None = 0,
        deadline_at: str | None = None,
    ) -> AsyncIterator[TranslateProductPayload]:
        self._require_kind(StageKind.TRANSLATE)
        await self._ensure_started()
        utt = utterance_id or str(uuid4())
        utt_seq = 0 if utterance_sequence is None else utterance_sequence
        dl = deadline_at or _deadline_after(self.default_deadline_s)

        async def _produce() -> None:
            try:
                async for raw in requests:
                    check_deadline(dl)
                    req = (
                        raw
                        if isinstance(raw, TranslateRequestPayload)
                        else TranslateRequestPayload.model_validate(raw)
                    )
                    env = self._make_envelope(
                        EventType.TRANSLATE_REQUEST,
                        req.model_dump(mode="json"),
                        deadline_at=dl,
                        utterance_id=utt,
                        utterance_sequence=utt_seq,
                    )
                    await self._enqueue_outbound(env, stream_id=stream_id)
                eos = self._make_envelope(
                    EventType.EOS,
                    EosPayload(stream_id=stream_id, utterance_id=utt).model_dump(
                        mode="json", exclude_none=True
                    ),
                    deadline_at=dl,
                    utterance_id=utt,
                    utterance_sequence=utt_seq,
                )
                await self._enqueue_outbound(eos, stream_id=stream_id, require_credits=False)
            except Exception as exc:
                self._fail_pump(exc)
                raise

        produce_task = asyncio.create_task(_produce(), name="stage-v1-translate-produce")
        try:
            async for item in self._iter_products(
                accept={EventType.TRANSLATE_PRODUCT},
                terminal_on_final=True,
                deadline_at=dl,
            ):
                payload = TranslateProductPayload.model_validate(item.envelope.payload)
                yield payload
                if payload.is_final:
                    break
        finally:
            await self._stop_task(produce_task)

    async def speak(
        self,
        requests: AsyncIterator[SpeakRequestPayload | dict[str, Any]],
        *,
        stream_id: str = "translated:main",
        utterance_id: str | None = None,
        utterance_sequence: int | None = 0,
        deadline_at: str | None = None,
    ) -> AsyncIterator[tuple[bytes, EventEnvelope] | SpeakCompletePayload]:
        self._require_kind(StageKind.SPEAK)
        await self._ensure_started()
        utt = utterance_id or str(uuid4())
        utt_seq = 0 if utterance_sequence is None else utterance_sequence
        dl = deadline_at or _deadline_after(self.default_deadline_s)

        async def _produce() -> None:
            try:
                async for raw in requests:
                    check_deadline(dl)
                    req = (
                        raw
                        if isinstance(raw, SpeakRequestPayload)
                        else SpeakRequestPayload.model_validate(raw)
                    )
                    env = self._make_envelope(
                        EventType.SPEAK_REQUEST,
                        req.model_dump(mode="json"),
                        deadline_at=dl,
                        utterance_id=utt,
                        utterance_sequence=utt_seq,
                    )
                    await self._enqueue_outbound(env, stream_id="default")
                eos = self._make_envelope(
                    EventType.EOS,
                    EosPayload(stream_id="default", utterance_id=utt).model_dump(
                        mode="json", exclude_none=True
                    ),
                    deadline_at=dl,
                    utterance_id=utt,
                    utterance_sequence=utt_seq,
                )
                await self._enqueue_outbound(eos, stream_id="default", require_credits=False)
            except Exception as exc:
                self._fail_pump(exc)
                raise

        produce_task = asyncio.create_task(_produce(), name="stage-v1-speak-produce")
        try:
            async for item in self._iter_products(
                accept={EventType.SPEAK_AUDIO, EventType.SPEAK_COMPLETE},
                terminal_on_final=False,
                deadline_at=dl,
                stop_event_types={EventType.SPEAK_COMPLETE},
            ):
                if item.envelope.event_type == EventType.SPEAK_AUDIO:
                    if item.pcm is None:
                        raise StageClientError(
                            StageErrorCode.INVALID_ARGUMENT, "speak.audio missing pcm"
                        )
                    yield item.pcm, item.envelope
                else:
                    yield SpeakCompletePayload.model_validate(item.envelope.payload)
                    break
        finally:
            await self._stop_task(produce_task)

    async def _ensure_started(self) -> None:
        if not self._started:
            await self.start()
        self._raise_if_failed()

    def _require_kind(self, kind: StageKind) -> None:
        if self.stage_kind != kind:
            raise StageClientError(
                StageErrorCode.INVALID_ARGUMENT,
                f"client stage_kind is {self.stage_kind.value}, expected {kind.value}",
            )

    def _make_envelope(
        self,
        event_type: EventType,
        payload: dict[str, Any],
        *,
        deadline_at: str | None = None,
        utterance_id: str | None = None,
        utterance_sequence: int | None = None,
        include_worker_identity: bool = True,
    ) -> EventEnvelope:
        data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event_type": event_type.value,
            "message_id": str(uuid4()),
            "event_sequence": self._next_out_seq,
            "created_at": _utc_now_rfc3339(),
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "owner_generation": self.owner_generation,
            "stage_kind": self.stage_kind.value,
            "stage_id": self.stage_id,
            "attempt_id": self.attempt_id,
            "cancel_id": self.cancel_id,
            "payload": payload,
        }
        if deadline_at is not None:
            data["deadline_at"] = deadline_at
        if utterance_id is not None:
            data["utterance_id"] = utterance_id
            data["utterance_sequence"] = (
                utterance_sequence if utterance_sequence is not None else 0
            )
        if include_worker_identity and self._stage_instance_id is not None:
            data["stage_instance_id"] = self._stage_instance_id
            data["stage_version"] = self._stage_version
            data["model_revision"] = self._model_revision
            data["model_artifact_digest"] = self._model_artifact_digest
            data["provenance_id"] = self._provenance_id
        self._next_out_seq += 1
        return parse_event(data)

    async def _enqueue_outbound(
        self,
        envelope: EventEnvelope,
        *,
        pcm: bytes | None = None,
        stream_id: str = "default",
        require_credits: bool = True,
    ) -> None:
        self._raise_if_failed()
        if self._closed:
            raise StageClientError(StageErrorCode.CANCELLED, "client closed")
        check_deadline(envelope.deadline_at)
        timeout = _deadline_timeout_s(envelope.deadline_at, fallback=self.default_deadline_s)
        if require_credits and envelope.event_type not in (
            EventType.HELLO,
            EventType.CANCEL,
            EventType.ACK,
            EventType.WINDOW,
        ):
            await asyncio.wait_for(
                self._wait_credits(stream_id, events=1, bytes_=len(pcm or b"")),
                timeout=timeout,
            )
        item = OutboundItem(envelope=envelope, pcm=pcm)
        try:
            await asyncio.wait_for(self._outbound.put(item), timeout=timeout)
        except TimeoutError as exc:
            raise StageClientError(
                StageErrorCode.DEADLINE_EXCEEDED,
                "outbound queue put exceeded deadline",
            ) from exc
        self._outbound_high_water = max(self._outbound_high_water, self._outbound.qsize())

    async def _wait_credits(self, stream_id: str, *, events: int, bytes_: int) -> None:
        while True:
            win = self.window(stream_id if stream_id in self._windows else "default")
            if win.available_events >= events and win.available_bytes >= bytes_:
                win.consume(events=events, bytes_=bytes_)
                if win.available_events == 0 or win.available_bytes == 0:
                    self._credit_event.clear()
                return
            self._credit_event.clear()
            await self._credit_event.wait()

    async def _sender_loop(self) -> None:
        try:
            while True:
                item = await self._outbound.get()
                if item is None:
                    return
                check_deadline(item.envelope.deadline_at)
                self._out_seq.observe(item.envelope.event_sequence)
                if item.pcm is not None:
                    frame = encode_binary_frame(
                        item.envelope,
                        item.pcm,
                        max_frame_bytes=self.max_frame_bytes,
                    )
                    await self._transport.send_bytes(frame)
                else:
                    raw = item.envelope.model_dump_json(exclude_none=True)
                    validate_text_frame_size(raw)
                    await self._transport.send_text(raw)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_pump(exc)

    async def _receiver_loop(self) -> None:
        try:
            while not self._closed:
                raw = await self._transport.recv()
                item = self._decode_inbound(raw)
                env = item.envelope
                # Best-effort inbound sequence tracking (scripted peers may skip).
                with contextlib.suppress(ValidationError):
                    self._in_seq.observe(env.event_sequence)
                if env.event_type == EventType.WINDOW:
                    self._apply_window(env)
                    continue
                if env.event_type == EventType.ACK:
                    continue
                if env.event_type == EventType.HEALTH:
                    continue
                if env.event_type == EventType.ERROR:
                    err = ErrorPayload.model_validate(env.payload)
                    self._fail_pump(StageClientError(err.code, err.message))
                    # Still deliver error to consumers.
                if self._fence is not None and env.event_type not in (
                    EventType.ACCEPTED,
                    EventType.ERROR,
                ):
                    with contextlib.suppress(ValidationError):
                        # Soft check; stale products are dropped by consumer fence.
                        if env.stage_instance_id is not None:
                            check_fence(self._fence, env, require_instance=True)
                check_deadline(env.deadline_at)
                timeout = _deadline_timeout_s(env.deadline_at, fallback=self.default_deadline_s)
                try:
                    await asyncio.wait_for(self._inbound.put(item), timeout=timeout)
                except TimeoutError as exc:
                    raise StageClientError(
                        StageErrorCode.DEADLINE_EXCEEDED,
                        "inbound queue put exceeded deadline",
                    ) from exc
                self._inbound_high_water = max(self._inbound_high_water, self._inbound.qsize())
                # Application-level ack of highest contiguous inbound event sequence.
                with contextlib.suppress(Exception):
                    await self._send_ack(env)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_pump(exc)
        finally:
            with contextlib.suppress(asyncio.QueueFull):
                self._inbound.put_nowait(None)

    def _decode_inbound(self, raw: str | bytes) -> InboundItem:
        if isinstance(raw, bytes) and raw.startswith(b"STG1"):
            try:
                decoded: DecodedBinaryFrame = decode_binary_frame(
                    raw, max_frame_bytes=self.max_frame_bytes
                )
            except FramingError as exc:
                raise StageClientError(exc.code, exc.message) from exc
            return InboundItem(envelope=decoded.envelope, pcm=decoded.pcm)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        validate_text_frame_size(raw)
        return InboundItem(envelope=parse_event_json(raw))

    def _apply_window(self, envelope: EventEnvelope) -> None:
        payload = WindowPayload.model_validate(envelope.payload)
        win = self.window(payload.stream_id)
        win.available_events = payload.available_events
        win.available_bytes = payload.available_bytes
        win.credit_epoch = payload.credit_epoch
        win.oldest_queue_age_ms = payload.oldest_queue_age_ms
        self._credit_event.set()

    async def _send_ack(self, env: EventEnvelope) -> None:
        if not self._started or self._closed:
            return
        ack = self._make_envelope(
            EventType.ACK,
            {
                "stream_id": "default",
                "event_sequence": env.event_sequence,
            },
            include_worker_identity=self._stage_instance_id is not None,
        )
        # Best-effort; do not block receiver on full outbound queue.
        item = OutboundItem(envelope=ack)
        with contextlib.suppress(asyncio.QueueFull):
            self._outbound.put_nowait(item)

    async def _recv_control(
        self,
        *,
        expected: set[EventType],
        timeout: float,
    ) -> InboundItem:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            self._raise_if_failed()
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise StageClientError(StageErrorCode.DEADLINE_EXCEEDED, "control recv timeout")
            try:
                item = await asyncio.wait_for(self._inbound.get(), timeout=remaining)
            except TimeoutError as exc:
                raise StageClientError(
                    StageErrorCode.DEADLINE_EXCEEDED, "control recv timeout"
                ) from exc
            if item is None:
                self._raise_if_failed()
                raise StageClientError(StageErrorCode.INTERNAL, "connection closed")
            if item.envelope.event_type in expected:
                return item
            if item.envelope.event_type == EventType.ERROR:
                err = ErrorPayload.model_validate(item.envelope.payload)
                raise StageClientError(err.code, err.message)
            # Ignore non-matching control (e.g. window already applied in receiver).

    async def _iter_products(
        self,
        *,
        accept: set[EventType],
        terminal_on_final: bool,
        deadline_at: str | None,
        stop_event_types: set[EventType] | None = None,
    ) -> AsyncIterator[InboundItem]:
        stop_types = stop_event_types or set()
        timeout = _deadline_timeout_s(deadline_at, fallback=self.default_deadline_s)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while True:
            self._raise_if_failed()
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise StageClientError(StageErrorCode.DEADLINE_EXCEEDED, "product recv timeout")
            try:
                item = await asyncio.wait_for(self._inbound.get(), timeout=remaining)
            except TimeoutError as exc:
                raise StageClientError(
                    StageErrorCode.DEADLINE_EXCEEDED, "product recv timeout"
                ) from exc
            if item is None:
                self._raise_if_failed()
                return
            et = item.envelope.event_type
            if et == EventType.ERROR:
                err = ErrorPayload.model_validate(item.envelope.payload)
                raise StageClientError(err.code, err.message)
            if et == EventType.CANCELLED:
                raise StageClientError(StageErrorCode.CANCELLED, "attempt cancelled")
            if et == EventType.EOS:
                return
            if et in accept:
                if self._fence is not None and item.envelope.stage_instance_id is not None:
                    check_fence(self._fence, item.envelope, require_instance=True)
                yield item
                if et in stop_types:
                    return
                if terminal_on_final:
                    # caller decides break on is_final
                    continue
                continue
            # gap/dropped/health already filtered; ignore unknowns additively
            if et in (EventType.GAP, EventType.DROPPED, EventType.HEALTH, EventType.DRAINING):
                continue

    def _fail_pump(self, exc: BaseException) -> None:
        if self._pump_error is None:
            self._pump_error = exc
        with contextlib.suppress(asyncio.QueueFull):
            self._inbound.put_nowait(None)
        self._credit_event.set()

    def _raise_if_failed(self) -> None:
        if self._pump_error is None:
            return
        if isinstance(self._pump_error, StageClientError):
            raise self._pump_error
        if isinstance(self._pump_error, ValidationError):
            raise StageClientError(
                self._pump_error.code,
                self._pump_error.message,
            ) from self._pump_error
        raise StageClientError(
            StageErrorCode.INTERNAL,
            str(self._pump_error),
        ) from self._pump_error

    async def _stop_task(self, task: asyncio.Task[Any]) -> None:
        """Allow the producer to finish (e.g. latched EOS) before cancelling."""
        if task.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=self.default_deadline_s)
        except TimeoutError:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        except Exception:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


def duplex_tasks_live(client: StageV1Client) -> bool:
    """Anti-cheat helper: sender and receiver tasks must both be running."""
    sender = client.sender_task
    receiver = client.receiver_task
    return (
        sender is not None
        and receiver is not None
        and not sender.done()
        and not receiver.done()
    )


async def wait_first(
    awaitable: Awaitable[Any],
    *,
    timeout: float,
    on_timeout: str,
) -> Any:
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except TimeoutError as exc:
        raise AssertionError(on_timeout) from exc
