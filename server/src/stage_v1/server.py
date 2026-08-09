"""Live stage.v1 WebSocket server: GET /stage/v1/stream.

Negotiates subprotocol ``stage.v1``, enforces D11 auth before admission, and
serves full-duplex sessions against a warm StageHost (D6).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from src.models import ListenProduct, TranslateProduct
from src.stage_v1.adapters import (
    listen_product_to_payload,
    open_edge_tts_session_stage,
    open_opus_mt_session_stage,
    open_pocket_tts_session_stage,
    open_whisper_session_stage,
    translate_product_to_payload,
)
from src.stage_v1.auth import (
    STAGE_V1_SUBPROTOCOL,
    AuthDecision,
    StageV1AuthConfig,
    authorize_stage_upgrade,
    client_host_from_scope,
    headers_from_scope,
    load_stage_v1_auth_config,
    url_scheme_from_scope,
)
from src.stage_v1.framing import (
    FramingError,
    decode_binary_frame,
    encode_binary_frame,
    validate_text_frame_size,
)
from src.stage_v1.host import SessionState, StageHost, StageHostError
from src.stage_v1.models import (
    BASELINE_CHANNELS,
    BASELINE_SAMPLE_RATE_HZ,
    DEFAULT_MAX_FRAME_BYTES,
    SCHEMA_VERSION,
    AcceptedPayload,
    AudioFormat,
    BinaryAudioPayload,
    CancelledPayload,
    CancelPayload,
    CancelScope,
    ErrorPayload,
    ErrorScope,
    EventEnvelope,
    EventType,
    HelloPayload,
    LimitsAdvertised,
    OpenedPayload,
    OpenPayload,
    ProsodyReport,
    ProsodyStatus,
    SpeakCompletePayload,
    SpeakRequestPayload,
    StageErrorCode,
    StageKind,
    TranslateRequestPayload,
    WindowPayload,
    parse_event,
    parse_event_json,
)
from src.stage_v1.peer import CreditWindow
from src.stage_v1.validation import (
    Fence,
    ValidationError,
    check_deadline,
    check_fence,
    check_schema_version,
)

logger = logging.getLogger(__name__)

OpenSessionStage = Callable[[StageHost, SessionState], Any]


def _utc_now_rfc3339() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def resolve_session_opener(host: StageHost) -> OpenSessionStage:
    """Map host stage_id/kind to warm session binder (never cold factory.create)."""
    stage_id = host.stage_id
    kind = host.stage_kind

    def _sample_rate() -> int:
        raw = host.runtime_versions.get("sample_rate_hz")
        if raw is None:
            return BASELINE_SAMPLE_RATE_HZ
        try:
            return int(raw)
        except ValueError:
            return BASELINE_SAMPLE_RATE_HZ

    if stage_id == "whisper-listen" or kind == StageKind.LISTEN:
        def open_listen(h: StageHost, s: SessionState) -> Any:
            return open_whisper_session_stage(h, s, sample_rate=_sample_rate())

        return open_listen
    if stage_id == "opus-mt-en-es" or kind == StageKind.TRANSLATE:
        return open_opus_mt_session_stage
    if stage_id == "edge-tts-es":
        def open_edge(h: StageHost, s: SessionState) -> Any:
            return open_edge_tts_session_stage(h, s, sample_rate=_sample_rate())

        return open_edge
    if stage_id == "pocket-tts-spanish-24l":
        def open_pocket(h: StageHost, s: SessionState) -> Any:
            return open_pocket_tts_session_stage(h, s, sample_rate=_sample_rate())

        return open_pocket
    if kind == StageKind.SPEAK:
        def open_speak(h: StageHost, s: SessionState) -> Any:
            return open_edge_tts_session_stage(h, s, sample_rate=_sample_rate())

        return open_speak
    raise ValueError(f"no warm session opener for stage_id={stage_id!r} kind={kind!r}")


class StageV1ServerSession:
    """One accepted stage.v1 connection bound to a warm host session."""

    def __init__(
        self,
        ws: WebSocket,
        host: StageHost,
        *,
        open_stage: OpenSessionStage | None = None,
        initial_event_credits: int | None = None,
        initial_byte_credits: int | None = None,
    ) -> None:
        self.ws = ws
        self.host = host
        self._open_stage = open_stage or resolve_session_opener(host)
        self.max_frame_bytes = host.limits.max_frame_bytes or DEFAULT_MAX_FRAME_BYTES
        events = (
            initial_event_credits
            if initial_event_credits is not None
            else host.limits.max_inflight_events
        )
        bytes_ = (
            initial_byte_credits
            if initial_byte_credits is not None
            else host.limits.max_inflight_bytes
        )
        self._window = CreditWindow(
            stream_id="default",
            available_events=events,
            available_bytes=bytes_,
        )
        self._out_seq = 0
        self._accepted = False
        self._cancelled = False
        self._fence: Fence | None = None
        self._session: SessionState | None = None
        self._stage: Any | None = None
        self._hello: EventEnvelope | None = None
        self._closed = False
        self._sample_rate = BASELINE_SAMPLE_RATE_HZ
        self._channels = BASELINE_CHANNELS
        self._listen_revision = 0
        self._translate_revision = 0
        self._input_q: asyncio.Queue[Any | None] = asyncio.Queue(maxsize=32)
        self._cancel_event = asyncio.Event()
        self._runner_task: asyncio.Task[None] | None = None
        self._recv_task: asyncio.Task[None] | None = None
        self._opened = False

    @property
    def session_state_id(self) -> str | None:
        return None if self._session is None else self._session.session_state_id

    async def run(self) -> None:
        try:
            first = await self._recv_raw()
            if first is None:
                return
            if isinstance(first, bytes):
                await self._emit_error_standalone(
                    StageErrorCode.INVALID_ARGUMENT,
                    "hello must be a JSON text frame",
                )
                return
            try:
                envelope = parse_event_json(first)
            except Exception as exc:
                await self._emit_error_standalone(
                    StageErrorCode.INVALID_ARGUMENT,
                    f"invalid hello envelope: {exc}",
                )
                return
            if envelope.event_type != EventType.HELLO:
                await self._emit_error_from_base(
                    envelope,
                    StageErrorCode.INVALID_ARGUMENT,
                    "hello required first",
                    terminal=True,
                )
                return
            ok = await self._handle_hello(envelope)
            if not ok:
                return
            self._recv_task = asyncio.create_task(self._recv_loop(), name="stage-v1-recv")
            self._runner_task = asyncio.create_task(self._run_stage(), name="stage-v1-run")
            done, _pending = await asyncio.wait(
                {self._recv_task, self._runner_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        except WebSocketDisconnect:
            logger.info(
                "stage.v1 client disconnected stage_id=%s",
                self.host.stage_id,
            )
        except Exception:
            logger.exception("stage.v1 session failed stage_id=%s", self.host.stage_id)
            with contextlib.suppress(Exception):
                if self._hello is not None:
                    await self._emit_error_from_base(
                        self._hello,
                        StageErrorCode.INTERNAL,
                        "internal session failure",
                        terminal=True,
                    )
        finally:
            await self._cleanup()

    async def _cleanup(self) -> None:
        self._closed = True
        self._cancel_event.set()
        with contextlib.suppress(asyncio.QueueFull):
            self._input_q.put_nowait(None)
        for task in (self._recv_task, self._runner_task):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        stage = self._stage
        self._stage = None
        if stage is not None:
            with contextlib.suppress(Exception):
                await stage.stop()
        session = self._session
        self._session = None
        if session is not None:
            with contextlib.suppress(Exception):
                await self.host.close_session(session.session_state_id)
        if self.ws.client_state == WebSocketState.CONNECTED:
            with contextlib.suppress(Exception):
                await self.ws.close()

    async def _handle_hello(self, envelope: EventEnvelope) -> bool:
        self._hello = envelope
        try:
            check_schema_version(str(envelope.schema_version))
        except ValidationError as exc:
            await self._emit_error_from_base(
                envelope, exc.code, exc.message, terminal=True
            )
            return False

        try:
            hello = HelloPayload.model_validate(envelope.payload)
        except Exception as exc:
            await self._emit_error_from_base(
                envelope,
                StageErrorCode.INVALID_ARGUMENT,
                f"invalid hello payload: {exc}",
                terminal=True,
            )
            return False

        if hello.resume is not None:
            await self._emit_error_from_base(
                envelope,
                StageErrorCode.RESUME_UNSUPPORTED,
                "resume unsupported in baseline stage.v1",
                terminal=True,
            )
            return False

        if envelope.stage_kind != self.host.stage_kind:
            await self._emit_error_from_base(
                envelope,
                StageErrorCode.INVALID_ARGUMENT,
                (
                    f"stage_kind mismatch: hello={envelope.stage_kind.value} "
                    f"host={self.host.stage_kind.value}"
                ),
                terminal=True,
            )
            return False
        if envelope.stage_id != self.host.stage_id:
            await self._emit_error_from_base(
                envelope,
                StageErrorCode.INVALID_ARGUMENT,
                (
                    f"stage_id mismatch: hello={envelope.stage_id!r} "
                    f"host={self.host.stage_id!r}"
                ),
                terminal=True,
            )
            return False

        if hello.audio_formats:
            fmt = hello.audio_formats[0]
            self._sample_rate = fmt.sample_rate_hz
            self._channels = fmt.channels

        req_max = hello.limits_requested.max_frame_bytes or self.max_frame_bytes
        self.max_frame_bytes = min(self.max_frame_bytes, req_max)

        # Allocate session only after hello validation (auth already passed).
        try:
            session = await self.host.open_session(
                attempt_id=envelope.attempt_id,
                cancel_id=envelope.cancel_id,
                session_id=envelope.session_id,
            )
        except StageHostError as exc:
            await self._emit_error_from_base(
                envelope,
                exc.payload.code,
                exc.payload.message,
                terminal=True,
                retryable=exc.payload.retryable,
            )
            return False

        self._session = session
        try:
            self._stage = self._open_stage(self.host, session)
        except Exception as exc:
            await self.host.close_session(session.session_state_id)
            self._session = None
            await self._emit_error_from_base(
                envelope,
                StageErrorCode.MODEL_UNAVAILABLE,
                f"failed to bind warm stage: {type(exc).__name__}: {exc}",
                terminal=True,
            )
            return False

        if self.host.provenance is None or self.host.provenance_id is None:
            self.host._rebuild_provenance()  # noqa: SLF001 — ensure accepted provenance
        provenance = self.host.provenance
        assert provenance is not None
        prov_id = self.host.provenance_id
        assert prov_id is not None

        limits = LimitsAdvertised(
            max_sessions=self.host.max_sessions,
            max_inflight_events=self._window.available_events,
            max_inflight_bytes=self._window.available_bytes,
            max_frame_bytes=self.max_frame_bytes,
            input_queue_capacity=self._input_q.maxsize,
            max_queue_age_ms=self.host.limits.max_queue_age_ms,
        )
        accepted_payload = AcceptedPayload(
            stage_version=self.host.stage_version,
            model_revision=self.host.model_revision,
            model_artifact_digest=self.host.model_artifact_digest,
            stage_instance_id=self.host.stage_instance_id,
            boot_id=self.host.boot_id,
            capabilities=["warm-host", "stage.v1"],
            audio_formats=[
                AudioFormat(
                    sample_rate_hz=self._sample_rate,
                    channels=self._channels,
                )
            ],
            limits=limits,
            provenance=provenance,
            provenance_id=prov_id,
        )
        out = self._base_envelope(
            EventType.ACCEPTED,
            envelope,
            accepted_payload.model_dump(mode="json"),
            stage_instance_id=self.host.stage_instance_id,
            stage_version=self.host.stage_version,
            model_revision=self.host.model_revision,
            model_artifact_digest=self.host.model_artifact_digest,
            provenance_id=prov_id,
        )
        self._fence = Fence.from_envelope(out)
        self._accepted = True
        await self._send_envelope(out)

        window_env = self._base_envelope(
            EventType.WINDOW,
            envelope,
            WindowPayload(
                stream_id="default",
                available_events=self._window.available_events,
                available_bytes=self._window.available_bytes,
                credit_epoch=self._window.credit_epoch,
                oldest_queue_age_ms=self._window.oldest_queue_age_ms,
            ).model_dump(mode="json"),
            stage_instance_id=self.host.stage_instance_id,
            stage_version=self.host.stage_version,
            model_revision=self.host.model_revision,
            model_artifact_digest=self.host.model_artifact_digest,
            provenance_id=prov_id,
        )
        await self._send_envelope(window_env)
        return True

    async def _recv_loop(self) -> None:
        assert self._hello is not None
        try:
            while not self._closed and not self._cancel_event.is_set():
                raw = await self._recv_raw()
                if raw is None:
                    await self._input_q.put(None)
                    return
                try:
                    if isinstance(raw, bytes):
                        frame = decode_binary_frame(
                            raw, max_frame_bytes=self.max_frame_bytes
                        )
                        envelope, pcm = frame.envelope, frame.pcm
                    else:
                        validate_text_frame_size(raw)
                        envelope = parse_event_json(raw)
                        pcm = None
                except FramingError as exc:
                    await self._emit_error_from_base(
                        self._hello, exc.code, exc.message
                    )
                    continue
                except Exception as exc:
                    await self._emit_error_from_base(
                        self._hello,
                        StageErrorCode.INVALID_ARGUMENT,
                        f"invalid frame: {exc}",
                    )
                    continue

                if not await self._accept_inbound(envelope, pcm):
                    continue
        except WebSocketDisconnect:
            await self._input_q.put(None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("stage.v1 recv loop error")
            await self._emit_error_from_base(
                self._hello,
                StageErrorCode.INTERNAL,
                f"recv failed: {type(exc).__name__}",
            )
            await self._input_q.put(None)

    async def _accept_inbound(
        self, envelope: EventEnvelope, pcm: bytes | None
    ) -> bool:
        assert self._hello is not None
        if self._cancelled and envelope.event_type != EventType.CANCEL:
            await self._emit_error_from_base(
                envelope, StageErrorCode.CANCELLED, "attempt cancelled"
            )
            return False

        if self._fence is not None and envelope.event_type != EventType.HELLO:
            try:
                require_instance = envelope.event_type != EventType.OPEN
                if envelope.stage_instance_id is not None or require_instance:
                    # open may omit instance; after accepted other events should match
                    if envelope.event_type == EventType.OPEN:
                        check_fence(self._fence, envelope, require_instance=False)
                    else:
                        check_fence(self._fence, envelope, require_instance=False)
            except ValidationError as exc:
                await self._emit_error_from_base(envelope, exc.code, exc.message)
                return False

        try:
            check_deadline(envelope.deadline_at)
        except ValidationError as exc:
            await self._emit_error_from_base(envelope, exc.code, exc.message)
            return False

        nbytes = len(pcm) if pcm is not None else 0
        if envelope.event_type not in {
            EventType.CANCEL,
            EventType.ACK,
            EventType.EOS,
            EventType.OPEN,
        }:
            try:
                self._window.consume(events=1, bytes_=nbytes)
            except ValidationError as exc:
                await self._emit_error_from_base(envelope, exc.code, exc.message)
                return False

        if envelope.event_type == EventType.CANCEL:
            await self._handle_cancel(envelope)
            return False

        if envelope.event_type == EventType.OPEN:
            await self._handle_open(envelope)
            return False

        if envelope.event_type == EventType.ACK:
            # Credits are re-granted opportunistically.
            self._window.grant(events=1, bytes_=0)
            return False

        if envelope.event_type == EventType.EOS:
            await self._input_q.put(None)
            return True

        if envelope.event_type == EventType.LISTEN_AUDIO:
            if pcm is None:
                await self._emit_error_from_base(
                    envelope,
                    StageErrorCode.INVALID_ARGUMENT,
                    "listen.audio requires binary STG1 pcm",
                )
                return False
            await self._input_q.put(pcm)
            return True

        if envelope.event_type == EventType.TRANSLATE_REQUEST:
            try:
                req = TranslateRequestPayload.model_validate(envelope.payload)
            except Exception as exc:
                await self._emit_error_from_base(
                    envelope,
                    StageErrorCode.INVALID_ARGUMENT,
                    f"invalid translate.request: {exc}",
                )
                return False
            await self._input_q.put(req)
            return True

        if envelope.event_type == EventType.SPEAK_REQUEST:
            try:
                req = SpeakRequestPayload.model_validate(envelope.payload)
            except Exception as exc:
                await self._emit_error_from_base(
                    envelope,
                    StageErrorCode.INVALID_ARGUMENT,
                    f"invalid speak.request: {exc}",
                )
                return False
            await self._input_q.put(req)
            return True

        # Ignore unknown additive control events under same major.
        return False

    async def _handle_open(self, envelope: EventEnvelope) -> None:
        OpenPayload.model_validate(envelope.payload or {})
        self._opened = True
        out = self._base_envelope(
            EventType.OPENED,
            envelope,
            OpenedPayload(ready=True).model_dump(mode="json"),
        )
        await self._send_envelope(out)

    async def _handle_cancel(self, envelope: EventEnvelope) -> None:
        self._cancelled = True
        self._cancel_event.set()
        if self._session is not None:
            self._session.cancel()
        try:
            payload = CancelPayload.model_validate(envelope.payload)
            scope = payload.scope
            reason = payload.reason
        except Exception:
            scope = CancelScope.ATTEMPT
            reason = "cancel"
        out = self._base_envelope(
            EventType.CANCELLED,
            envelope,
            CancelledPayload(scope=scope, reason=reason, disposed=True).model_dump(
                mode="json"
            ),
        )
        await self._send_envelope(out)
        with contextlib.suppress(asyncio.QueueFull):
            self._input_q.put_nowait(None)

    async def _run_stage(self) -> None:
        assert self._stage is not None
        assert self._hello is not None
        kind = self.host.stage_kind
        try:
            if kind == StageKind.LISTEN:
                await self._run_listen()
            elif kind == StageKind.TRANSLATE:
                await self._run_translate()
            elif kind == StageKind.SPEAK:
                await self._run_speak()
            else:
                await self._emit_error_from_base(
                    self._hello,
                    StageErrorCode.UNSUPPORTED_CAPABILITY,
                    f"unsupported stage kind on stream path: {kind}",
                    terminal=True,
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("stage runner failed")
            await self._emit_error_from_base(
                self._hello,
                StageErrorCode.INFERENCE_FAILED,
                f"{type(exc).__name__}: {exc}",
            )

    async def _iter_inputs(self) -> AsyncIterator[Any]:
        while True:
            if self._cancel_event.is_set() and self._input_q.empty():
                return
            item = await self._input_q.get()
            if item is None:
                return
            yield item

    async def _run_listen(self) -> None:
        assert self._stage is not None
        assert self._hello is not None
        stage = self._stage
        await stage.start()
        try:
            async for product in stage.transcribe(self._iter_inputs()):
                if self._cancel_event.is_set():
                    break
                if not isinstance(product, ListenProduct):
                    continue
                if not product.text and not product.is_final:
                    continue
                payload = listen_product_to_payload(
                    product,
                    revision=self._listen_revision,
                    sample_rate=self._sample_rate,
                    commit_full_text=True,
                )
                self._listen_revision += 1
                if not payload.text and not payload.is_final:
                    continue
                env = self._product_envelope(
                    EventType.LISTEN_PRODUCT,
                    payload.model_dump(mode="json"),
                    utterance_id=product.utterance_id,
                )
                await self._send_envelope(env)
                if payload.is_final:
                    break
        finally:
            await stage.stop()
            # stop clears session state only; host model stays warm

    async def _run_translate(self) -> None:
        assert self._stage is not None
        assert self._hello is not None
        stage = self._stage

        async def listen_products() -> AsyncIterator[ListenProduct]:
            async for req in self._iter_inputs():
                if not isinstance(req, TranslateRequestPayload):
                    continue
                yield ListenProduct(
                    sequence=req.source_revision,
                    utterance_id=req.source_span_id,
                    text=req.text,
                    is_final=True,
                    language=req.source_language or "en",
                )

        await stage.start()
        try:
            async for product in stage.translate(listen_products()):
                if self._cancel_event.is_set():
                    break
                if not isinstance(product, TranslateProduct):
                    continue
                if not product.text.strip() and not product.is_final:
                    continue
                payload = translate_product_to_payload(
                    product,
                    revision=self._translate_revision,
                )
                self._translate_revision += 1
                env = self._product_envelope(
                    EventType.TRANSLATE_PRODUCT,
                    payload.model_dump(mode="json"),
                    utterance_id=product.source_utterance_id,
                )
                await self._send_envelope(env)
                if payload.is_final:
                    # continue until input EOS; finals may be per-utterance
                    pass
        finally:
            await stage.stop()

    async def _run_speak(self) -> None:
        assert self._stage is not None
        assert self._hello is not None
        stage = self._stage
        media_seq = 0
        start_sample = 0
        chunk_count = 0
        sample_count_total = 0
        last_span: str | None = None
        seq = 0

        async def translate_products() -> AsyncIterator[TranslateProduct]:
            nonlocal seq, last_span
            async for req in self._iter_inputs():
                if not isinstance(req, SpeakRequestPayload):
                    continue
                last_span = req.target_span_id
                product = TranslateProduct(
                    sequence=seq,
                    source_utterance_id=req.target_span_id,
                    target_utterance_id=req.target_span_id,
                    text=req.text,
                    is_final=True,
                )
                seq += 1
                yield product

        await stage.start()
        try:
            async for pcm in stage.synthesize(translate_products()):
                if self._cancel_event.is_set():
                    break
                if not pcm:
                    continue
                sample_count = len(pcm) // (2 * self._channels)
                span_id = last_span or "span-0"
                audio_payload = BinaryAudioPayload(
                    stream_id="translated:main",
                    media_sequence=media_seq,
                    start_sample=start_sample,
                    sample_count=sample_count,
                    payload_bytes=len(pcm),
                    format=AudioFormat(
                        sample_rate_hz=self._sample_rate,
                        channels=self._channels,
                    ),
                    target_span_id=span_id,
                    audio_chunk_sequence=chunk_count,
                    discontinuity=False,
                )
                env = self._product_envelope(
                    EventType.SPEAK_AUDIO,
                    audio_payload.model_dump(mode="json"),
                )
                await self._send_binary(env, pcm)
                media_seq += 1
                start_sample += sample_count
                chunk_count += 1
                sample_count_total += sample_count

            duration_ms = (
                (sample_count_total / float(self._sample_rate)) * 1000.0
                if self._sample_rate
                else 0.0
            )
            complete = SpeakCompletePayload(
                target_span_id=last_span or "span-0",
                chunk_count=chunk_count,
                sample_count=sample_count_total,
                duration_ms=duration_ms,
                is_final=True,
                prosody_report=ProsodyReport(
                    prosody_status=ProsodyStatus.UNSUPPORTED,
                    consumed_fields=[],
                    ignored_fields=["prosody"],
                ),
            )
            env = self._product_envelope(
                EventType.SPEAK_COMPLETE,
                complete.model_dump(mode="json"),
            )
            await self._send_envelope(env)
        finally:
            await stage.stop()

    def _product_envelope(
        self,
        event_type: EventType,
        payload: dict[str, Any],
        *,
        utterance_id: str | None = None,
    ) -> EventEnvelope:
        assert self._hello is not None
        return self._base_envelope(
            event_type,
            self._hello,
            payload,
            stage_instance_id=self.host.stage_instance_id,
            stage_version=self.host.stage_version,
            model_revision=self.host.model_revision,
            model_artifact_digest=self.host.model_artifact_digest,
            provenance_id=self.host.provenance_id,
            utterance_id=utterance_id,
            utterance_sequence=0 if utterance_id is not None else None,
        )

    def _base_envelope(
        self,
        event_type: EventType,
        base: EventEnvelope,
        payload: dict[str, Any],
        **overrides: Any,
    ) -> EventEnvelope:
        data: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event_type": event_type.value,
            "message_id": str(uuid4()),
            "event_sequence": self._out_seq,
            "created_at": _utc_now_rfc3339(),
            "correlation_id": base.correlation_id,
            "session_id": base.session_id,
            "owner_generation": base.owner_generation,
            "stage_kind": self.host.stage_kind.value,
            "stage_id": self.host.stage_id,
            "attempt_id": base.attempt_id,
            "cancel_id": base.cancel_id,
            "payload": payload,
        }
        data.update({k: v for k, v in overrides.items() if v is not None})
        self._out_seq += 1
        return parse_event(data)

    async def _send_envelope(self, envelope: EventEnvelope) -> None:
        if self._closed or self.ws.client_state != WebSocketState.CONNECTED:
            return
        text = envelope.model_dump_json(exclude_none=True)
        validate_text_frame_size(text)
        await self.ws.send_text(text)

    async def _send_binary(self, envelope: EventEnvelope, pcm: bytes) -> None:
        if self._closed or self.ws.client_state != WebSocketState.CONNECTED:
            return
        frame = encode_binary_frame(
            envelope, pcm, max_frame_bytes=self.max_frame_bytes
        )
        await self.ws.send_bytes(frame)

    async def _emit_error_from_base(
        self,
        base: EventEnvelope,
        code: StageErrorCode,
        message: str,
        *,
        terminal: bool = False,
        retryable: bool = False,
    ) -> None:
        env = self._base_envelope(
            EventType.ERROR,
            base,
            ErrorPayload(
                code=code,
                message=message,
                retryable=retryable,
                scope=ErrorScope.CONNECTION if terminal else ErrorScope.ATTEMPT,
            ).model_dump(mode="json"),
            stage_instance_id=(
                self.host.stage_instance_id if self._accepted else None
            ),
            stage_version=self.host.stage_version if self._accepted else None,
            model_revision=self.host.model_revision if self._accepted else None,
            model_artifact_digest=(
                self.host.model_artifact_digest if self._accepted else None
            ),
        )
        with contextlib.suppress(Exception):
            await self._send_envelope(env)
        if terminal and self.ws.client_state == WebSocketState.CONNECTED:
            with contextlib.suppress(Exception):
                await self.ws.close(code=1008)

    async def _emit_error_standalone(
        self, code: StageErrorCode, message: str
    ) -> None:
        # Minimal error when we lack a hello envelope.
        now = _utc_now_rfc3339()
        data = {
            "schema_version": SCHEMA_VERSION,
            "event_type": EventType.ERROR.value,
            "message_id": str(uuid4()),
            "event_sequence": self._out_seq,
            "created_at": now,
            "correlation_id": "unauthenticated",
            "session_id": "unknown",
            "owner_generation": 0,
            "stage_kind": self.host.stage_kind.value,
            "stage_id": self.host.stage_id,
            "attempt_id": "unknown",
            "cancel_id": "unknown",
            "payload": ErrorPayload(
                code=code,
                message=message,
                retryable=False,
                scope=ErrorScope.CONNECTION,
            ).model_dump(mode="json"),
        }
        self._out_seq += 1
        with contextlib.suppress(Exception):
            await self.ws.send_text(
                parse_event(data).model_dump_json(exclude_none=True)
            )
        with contextlib.suppress(Exception):
            await self.ws.close(code=1008)

    async def _recv_raw(self) -> str | bytes | None:
        if self.ws.client_state != WebSocketState.CONNECTED:
            return None
        message = await self.ws.receive()
        if message.get("type") == "websocket.disconnect":
            return None
        if "text" in message and message["text"] is not None:
            return str(message["text"])
        if "bytes" in message and message["bytes"] is not None:
            return bytes(message["bytes"])
        return None


async def reject_websocket(
    ws: WebSocket,
    decision: AuthDecision,
    *,
    host: StageHost | None = None,
) -> None:
    """Reject upgrade after accept with AUTHENTICATION_FAILED (no session alloc)."""
    # Starlette requires accept before send; reject without host session.
    subprotocol = None
    requested = ws.headers.get("sec-websocket-protocol", "")
    if STAGE_V1_SUBPROTOCOL in {p.strip() for p in requested.split(",") if p.strip()}:
        subprotocol = STAGE_V1_SUBPROTOCOL
    with contextlib.suppress(Exception):
        await ws.accept(subprotocol=subprotocol)
    code = decision.code or StageErrorCode.AUTHENTICATION_FAILED
    message = decision.message or "authentication failed"
    now = _utc_now_rfc3339()
    stage_kind = host.stage_kind.value if host is not None else StageKind.LISTEN.value
    stage_id = host.stage_id if host is not None else "unknown"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "event_type": EventType.ERROR.value,
        "message_id": str(uuid4()),
        "event_sequence": 0,
        "created_at": now,
        "correlation_id": "auth",
        "session_id": "unauthorized",
        "owner_generation": 0,
        "stage_kind": stage_kind,
        "stage_id": stage_id,
        "attempt_id": "unauthorized",
        "cancel_id": "unauthorized",
        "payload": ErrorPayload(
            code=code,
            message=message,
            retryable=False,
            scope=ErrorScope.CONNECTION,
        ).model_dump(mode="json"),
    }
    with contextlib.suppress(Exception):
        await ws.send_text(parse_event(payload).model_dump_json(exclude_none=True))
    with contextlib.suppress(Exception):
        await ws.close(code=decision.close_code)


def build_stage_v1_router(
    host: StageHost,
    *,
    auth: StageV1AuthConfig | None = None,
) -> APIRouter:
    """Build router with GET /stage/v1/stream WebSocket endpoint."""
    router = APIRouter(tags=["stage.v1"])
    auth_cfg = auth or load_stage_v1_auth_config()

    @router.websocket("/stage/v1/stream")
    async def stage_v1_stream(ws: WebSocket) -> None:
        headers = headers_from_scope(ws.scope)
        # Prefer Starlette header mapping when present.
        for key, value in ws.headers.items():
            headers[key] = value
        decision = authorize_stage_upgrade(
            config=auth_cfg,
            headers=headers,
            url_scheme=url_scheme_from_scope(ws.scope),
            client_host=client_host_from_scope(ws.scope),
        )
        if not decision.ok:
            await reject_websocket(ws, decision, host=host)
            return

        requested = [
            p.strip()
            for p in (ws.headers.get("sec-websocket-protocol") or "").split(",")
            if p.strip()
        ]
        if STAGE_V1_SUBPROTOCOL not in requested:
            await reject_websocket(
                ws,
                AuthDecision(
                    ok=False,
                    code=StageErrorCode.INVALID_ARGUMENT,
                    message=f"required subprotocol {STAGE_V1_SUBPROTOCOL}",
                ),
                host=host,
            )
            return

        await ws.accept(subprotocol=STAGE_V1_SUBPROTOCOL)
        session = StageV1ServerSession(ws, host)
        await session.run()

    return router


def mount_stage_v1_routes(
    app: FastAPI,
    host: StageHost,
    *,
    auth: StageV1AuthConfig | None = None,
) -> None:
    """Attach /stage/v1/stream to a FastAPI app."""
    app.include_router(build_stage_v1_router(host, auth=auth))
