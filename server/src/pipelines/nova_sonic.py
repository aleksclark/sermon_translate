"""Nova Sonic pipeline — AWS Bedrock speech-to-speech translation.

Streams audio at real-time pace to Nova Sonic which handles ASR +
translation + voice synthesis. Produces Spanish audio and transcripts
as Nova Sonic detects sentence boundaries.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from src.models import PipelineInfo, Session
from src.pipelines._audio import downsample
from src.pipelines.base import BasePipeline, OutputStreamDescriptor, OutputStreamKind

logger = logging.getLogger(__name__)

NOVA_MODEL_ID = "amazon.nova-sonic-v1:0"
NOVA_REGION = "us-east-1"
NOVA_INPUT_RATE = 16000
NOVA_OUTPUT_RATE = 24000
NOVA_VOICE = "lupe"


class NovaSonicPipeline(BasePipeline):
    """Real-time English → Spanish via Amazon Nova Sonic.

    Streams audio to Nova Sonic at real-time pace. Nova Sonic
    handles ASR, translation, and speech synthesis in a single
    bidirectional stream. Audio is paced internally so the pipeline
    works with both real-time WebRTC and instant-feed harness.
    """

    def __init__(self, sample_rate: int = 48000) -> None:
        super().__init__()
        self._sample_rate = sample_rate
        self._en_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._es_queue: asyncio.Queue[str | None] = asyncio.Queue()

    @property
    def info(self) -> PipelineInfo:
        return PipelineInfo(
            id="nova-sonic",
            name="Nova Sonic (AWS Bedrock S2S)",
            description=(
                "Real-time English → Spanish via Amazon Nova Sonic. "
                "Single streaming API for ASR + translation + voice."
            ),
            output_streams=self._build_output_stream_info(),
        )

    @property
    def output_streams(self) -> list[OutputStreamDescriptor]:
        return [
            OutputStreamDescriptor(
                name="audio", kind=OutputStreamKind.AUDIO, label="Spanish Audio",
            ),
            OutputStreamDescriptor(
                name="en-transcript", kind=OutputStreamKind.TEXT, label="English",
            ),
            OutputStreamDescriptor(
                name="es-transcript", kind=OutputStreamKind.TEXT, label="Spanish",
            ),
        ]

    async def _do_start(self) -> None:
        pass

    async def _do_stop(self) -> None:
        await self._en_queue.put(None)
        await self._es_queue.put(None)

    async def process(
        self, audio_stream: AsyncIterator[bytes], session: Session | None = None,
    ) -> AsyncIterator[bytes]:
        from aws_sdk_bedrock_runtime.client import (
            BedrockRuntimeClient,
            InvokeModelWithBidirectionalStreamOperationInput,
        )
        from aws_sdk_bedrock_runtime.config import Config
        from aws_sdk_bedrock_runtime.models import (
            BidirectionalInputPayloadPart,
            InvokeModelWithBidirectionalStreamInputChunk,
        )
        from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver

        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{NOVA_REGION}.amazonaws.com",
            region=NOVA_REGION,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        client = BedrockRuntimeClient(config=config)
        bidi = await client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=NOVA_MODEL_ID),
        )

        pn = str(uuid.uuid4())
        cn_sys = str(uuid.uuid4())
        cn_audio = str(uuid.uuid4())

        async def send(evt: dict[str, Any]) -> None:
            await bidi.input_stream.send(
                InvokeModelWithBidirectionalStreamInputChunk(
                    value=BidirectionalInputPayloadPart(
                        bytes_=json.dumps({"event": evt}).encode(),
                    ),
                ),
            )

        # Session + prompt + system + audio-content setup
        await send({"sessionStart": {"inferenceConfiguration": {
            "maxTokens": 1024, "topP": 0.9, "temperature": 0.7,
        }}})
        await send({"promptStart": {
            "promptName": pn,
            "textOutputConfiguration": {"mediaType": "text/plain"},
            "audioOutputConfiguration": {
                "mediaType": "audio/lpcm",
                "sampleRateHertz": NOVA_OUTPUT_RATE,
                "sampleSizeBits": 16, "channelCount": 1,
                "voiceId": NOVA_VOICE, "encoding": "base64",
                "audioType": "SPEECH",
            },
        }})
        await send({"contentStart": {
            "promptName": pn, "contentName": cn_sys,
            "type": "TEXT", "interactive": False, "role": "SYSTEM",
            "textInputConfiguration": {"mediaType": "text/plain"},
        }})
        await send({"textInput": {
            "promptName": pn, "contentName": cn_sys,
            "content": (
                "You are a real-time English to Spanish translator. "
                "Translate the speech to natural Spanish. "
                "Only output the Spanish translation."
            ),
        }})
        await send({"contentEnd": {"promptName": pn, "contentName": cn_sys}})
        await send({"contentStart": {
            "promptName": pn, "contentName": cn_audio,
            "type": "AUDIO", "interactive": True, "role": "USER",
            "audioInputConfiguration": {
                "mediaType": "audio/lpcm",
                "sampleRateHertz": NOVA_INPUT_RATE,
                "sampleSizeBits": 16, "channelCount": 1,
                "audioType": "SPEECH", "encoding": "base64",
            },
        }})

        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        done = asyncio.Event()

        async def receive_responses() -> None:
            try:
                while not done.is_set():
                    output = await asyncio.wait_for(
                        bidi.await_output(), timeout=20,
                    )
                    result = await output[1].receive()
                    if not (result.value and result.value.bytes_):
                        continue
                    data = json.loads(result.value.bytes_.decode())
                    if "event" not in data:
                        continue
                    evt = data["event"]

                    if "textOutput" in evt:
                        role = evt["textOutput"].get("role", "")
                        text = evt["textOutput"]["content"]
                        if role == "ASSISTANT":
                            await self._es_queue.put(text)
                        elif role == "USER":
                            await self._en_queue.put(text)

                    elif "audioOutput" in evt:
                        pcm_24k = base64.b64decode(evt["audioOutput"]["content"])
                        arr = (
                            np.frombuffer(pcm_24k, dtype=np.int16)
                            .astype(np.float32) / 32768.0
                        )
                        resampled = downsample(arr, NOVA_OUTPUT_RATE, self._sample_rate)
                        pcm = (
                            (resampled * 32767).clip(-32768, 32767)
                            .astype(np.int16).tobytes()
                        )
                        await audio_queue.put(pcm)

                    elif "completionEnd" in evt:
                        done.set()
                        return

            except TimeoutError:
                logger.warning("Nova Sonic receive timeout")
            except Exception:
                logger.exception("Nova Sonic receive error")
            finally:
                done.set()

        recv_task = asyncio.create_task(receive_responses())

        async def feed_audio() -> None:
            import time as _time

            chunk_16k_bytes = NOVA_INPUT_RATE // 10 * 2  # 100ms of s16le
            stream_start = _time.monotonic()
            samples_sent = 0

            async for raw in audio_stream:
                pcm_int16 = np.frombuffer(raw, dtype=np.int16)
                pcm_float = pcm_int16.astype(np.float32) / 32768.0
                ds = downsample(pcm_float, self._sample_rate, NOVA_INPUT_RATE)
                pcm_16k = (
                    (ds * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
                )

                for i in range(0, len(pcm_16k), chunk_16k_bytes):
                    chunk = pcm_16k[i : i + chunk_16k_bytes]
                    b64 = base64.b64encode(chunk).decode()
                    await send({"audioInput": {
                        "promptName": pn, "contentName": cn_audio,
                        "content": b64,
                    }})
                    samples_sent += len(chunk) // 2
                    target = stream_start + samples_sent / NOVA_INPUT_RATE
                    wait = target - _time.monotonic()
                    if wait > 0:
                        await asyncio.sleep(wait)

            # Close audio + prompt + session
            await send({"contentEnd": {"promptName": pn, "contentName": cn_audio}})
            import contextlib

            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(done.wait(), timeout=15)
            try:
                await send({"promptEnd": {"promptName": pn}})
                await send({"sessionEnd": {}})
                await bidi.input_stream.close()
            except Exception:
                pass
            await audio_queue.put(None)

        feed_task = asyncio.create_task(feed_audio())

        while True:
            data = await audio_queue.get()
            if data is None:
                break
            yield data

        await feed_task
        recv_task.cancel()
        await self._en_queue.put(None)
        await self._es_queue.put(None)

    def iter_stream(
        self, name: str, audio_stream: AsyncIterator[bytes],
    ) -> AsyncIterator[str] | AsyncIterator[bytes] | None:
        if name == "en-transcript":
            return self._drain_queue(self._en_queue)
        if name == "es-transcript":
            return self._drain_queue(self._es_queue)
        return None
