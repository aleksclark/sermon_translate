from __future__ import annotations

import asyncio
import fractions

import numpy as np
from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame

OPUS_SAMPLE_RATE = 48000
FRAME_DURATION = 0.020


class OutputAudioTrack(MediaStreamTrack):
    """Sends pipeline PCM output to a WebRTC peer via Opus/RTP.

    Generates silence frames on a 20 ms wall-clock timer so the RTP stream
    never goes quiet. When real PCM data is available it is played instead.
    """

    kind = "audio"

    def __init__(self, sample_rate: int = OPUS_SAMPLE_RATE) -> None:
        super().__init__()
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._sample_rate = sample_rate
        self._pts = 0
        self._samples_per_frame = int(sample_rate * FRAME_DURATION)
        self._leftover = b""
        self._started = False
        self._start_time = 0.0

    async def recv(self) -> AudioFrame:  # type: ignore[override]
        frame_bytes = self._samples_per_frame * 2

        if not self._started:
            self._started = True
            self._start_time = asyncio.get_event_loop().time()

        target_time = self._start_time + (self._pts / self._sample_rate)
        now = asyncio.get_event_loop().time()
        wait = target_time - now
        if wait > 0:
            await asyncio.sleep(wait)

        while len(self._leftover) < frame_bytes:
            try:
                data = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if data is None:
                if self._leftover:
                    self._leftover += b"\x00" * (frame_bytes - len(self._leftover))
                    break
                self.stop()
                raise MediaStreamError
            self._leftover += data

        if len(self._leftover) >= frame_bytes:
            chunk = self._leftover[:frame_bytes]
            self._leftover = self._leftover[frame_bytes:]
        else:
            chunk = b"\x00" * frame_bytes

        samples = len(chunk) // 2
        arr = np.frombuffer(chunk, dtype=np.int16).reshape(1, -1)
        frame = AudioFrame.from_ndarray(arr, format="s16", layout="mono")
        frame.sample_rate = self._sample_rate
        frame.pts = self._pts
        frame.time_base = fractions.Fraction(1, self._sample_rate)
        self._pts += samples
        return frame

    def push(self, data: bytes) -> None:
        self._queue.put_nowait(data)

    def finish(self) -> None:
        self._queue.put_nowait(None)

    @property
    def queued_bytes(self) -> int:
        return self._queue.qsize()


def frame_to_mono_pcm(frame: AudioFrame) -> bytes:
    """Decode an aiortc audio frame to mono s16le PCM.

    aiortc decodes Opus to interleaved stereo s16; downmix so downstream
    pipelines and OutputAudioTrack agree on a single channel.
    """
    arr = frame.to_ndarray()
    if arr.ndim == 2:
        arr = arr[0]
    if frame.layout.name != "mono" and arr.size % 2 == 0:
        arr = arr.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return arr.astype(np.int16).tobytes()
