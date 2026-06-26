"""Microphone capture -> publish PCM frames to Redis event bus."""
from __future__ import annotations

import asyncio
import base64
from contextlib import suppress
from typing import Any

import numpy as np
import sounddevice as sd

from openjarvis.bus.client import BusClient
from openjarvis.bus.schemas import AudioChunk
from openjarvis.system.config import AudioConfig


class AudioCapture:
    def __init__(self, config: AudioConfig, bus: BusClient, trace_id: str = "") -> None:
        self._config = config
        self._bus = bus
        self._trace_id = trace_id
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._stream: sd.InputStream | None = None
        self._publish_task: asyncio.Task[None] | None = None

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time: Any,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"[AudioCapture] {status}")
        pcm = indata.copy().tobytes()
        with suppress(asyncio.QueueFull):
            self._queue.put_nowait(pcm)  # drop frame rather than block

    async def start(self) -> None:
        cfg = self._config
        self._stream = sd.InputStream(
            samplerate=cfg.sample_rate,
            channels=cfg.channels,
            dtype="int16",
            blocksize=int(cfg.sample_rate * cfg.frame_ms / 1000),
            device=cfg.device,
            callback=self._callback,
        )
        self._stream.start()
        self._publish_task = asyncio.create_task(self._publish_loop())

    async def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
        if self._publish_task:
            self._publish_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._publish_task

    async def _publish_loop(self) -> None:
        cfg = self._config
        while True:
            pcm = await self._queue.get()
            chunk = AudioChunk(
                source="audio",
                trace_id=self._trace_id,
                pcm_b64=base64.b64encode(pcm).decode(),
                sample_rate=cfg.sample_rate,
                channels=cfg.channels,
                frame_ms=cfg.frame_ms,
            )
            await self._bus.publish("jarvis:audio:chunk", chunk)
