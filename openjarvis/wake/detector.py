"""Wake-word detection using openWakeWord."""
from __future__ import annotations

import asyncio
import base64

import numpy as np
from openwakeword.model import Model  # type: ignore[import-untyped]

from openjarvis.bus.client import BusClient
from openjarvis.bus.schemas import AudioChunk, WakeEvent
from openjarvis.system.config import WakeConfig


class WakeDetector:
    def __init__(self, config: WakeConfig, bus: BusClient) -> None:
        self._config = config
        self._bus = bus
        self._model = Model(wakeword_models=config.models, inference_framework="onnx")
        self._cooldown = False

    async def start(self) -> None:
        await self._bus.subscribe("jarvis:audio:chunk", AudioChunk, self._handle_chunk)

    async def _handle_chunk(self, chunk: AudioChunk) -> None:
        if self._cooldown:
            return
        pcm_bytes = base64.b64decode(chunk.pcm_b64)
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        self._model.predict(audio)
        for model_name in self._config.models:
            score = float(self._model.prediction_buffer[model_name][-1])
            if score >= self._config.threshold:
                await self._trigger(chunk.trace_id, model_name, score)
                break

    async def _trigger(self, trace_id: str, model_name: str, score: float) -> None:
        self._cooldown = True
        event = WakeEvent(
            source="wake",
            trace_id=trace_id,
            model_name=model_name,
            score=score,
        )
        await self._bus.publish("jarvis:wake:detected", event)
        await asyncio.sleep(self._config.cooldown_ms / 1000)
        self._cooldown = False
