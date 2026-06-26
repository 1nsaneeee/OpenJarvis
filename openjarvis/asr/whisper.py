"""ASR: buffer PCM after wake event, transcribe with faster-whisper."""
from __future__ import annotations

import base64

import numpy as np
from faster_whisper import WhisperModel  # type: ignore[import-untyped]

from openjarvis.bus.client import BusClient
from openjarvis.bus.schemas import AsrFinal, AudioChunk, WakeEvent
from openjarvis.system.config import AsrConfig, WakeConfig

SILENCE_FRAMES = 40   # ~1.2 s of silence ends the utterance
SILENCE_ENERGY_THRESHOLD = 200  # tune per mic


class WhisperASR:
    def __init__(
        self,
        asr_config: AsrConfig,
        wake_config: WakeConfig,
        bus: BusClient,
    ) -> None:
        self._config = asr_config
        self._wake_config = wake_config
        self._bus = bus
        self._model = WhisperModel(
            asr_config.model_size,
            device=asr_config.device,
            compute_type=asr_config.compute_type,
        )
        self._listening = False
        self._buffer: list[bytes] = []
        self._silence_counter = 0
        self._current_trace: str = ""

    async def start(self) -> None:
        await self._bus.subscribe("jarvis:wake:detected", WakeEvent, self._on_wake)
        await self._bus.subscribe("jarvis:audio:chunk", AudioChunk, self._on_chunk)

    async def _on_wake(self, event: WakeEvent) -> None:
        self._listening = True
        self._buffer = []
        self._silence_counter = 0
        self._current_trace = event.trace_id

    async def _on_chunk(self, chunk: AudioChunk) -> None:
        if not self._listening:
            return
        pcm = base64.b64decode(chunk.pcm_b64)
        self._buffer.append(pcm)

        # Simple energy-based VAD for end-of-speech
        audio = np.frombuffer(pcm, dtype=np.int16)
        energy = float(np.abs(audio).mean())
        if energy < SILENCE_ENERGY_THRESHOLD:
            self._silence_counter += 1
        else:
            self._silence_counter = 0

        if self._silence_counter >= SILENCE_FRAMES:
            self._listening = False
            await self._transcribe(self._current_trace)

    async def _transcribe(self, trace_id: str) -> None:
        if not self._buffer:
            return
        raw = b"".join(self._buffer)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        duration_s = len(audio) / 16000.0

        segments, info = self._model.transcribe(
            audio,
            language=self._config.language,
            vad_filter=self._config.vad_filter,
        )
        text = " ".join(s.text for s in segments).strip()
        if not text:
            return

        event = AsrFinal(
            source="asr",
            trace_id=trace_id,
            text=text,
            language=info.language,
            duration_s=duration_s,
        )
        await self._bus.xadd("jarvis:asr:final", event)
        await self._bus.publish("jarvis:asr:final", event)
