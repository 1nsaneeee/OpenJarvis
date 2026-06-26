"""Pydantic event models for the OpenJarvis Redis event bus."""

from __future__ import annotations

import time
from enum import StrEnum
from typing import Any

import ulid
from pydantic import BaseModel, Field


def _ulid() -> str:
    return str(ulid.new())


class Envelope(BaseModel):
    """Universal outer wrapper for all bus events."""

    id: str = Field(default_factory=_ulid)
    ts: float = Field(default_factory=time.time)
    source: str
    trace_id: str
    type: str
    payload: dict[str, Any] = Field(default_factory=dict)


# ── Audio ──────────────────────────────────────────────────────────────────


class AudioChunk(Envelope):
    type: str = "audio.chunk"
    pcm_b64: str  # base64-encoded PCM bytes
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 30


class VadEvent(Envelope):
    type: str = "audio.vad"
    is_speech: bool


# ── Wake ───────────────────────────────────────────────────────────────────


class WakeEvent(Envelope):
    type: str = "wake.detected"
    model_name: str
    score: float


# ── ASR ────────────────────────────────────────────────────────────────────


class AsrPartial(Envelope):
    type: str = "asr.partial"
    text: str


class AsrFinal(Envelope):
    type: str = "asr.final"
    text: str
    language: str | None = None
    duration_s: float | None = None


# ── Conversation ───────────────────────────────────────────────────────────


class ConvState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    EXECUTING = "executing"
    CONFIRMING = "confirming"
    RESPONDING = "responding"
    COOLDOWN = "cooldown"


class ConvStateEvent(Envelope):
    type: str = "conv.state"
    state: ConvState
    previous: ConvState | None = None


# ── LLM ────────────────────────────────────────────────────────────────────


class LlmRequest(Envelope):
    type: str = "llm.request"
    messages_json: str  # JSON-serialised list[Message]
    tools_json: str = "[]"
    model: str
    max_tokens: int = 4096
    temperature: float = 0.7


class LlmDeltaEvent(Envelope):
    type: str = "llm.delta"
    text: str | None = None


class LlmResponse(Envelope):
    type: str = "llm.response"
    text: str | None = None
    tool_calls_json: str = "[]"  # JSON list of ToolCall dicts
    finish_reason: str = "stop"
    tokens_in: int = 0
    tokens_out: int = 0


# ── Tools ──────────────────────────────────────────────────────────────────


class ToolCallEvent(Envelope):
    type: str = "tool.call"
    tool_call_id: str
    name: str
    arguments_json: str  # JSON dict


class ToolResultEvent(Envelope):
    type: str = "tool.result"
    tool_call_id: str
    name: str
    result_json: str  # JSON-serialised result value
    success: bool = True
    error: str | None = None


class ToolConfirmEvent(Envelope):
    type: str = "tool.confirm"
    tool_call_id: str
    name: str
    arguments_json: str
    risk: str = "high"


# ── System ─────────────────────────────────────────────────────────────────


class ShutdownEvent(Envelope):
    type: str = "system.shutdown"
    reason: str = "user_request"
