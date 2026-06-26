# OpenJarvis Walking Skeleton — v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** End-to-end voice pipeline — say "Hey Jarvis, what time is it?" and receive a printed answer in the terminal.

**Architecture:** Local microphone → openWakeWord wake detection → faster-whisper ASR → Redis event bus → ConversationManager state machine → Anthropic LLM with tool calling → ToolExecutor (builtin `get_time`) → terminal output. All cross-module communication via Redis pub/sub or streams using a shared `BusClient`.

**Tech Stack:** Python 3.11+, Redis 7+, `faster-whisper`, `openwakeword`, `sounddevice`, `anthropic`, `pydantic` v2, `aiosqlite` (stub only in v0.1), `structlog`, `rich`, `click`, `ulid-py`

**Scope (MVP — deliberately excluded):**
- No streaming ASR or streaming LLM (batch only)
- No SQLite writes (in-memory message list)
- No COOLDOWN or CONFIRMING states
- No MCP, no multi-provider
- No TTS

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `openjarvis/bus/schemas.py` | **Create** | All Pydantic event models + Envelope |
| `openjarvis/bus/client.py` | **Create** | BusClient: publish, subscribe, xadd, xread |
| `openjarvis/audio/capture.py` | **Create** | Mic → PCM frames → `jarvis:audio:chunk` |
| `openjarvis/wake/detector.py` | **Create** | openWakeWord → `jarvis:wake:detected` |
| `openjarvis/asr/whisper.py` | **Create** | Buffer PCM after wake → transcribe → `jarvis:asr:final` |
| `openjarvis/llm/base.py` | **Create** | `BaseProvider`, `Message`, `ToolCall`, `ToolSpec`, `LlmDelta` |
| `openjarvis/llm/providers/anthropic.py` | **Create** | `AnthropicProvider(BaseProvider)` |
| `openjarvis/llm/registry.py` | **Create** | `load_provider(name)` registry |
| `openjarvis/tools/registry.py` | **Create** | `@tool` decorator + `ToolRegistry` |
| `openjarvis/tools/builtin/time_tool.py` | **Create** | `get_time()` builtin tool |
| `openjarvis/tools/executor.py` | **Create** | Dispatch `jarvis:tool:call` → run → `jarvis:tool:result` |
| `openjarvis/conversation/manager.py` | **Create** | State machine: IDLE→LISTEN→THINK→EXECUTE→RESPOND→IDLE |
| `openjarvis/system/config.py` | **Create** | Load `config.yaml` + `.env` → frozen `AppConfig` |
| `openjarvis/__main__.py` | **Modify** | Bootstrap all coroutines with asyncio |
| `tests/unit/test_bus_schemas.py` | **Create** | Envelope round-trip, field validation |
| `tests/unit/test_bus_client.py` | **Create** | publish/subscribe, xadd/xread (real Redis) |
| `tests/unit/test_llm_base.py` | **Create** | Message, ToolCall, ToolSpec validation |
| `tests/unit/test_tool_registry.py` | **Create** | `@tool` decorator, registry lookup |
| `tests/unit/test_tool_executor.py` | **Create** | get_time dispatch end-to-end |
| `tests/unit/test_conversation_manager.py` | **Create** | State transition assertions |
| `tests/unit/test_config.py` | **Create** | Config loading with env override |
| `scripts/sniff_bus.py` | **Create** | Dev tool: print all Redis events in real time |
| `config/prompts/system.md` | **Create** | System prompt for the LLM |

---

## Task 1: Event Bus Schemas

**Files:**
- Create: `openjarvis/bus/schemas.py`
- Create: `tests/unit/test_bus_schemas.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_bus_schemas.py
import time
import pytest
from openjarvis.bus.schemas import (
    Envelope, AudioChunk, WakeEvent, AsrFinal,
    LlmRequest, LlmResponse, ToolCallEvent, ToolResultEvent,
    ConvStateEvent, ConvState,
)

def test_envelope_round_trip():
    env = Envelope(
        source="test",
        trace_id="trace_abc",
        type="test.event",
        payload={"key": "value"},
    )
    data = env.model_dump_json()
    restored = Envelope.model_validate_json(data)
    assert restored.source == "test"
    assert restored.payload == {"key": "value"}
    assert len(restored.id) > 0   # ULID generated
    assert restored.ts > 0

def test_envelope_id_is_unique():
    a = Envelope(source="x", trace_id="t", type="t.e", payload={})
    b = Envelope(source="x", trace_id="t", type="t.e", payload={})
    assert a.id != b.id

def test_audio_chunk_fields():
    chunk = AudioChunk(
        source="audio",
        trace_id="",
        pcm_b64="AAAA",
        sample_rate=16000,
        channels=1,
        frame_ms=30,
    )
    assert chunk.type == "audio.chunk"

def test_wake_event_fields():
    ev = WakeEvent(source="wake", trace_id="t1", model_name="hey_jarvis", score=0.9)
    assert ev.type == "wake.detected"
    assert ev.score == pytest.approx(0.9)

def test_asr_final_fields():
    ev = AsrFinal(source="asr", trace_id="t1", text="what time is it", language="en", duration_s=2.1)
    assert ev.type == "asr.final"
    assert ev.text == "what time is it"

def test_conv_state_event():
    ev = ConvStateEvent(source="conv", trace_id="t", state=ConvState.LISTENING)
    assert ev.state == ConvState.LISTENING
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cd ~/OpenJarvis && python -m pytest tests/unit/test_bus_schemas.py -v 2>&1 | head -30
```
Expected: `ImportError` or `ModuleNotFoundError` (schemas not yet written)

- [ ] **Step 3: Write `openjarvis/bus/schemas.py`**

```python
"""Pydantic event models for the OpenJarvis Redis event bus."""
from __future__ import annotations

import time
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from ulid import ULID


def _ulid() -> str:
    return str(ULID())


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
    pcm_b64: str        # base64-encoded PCM bytes
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

class ConvState(str, Enum):
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
    messages_json: str   # JSON-serialised list[Message]
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
    tool_calls_json: str = "[]"   # JSON list of ToolCall dicts
    finish_reason: str = "stop"
    tokens_in: int = 0
    tokens_out: int = 0


# ── Tools ──────────────────────────────────────────────────────────────────

class ToolCallEvent(Envelope):
    type: str = "tool.call"
    tool_call_id: str
    name: str
    arguments_json: str   # JSON dict


class ToolResultEvent(Envelope):
    type: str = "tool.result"
    tool_call_id: str
    name: str
    result_json: str      # JSON-serialised result value
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
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd ~/OpenJarvis && python -m pytest tests/unit/test_bus_schemas.py -v
```
Expected: 6 tests passing, 0 failures.

- [ ] **Step 5: Commit**

```bash
cd ~/OpenJarvis && git add openjarvis/bus/schemas.py tests/unit/test_bus_schemas.py && git commit -m "feat(bus): add Pydantic event schemas for all bus channels"
```

---

## Task 2: Bus Client

**Files:**
- Create: `openjarvis/bus/client.py`
- Create: `tests/unit/test_bus_client.py`

> **Prerequisite:** Redis running on `localhost:6379`. Tests use a real Redis connection (integration-style unit tests). Flush test keys after each test.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_bus_client.py
import asyncio
import pytest
import pytest_asyncio
from openjarvis.bus.client import BusClient
from openjarvis.bus.schemas import WakeEvent

REDIS_URL = "redis://localhost:6379/1"   # DB 1 = test isolation


@pytest_asyncio.fixture
async def bus():
    client = BusClient(REDIS_URL)
    await client.connect()
    yield client
    await client.flush_db()   # clean up after each test
    await client.close()


@pytest.mark.asyncio
async def test_publish_and_subscribe(bus: BusClient):
    received: list[WakeEvent] = []

    async def handler(ev: WakeEvent) -> None:
        received.append(ev)

    await bus.subscribe("jarvis:wake:detected", WakeEvent, handler)
    ev = WakeEvent(source="test", trace_id="t1", model_name="hey_jarvis", score=0.8)
    await bus.publish("jarvis:wake:detected", ev)
    await asyncio.sleep(0.1)   # let subscriber fire
    assert len(received) == 1
    assert received[0].trace_id == "t1"


@pytest.mark.asyncio
async def test_xadd_and_xread(bus: BusClient):
    from openjarvis.bus.schemas import AsrFinal
    ev = AsrFinal(source="asr", trace_id="t2", text="hello world", language="en")
    await bus.xadd("jarvis:asr:final", ev)
    results = await bus.xread("jarvis:asr:final", count=1)
    assert len(results) == 1
    assert results[0]["text"] == "hello world"


@pytest.mark.asyncio
async def test_set_and_get_state(bus: BusClient):
    from openjarvis.bus.schemas import ConvState
    await bus.set_state("jarvis:conv:state", ConvState.LISTENING)
    state = await bus.get_state("jarvis:conv:state", ConvState)
    assert state == ConvState.LISTENING
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cd ~/OpenJarvis && python -m pytest tests/unit/test_bus_client.py -v 2>&1 | head -20
```
Expected: `ImportError` (BusClient not yet written)

- [ ] **Step 3: Write `openjarvis/bus/client.py`**

```python
"""Redis event bus client — pub/sub, streams, and key-value state."""
from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Coroutine, Type, TypeVar

import redis.asyncio as aioredis
from pydantic import BaseModel

from openjarvis.bus.schemas import Envelope

T = TypeVar("T", bound=Envelope)
Handler = Callable[[Any], Coroutine[Any, Any, None]]


class BusClient:
    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        self._url = url
        self._redis: aioredis.Redis | None = None
        self._pubsub: aioredis.client.PubSub | None = None
        self._subscriptions: dict[str, tuple[Type[Envelope], Handler]] = {}
        self._listener_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        self._redis = await aioredis.from_url(self._url, decode_responses=True)
        self._pubsub = self._redis.pubsub()

    async def close(self) -> None:
        if self._listener_task:
            self._listener_task.cancel()
        if self._pubsub:
            await self._pubsub.close()
        if self._redis:
            await self._redis.aclose()

    async def flush_db(self) -> None:
        """Test helper — flush the current Redis DB."""
        if self._redis:
            await self._redis.flushdb()

    # ── Pub/Sub ──────────────────────────────────────────────────────────

    async def publish(self, channel: str, event: Envelope) -> None:
        assert self._redis is not None
        await self._redis.publish(channel, event.model_dump_json())

    async def subscribe(
        self,
        channel: str,
        model: Type[T],
        handler: Callable[[T], Coroutine[Any, Any, None]],
    ) -> None:
        assert self._pubsub is not None
        self._subscriptions[channel] = (model, handler)  # type: ignore[assignment]
        await self._pubsub.subscribe(channel)
        if not self._listener_task or self._listener_task.done():
            self._listener_task = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        assert self._pubsub is not None
        async for message in self._pubsub.listen():
            if message["type"] != "message":
                continue
            channel = message["channel"]
            if channel not in self._subscriptions:
                continue
            model_cls, handler = self._subscriptions[channel]
            try:
                event = model_cls.model_validate_json(message["data"])
                await handler(event)
            except Exception as exc:  # noqa: BLE001
                # Never crash the listener on a bad message
                print(f"[BusClient] Error handling {channel}: {exc}")

    # ── Streams ──────────────────────────────────────────────────────────

    async def xadd(self, stream: str, event: Envelope) -> None:
        assert self._redis is not None
        data = json.loads(event.model_dump_json())
        # Redis stream fields must be str → str
        flat = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in data.items()}
        await self._redis.xadd(stream, flat)

    async def xread(
        self,
        stream: str,
        count: int = 10,
        last_id: str = "0",
    ) -> list[dict[str, Any]]:
        assert self._redis is not None
        results = await self._redis.xread({stream: last_id}, count=count)
        if not results:
            return []
        entries = []
        for _stream, messages in results:
            for _msg_id, fields in messages:
                decoded = {}
                for k, v in fields.items():
                    try:
                        decoded[k] = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        decoded[k] = v
                entries.append(decoded)
        return entries

    # ── Key-value state ──────────────────────────────────────────────────

    async def set_state(self, key: str, value: Any) -> None:
        assert self._redis is not None
        await self._redis.set(key, value if isinstance(value, str) else value.value)

    async def get_state(self, key: str, model: Type[T] | None = None) -> Any:
        assert self._redis is not None
        raw = await self._redis.get(key)
        if raw is None:
            return None
        if model is not None:
            return model(raw)
        return raw
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd ~/OpenJarvis && python -m pytest tests/unit/test_bus_client.py -v
```
Expected: 3 tests passing. If Redis is not running, start it:
```bash
docker run -d --name openjarvis-redis -p 6379:6379 redis:alpine
```

- [ ] **Step 5: Add sniff script**

```python
# scripts/sniff_bus.py
"""Dev tool: print every Redis pub/sub and stream event in real time."""
import asyncio
import sys

import redis.asyncio as aioredis


async def sniff(url: str = "redis://localhost:6379/0") -> None:
    r = await aioredis.from_url(url, decode_responses=True)
    ps = r.pubsub()
    await ps.psubscribe("jarvis:*")
    print(f"Sniffing all jarvis:* channels on {url} — Ctrl+C to stop\n")
    async for msg in ps.listen():
        if msg["type"] in ("pmessage", "message"):
            print(f"[{msg.get('channel', msg.get('pattern'))}] {msg['data'][:200]}")


if __name__ == "__main__":
    asyncio.run(sniff(*sys.argv[1:]))
```

- [ ] **Step 6: Commit**

```bash
cd ~/OpenJarvis && git add openjarvis/bus/client.py tests/unit/test_bus_client.py scripts/sniff_bus.py && git commit -m "feat(bus): add async Redis BusClient with pub/sub, streams, and state"
```

---

## Task 3: System Config

**Files:**
- Create: `openjarvis/system/config.py`
- Create: `config/prompts/system.md`
- Create: `tests/unit/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_config.py
import os
import pytest
from openjarvis.system.config import load_config, AppConfig


def test_load_defaults(tmp_path):
    """load_config with a minimal yaml returns valid AppConfig."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("llm:\n  provider: openai\n  model: gpt-4o\n")
    cfg = load_config(str(cfg_file))
    assert isinstance(cfg, AppConfig)
    assert cfg.llm.provider == "openai"
    assert cfg.audio.sample_rate == 16000   # default


def test_env_override(tmp_path, monkeypatch):
    """REDIS_URL env var overrides config file value."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("")
    monkeypatch.setenv("REDIS_URL", "redis://myhost:6380/2")
    cfg = load_config(str(cfg_file))
    assert cfg.redis_url == "redis://myhost:6380/2"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nonexistent.yaml"))
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cd ~/OpenJarvis && python -m pytest tests/unit/test_config.py -v 2>&1 | head -20
```

- [ ] **Step 3: Write `openjarvis/system/config.py`**

```python
"""Config loading: YAML file + .env overrides → frozen AppConfig."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    frame_ms: int = 30
    device: int | str | None = None


class WakeConfig(BaseModel):
    enabled: bool = True
    models: list[str] = ["hey_jarvis"]
    threshold: float = 0.5
    cooldown_ms: int = 1500


class AsrConfig(BaseModel):
    model_size: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None
    vad_filter: bool = True


class LlmConfig(BaseModel):
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-5"
    max_tokens: int = 4096
    temperature: float = 0.7
    provider_options: dict[str, Any] = {}


class ConversationConfig(BaseModel):
    silence_timeout_ms: int = 8000
    max_turn_history: int = 20
    system_prompt_file: str = "config/prompts/system.md"


class AppConfig(BaseModel, frozen=True):
    audio: AudioConfig = AudioConfig()
    wake: WakeConfig = WakeConfig()
    asr: AsrConfig = AsrConfig()
    llm: LlmConfig = LlmConfig()
    conversation: ConversationConfig = ConversationConfig()
    redis_url: str = "redis://localhost:6379/0"
    log_level: str = "INFO"
    data_dir: str = "./data"


def load_config(path: str) -> AppConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw: dict[str, Any] = {}
    content = p.read_text(encoding="utf-8").strip()
    if content:
        raw = yaml.safe_load(content) or {}

    # Env overrides
    if url := os.getenv("REDIS_URL"):
        raw["redis_url"] = url
    if level := os.getenv("OPENJARVIS_LOG_LEVEL"):
        raw["log_level"] = level
    if data := os.getenv("OPENJARVIS_DATA_DIR"):
        raw["data_dir"] = data

    return AppConfig.model_validate(raw)
```

- [ ] **Step 4: Write `config/prompts/system.md`**

```markdown
You are Jarvis, an always-on AI operating assistant running on the user's personal computer.

## Personality
- Concise and direct. No filler phrases like "Certainly!" or "Of course!".
- Proactive but not intrusive — if you notice something useful, mention it briefly.
- Honest about uncertainty and limitations.

## Tool use rules
1. Prefer reading before writing.
2. For any destructive operation (delete, send, execute shell), state what you are about to do before calling the tool — the user can hear you.
3. Never invoke `shell.run` unless the user explicitly asked for shell access.
4. If a tool fails, explain to the user and ask before retrying.

## Response style
- Keep responses short — this is a voice interface. One to three sentences is ideal.
- Avoid markdown formatting (bullet points, headers) — they don't render in speech.
- If you need to show code or a long result, say "I'll show that on screen" and print it.
```

- [ ] **Step 5: Run tests — expect green**

```bash
cd ~/OpenJarvis && python -m pytest tests/unit/test_config.py -v
```
Expected: 3 tests passing.

- [ ] **Step 6: Commit**

```bash
cd ~/OpenJarvis && git add openjarvis/system/config.py config/prompts/system.md tests/unit/test_config.py && git commit -m "feat(system): add config loader with YAML + env override"
```

---

## Task 4: LLM Base + Anthropic Provider

**Files:**
- Create: `openjarvis/llm/base.py`
- Create: `openjarvis/llm/providers/anthropic.py`
- Create: `openjarvis/llm/registry.py`
- Create: `tests/unit/test_llm_base.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_llm_base.py
import pytest
from openjarvis.llm.base import Message, ToolCall, ToolSpec, LlmDelta


def test_message_roles():
    for role in ("system", "user", "assistant", "tool"):
        m = Message(role=role, content="hello")
        assert m.role == role


def test_message_invalid_role():
    with pytest.raises(Exception):
        Message(role="robot", content="hi")


def test_tool_call_fields():
    tc = ToolCall(id="tc_1", name="get_time", arguments={})
    assert tc.name == "get_time"


def test_tool_spec_json_schema():
    spec = ToolSpec(
        name="get_time",
        description="Returns the current local time.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    )
    assert spec.name == "get_time"


def test_llm_delta_text():
    d = LlmDelta(text="hello")
    assert d.text == "hello"
    assert d.tool_call is None


def test_llm_delta_tool_call():
    tc = ToolCall(id="x", name="get_time", arguments={})
    d = LlmDelta(tool_call=tc, finish_reason="tool_use")
    assert d.tool_call is not None
    assert d.finish_reason == "tool_use"
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cd ~/OpenJarvis && python -m pytest tests/unit/test_llm_base.py -v 2>&1 | head -20
```

- [ ] **Step 3: Write `openjarvis/llm/base.py`**

```python
"""Abstract LLM provider interface and shared data models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = []
    tool_call_id: str | None = None   # set when role=="tool"
    name: str | None = None           # tool name for role=="tool"


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]    # JSON Schema object


class LlmDelta(BaseModel):
    text: str | None = None
    tool_call: ToolCall | None = None
    finish_reason: Literal["stop", "tool_use", "length"] | None = None


class ProviderError(Exception):
    """Raised by providers on API or parsing failure."""


class BaseProvider(ABC):
    name: str = "base"

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **provider_options: Any,
    ) -> AsyncIterator[LlmDelta]:
        """Stream chat completion. Yields LlmDelta until finish_reason is set."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable."""
        ...
```

- [ ] **Step 4: Write `openjarvis/llm/providers/anthropic.py`**

```python
"""Anthropic (Claude) provider implementation."""
from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

import anthropic

from openjarvis.llm.base import (
    BaseProvider, LlmDelta, Message, ProviderError, ToolCall, ToolSpec,
)


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        self._client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )

    async def health_check(self) -> bool:
        try:
            await self._client.models.list()
            return True
        except Exception:
            return False

    async def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
        *,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **provider_options: Any,
    ) -> AsyncIterator[LlmDelta]:
        # Split system prompt
        system_parts = [m.content for m in messages if m.role == "system" and m.content]
        system = "\n\n".join(system_parts) if system_parts else anthropic.NOT_GIVEN

        # Build Anthropic messages (no system)
        api_msgs: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "assistant" and m.tool_calls:
                api_msgs.append({
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        }
                        for tc in m.tool_calls
                    ],
                })
            elif m.role == "tool":
                api_msgs.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content or "",
                    }],
                })
            else:
                api_msgs.append({"role": m.role, "content": m.content or ""})

        # Build tools
        api_tools = anthropic.NOT_GIVEN
        if tools:
            api_tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.parameters,
                }
                for t in tools
            ]

        return self._stream(
            model=model,
            system=system,
            messages=api_msgs,
            tools=api_tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    async def _stream(self, **kwargs: Any) -> AsyncIterator[LlmDelta]:  # type: ignore[override]
        # Collect complete response (no true streaming in MVP)
        try:
            response = await self._client.messages.create(stream=False, **kwargs)
        except anthropic.APIError as exc:
            raise ProviderError(str(exc)) from exc

        # Yield text delta
        full_text = ""
        pending_tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                full_text = block.text
            elif block.type == "tool_use":
                pending_tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=block.input)
                )

        if full_text:
            yield LlmDelta(text=full_text)

        for tc in pending_tool_calls:
            yield LlmDelta(tool_call=tc, finish_reason="tool_use")

        if not pending_tool_calls:
            yield LlmDelta(finish_reason=response.stop_reason or "stop")
```

- [ ] **Step 5: Write `openjarvis/llm/registry.py`**

```python
"""LLM provider registry."""
from __future__ import annotations

from openjarvis.llm.base import BaseProvider
from openjarvis.llm.providers.anthropic import AnthropicProvider

PROVIDERS: dict[str, type[BaseProvider]] = {
    "anthropic": AnthropicProvider,
}


def load_provider(name: str, **kwargs: object) -> BaseProvider:
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: '{name}'. Available: {sorted(PROVIDERS)}"
        )
    return PROVIDERS[name](**kwargs)  # type: ignore[arg-type]
```

- [ ] **Step 6: Run tests — expect green**

```bash
cd ~/OpenJarvis && python -m pytest tests/unit/test_llm_base.py -v
```
Expected: 6 tests passing.

- [ ] **Step 7: Commit**

```bash
cd ~/OpenJarvis && git add openjarvis/llm/base.py openjarvis/llm/providers/anthropic.py openjarvis/llm/registry.py tests/unit/test_llm_base.py && git commit -m "feat(llm): add BaseProvider, AnthropicProvider, and registry"
```

---

## Task 5: Tool Registry + `get_time` Builtin

**Files:**
- Create: `openjarvis/tools/registry.py`
- Create: `openjarvis/tools/builtin/time_tool.py`
- Create: `openjarvis/tools/executor.py`
- Create: `tests/unit/test_tool_registry.py`
- Create: `tests/unit/test_tool_executor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_tool_registry.py
import pytest
from openjarvis.tools.registry import tool, ToolRegistry


def test_decorator_registers_tool():
    registry = ToolRegistry()

    @tool(name="test.hello", description="Say hello.", registry=registry)
    async def hello(name: str) -> str:
        return f"Hello, {name}!"

    assert "test.hello" in registry
    spec = registry.get_spec("test.hello")
    assert spec.description == "Say hello."


def test_registry_call(event_loop):
    import asyncio
    registry = ToolRegistry()

    @tool(name="test.add", description="Add two numbers.", registry=registry)
    async def add(a: int, b: int) -> int:
        return a + b

    result = asyncio.get_event_loop().run_until_complete(
        registry.call("test.add", {"a": 2, "b": 3})
    )
    assert result == 5


def test_unknown_tool_raises():
    registry = ToolRegistry()
    import asyncio
    with pytest.raises(KeyError):
        asyncio.get_event_loop().run_until_complete(
            registry.call("no.such.tool", {})
        )
```

```python
# tests/unit/test_tool_executor.py
import asyncio
import pytest
from openjarvis.tools.builtin.time_tool import register_time_tool
from openjarvis.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_get_time_returns_string():
    registry = ToolRegistry()
    register_time_tool(registry)
    result = await registry.call("get_time", {})
    assert isinstance(result, str)
    assert len(result) > 0
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cd ~/OpenJarvis && python -m pytest tests/unit/test_tool_registry.py tests/unit/test_tool_executor.py -v 2>&1 | head -20
```

- [ ] **Step 3: Write `openjarvis/tools/registry.py`**

```python
"""Tool registry: @tool decorator + ToolRegistry."""
from __future__ import annotations

import inspect
from typing import Any, Callable

from openjarvis.llm.base import ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._fns: dict[str, Callable[..., Any]] = {}
        self._specs: dict[str, ToolSpec] = {}

    def register(self, name: str, fn: Callable[..., Any], spec: ToolSpec) -> None:
        self._fns[name] = fn
        self._specs[name] = spec

    def get_spec(self, name: str) -> ToolSpec:
        return self._specs[name]

    def all_specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def __contains__(self, name: str) -> bool:
        return name in self._fns

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._fns:
            raise KeyError(f"Tool not found: {name}")
        fn = self._fns[name]
        if inspect.iscoroutinefunction(fn):
            return await fn(**arguments)
        return fn(**arguments)


# Global default registry
_default_registry = ToolRegistry()


def tool(
    *,
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
    registry: ToolRegistry | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a function as a tool."""
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        # Auto-generate JSON schema from type hints if not provided
        schema = parameters or _infer_schema(fn)
        spec = ToolSpec(name=name, description=description, parameters=schema)
        target = registry if registry is not None else _default_registry
        target.register(name, fn, spec)
        return fn
    return decorator


def _infer_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build a minimal JSON Schema from function signature type hints."""
    import typing
    hints = typing.get_type_hints(fn)
    hints.pop("return", None)
    type_map: dict[Any, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}
    props: dict[str, Any] = {}
    for param, hint in hints.items():
        props[param] = {"type": type_map.get(hint, "string")}
    return {"type": "object", "properties": props, "required": list(props)}
```

- [ ] **Step 4: Write `openjarvis/tools/builtin/time_tool.py`**

```python
"""Built-in tool: get_time — returns the current local time."""
from __future__ import annotations

from datetime import datetime

from openjarvis.tools.registry import ToolRegistry, tool


async def _get_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def register_time_tool(registry: ToolRegistry) -> None:
    """Register get_time into the given registry."""
    tool(
        name="get_time",
        description="Returns the current local date and time as a string.",
        parameters={"type": "object", "properties": {}, "required": []},
        registry=registry,
    )(_get_time)
```

- [ ] **Step 5: Write `openjarvis/tools/executor.py`**

```python
"""Tool executor: dispatches tool calls from the LLM to registered tools."""
from __future__ import annotations

import json

from openjarvis.tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(self, name: str, arguments: dict) -> tuple[bool, str]:
        """
        Run a tool by name.

        Returns (success, result_json).
        """
        try:
            result = await self._registry.call(name, arguments)
            return True, json.dumps(result)
        except KeyError:
            return False, json.dumps({"error": f"Unknown tool: {name}"})
        except Exception as exc:  # noqa: BLE001
            return False, json.dumps({"error": str(exc)})
```

- [ ] **Step 6: Run tests — expect green**

```bash
cd ~/OpenJarvis && python -m pytest tests/unit/test_tool_registry.py tests/unit/test_tool_executor.py -v
```
Expected: 4 tests passing.

- [ ] **Step 7: Commit**

```bash
cd ~/OpenJarvis && git add openjarvis/tools/registry.py openjarvis/tools/builtin/time_tool.py openjarvis/tools/executor.py tests/unit/test_tool_registry.py tests/unit/test_tool_executor.py && git commit -m "feat(tools): add ToolRegistry, @tool decorator, get_time builtin, ToolExecutor"
```

---

## Task 6: Audio Capture + Wake Detector + ASR

**Files:**
- Create: `openjarvis/audio/capture.py`
- Create: `openjarvis/wake/detector.py`
- Create: `openjarvis/asr/whisper.py`

> These three modules deal with hardware (mic) and large model files. Tests are integration-style and require a working mic + Redis. They are not run in CI — manual verification only.

- [ ] **Step 1: Write `openjarvis/audio/capture.py`**

```python
"""Microphone capture → publish PCM frames to Redis event bus."""
from __future__ import annotations

import asyncio
import base64
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
        try:
            self._queue.put_nowait(pcm)
        except asyncio.QueueFull:
            pass  # drop frame rather than block

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
        asyncio.create_task(self._publish_loop())

    async def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()

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
```

- [ ] **Step 2: Write `openjarvis/wake/detector.py`**

```python
"""Wake-word detection using openWakeWord."""
from __future__ import annotations

import asyncio
import base64
from typing import Callable, Coroutine, Any

import numpy as np
from openwakeword.model import Model  # type: ignore[import]

from openjarvis.bus.client import BusClient
from openjarvis.bus.schemas import AudioChunk, WakeEvent
from openjarvis.system.config import WakeConfig


WakeCallback = Callable[[WakeEvent], Coroutine[Any, Any, None]]


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
```

- [ ] **Step 3: Write `openjarvis/asr/whisper.py`**

```python
"""ASR: buffer PCM after wake event, transcribe with faster-whisper."""
from __future__ import annotations

import asyncio
import base64
import io
from typing import Any

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel  # type: ignore[import]

from openjarvis.bus.client import BusClient
from openjarvis.bus.schemas import AsrFinal, AudioChunk, WakeEvent
from openjarvis.system.config import AsrConfig, WakeConfig


SILENCE_FRAMES = 40   # ~1.2 s of silence ends the utterance


class WhisperASR:
    def __init__(
        self,
        asr_config: AsrConfig,
        wake_config: WakeConfig,
        bus: BusClient,
    ) -> None:
        self._config = asr_config
        self._bus = bus
        self._model = WhisperModel(
            asr_config.model_size,
            device=asr_config.device,
            compute_type=asr_config.compute_type,
        )
        self._listening = False
        self._buffer: list[bytes] = []
        self._silence_counter = 0

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
        if energy < 200:   # silence threshold (tune per mic)
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

        # faster-whisper expects a file-like or numpy array
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
```

- [ ] **Step 4: Manual smoke test**

```bash
# Start Redis, then in one terminal:
python scripts/sniff_bus.py

# In another terminal:
cd ~/OpenJarvis && python -c "
import asyncio
from openjarvis.bus.client import BusClient
from openjarvis.system.config import AudioConfig, WakeConfig
from openjarvis.audio.capture import AudioCapture
from openjarvis.wake.detector import WakeDetector

async def main():
    bus = BusClient()
    await bus.connect()
    cap = AudioCapture(AudioConfig(), bus)
    wake = WakeDetector(WakeConfig(), bus)
    await wake.start()
    await cap.start()
    print('Listening for wake word... say Hey Jarvis')
    await asyncio.sleep(15)

asyncio.run(main())
"
```
Expected: sniff_bus terminal prints `[jarvis:wake:detected]` when you say "Hey Jarvis".

- [ ] **Step 5: Commit**

```bash
cd ~/OpenJarvis && git add openjarvis/audio/capture.py openjarvis/wake/detector.py openjarvis/asr/whisper.py && git commit -m "feat(audio/wake/asr): add mic capture, openWakeWord detection, faster-whisper ASR"
```

---

## Task 7: ConversationManager (Simplified State Machine)

**Files:**
- Create: `openjarvis/conversation/manager.py`
- Create: `tests/unit/test_conversation_manager.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_conversation_manager.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from openjarvis.conversation.manager import ConversationManager, State
from openjarvis.bus.schemas import AsrFinal, WakeEvent


@pytest.fixture
def mock_bus():
    bus = AsyncMock()
    bus.subscribe = AsyncMock()
    bus.publish = AsyncMock()
    bus.xadd = AsyncMock()
    bus.set_state = AsyncMock()
    bus.xread = AsyncMock(return_value=[])
    return bus


@pytest.fixture
def mock_provider():
    from openjarvis.llm.base import LlmDelta
    provider = AsyncMock()

    async def fake_chat(*args, **kwargs):
        yield LlmDelta(text="The time is 10:00 AM.", finish_reason="stop")

    provider.chat = fake_chat
    return provider


@pytest.fixture
def mock_executor():
    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=(True, '"10:00 AM"'))
    return executor


@pytest.mark.asyncio
async def test_initial_state_is_idle(mock_bus, mock_provider, mock_executor):
    mgr = ConversationManager(mock_bus, mock_provider, mock_executor, model="test-model")
    assert mgr.state == State.IDLE


@pytest.mark.asyncio
async def test_wake_transitions_to_listening(mock_bus, mock_provider, mock_executor):
    mgr = ConversationManager(mock_bus, mock_provider, mock_executor, model="test-model")
    wake_ev = WakeEvent(source="wake", trace_id="t1", model_name="hey_jarvis", score=0.9)
    await mgr._on_wake(wake_ev)
    assert mgr.state == State.LISTENING


@pytest.mark.asyncio
async def test_asr_final_triggers_thinking(mock_bus, mock_provider, mock_executor):
    mgr = ConversationManager(mock_bus, mock_provider, mock_executor, model="test-model")
    mgr._state = State.LISTENING
    mgr._trace_id = "t1"
    asr_ev = AsrFinal(source="asr", trace_id="t1", text="what time is it")
    await mgr._on_asr_final(asr_ev)
    # Should have called the provider
    assert mgr.state == State.IDLE   # ended up back at idle after responding
```

- [ ] **Step 2: Run tests — expect failure**

```bash
cd ~/OpenJarvis && python -m pytest tests/unit/test_conversation_manager.py -v 2>&1 | head -20
```

- [ ] **Step 3: Write `openjarvis/conversation/manager.py`**

```python
"""Simplified ConversationManager for MVP walking skeleton.

States: IDLE → LISTENING → THINKING → (EXECUTING)* → RESPONDING → IDLE
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from openjarvis.bus.client import BusClient
from openjarvis.bus.schemas import (
    AsrFinal, ConvState, ConvStateEvent, LlmResponse,
    ToolCallEvent, ToolResultEvent, WakeEvent,
)
from openjarvis.llm.base import BaseProvider, Message, ToolSpec
from openjarvis.tools.executor import ToolExecutor
from openjarvis.llm.base import ToolCall


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    EXECUTING = "executing"
    RESPONDING = "responding"


class ConversationManager:
    def __init__(
        self,
        bus: BusClient,
        provider: BaseProvider,
        executor: ToolExecutor,
        *,
        model: str,
        max_history: int = 20,
        system_prompt_file: str = "config/prompts/system.md",
        tools: list[ToolSpec] | None = None,
    ) -> None:
        self._bus = bus
        self._provider = provider
        self._executor = executor
        self._model = model
        self._max_history = max_history
        self._tools = tools or []
        self._state = State.IDLE
        self._trace_id = ""
        self._history: list[Message] = []
        self._system_prompt = self._load_system_prompt(system_prompt_file)

    @staticmethod
    def _load_system_prompt(path: str) -> str:
        p = Path(path)
        return p.read_text(encoding="utf-8") if p.exists() else "You are Jarvis, an AI assistant."

    @property
    def state(self) -> State:
        return self._state

    async def start(self) -> None:
        await self._bus.subscribe("jarvis:wake:detected", WakeEvent, self._on_wake)
        await self._bus.subscribe("jarvis:asr:final", AsrFinal, self._on_asr_final)

    async def _set_state(self, new_state: State) -> None:
        old = self._state
        self._state = new_state
        ev = ConvStateEvent(
            source="conversation",
            trace_id=self._trace_id,
            state=ConvState(new_state.value),
            previous=ConvState(old.value),
        )
        await self._bus.publish("jarvis:conv:state", ev)
        await self._bus.set_state("jarvis:conv:state", ConvState(new_state.value))

    async def _on_wake(self, event: WakeEvent) -> None:
        if self._state != State.IDLE:
            return
        self._trace_id = event.id   # use wake event id as trace
        await self._set_state(State.LISTENING)
        print("\n[Jarvis] Listening...")

    async def _on_asr_final(self, event: AsrFinal) -> None:
        if self._state != State.LISTENING:
            return
        print(f"[You] {event.text}")
        self._history.append(Message(role="user", content=event.text))
        # Trim history
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]
        await self._think()

    async def _think(self) -> None:
        await self._set_state(State.THINKING)
        messages = [Message(role="system", content=self._system_prompt)] + self._history

        # Collect full response (MVP: non-streaming)
        full_text = ""
        pending_calls: list[ToolCall] = []

        async for delta in self._provider.chat(
            messages,
            tools=self._tools or None,
            model=self._model,
        ):
            if delta.text:
                full_text += delta.text
            if delta.tool_call:
                pending_calls.append(delta.tool_call)

        if pending_calls:
            # Append assistant turn with tool calls
            self._history.append(Message(role="assistant", content=None, tool_calls=pending_calls))
            await self._execute_tools(pending_calls)
        else:
            self._history.append(Message(role="assistant", content=full_text))
            await self._respond(full_text)

    async def _execute_tools(self, calls: list[ToolCall]) -> None:
        await self._set_state(State.EXECUTING)
        for tc in calls:
            success, result_json = await self._executor.execute(tc.name, tc.arguments)
            self._history.append(Message(
                role="tool",
                content=result_json,
                tool_call_id=tc.id,
                name=tc.name,
            ))
        # Re-enter thinking with tool results
        await self._think()

    async def _respond(self, text: str) -> None:
        await self._set_state(State.RESPONDING)
        print(f"\n[Jarvis] {text}\n")
        await self._set_state(State.IDLE)
```

- [ ] **Step 4: Run tests — expect green**

```bash
cd ~/OpenJarvis && python -m pytest tests/unit/test_conversation_manager.py -v
```
Expected: 3 tests passing.

- [ ] **Step 5: Commit**

```bash
cd ~/OpenJarvis && git add openjarvis/conversation/manager.py tests/unit/test_conversation_manager.py && git commit -m "feat(conversation): add simplified ConversationManager state machine"
```

---

## Task 8: Wire Up `__main__.py` + End-to-End Smoke Test

**Files:**
- Modify: `openjarvis/__main__.py`

- [ ] **Step 1: Rewrite `openjarvis/__main__.py`**

```python
"""Entry point: bootstrap all coroutines and run the event loop."""
from __future__ import annotations

import asyncio
import signal
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.command()
@click.option("--config", default="config/config.yaml", help="Path to config file.")
def main(config: str) -> None:
    """Launch the OpenJarvis voice assistant."""
    cfg_path = Path(config)
    if not cfg_path.exists():
        console.print(f"[yellow]Config not found at {config}, using defaults.[/yellow]")
        console.print("[dim]Tip: cp config/config.example.yaml config/config.yaml[/dim]")

    asyncio.run(_run(config if cfg_path.exists() else None))


async def _run(config_path: str | None) -> None:
    from openjarvis.bus.client import BusClient
    from openjarvis.system.config import AppConfig, load_config
    from openjarvis.audio.capture import AudioCapture
    from openjarvis.wake.detector import WakeDetector
    from openjarvis.asr.whisper import WhisperASR
    from openjarvis.llm.registry import load_provider
    from openjarvis.tools.registry import ToolRegistry
    from openjarvis.tools.builtin.time_tool import register_time_tool
    from openjarvis.tools.executor import ToolExecutor
    from openjarvis.conversation.manager import ConversationManager

    # Load config
    if config_path:
        cfg = load_config(config_path)
    else:
        cfg = AppConfig()

    console.rule("[bold green]OpenJarvis v0.1[/bold green]")
    console.print(f"  Provider : [cyan]{cfg.llm.provider}[/cyan]  Model: [cyan]{cfg.llm.model}[/cyan]")
    console.print(f"  Redis    : [cyan]{cfg.redis_url}[/cyan]")
    console.print(f"  Wake     : [cyan]{cfg.wake.models}[/cyan]")
    console.print()

    # Event bus
    bus = BusClient(cfg.redis_url)
    await bus.connect()

    # Tools
    registry = ToolRegistry()
    register_time_tool(registry)

    # LLM provider
    provider = load_provider(cfg.llm.provider)

    # Executor
    executor = ToolExecutor(registry)

    # Conversation manager
    mgr = ConversationManager(
        bus,
        provider,
        executor,
        model=cfg.llm.model,
        max_history=cfg.conversation.max_turn_history,
        system_prompt_file=cfg.conversation.system_prompt_file,
        tools=registry.all_specs(),
    )
    await mgr.start()

    # ASR
    asr = WhisperASR(cfg.asr, cfg.wake, bus)
    await asr.start()

    # Wake detector
    wake = WakeDetector(cfg.wake, bus)
    await wake.start()

    # Mic capture
    capture = AudioCapture(cfg.audio, bus)
    await capture.start()

    console.print("[bold green]✓ OpenJarvis is running.[/bold green]")
    console.print(f"  Say [bold]'{cfg.wake.models[0].replace('_', ' ')}'[/bold] to wake me up.")
    console.print("  Press [bold]Ctrl+C[/bold] to quit.\n")

    # Run until interrupted
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGINT, stop.set)
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    await stop.wait()

    console.print("\n[yellow]Shutting down...[/yellow]")
    await capture.stop()
    await bus.close()
    console.print("[green]Goodbye.[/green]")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Install package in editable mode and verify CLI works**

```bash
cd ~/OpenJarvis && pip install -e ".[dev]" --quiet && openjarvis --help
```
Expected output contains: `Usage: openjarvis [OPTIONS]`

- [ ] **Step 3: Run all unit tests — full green suite**

```bash
cd ~/OpenJarvis && python -m pytest tests/unit/ -v
```
Expected: All tests pass (no hardware needed for unit tests).

- [ ] **Step 4: End-to-end manual smoke test**

```bash
# Prerequisites:
# 1. Redis running:  docker run -d -p 6379:6379 redis:alpine
# 2. ANTHROPIC_API_KEY set in .env
# 3. Microphone attached

cd ~/OpenJarvis
cp config/config.example.yaml config/config.yaml
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY=sk-ant-...

openjarvis
# Say: "Hey Jarvis" → wait → "What time is it?"
# Expected: terminal prints "[Jarvis] The current time is HH:MM:SS."
```

- [ ] **Step 5: Commit**

```bash
cd ~/OpenJarvis && git add openjarvis/__main__.py && git commit -m "feat: wire up __main__.py — walking skeleton complete"
```

- [ ] **Step 6: Tag v0.0.1-alpha and push**

```bash
cd ~/OpenJarvis && git tag v0.0.1-alpha && git push && git push --tags
```

---

## Verification Checklist (After All Tasks)

- [ ] `python -m pytest tests/unit/ -v` → all green, no skips
- [ ] `ruff check openjarvis/` → no errors
- [ ] `openjarvis --help` → shows usage
- [ ] End-to-end smoke test passes (manual, requires mic + API key)
- [ ] GitHub shows tag `v0.0.1-alpha`
- [ ] README Quick Start instructions work on a fresh clone

---

## Notes for Implementers

- **Redis must be running** before any test that touches BusClient
- **No SQLite writes** in v0.1 — history lives in `ConversationManager._history` (in-memory list)
- **No streaming** in v0.1 — `AnthropicProvider._stream` collects the full response then yields
- The `@tool` decorator uses a `registry=` parameter to avoid global state in tests
- All async tests use `pytest-asyncio` with `asyncio_mode = "auto"` (set in `pyproject.toml`)
- `scripts/sniff_bus.py` is a great debug tool — run it in a separate terminal while testing
