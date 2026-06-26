# Redis Event Bus Contract

All inter-module communication flows through Redis. This document is the **source of truth** for channel names and event payloads.

## Channel naming

Pattern: `jarvis:<domain>:<event>`

| Channel | Transport | Payload type |
|---|---|---|
| `jarvis:audio:chunk` | Pub/Sub | `AudioChunk` |
| `jarvis:audio:vad` | Pub/Sub | `VadEvent` |
| `jarvis:wake:detected` | Pub/Sub | `WakeEvent` |
| `jarvis:asr:partial` | Pub/Sub | `AsrPartial` |
| `jarvis:asr:final` | Stream | `AsrFinal` |
| `jarvis:conv:state` | Key + Pub/Sub | `ConvState` |
| `jarvis:llm:request` | Stream | `LlmRequest` |
| `jarvis:llm:delta` | Pub/Sub | `LlmDelta` |
| `jarvis:llm:response` | Stream | `LlmResponse` |
| `jarvis:tool:call` | Stream | `ToolCall` |
| `jarvis:tool:result` | Stream | `ToolResult` |
| `jarvis:tool:confirm` | Pub/Sub | `ToolConfirm` |
| `jarvis:memory:write` | Stream | `MemoryWrite` |
| `jarvis:system:log` | Pub/Sub | `LogEntry` |
| `jarvis:system:shutdown` | Pub/Sub | `ShutdownSignal` |

## Transport choice rules

- **Pub/Sub** — fire-and-forget, no replay. Use for high-frequency or ephemeral events (audio frames, partial ASR, log lines).
- **Stream** (`XADD`/`XREAD`) — persistent, replayable, consumer groups. Use for anything that affects state or must survive a crash (final ASR, LLM I/O, tool calls).
- **Key** — current value snapshot. Use for state that consumers may join late and need to read (e.g. current conversation state).

## Envelope

```python
class Envelope(BaseModel):
    id: str          # ULID
    ts: float        # unix epoch seconds
    source: str      # module name
    trace_id: str    # groups events of one logical turn
    type: str        # mirrors the channel suffix
    payload: dict    # event-specific schema
```

## Trace ID

Every event belonging to one "turn" (wake → final answer) shares the same `trace_id`. The `ConversationManager` generates a new one on each wake event.
