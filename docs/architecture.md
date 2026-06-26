# OpenJarvis Architecture

## Overview

OpenJarvis is built as a set of **independent coroutines** communicating through a **Redis event bus**. No module imports another module's runtime code directly — they only share schemas (Pydantic models in `openjarvis.bus.schemas`).

```
┌─────────────┐  audio   ┌─────────────┐  wake   ┌──────────────────┐
│ AudioCapture├─────────►│ WakeDetect  ├────────►│ ConversationMgr  │
└─────────────┘          └─────────────┘         │  (state machine) │
                                                  └────────┬─────────┘
                                                           │
                ┌──────────────────────────────────────────┼────────────┐
                ▼                                          ▼            ▼
        ┌──────────────┐                         ┌──────────────┐  ┌──────────┐
        │ ASR (FW)     │   asr.final             │ LLM Provider │  │ Memory   │
        │ (streaming)  ├────────────────────────►│ (adapter)    │  │ (SQLite) │
        └──────────────┘                         └──────┬───────┘  └────▲─────┘
                                                        │ tool.call     │
                                                        ▼               │
                                                ┌──────────────┐        │
                                                │ ToolExecutor │────────┘
                                                │ (local + MCP)│
                                                └──────────────┘
```

## Module responsibilities

| Module | Subscribes to | Publishes to |
|---|---|---|
| `audio` | — | `jarvis:audio:chunk`, `jarvis:audio:vad` |
| `wake` | `jarvis:audio:chunk` | `jarvis:wake:detected` |
| `asr` | `jarvis:audio:chunk` (after wake) | `jarvis:asr:partial`, `jarvis:asr:final` |
| `conversation` | `jarvis:wake:detected`, `jarvis:asr:final`, `jarvis:tool:result` | `jarvis:conv:state`, `jarvis:llm:request` |
| `llm` | `jarvis:llm:request` | `jarvis:llm:delta`, `jarvis:llm:response` |
| `tools` | `jarvis:tool:call` | `jarvis:tool:result`, `jarvis:tool:confirm` |
| `memory` | `jarvis:asr:final`, `jarvis:llm:response`, `jarvis:tool:result` | — |

## Event envelope

All events share the same outer shape:

```json
{
  "id": "evt_01HXYZ...",
  "ts": 1719360000.123,
  "source": "asr",
  "trace_id": "conv_abc123",
  "type": "asr.final",
  "payload": { ... }
}
```

See `openjarvis/bus/schemas.py` for full Pydantic definitions.

## Why Redis?

- **Decoupling** — modules can be restarted independently
- **Observability** — `redis-cli MONITOR` shows everything in real time
- **Replay** — Streams persist; crashed modules resume from last offset
- **Future** — easy migration to multi-host if ever needed
