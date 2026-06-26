# OpenJarvis Design Decisions

> This document captures **every locked decision** made during the brainstorming phase, the **alternatives considered**, and **why we chose what we chose**. It is the source of truth for "why is it built this way?" — both for future contributors and for our own future selves.
>
> **Status:** v0.1 design locked · ready for implementation.

---

## 0. Product Identity

| Field | Value |
|---|---|
| **Name** | OpenJarvis |
| **One-liner** | An open-source, model-agnostic voice AI operating assistant for the desktop. |
| **License** | MIT |
| **Primary OS** | Windows (cross-platform code style, Linux/macOS as best-effort) |
| **Audience** | Developers comfortable cloning a repo and editing `config.yaml` |
| **Languages** | Code in English; UI/docs bilingual (EN + ZH) |

### Use cases (in priority order)
1. Daily office automation
2. Coding / development assistance
3. Always-on companion-style assistant
4. Accessibility / hands-free operation

---

## 1. Top-level Architecture

**Decision:** **Plan B — Event-bus + coroutines**, using **Redis** as the bus.

### Alternatives considered

| Plan | Verdict | Why rejected |
|---|---|---|
| A — Monolithic pipeline | Rejected | Too rigid for the "always-on, partly autonomous" vision; modules can't run independently |
| **B — Redis event bus + asyncio coroutines** | **Chosen** | Modules are independently testable/swappable; matches the eventual visual/background-agent expansion; MCP is naturally event-shaped |
| C — Microservices / gRPC | Rejected | Premature; tanks contributor onboarding for an early-stage OSS project |

### Why Redis specifically
- Simpler than building our own event bus
- `MONITOR` gives instant observability for debugging
- Streams give crash-safe replay for state-affecting events
- Trivially upgradable to multi-host later if ever needed
- Considered alternatives: `asyncio.Queue` (rejected: kills cross-process isolation), `blinker` (rejected: same), in-memory only (rejected: same)

---

## 2. Deployment Model

**Decision:** **Local agent + cloud brain.**

Local: audio capture, wake-word, ASR, OS control, privacy filtering, tool execution.
Cloud (or local LLM): the actual reasoning model.

### Alternatives considered
| Option | Verdict | Why |
|---|---|---|
| 100% local (incl. LLM) | Optional via Ollama, not the default | Hardware cost too high to be the default |
| Local audio + cloud LLM | **Chosen as default** | Best privacy/intelligence/cost balance today |
| 100% cloud | Rejected | Privacy unacceptable; "always-on" should not stream audio offsite |

### Privacy invariant (non-negotiable)
- **Raw audio never leaves the machine.**
- Only **transcribed text** is sent to cloud LLM providers.
- Local keyword/regex filter can redact before any network call.

---

## 3. Control Depth

**Decision:** **Hybrid mode.**

- Routine tasks → scripts / tool calls (fast, safe, deterministic)
- Unknown tasks → escalate to GUI automation (V2+)

### Alternatives considered
| Option | Verdict |
|---|---|
| Advisory only | Too weak — not "taking over" |
| Tool calls only (Claude Code style) | Strong MVP baseline |
| Full Computer Use (mouse + keyboard) | V2/V3; too risky and expensive for MVP |
| **Hybrid** | **Chosen** |

---

## 4. Wake & Session Model

**Decision:** **Wake word + continuous session.**

- openWakeWord detects the wake word
- After wake, enter a continuous conversation (no need to repeat the wake word)
- A `COOLDOWN` state (default **8 seconds** of silence) keeps the conversation open
- Returns to `IDLE` after silence timeout

### Alternatives considered
| Option | Why rejected |
|---|---|
| Wake-word only (every turn) | Too annoying for real conversation |
| Always-listening | Privacy nightmare + false-trigger hell |
| Push-to-talk | Defeats "hands-free" goal |

---

## 5. Memory & Proactivity

### Memory
**Decision:** **Long-term memory + active task tracking** stored in **SQLite**.

Three memory tiers:
- **Short-term** — `messages` table, loaded into LLM context
- **Mid-term** — `conversations.summary` (one-line digest of each session)
- **Long-term** — `user_profile` (key-value preferences) + `facts` (free-form semantic memories)

### Proactivity
**Decision:** **MVP = reactive + scheduled only.** Visual proactivity is V3.

| Stage | Behavior |
|---|---|
| MVP | React to user input; scheduled reminders |
| V2 | Background long-running agent tasks |
| V3 | Event-driven visual (screen capture on demand only — not 24×7 streaming) |

Rationale for *not* doing 24×7 visual: cost scales linearly with frequency, and "barging in" UX is the harder problem than the model capability.

---

## 6. LLM Provider Abstraction

**Decision:** **`BaseProvider` interface with stream-only API and unified tool-calling.**

### Locked choices
- **Streaming only** — non-streaming = collect a stream and join (helper provided)
- **Tool-call deltas yielded only when JSON is complete** — no partial-JSON parsing
- **MVP does not expose** extended-thinking blocks or multimodal input (interface reserved)
- **Errors raise `ProviderError`** — caught uniformly upstream and turned into voice-friendly messages

### Initial providers
- `AnthropicProvider`, `OpenAIProvider`, `GeminiProvider`, `OllamaProvider`
- Adding a new provider = one file + one line in the registry

### Rejected
- Binding to a single SDK (Claude-only) — explicitly out, because we are an OSS project
- LangChain / LiteLLM as the abstraction — adds dependency weight, conflicts with our minimal-deps stance

---

## 7. Tool Execution

**Decision:** **Three-layer executor: built-in Python tools + MCP servers + opt-in shell.**

### Built-in tools (MVP, 8 items)
`filesystem.read`, `filesystem.list`, `filesystem.write` (opt-in), `clipboard.read`, `clipboard.write`, `app.open`, `web.fetch`, `notify`.

### MCP
- Spawned as stdio subprocesses per server (the official MCP pattern)
- Lets users immediately reuse the entire Claude Code MCP ecosystem
- Tools are merged into one flat list before being shown to the LLM — provider-agnostic

### Shell
- **Disabled by default** in `config.yaml`.
- When enabled: `deny_list` always applied, optional `allow_list`, `require_confirmation: true` by default.
- Every execution audited to SQLite `tool_audit`.

### Safety pattern
- Any tool flagged `risk="high"` (and any shell call) triggers the `CONFIRMING` state machine branch.
- The LLM does **not** see whether something is local / MCP / shell — it just sees `ToolSpec`s.

---

## 8. Conversation State Machine

**Decision:** **7 states, hand-rolled (no library).**

```
IDLE → LISTENING → THINKING → (EXECUTING ↔ CONFIRMING)* → RESPONDING → COOLDOWN → IDLE
```

### Locked details
- `trace_id` is generated on `IDLE → LISTENING` and shared by every Redis event in that turn
- Current state is mirrored to Redis key `jarvis:conv:state` so other modules can read it
- `CONFIRMING` is a dedicated state (chosen over reusing `LISTENING`) — keeps the LISTEN handler pure
- Confirmation timeout = default refuse → goes back to THINKING with a "user declined" tool result
- COOLDOWN = 8s (configurable)

### Rejected
- `transitions` / `python-statemachine` libraries — clear `dict[State, dict[Event, Handler]]` reads better than a DSL for this state count

---

## 9. Event Bus Contract

**Decision:** Channels follow `jarvis:<domain>:<event>`. Three transports used by intent:

| Transport | Used for |
|---|---|
| **Pub/Sub** | Ephemeral, high-frequency (audio frames, partial ASR, logs) |
| **Streams** (`XADD`/`XREAD`) | Anything state-affecting or crash-recoverable (final ASR, LLM I/O, tool calls) |
| **Keys** | Snapshot state readable by latecomers (current conv state) |

### Envelope
```json
{ "id": "ulid", "ts": float, "source": "...", "trace_id": "...", "type": "...", "payload": {...} }
```

### Rejected
- All-Streams (uniform but heavy)
- Protobuf payloads (overkill for OSS debuggability — JSON wins)

---

## 10. Storage / SQLite

**Decision:** **Single SQLite file with 7 tables**, WAL mode + `synchronous=NORMAL`.

Tables: `conversations`, `messages`, `user_profile`, `facts`, `tasks`, `tool_audit`, `meta`.

### Locked
- All timestamps = unix epoch (REAL)
- All writes go through one subscriber (`MemoryWriter`); no module writes the DB directly
- Default retention: 90 days for `conversations` (configurable)
- Schema versioning via `meta.schema_version` + simple if-chain migrations (no Alembic)

### Rejected
- DuckDB / Postgres (deployment friction for OSS users)
- Multi-user schema (V1 is single-user-single-machine)
- Default encryption (documented opt-in via SQLCipher instead)
- Embedded vector store at MVP (V2 = `sqlite-vec`)

---

## 11. Non-Functional Requirements

| Concern | Decision |
|---|---|
| Latency | **No hard target.** Correctness first. |
| Availability | Single-process; crash → user restarts. |
| Privacy | Audio stays local; only text crosses network. Sensitive-keyword local filter available. |
| Security | Destructive ops require voice confirmation; configurable allow/deny lists; full tool audit log. |
| Maintainer | OSS project — code must be modular, tested, documented. |
| Observability | Structured JSON logs; Redis bus is itself the trace. |
| Cost | Default config should run on the **cheapest** viable LLM (e.g. Gemini Flash or local Ollama). |

---

## 12. Tech Stack (Final)

| Concern | Tool |
|---|---|
| Language | Python 3.11+ |
| Async | `asyncio` |
| Event bus | Redis 7+ |
| ASR | `faster-whisper` |
| Wake | `openWakeWord` |
| Audio I/O | `sounddevice` |
| LLM SDKs | `anthropic`, `openai`, `google-generativeai`, `ollama` |
| MCP | official `mcp` Python SDK |
| Memory | `aiosqlite` |
| Models | `pydantic` v2 |
| Config | `pyyaml` + `python-dotenv` |
| Logging | `structlog` |
| CLI | `click` + `rich` |
| Tests | `pytest` + `pytest-asyncio` |
| Lint/Type | `ruff` + `mypy --strict` |

### Explicitly **not** in the stack
- LangChain / LlamaIndex (over-abstraction)
- Celery (Redis is enough)
- FastAPI (no HTTP server in MVP)
- DuckDB / Postgres (SQLite is enough)
- transitions / python-statemachine (hand-rolled is clearer)

---

## 13. Roadmap

| Phase | Scope |
|---|---|
| **v0.1 (MVP)** | Wake → ASR → 1 LLM (Claude) → 1 tool (`get_time`) → text output. Public GitHub release end of week 1. |
| **v0.2** | Built-in tools (filesystem/clipboard/web), all 4 providers, MCP, full state machine, SQLite memory |
| **v0.3** | TTS voice output, background long-running agent tasks |
| **v0.4+** | Event-driven visual context (screen capture on demand), tray/GUI runtime |

---

## 14. The Walking Skeleton (Week 1)

**Goal:** "Hey Jarvis, what time is it?" → spoken-then-printed answer, working end to end.

Deliberately excluded from week 1:
- Multi-provider (Claude only)
- MCP
- Streaming ASR / LLM
- SQLite (in-memory list is fine)
- COOLDOWN, CONFIRMING
- Structured logging

**Key property:** the skeleton is not throwaway code. Every line lives in the final architecture's modules — just with fewer states, fewer providers, fewer tools.

Public GitHub push targeted for **Day 7**.

---

## 15. Open Questions (V2+)

These are intentionally not decided now:

- **Embedding model** for `facts` semantic search (likely local `bge-small` or provider-side)
- **TTS engine** choice (Edge TTS / Piper / ElevenLabs) — pick based on Day-30 user feedback
- **Visual perception trigger heuristics** (when does the agent decide to look at the screen?)
- **Multi-machine sync** (probably never in core; left to user via `litestream` or similar)
- **Plugin marketplace** — wait until there's a real ecosystem before designing one

---

*Document version: 1.0 · Authored during brainstorming session · See git history for revisions.*
