# OpenJarvis v0.0.1-alpha 交接文档

> 标签：`v0.0.1-alpha` | 分支：`master` | 日期：2026-06-26

---

## 一、项目目标

OpenJarvis 是一个**开源、模型无关的语音 AI 操作助手**。用户说出唤醒词，助手转录语音、调用 LLM、执行工具调用，最终将结果打印到终端（TTS 为后续里程碑）。

本版本（walking skeleton）的目标是打通端到端链路，验证所有子系统可以协同工作，**不追求功能完整，只追求链路畅通**。

---

## 二、架构概览

```
麦克风
  │  (int16 PCM)
  ▼
AudioCapture ──publish──▶ Redis pub/sub: jarvis:audio:chunk
                                │
                    ┌───────────┴────────────┐
                    ▼                        ▼
             WakeDetector            WhisperASR
             (openWakeWord)          (faster-whisper)
                    │                        │
                    │ jarvis:wake:detected   │ jarvis:asr:final
                    └──────────┬─────────────┘
                               ▼
                      ConversationManager
                      (状态机: IDLE→LISTEN→THINK→EXECUTE→RESPOND)
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
             BaseProvider           ToolExecutor
             (Anthropic Claude)     (get_time 等)
                    │
                    ▼
              终端打印响应
```

**事件总线通道一览：**

| 通道 | 发布方 | 订阅方 |
|---|---|---|
| `jarvis:audio:chunk` | AudioCapture | WakeDetector, WhisperASR |
| `jarvis:wake:detected` | WakeDetector | WhisperASR, ConversationManager |
| `jarvis:asr:final` | WhisperASR | ConversationManager |
| `jarvis:conv:state` | ConversationManager | （监控用，无订阅方） |

---

## 三、代码结构

```
openjarvis/
├── __main__.py              # CLI 入口，bootstrap 所有协程
├── bus/
│   ├── schemas.py           # Pydantic 事件模型（14 个事件类型 + ConvState）
│   └── client.py            # BusClient：pub/sub + streams + state KV
├── system/
│   └── config.py            # 冻结 AppConfig，YAML + 环境变量覆盖
├── llm/
│   ├── base.py              # BaseProvider ABC, Message, LlmDelta, ToolSpec, ToolCall
│   ├── registry.py          # load_provider("anthropic") 工厂
│   └── providers/
│       └── anthropic.py     # AnthropicProvider（非流式，stream=False）
├── tools/
│   ├── registry.py          # ToolRegistry + @tool 装饰器 + _infer_schema
│   ├── executor.py          # ToolExecutor.execute(name, args) → (bool, json)
│   └── builtin/
│       └── time_tool.py     # get_time 内置工具
├── audio/
│   └── capture.py           # AudioCapture：sounddevice → Redis pub/sub
├── wake/
│   └── detector.py          # WakeDetector：openWakeWord ONNX 模型
├── asr/
│   └── whisper.py           # WhisperASR：faster-whisper + 能量 VAD
└── conversation/
    └── manager.py           # ConversationManager：状态机 + 历史管理
```

```
tests/unit/
├── test_bus_schemas.py       # 18 tests
├── test_bus_client.py        # 3 tests（需要 Redis）
├── test_config.py            # 3 tests
├── test_llm_base.py          # 7 tests
├── test_anthropic_provider.py # 20 tests（含参数化 stop_reason 映射）
├── test_tool_registry.py     # 4 tests
├── test_tool_executor.py     # 4 tests
└── test_conversation_manager.py # 9 tests
```

**共 61 个单元测试，全部 passing。**

---

## 四、快速上手

### 前置依赖

```bash
# 1. Python 3.11+
# 2. Redis（本地）
#    Windows: C:/Users/zhouzixi/redis/redis-server.exe --port 6379
#    Linux/Mac: redis-server
# 3. 麦克风
```

### 安装

```bash
git clone https://github.com/1nsaneeee/OpenJarvis.git
cd OpenJarvis
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Unix
pip install -e ".[dev]"
```

### 配置

```bash
cp config/config.example.yaml config/config.yaml
cp .env.example .env
# 编辑 .env，填入：
# ANTHROPIC_API_KEY=sk-ant-...
```

### 运行

```bash
# 先确保 Redis 在运行
openjarvis
# 或：python -m openjarvis

# 说 "Hey Jarvis"，然后说 "What time is it?"
# 预期输出：[Jarvis] The current time is 14:30:00.
```

### 调试总线事件

```bash
# 另开一个终端
python scripts/sniff_bus.py
# 会打印所有经过 Redis 的事件
```

### 只跑测试（不需要麦克风）

```bash
# 需要 Redis 在运行（test_bus_client.py 需要）
python -m pytest tests/unit/ -v

# 跳过 Bus Client 集成测试（完全离线）
python -m pytest tests/unit/ -v --ignore=tests/unit/test_bus_client.py
```

---

## 五、关键设计决策

### Redis 协议兼容性
redis-py 8.x 默认使用 RESP3（HELLO 命令），但本机安装的是 Redis 5.0.14，不支持 RESP3。
**解决方案：** `from_url(..., protocol=2)` 强制 RESP2。注释在 `bus/client.py` 中有说明。

### AnthropicProvider 停止原因映射
Anthropic API 返回 `"end_turn"` / `"stop_sequence"` 等原始值，但 `LlmDelta.finish_reason` 是 `Literal["stop", "tool_use", "length"]`。
**解决方案：** `_STOP_REASON_MAP` 字典做翻译，避免 Pydantic ValidationError。

### `BaseProvider.chat()` 是同步方法
`chat()` 返回 `AsyncIterator[LlmDelta]`，但方法本身是**同步**的（不需要 `await`）。调用方式：
```python
async for delta in provider.chat(messages, model=...):
    ...
```

### `asyncio.to_thread` for Whisper
`faster_whisper.transcribe()` 是同步 CPU 密集操作，会阻塞事件循环。
**解决方案：** `_do_transcribe()` 内部闭包（含 segment 迭代）通过 `await asyncio.to_thread(...)` 在线程池执行。

### ConversationManager 工具调用防递归
`_think()` → `_execute_tools()` → `_think()` 可能无限递归。
**解决方案：** `max_tool_rounds=5` 参数，超限后打印错误信息并回到 IDLE。

### Windows 信号处理
`loop.add_signal_handler()` 在 Windows 上抛 `NotImplementedError`。
**解决方案：** `asyncio.Event().wait()` 永久阻塞，`asyncio.run()` 外层 `except KeyboardInterrupt` 捕获 Ctrl+C，`finally` 块执行清理。

---

## 六、配置说明

`config/config.yaml`（从 `config.example.yaml` 复制）主要字段：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `audio.sample_rate` | 16000 | 麦克风采样率 |
| `wake.models` | `["hey_jarvis"]` | openWakeWord 模型名 |
| `wake.threshold` | 0.5 | 唤醒置信度阈值 |
| `asr.model_size` | `"base"` | Whisper 模型大小 |
| `asr.device` | `"cpu"` | `"cpu"` 或 `"cuda"` |
| `llm.provider` | `"anthropic"` | 目前只有 `"anthropic"` 可用 |
| `llm.model` | `"claude-sonnet-4-5"` | Claude 模型 ID |
| `conversation.max_turn_history` | 20 | 上下文窗口保留的轮次 |
| `redis_url` | `"redis://localhost:6379/0"` | 可用 `REDIS_URL` 环境变量覆盖 |

> ⚠️ `config.example.yaml` 中的 `tools`、`memory`、`logging` 三个节在当前版本**未被解析**，配置了也不生效。

---

## 七、已知限制（v0.0.1-alpha）

| 限制 | 规划版本 |
|---|---|
| 无 TTS，响应只打印到终端 | v0.2 |
| AnthropicProvider 非流式（完整响应后一次性 yield） | v0.2 |
| 仅 Anthropic 一个 Provider | v0.2（OpenAI、Ollama） |
| 仅 `get_time` 一个内置工具 | v0.2 |
| 对话历史纯内存，重启丢失 | v0.2（SQLite） |
| `silence_timeout_ms` 配置项已定义但未接入 | v0.2 |
| `trace_id` 在 AudioChunk 中为空字符串 | v0.2 |
| `tools`/`memory`/`logging` 配置节未解析 | v0.2 |

---

## 八、下一步开发建议

优先级排序（基于走骨架设计文档）：

1. **TTS 集成**（`openjarvis/tts/`）— 让 Jarvis 真正"说话"
2. **流式 LLM 响应** — `AnthropicProvider` 改为真正流式，减少首字节延迟
3. **更多 Provider**（OpenAI、Gemini、Ollama）— `llm/providers/` 下新增
4. **SQLite 记忆层**（`openjarvis/memory/`）— 持久化对话历史
5. **更多内置工具**（文件读写、剪贴板、Web fetch）— `tools/builtin/` 下新增
6. **修复 `config.example.yaml` 与 AppConfig 的字段漂移**

---

## 九、提交历史摘要

```
3da9e16  feat: OpenJarvis walking skeleton v0.0.1-alpha   ← merge commit / tag
8daa6e6  fix(__main__): graceful shutdown cleanup, Redis connect error handling
1a43d6d  feat: wire up __main__.py — walking skeleton complete
f0f532a  fix(conversation): bound tool recursion, recover from provider errors, trim history on every append
ca895cc  feat(conversation): add simplified ConversationManager state machine
61f5ab7  fix(audio/wake/asr): unblock event loop, configurable VAD thresholds, audio publish error recovery
cb2542e  feat(audio/wake/asr): add mic capture, openWakeWord detection, faster-whisper ASR
04d6406  fix(tools): narrow executor exception handling, fix _infer_schema required fields, add missing test
57209bb  feat(tools): add ToolRegistry, @tool decorator, get_time builtin, ToolExecutor
d2e7a3e  test(llm): add AnthropicProvider tests; tighten types and exception scope
cfb2c4d  fix(llm): map anthropic stop_reason to LlmDelta finish_reason Literal
a875507  feat(llm): add BaseProvider, AnthropicProvider, and registry
a5df208  refactor(system): freeze sub-configs, document load_config
3368b53  feat(system): add config loader with YAML + env override
959eefb  refactor(bus): use structlog, narrow set_state type, document xread cursor
b710452  refactor(bus): address review feedback — await listener task, dedup xadd serialization
b88ffee  feat(bus): add async Redis BusClient with pub/sub, streams, and state
1cc2d4d  fix(bus/schemas): improve test coverage, add frozen model config, __all__
3a971e6  feat(bus): add Pydantic event schemas for all bus channels
```

---

*文档生成时间：2026-06-26 | 作者：Claude (subagent-driven development)*
