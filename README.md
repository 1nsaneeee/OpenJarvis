# 🤖 OpenJarvis

> An open-source, model-agnostic voice AI operating assistant — your always-on Jarvis for the desktop.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[English](#english) | [中文](#中文)

---

## English

### What is OpenJarvis?

OpenJarvis is a local-first voice AI assistant framework that lets you control your computer by talking to it. It is:

- **Model-agnostic** — plug in Claude, GPT-4o, Gemini, or a local Ollama model
- **Privacy-first** — audio never leaves your machine; only transcribed text is sent to cloud APIs
- **Extensible** — add tools via MCP servers or local Python functions
- **Open** — MIT licensed, built for contributors

### Architecture

```
Microphone → Wake Word (openWakeWord) → ASR (faster-whisper)
                                              ↓
                                    Redis Event Bus
                                              ↓
                              ConversationManager (state machine)
                                              ↓
                               LLM Provider (pluggable adapter)
                                    ↙           ↘
                           Tool Executor      Memory (SQLite)
                          (local + MCP)
```

### Roadmap

| Phase | Features |
|-------|----------|
| **MVP (v0.1)** | Wake word → ASR → LLM → tool execution → terminal output |
| **v0.2** | TTS voice output, background task agent |
| **v0.3** | Screen capture, event-driven visual context |

### Quick Start

```bash
# 1. Clone
git clone https://github.com/1nsaneeee/OpenJarvis.git
cd OpenJarvis

# 2. Install
pip install -e ".[dev]"

# 3. Configure
cp config/config.example.yaml config/config.yaml
cp .env.example .env
# Edit .env with your API keys

# 4. Start Redis
docker run -d -p 6379:6379 redis:alpine

# 5. Run
python -m openjarvis
```

### Requirements

- Python 3.11+
- Redis 7+
- A microphone
- An LLM API key (Claude / OpenAI / Gemini) **or** Ollama running locally

---

## 中文

### OpenJarvis 是什么?

OpenJarvis 是一个本地优先的语音 AI 助手框架，让你通过说话来控制电脑。

- **模型无关** — 支持 Claude、GPT-4o、Gemini 或本地 Ollama 模型
- **隐私优先** — 音频不离开本机，只有转写后的文字才会发送到云端
- **可扩展** — 通过 MCP server 或本地 Python 函数添加工具
- **开源** — MIT 协议，欢迎贡献

### 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/1nsaneeee/OpenJarvis.git
cd OpenJarvis

# 2. 安装依赖
pip install -e ".[dev]"

# 3. 配置
cp config/config.example.yaml config/config.yaml
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 4. 启动 Redis
docker run -d -p 6379:6379 redis:alpine

# 5. 运行
python -m openjarvis
```

### 系统要求

- Python 3.11+
- Redis 7+
- 麦克风
- LLM API Key（Claude / OpenAI / Gemini）**或** 本地运行的 Ollama

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). All contributions welcome — new LLM providers, tools, bug fixes, docs.

## License

MIT © OpenJarvis Contributors
