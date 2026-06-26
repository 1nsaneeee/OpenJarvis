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
