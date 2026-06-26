"""LLM provider registry."""
from __future__ import annotations

from typing import Any

from openjarvis.llm.base import BaseProvider
from openjarvis.llm.providers.anthropic import AnthropicProvider

PROVIDERS: dict[str, type[BaseProvider]] = {
    "anthropic": AnthropicProvider,
}


def load_provider(name: str, **kwargs: Any) -> BaseProvider:
    if name not in PROVIDERS:
        raise ValueError(
            f"Unknown provider: '{name}'. Available: {sorted(PROVIDERS)}"
        )
    return PROVIDERS[name](**kwargs)
