"""Abstract LLM provider interface and shared data models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]    # JSON Schema object


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None   # set when role=="tool"
    name: str | None = None           # tool name for role=="tool"


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


__all__ = ["BaseProvider", "LlmDelta", "Message", "ProviderError", "ToolCall", "ToolSpec"]
