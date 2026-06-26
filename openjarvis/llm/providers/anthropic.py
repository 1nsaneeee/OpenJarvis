"""Anthropic (Claude) provider implementation."""
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from openjarvis.llm.base import (
    BaseProvider,
    LlmDelta,
    Message,
    ProviderError,
    ToolCall,
    ToolSpec,
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
        api_tools: Any = anthropic.NOT_GIVEN
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

    async def _stream(self, **kwargs: Any) -> AsyncIterator[LlmDelta]:
        # Collect complete response (no true streaming in MVP)
        try:
            response = await self._client.messages.create(stream=False, **kwargs)
        except anthropic.APIError as exc:
            raise ProviderError(str(exc)) from exc

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
