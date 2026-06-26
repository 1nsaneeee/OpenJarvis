"""Tests for AnthropicProvider message conversion, stop_reason mapping, and errors."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import pytest

from openjarvis.llm.base import LlmDelta, Message, ProviderError, ToolSpec
from openjarvis.llm.providers.anthropic import _STOP_REASON_MAP, AnthropicProvider
from openjarvis.llm.registry import load_provider


def _fake_response(content_blocks: list[SimpleNamespace], stop_reason: str) -> SimpleNamespace:
    return SimpleNamespace(content=content_blocks, stop_reason=stop_reason)


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_block(id: str, name: str, input: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def _fake_request() -> httpx.Request:
    return httpx.Request("GET", "https://api.anthropic.com/v1/messages")


@pytest.mark.asyncio
async def test_chat_yields_text_delta_and_finish_reason() -> None:
    provider = AnthropicProvider(api_key="test")
    fake = _fake_response([_text_block("hello world")], stop_reason="end_turn")
    with patch.object(provider._client.messages, "create", new=AsyncMock(return_value=fake)):
        deltas = [d async for d in provider.chat(
            [Message(role="user", content="hi")],
            model="claude-test",
        )]
    assert any(d.text == "hello world" for d in deltas)
    assert deltas[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_chat_yields_tool_call_delta() -> None:
    provider = AnthropicProvider(api_key="test")
    fake = _fake_response(
        [_tool_block("tu_1", "get_time", {})],
        stop_reason="tool_use",
    )
    with patch.object(provider._client.messages, "create", new=AsyncMock(return_value=fake)):
        deltas = [d async for d in provider.chat(
            [Message(role="user", content="time?")],
            tools=[ToolSpec(name="get_time", description="x", parameters={"type": "object"})],
            model="claude-test",
        )]
    tool_deltas = [d for d in deltas if d.tool_call is not None]
    assert len(tool_deltas) == 1
    assert tool_deltas[0].tool_call.name == "get_time"  # type: ignore[union-attr]
    assert tool_deltas[0].finish_reason == "tool_use"


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,mapped", [
    ("end_turn", "stop"),
    ("stop_sequence", "stop"),
    ("max_tokens", "length"),
    ("pause_turn", "stop"),
    ("refusal", "stop"),
])
async def test_stop_reason_mapping(raw: str, mapped: str) -> None:
    provider = AnthropicProvider(api_key="test")
    fake = _fake_response([_text_block("ok")], stop_reason=raw)
    with patch.object(provider._client.messages, "create", new=AsyncMock(return_value=fake)):
        deltas = [d async for d in provider.chat(
            [Message(role="user", content="x")],
            model="claude-test",
        )]
    finish = [d.finish_reason for d in deltas if d.finish_reason is not None]
    assert finish[-1] == mapped


def test_stop_reason_map_values_are_all_valid() -> None:
    """Every mapped value must be a valid LlmDelta.finish_reason."""
    for value in _STOP_REASON_MAP.values():
        d = LlmDelta(finish_reason=value)
        assert d.finish_reason == value


@pytest.mark.asyncio
async def test_chat_raises_provider_error_on_api_error() -> None:
    provider = AnthropicProvider(api_key="test")
    err = anthropic.APIError("boom", request=_fake_request(), body=None)
    with patch.object(provider._client.messages, "create", new=AsyncMock(side_effect=err)), \
            pytest.raises(ProviderError):
            async for _ in provider.chat(
                [Message(role="user", content="x")],
                model="claude-test",
            ):
                pass


@pytest.mark.asyncio
async def test_health_check_returns_false_on_api_error() -> None:
    provider = AnthropicProvider(api_key="test")
    err = anthropic.APIError("nope", request=_fake_request(), body=None)
    with patch.object(provider._client.models, "list", new=AsyncMock(side_effect=err)):
        result = await provider.health_check()
    assert result is False


@pytest.mark.asyncio
async def test_health_check_returns_true_on_success() -> None:
    provider = AnthropicProvider(api_key="test")
    with patch.object(
        provider._client.models, "list", new=AsyncMock(return_value=SimpleNamespace())
    ):
        result = await provider.health_check()
    assert result is True


def test_load_provider_anthropic() -> None:
    p = load_provider("anthropic", api_key="test")
    assert isinstance(p, AnthropicProvider)


def test_load_provider_unknown_raises() -> None:
    with pytest.raises(ValueError):
        load_provider("does_not_exist")
