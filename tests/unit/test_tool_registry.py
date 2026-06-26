"""Tests for ToolRegistry and @tool decorator."""
import asyncio

import pytest

from openjarvis.tools.registry import ToolRegistry, tool


def test_decorator_registers_tool() -> None:
    registry = ToolRegistry()

    @tool(name="test.hello", description="Say hello.", registry=registry)
    async def hello(name: str) -> str:
        return f"Hello, {name}!"

    assert "test.hello" in registry
    spec = registry.get_spec("test.hello")
    assert spec.description == "Say hello."


def test_registry_call() -> None:
    registry = ToolRegistry()

    @tool(name="test.add", description="Add two numbers.", registry=registry)
    async def add(a: int, b: int) -> int:
        return a + b

    result = asyncio.run(registry.call("test.add", {"a": 2, "b": 3}))
    assert result == 5


def test_unknown_tool_raises() -> None:
    registry = ToolRegistry()
    with pytest.raises(KeyError):
        asyncio.run(registry.call("no.such.tool", {}))


def test_all_specs_returns_list() -> None:
    registry = ToolRegistry()

    @tool(name="test.noop", description="Does nothing.", registry=registry)
    async def noop() -> None:
        pass

    specs = registry.all_specs()
    assert len(specs) == 1
    assert specs[0].name == "test.noop"
