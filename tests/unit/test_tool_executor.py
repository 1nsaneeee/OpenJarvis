"""Tests for ToolExecutor."""
import pytest

from openjarvis.tools.builtin.time_tool import register_time_tool
from openjarvis.tools.executor import ToolExecutor
from openjarvis.tools.registry import ToolRegistry, tool


@pytest.mark.asyncio
async def test_get_time_returns_string() -> None:
    registry = ToolRegistry()
    register_time_tool(registry)
    result = await registry.call("get_time", {})
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_executor_success() -> None:
    registry = ToolRegistry()
    register_time_tool(registry)
    executor = ToolExecutor(registry)
    ok, result_json = await executor.execute("get_time", {})
    assert ok is True
    assert len(result_json) > 0


@pytest.mark.asyncio
async def test_executor_unknown_tool() -> None:
    registry = ToolRegistry()
    executor = ToolExecutor(registry)
    ok, result_json = await executor.execute("no.such.tool", {})
    assert ok is False
    assert "Unknown tool" in result_json


@pytest.mark.asyncio
async def test_executor_tool_raises() -> None:
    """Executor converts tool runtime errors to (False, error_json)."""
    registry = ToolRegistry()

    @tool(name="test.broken", description="Broken tool.", registry=registry)
    async def broken() -> str:
        raise ValueError("simulated failure")

    executor = ToolExecutor(registry)
    ok, result_json = await executor.execute("test.broken", {})
    assert ok is False
    assert "simulated failure" in result_json
