"""Built-in tool: get_time — returns the current local time."""
from __future__ import annotations

from datetime import datetime

from openjarvis.tools.registry import ToolRegistry, tool


async def _get_time() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def register_time_tool(registry: ToolRegistry) -> None:
    """Register get_time into the given registry."""
    tool(
        name="get_time",
        description="Returns the current local date and time as a string.",
        parameters={"type": "object", "properties": {}, "required": []},
        registry=registry,
    )(_get_time)
