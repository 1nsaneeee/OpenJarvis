"""Tool executor: dispatches tool calls from the LLM to registered tools."""
from __future__ import annotations

import json
from typing import Any

from openjarvis.tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    async def execute(self, name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        """Run a tool by name. Returns (success, result_json)."""
        try:
            result = await self._registry.call(name, arguments)
            return True, json.dumps(result)
        except KeyError:
            return False, json.dumps({"error": f"Unknown tool: {name}"})
        except Exception as exc:  # noqa: BLE001
            return False, json.dumps({"error": str(exc)})
