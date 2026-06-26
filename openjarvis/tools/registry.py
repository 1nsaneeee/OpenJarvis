"""Tool registry: @tool decorator + ToolRegistry."""
from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from typing import Any

from openjarvis.llm.base import ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._fns: dict[str, Callable[..., Any]] = {}
        self._specs: dict[str, ToolSpec] = {}

    def register(self, name: str, fn: Callable[..., Any], spec: ToolSpec) -> None:
        self._fns[name] = fn
        self._specs[name] = spec

    def get_spec(self, name: str) -> ToolSpec:
        return self._specs[name]

    def all_specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def __contains__(self, name: str) -> bool:
        return name in self._fns

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._fns:
            raise KeyError(f"Tool not found: {name}")
        fn = self._fns[name]
        if inspect.iscoroutinefunction(fn):
            return await fn(**arguments)
        return fn(**arguments)


# Global default registry
_default_registry = ToolRegistry()


def tool(
    *,
    name: str,
    description: str,
    parameters: dict[str, Any] | None = None,
    registry: ToolRegistry | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that registers a function as a tool."""
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        schema = parameters or _infer_schema(fn)
        spec = ToolSpec(name=name, description=description, parameters=schema)
        target = registry if registry is not None else _default_registry
        target.register(name, fn, spec)
        return fn
    return decorator


def _infer_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Build a minimal JSON Schema from function signature type hints."""
    hints = typing.get_type_hints(fn)
    hints.pop("return", None)
    sig = inspect.signature(fn)
    type_map: dict[type, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
    }
    props: dict[str, Any] = {}
    for param, hint in hints.items():
        props[param] = {"type": type_map.get(hint, "string")}
    required = [
        p for p in props
        if sig.parameters[p].default is inspect.Parameter.empty
    ]
    return {"type": "object", "properties": props, "required": required}
