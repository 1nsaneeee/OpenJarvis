import pytest
from pydantic import ValidationError

from openjarvis.llm.base import LlmDelta, Message, ToolCall, ToolSpec


def test_message_roles() -> None:
    for role in ("system", "user", "assistant", "tool"):
        m = Message(role=role, content="hello")
        assert m.role == role


def test_message_invalid_role() -> None:
    with pytest.raises(ValidationError):
        Message(role="robot", content="hi")  # type: ignore[arg-type]


def test_tool_call_fields() -> None:
    tc = ToolCall(id="tc_1", name="get_time", arguments={})
    assert tc.id == "tc_1"
    assert tc.name == "get_time"
    assert tc.arguments == {}


def test_tool_spec_json_schema() -> None:
    spec = ToolSpec(
        name="get_time",
        description="Returns the current local time.",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
    )
    assert spec.name == "get_time"
    assert spec.description == "Returns the current local time."
    assert spec.parameters == {"type": "object", "properties": {}, "required": []}


def test_llm_delta_text() -> None:
    d = LlmDelta(text="hello")
    assert d.text == "hello"
    assert d.tool_call is None


def test_llm_delta_tool_call() -> None:
    tc = ToolCall(id="x", name="get_time", arguments={})
    d = LlmDelta(tool_call=tc, finish_reason="tool_use")
    assert d.tool_call is not None
    assert d.finish_reason == "tool_use"


def test_llm_delta_finish_reason_values() -> None:
    """LlmDelta.finish_reason only accepts our mapped Literal values."""
    for valid in ("stop", "tool_use", "length"):
        d = LlmDelta(finish_reason=valid)
        assert d.finish_reason == valid
    with pytest.raises(ValidationError):
        LlmDelta(finish_reason="end_turn")  # type: ignore[arg-type]
