import pytest
from pydantic import ValidationError

from openjarvis.llm.base import LlmDelta, Message, ToolCall, ToolSpec


def test_message_roles():
    for role in ("system", "user", "assistant", "tool"):
        m = Message(role=role, content="hello")
        assert m.role == role


def test_message_invalid_role():
    with pytest.raises(ValidationError):
        Message(role="robot", content="hi")


def test_tool_call_fields():
    tc = ToolCall(id="tc_1", name="get_time", arguments={})
    assert tc.name == "get_time"


def test_tool_spec_json_schema():
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


def test_llm_delta_text():
    d = LlmDelta(text="hello")
    assert d.text == "hello"
    assert d.tool_call is None


def test_llm_delta_tool_call():
    tc = ToolCall(id="x", name="get_time", arguments={})
    d = LlmDelta(tool_call=tc, finish_reason="tool_use")
    assert d.tool_call is not None
    assert d.finish_reason == "tool_use"
