"""Tests for ConversationManager state machine."""
from unittest.mock import AsyncMock

import pytest

from openjarvis.bus.schemas import AsrFinal, WakeEvent
from openjarvis.conversation.manager import ConversationManager, State
from openjarvis.llm.base import LlmDelta


@pytest.fixture
def mock_bus() -> AsyncMock:
    bus = AsyncMock()
    bus.subscribe = AsyncMock()
    bus.publish = AsyncMock()
    bus.xadd = AsyncMock()
    bus.set_state = AsyncMock()
    bus.xread = AsyncMock(return_value=[])
    return bus


@pytest.fixture
def mock_provider() -> AsyncMock:
    provider = AsyncMock()

    async def fake_chat(*_args, **_kwargs):  # noqa: ANN001, ANN002, ANN003
        yield LlmDelta(text="The time is 10:00 AM.", finish_reason="stop")

    # provider.chat is a SYNC method returning an async iterator
    provider.chat = fake_chat
    return provider


@pytest.fixture
def mock_executor() -> AsyncMock:
    executor = AsyncMock()
    executor.execute = AsyncMock(return_value=(True, '"10:00 AM"'))
    return executor


@pytest.mark.asyncio
async def test_initial_state_is_idle(
    mock_bus: AsyncMock, mock_provider: AsyncMock, mock_executor: AsyncMock
) -> None:
    mgr = ConversationManager(mock_bus, mock_provider, mock_executor, model="test-model")
    assert mgr.state == State.IDLE


@pytest.mark.asyncio
async def test_wake_transitions_to_listening(
    mock_bus: AsyncMock, mock_provider: AsyncMock, mock_executor: AsyncMock
) -> None:
    mgr = ConversationManager(mock_bus, mock_provider, mock_executor, model="test-model")
    wake_ev = WakeEvent(source="wake", trace_id="t1", model_name="hey_jarvis", score=0.9)
    await mgr._on_wake(wake_ev)
    assert mgr.state == State.LISTENING


@pytest.mark.asyncio
async def test_asr_final_triggers_thinking_then_idle(
    mock_bus: AsyncMock, mock_provider: AsyncMock, mock_executor: AsyncMock
) -> None:
    mgr = ConversationManager(mock_bus, mock_provider, mock_executor, model="test-model")
    mgr._state = State.LISTENING
    mgr._trace_id = "t1"
    asr_ev = AsrFinal(source="asr", trace_id="t1", text="what time is it")
    await mgr._on_asr_final(asr_ev)
    # Ended back at IDLE after full think → respond cycle
    assert mgr.state == State.IDLE
    # History should now contain user turn + assistant turn
    assert len(mgr._history) == 2
    assert mgr._history[0].role == "user"
    assert mgr._history[1].role == "assistant"
    assert mgr._history[1].content == "The time is 10:00 AM."


@pytest.mark.asyncio
async def test_asr_final_ignored_when_not_listening(
    mock_bus: AsyncMock, mock_provider: AsyncMock, mock_executor: AsyncMock
) -> None:
    mgr = ConversationManager(mock_bus, mock_provider, mock_executor, model="test-model")
    # state is IDLE by default
    asr_ev = AsrFinal(source="asr", trace_id="t1", text="hello")
    await mgr._on_asr_final(asr_ev)
    assert mgr.state == State.IDLE
    assert len(mgr._history) == 0


@pytest.mark.asyncio
async def test_wake_ignored_when_not_idle(
    mock_bus: AsyncMock, mock_provider: AsyncMock, mock_executor: AsyncMock
) -> None:
    mgr = ConversationManager(mock_bus, mock_provider, mock_executor, model="test-model")
    mgr._state = State.THINKING
    wake_ev = WakeEvent(source="wake", trace_id="t1", model_name="hey_jarvis", score=0.9)
    await mgr._on_wake(wake_ev)
    assert mgr.state == State.THINKING  # unchanged


@pytest.mark.asyncio
async def test_tool_call_flow(
    mock_bus: AsyncMock, mock_executor: AsyncMock
) -> None:
    """Provider yields a tool call first, then on second pass returns text."""
    from openjarvis.llm.base import ToolCall

    provider = AsyncMock()
    call_count = {"n": 0}

    async def chat_with_tool(*_args, **_kwargs):  # noqa: ANN001, ANN002, ANN003
        call_count["n"] += 1
        if call_count["n"] == 1:
            yield LlmDelta(
                tool_call=ToolCall(id="tc1", name="get_time", arguments={}),
                finish_reason="tool_use",
            )
        else:
            yield LlmDelta(text="It's 10:00 AM.", finish_reason="stop")

    provider.chat = chat_with_tool

    mgr = ConversationManager(mock_bus, provider, mock_executor, model="test-model")
    mgr._state = State.LISTENING
    mgr._trace_id = "t1"
    asr_ev = AsrFinal(source="asr", trace_id="t1", text="what time is it")
    await mgr._on_asr_final(asr_ev)

    assert mgr.state == State.IDLE
    # History: user, assistant(tool_call), tool, assistant(final)
    assert len(mgr._history) == 4
    assert mgr._history[0].role == "user"
    assert mgr._history[1].role == "assistant"
    assert len(mgr._history[1].tool_calls) == 1
    assert mgr._history[2].role == "tool"
    assert mgr._history[2].tool_call_id == "tc1"
    assert mgr._history[3].role == "assistant"
    assert mgr._history[3].content == "It's 10:00 AM."
    # Executor was called
    mock_executor.execute.assert_called_once_with("get_time", {})


def test_state_enum_values() -> None:
    """State enum string values should match ConvState enum."""
    assert State.IDLE.value == "idle"
    assert State.LISTENING.value == "listening"
    assert State.THINKING.value == "thinking"
    assert State.EXECUTING.value == "executing"
    assert State.RESPONDING.value == "responding"
