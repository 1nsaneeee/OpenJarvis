import pytest
from pydantic import ValidationError

from openjarvis.bus.schemas import (
    AsrFinal,
    AsrPartial,
    AudioChunk,
    ConvState,
    ConvStateEvent,
    Envelope,
    LlmDeltaEvent,
    LlmRequest,
    LlmResponse,
    ShutdownEvent,
    ToolCallEvent,
    ToolConfirmEvent,
    ToolResultEvent,
    VadEvent,
    WakeEvent,
)


def test_envelope_round_trip():
    env = Envelope(
        source="test",
        trace_id="trace_abc",
        type="test.event",
        payload={"key": "value"},
    )
    data = env.model_dump_json()
    restored = Envelope.model_validate_json(data)
    assert restored.source == "test"
    assert restored.payload == {"key": "value"}
    assert len(restored.id) > 0  # ULID generated
    assert restored.ts > 0


def test_envelope_id_is_unique():
    a = Envelope(source="x", trace_id="t", type="t.e", payload={})
    b = Envelope(source="x", trace_id="t", type="t.e", payload={})
    assert a.id != b.id


def test_audio_chunk_fields():
    chunk = AudioChunk(
        source="audio",
        trace_id="",
        pcm_b64="AAAA",
        sample_rate=16000,
        channels=1,
        frame_ms=30,
    )
    assert chunk.type == "audio.chunk"


def test_wake_event_fields():
    ev = WakeEvent(source="wake", trace_id="t1", model_name="hey_jarvis", score=0.9)
    assert ev.type == "wake.detected"
    assert ev.score == pytest.approx(0.9)


def test_asr_final_fields():
    ev = AsrFinal(
        source="asr", trace_id="t1", text="what time is it", language="en", duration_s=2.1
    )
    assert ev.type == "asr.final"
    assert ev.text == "what time is it"


def test_conv_state_event():
    ev = ConvStateEvent(source="conv", trace_id="t", state=ConvState.LISTENING)
    assert ev.state == ConvState.LISTENING


# ── Missing model round-trip tests ──────────────────────────────────────


def test_vad_event_round_trip():
    ev = VadEvent(source="audio", trace_id="t1", type="audio.vad", is_speech=True)
    data = ev.model_dump_json()
    restored = VadEvent.model_validate_json(data)
    assert restored.is_speech is True
    assert restored.type == "audio.vad"


def test_asr_partial_round_trip():
    ev = AsrPartial(source="asr", trace_id="t1", text="hello")
    data = ev.model_dump_json()
    restored = AsrPartial.model_validate_json(data)
    assert restored.text == "hello"
    assert restored.type == "asr.partial"


def test_llm_delta_event_round_trip():
    ev = LlmDeltaEvent(source="llm", trace_id="t1", text="streaming token")
    data = ev.model_dump_json()
    restored = LlmDeltaEvent.model_validate_json(data)
    assert restored.text == "streaming token"
    assert restored.type == "llm.delta"


def test_llm_request_round_trip():
    ev = LlmRequest(
        source="llm",
        trace_id="t1",
        messages_json='[{"role":"user","content":"hi"}]',
        tools_json="[]",
        model="gpt-4",
        max_tokens=4096,
        temperature=0.7,
    )
    data = ev.model_dump_json()
    restored = LlmRequest.model_validate_json(data)
    assert restored.messages_json == '[{"role":"user","content":"hi"}]'
    assert restored.model == "gpt-4"
    assert restored.type == "llm.request"


def test_llm_response_round_trip():
    ev = LlmResponse(
        source="llm",
        trace_id="t1",
        text="Hello!",
        tool_calls_json="[]",
        finish_reason="stop",
        tokens_in=10,
        tokens_out=5,
    )
    data = ev.model_dump_json()
    restored = LlmResponse.model_validate_json(data)
    assert restored.text == "Hello!"
    assert restored.finish_reason == "stop"
    assert restored.tokens_in == 10
    assert restored.tokens_out == 5
    assert restored.type == "llm.response"


def test_tool_call_event_round_trip():
    ev = ToolCallEvent(
        source="tools",
        trace_id="t1",
        tool_call_id="call_123",
        name="search",
        arguments_json='{"query": "weather"}',
    )
    data = ev.model_dump_json()
    restored = ToolCallEvent.model_validate_json(data)
    assert restored.tool_call_id == "call_123"
    assert restored.name == "search"
    assert restored.arguments_json == '{"query": "weather"}'
    assert restored.type == "tool.call"


def test_tool_result_event_round_trip():
    ev = ToolResultEvent(
        source="tools",
        trace_id="t1",
        tool_call_id="call_123",
        name="search",
        result_json='{"temperature": 22}',
        success=True,
        error=None,
    )
    data = ev.model_dump_json()
    restored = ToolResultEvent.model_validate_json(data)
    assert restored.tool_call_id == "call_123"
    assert restored.result_json == '{"temperature": 22}'
    assert restored.success is True
    assert restored.error is None
    assert restored.type == "tool.result"


def test_tool_confirm_event_round_trip():
    ev = ToolConfirmEvent(
        source="tools",
        trace_id="t1",
        tool_call_id="call_123",
        name="delete_file",
        arguments_json='{"path": "/tmp/x"}',
        risk="high",
    )
    data = ev.model_dump_json()
    restored = ToolConfirmEvent.model_validate_json(data)
    assert restored.tool_call_id == "call_123"
    assert restored.risk == "high"
    assert restored.type == "tool.confirm"


def test_shutdown_event_round_trip():
    ev = ShutdownEvent(source="system", trace_id="t1", reason="user_request")
    data = ev.model_dump_json()
    restored = ShutdownEvent.model_validate_json(data)
    assert restored.reason == "user_request"
    assert restored.type == "system.shutdown"


# ── Enum membership test ────────────────────────────────────────────────


def test_conv_state_enum_members():
    assert ConvState.IDLE == "idle"
    assert ConvState.LISTENING == "listening"
    assert ConvState.THINKING == "thinking"
    assert ConvState.EXECUTING == "executing"
    assert ConvState.CONFIRMING == "confirming"
    assert ConvState.RESPONDING == "responding"
    assert ConvState.COOLDOWN == "cooldown"
    # All members should be reachable via enum iteration
    all_values = set(s.value for s in ConvState)
    assert len(all_values) == 7


# ── Optional None defaults ──────────────────────────────────────────────


def test_optional_none_defaults():
    """Optional fields should default to None when not provided."""
    # AsrFinal.language defaults to None
    ev1 = AsrFinal(source="asr", trace_id="t1", text="hello")
    assert ev1.language is None
    assert ev1.duration_s is None

    # ConvStateEvent.previous defaults to None
    ev2 = ConvStateEvent(source="conv", trace_id="t", state=ConvState.IDLE)
    assert ev2.previous is None

    # ShutdownEvent.reason has a string default, not None
    ev3 = ShutdownEvent(source="sys", trace_id="t")
    assert ev3.reason == "user_request"


# ── Envelope immutability (frozen=True) ─────────────────────────────────


def test_envelope_is_immutable():
    env = Envelope(source="test", trace_id="t1", type="test.event")
    with pytest.raises(ValidationError):
        env.source = "changed"  # frozen=True prevents mutation
