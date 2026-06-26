import time

import pytest

from openjarvis.bus.schemas import (
    AsrFinal,
    AudioChunk,
    ConvState,
    ConvStateEvent,
    Envelope,
    LlmRequest,
    LlmResponse,
    ToolCallEvent,
    ToolResultEvent,
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
