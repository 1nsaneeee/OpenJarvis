import asyncio

import pytest
import pytest_asyncio

from openjarvis.bus.client import BusClient
from openjarvis.bus.schemas import WakeEvent

REDIS_URL = "redis://localhost:6379/1"   # DB 1 = test isolation


@pytest_asyncio.fixture
async def bus():
    client = BusClient(REDIS_URL)
    await client.connect()
    yield client
    await client.flush_db()   # clean up after each test
    await client.close()


@pytest.mark.asyncio
async def test_publish_and_subscribe(bus: BusClient):
    received: list[WakeEvent] = []

    async def handler(ev: WakeEvent) -> None:
        received.append(ev)

    await bus.subscribe("jarvis:wake:detected", WakeEvent, handler)
    ev = WakeEvent(source="test", trace_id="t1", model_name="hey_jarvis", score=0.8)
    await bus.publish("jarvis:wake:detected", ev)
    await asyncio.sleep(0.1)   # let subscriber fire
    assert len(received) == 1
    assert received[0].trace_id == "t1"


@pytest.mark.asyncio
async def test_xadd_and_xread(bus: BusClient):
    from openjarvis.bus.schemas import AsrFinal
    ev = AsrFinal(source="asr", trace_id="t2", text="hello world", language="en")
    await bus.xadd("jarvis:asr:final", ev)
    results = await bus.xread("jarvis:asr:final", count=1)
    assert len(results) == 1
    assert results[0]["text"] == "hello world"


@pytest.mark.asyncio
async def test_set_and_get_state(bus: BusClient):
    from openjarvis.bus.schemas import ConvState
    await bus.set_state("jarvis:conv:state", ConvState.LISTENING)
    state = await bus.get_state("jarvis:conv:state", ConvState)
    assert state == ConvState.LISTENING
