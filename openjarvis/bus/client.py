"""Redis event bus client — pub/sub, streams, and key-value state."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

import redis.asyncio as aioredis

from openjarvis.bus.schemas import Envelope

T = TypeVar("T", bound=Envelope)
Handler = Callable[[Any], Coroutine[Any, Any, None]]


class BusClient:
    def __init__(self, url: str = "redis://localhost:6379/0") -> None:
        self._url = url
        self._redis: aioredis.Redis | None = None
        self._pubsub: aioredis.client.PubSub | None = None
        self._subscriptions: dict[str, tuple[type[Envelope], Handler]] = {}
        self._listener_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        self._redis = await aioredis.from_url(
            self._url, decode_responses=True, protocol=2
        )
        self._pubsub = self._redis.pubsub()

    async def close(self) -> None:
        if self._listener_task:
            self._listener_task.cancel()
        if self._pubsub:
            await self._pubsub.aclose()
        if self._redis:
            await self._redis.aclose()

    async def flush_db(self) -> None:
        """Test helper — flush the current Redis DB."""
        if self._redis:
            await self._redis.flushdb()

    # ── Pub/Sub ──────────────────────────────────────────────────────────

    async def publish(self, channel: str, event: Envelope) -> None:
        assert self._redis is not None
        await self._redis.publish(channel, event.model_dump_json())

    async def subscribe(
        self,
        channel: str,
        model: type[T],
        handler: Callable[[T], Coroutine[Any, Any, None]],
    ) -> None:
        assert self._pubsub is not None
        self._subscriptions[channel] = (model, handler)  # type: ignore[assignment]
        await self._pubsub.subscribe(channel)
        if not self._listener_task or self._listener_task.done():
            self._listener_task = asyncio.create_task(self._listen())

    async def _listen(self) -> None:
        assert self._pubsub is not None
        async for message in self._pubsub.listen():
            if message["type"] != "message":
                continue
            channel = message["channel"]
            if channel not in self._subscriptions:
                continue
            model_cls, handler = self._subscriptions[channel]
            try:
                event = model_cls.model_validate_json(message["data"])
                await handler(event)
            except Exception as exc:  # noqa: BLE001
                # Never crash the listener on a bad message
                print(f"[BusClient] Error handling {channel}: {exc}")

    # ── Streams ──────────────────────────────────────────────────────────

    async def xadd(self, stream: str, event: Envelope) -> None:
        assert self._redis is not None
        data = json.loads(event.model_dump_json())
        # Redis stream fields must be str → str
        flat = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in data.items()}
        await self._redis.xadd(stream, flat)

    async def xread(
        self,
        stream: str,
        count: int = 10,
        last_id: str = "0",
    ) -> list[dict[str, Any]]:
        assert self._redis is not None
        results = await self._redis.xread({stream: last_id}, count=count)
        if not results:
            return []
        entries = []
        for _stream, messages in results:
            for _msg_id, fields in messages:
                decoded = {}
                for k, v in fields.items():
                    try:
                        decoded[k] = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        decoded[k] = v
                entries.append(decoded)
        return entries

    # ── Key-value state ──────────────────────────────────────────────────

    async def set_state(self, key: str, value: Any) -> None:
        assert self._redis is not None
        await self._redis.set(key, value if isinstance(value, str) else value.value)

    async def get_state(self, key: str, model: type[T] | None = None) -> Any:
        assert self._redis is not None
        raw = await self._redis.get(key)
        if raw is None:
            return None
        if model is not None:
            return model(raw)
        return raw
