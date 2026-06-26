"""Redis event bus client — pub/sub, streams, and key-value state."""
from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from contextlib import suppress
from enum import StrEnum
from typing import Any, TypeVar

import redis.asyncio as aioredis
import structlog

from openjarvis.bus.schemas import Envelope

_logger = structlog.get_logger(__name__)

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
        # protocol=2 forces RESP2 for compatibility with Redis 5.0
        # (redis-py 8.x defaults to RESP3 HELLO which Redis 5.0 rejects).
        self._redis = await aioredis.from_url(
            self._url, decode_responses=True, protocol=2
        )
        self._pubsub = self._redis.pubsub()

    async def close(self) -> None:
        if self._listener_task:
            self._listener_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._listener_task
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
                _logger.warning("bus_message_error", channel=channel, error=str(exc))

    # ── Streams ──────────────────────────────────────────────────────────

    async def xadd(self, stream: str, event: Envelope) -> None:
        assert self._redis is not None
        data = event.model_dump(mode="json")
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
            for _msg_id, fields in messages:  # msg_id cursor discarded; use last_id param to resume
                decoded = {}
                for k, v in fields.items():
                    try:
                        decoded[k] = json.loads(v)
                    except (json.JSONDecodeError, TypeError):
                        decoded[k] = v
                entries.append(decoded)
        return entries

    # ── Key-value state ──────────────────────────────────────────────────

    async def set_state(self, key: str, value: str | StrEnum) -> None:
        assert self._redis is not None
        if isinstance(value, str):
            str_value = value
        elif isinstance(value, StrEnum):
            str_value = value.value
        else:
            raise TypeError(
                f"set_state value must be str or StrEnum, got {type(value).__name__}"
            )
        await self._redis.set(key, str_value)

    async def get_state(self, key: str, model: type[T] | None = None) -> Any:
        assert self._redis is not None
        raw = await self._redis.get(key)
        if raw is None:
            return None
        if model is not None:
            return model(raw)
        return raw
