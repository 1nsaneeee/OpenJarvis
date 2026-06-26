"""Dev tool: print every Redis pub/sub and stream event in real time."""
import asyncio
import sys

import redis.asyncio as aioredis


async def sniff(url: str = "redis://localhost:6379/0") -> None:
    r = await aioredis.from_url(url, decode_responses=True, protocol=2)
    ps = r.pubsub()
    await ps.psubscribe("jarvis:*")
    print(f"Sniffing all jarvis:* channels on {url} — Ctrl+C to stop\n")
    async for msg in ps.listen():
        if msg["type"] in ("pmessage", "message"):
            print(f"[{msg.get('channel', msg.get('pattern'))}] {msg['data'][:200]}")


if __name__ == "__main__":
    asyncio.run(sniff(*sys.argv[1:]))
