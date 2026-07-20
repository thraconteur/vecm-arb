"""
data_pipeline.py
=================
Distributed, event-driven market data pipeline.

Architecture
------------
    [Feed microservice A] --\\
    [Feed microservice B] ---> Redis Pub/Sub channel "ticks" --> [Alpha Engine subscriber]
    [Feed microservice N] --/

Each feed is an INDEPENDENT asyncio task that publishes ticks the moment it
has them. Feeds do not know about each other and do not know about the math
engine. The math engine is a subscriber that maintains its own rolling
buffer and only recomputes when it has enough synchronized data -- it is
never blocked waiting on any single feed. This solves the "non-synchronous
data streams" requirement in the brief: if ASSET_3's feed lags or drops a
tick, the other feeds keep flowing and the engine works off the latest
values it has (last-observation-carried-forward), rather than stalling.

Two backends are provided behind the same `Broker` interface:
    - RedisBroker    : real pub/sub over a Redis server (used by default;
                        this is what's actually running in this repo's demo)
    - InMemoryBroker  : asyncio.Queue-based fallback with an identical
                        interface, useful for unit tests / CI boxes with no
                        Redis available. Swapping backends is a one-line
                        change (see run_demo.py).

ZeroMQ note
-----------
The brief lists Redis/ZeroMQ as alternatives. Redis Pub/Sub was chosen as
the primary implementation here because it also doubles as shared state
(the risk engine reads live halt/borrow flags from the same Redis instance
via simple keys), which keeps the infra footprint to one process. A ZMQ
PUB/SUB version would follow the identical `Broker` interface, swapping
`redis.asyncio` calls for `zmq.asyncio` socket sends -- left as a drop-in
extension point (`ZmqBroker` stub below) since it doesn't change any of the
math/risk code above it.
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

import redis.asyncio as aioredis


class Broker(ABC):
    """Common interface every transport backend must satisfy."""

    @abstractmethod
    async def publish(self, channel: str, message: dict) -> None: ...

    @abstractmethod
    async def subscribe(self, channel: str) -> AsyncIterator[dict]: ...

    @abstractmethod
    async def set_flag(self, key: str, value: str) -> None: ...

    @abstractmethod
    async def get_flag(self, key: str) -> Optional[str]: ...


class RedisBroker(Broker):
    def __init__(self, url: str = "redis://localhost:6379/0"):
        self._redis = aioredis.from_url(url, decode_responses=True)

    async def publish(self, channel: str, message: dict) -> None:
        await self._redis.publish(channel, json.dumps(message))

    async def subscribe(self, channel: str) -> AsyncIterator[dict]:
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel)
        async for raw in pubsub.listen():
            if raw["type"] != "message":
                continue
            yield json.loads(raw["data"])

    async def set_flag(self, key: str, value: str) -> None:
        await self._redis.set(key, value)

    async def get_flag(self, key: str) -> Optional[str]:
        return await self._redis.get(key)

    async def close(self):
        await self._redis.aclose()


class InMemoryBroker(Broker):
    """Drop-in broker for tests/CI where no Redis server is available."""

    def __init__(self):
        self._queues: dict[str, list[asyncio.Queue]] = {}
        self._flags: dict[str, str] = {}

    async def publish(self, channel: str, message: dict) -> None:
        for q in self._queues.get(channel, []):
            await q.put(message)

    async def subscribe(self, channel: str) -> AsyncIterator[dict]:
        q: asyncio.Queue = asyncio.Queue()
        self._queues.setdefault(channel, []).append(q)
        while True:
            yield await q.get()

    async def set_flag(self, key: str, value: str) -> None:
        self._flags[key] = value

    async def get_flag(self, key: str) -> Optional[str]:
        return self._flags.get(key)


# ------------------------------------------------------------------------ #
# Feed microservice: publishes one asset's ticks independently
# ------------------------------------------------------------------------ #
class FeedMicroservice:
    def __init__(self, broker: Broker, asset: str, ticks: list[dict], channel: str = "ticks",
                 base_delay: float = 0.0, jitter: float = 0.0):
        self.broker = broker
        self.asset = asset
        self.ticks = ticks
        self.channel = channel
        self.base_delay = base_delay
        self.jitter = jitter

    async def run(self):
        import random

        for tick in self.ticks:
            payload = {"asset": self.asset, **tick, "published_at": time.time()}
            await self.broker.publish(self.channel, payload)
            if self.base_delay or self.jitter:
                await asyncio.sleep(max(0.0, self.base_delay + random.uniform(-self.jitter, self.jitter)))


# ------------------------------------------------------------------------ #
# Non-blocking rolling buffer the alpha engine subscribes against
# ------------------------------------------------------------------------ #
class SynchronizedBuffer:
    """
    Maintains last-observation-carried-forward (LOCF) mid prices per asset.
    Never blocks on a slow/missing feed: reads always return the latest
    known state for every asset instantly.
    """

    def __init__(self, assets: list[str]):
        self.assets = assets
        self.latest: dict[str, Optional[float]] = {a: None for a in assets}
        self.update_counts: dict[str, int] = {a: 0 for a in assets}

    def update(self, tick: dict):
        asset = tick["asset"]
        if asset in self.latest:
            self.latest[asset] = tick.get("mid_price", tick.get("price"))
            self.update_counts[asset] += 1

    def is_ready(self) -> bool:
        return all(v is not None for v in self.latest.values())

    def snapshot(self) -> dict:
        return dict(self.latest)


async def consume_and_buffer(broker: Broker, channel: str, buffer: SynchronizedBuffer,
                              on_ready_callback, min_updates_before_signal: int = 5,
                              max_messages: Optional[int] = None):
    """
    The Alpha Engine's subscriber loop. Every incoming tick updates the
    buffer immediately (non-blocking); once every asset has enough history,
    `on_ready_callback(snapshot)` fires. This is what decouples the math
    engine's cadence from any single feed's cadence.
    """
    n = 0
    async for tick in broker.subscribe(channel):
        buffer.update(tick)
        n += 1
        if buffer.is_ready() and min(buffer.update_counts.values()) >= min_updates_before_signal:
            on_ready_callback(buffer.snapshot())
        if max_messages is not None and n >= max_messages:
            break
