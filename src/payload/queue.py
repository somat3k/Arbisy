"""
Async priority message queue.

Wraps asyncio.PriorityQueue with typed message support and a size cap.
Items are ordered by (priority, timestamp) so lower Priority enum values
are dequeued first (CRITICAL=0 before HIGH=1, etc.).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

from src.payload.protocol import BaseMessage, Priority

T = TypeVar("T", bound=BaseMessage)


@dataclass(order=True)
class _QueueItem(Generic[T]):
    """Sortable wrapper for priority queue insertion."""
    sort_key: tuple
    message: T

    def __init__(self, message: T) -> None:
        self.sort_key = (int(message.priority), message.timestamp)
        self.message = message

    def __iter__(self):  # allow tuple-unpacking if needed
        return iter((self.sort_key, self.message))


class MessageQueue(Generic[T]):
    """
    Async priority queue for BaseMessage subclasses.

    Parameters
    ----------
    maxsize: Maximum items in queue (0 = unlimited).
    """

    def __init__(self, maxsize: int = 0) -> None:
        self._queue: asyncio.PriorityQueue[_QueueItem[T]] = asyncio.PriorityQueue(
            maxsize=maxsize
        )

    async def put(self, message: T) -> None:
        """Enqueue a message respecting priority."""
        await self._queue.put(_QueueItem(message))

    def put_nowait(self, message: T) -> None:
        """Non-blocking enqueue; raises QueueFull if at capacity."""
        self._queue.put_nowait(_QueueItem(message))

    async def get(self) -> T:
        """Dequeue and return the highest-priority message."""
        item = await self._queue.get()
        return item.message

    def get_nowait(self) -> Optional[T]:
        """Non-blocking dequeue; returns None if empty."""
        try:
            item = self._queue.get_nowait()
            return item.message
        except asyncio.QueueEmpty:
            return None

    def task_done(self) -> None:
        self._queue.task_done()

    async def join(self) -> None:
        await self._queue.join()

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        return self._queue.empty()
