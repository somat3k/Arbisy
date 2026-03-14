"""
Inter-component payload communicator.

Provides a publish/subscribe message bus backed by per-topic
async priority queues.  Components publish typed messages and
subscribe to specific MessageType topics.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Callable, Coroutine, Dict, List, Optional, Type

from src.payload.protocol import BaseMessage, MessageType
from src.payload.queue import MessageQueue
from src.utils.logger import get_logger

log = get_logger(__name__)

# Type alias for async subscriber callbacks
SubscriberCallback = Callable[[BaseMessage], Coroutine]


class PayloadCommunicator:
    """
    Async publish/subscribe message bus.

    Usage
    -----
    bus = PayloadCommunicator()

    # Subscribe
    async def on_opportunity(msg: OpportunityPayload):
        ...
    bus.subscribe(MessageType.OPPORTUNITY, on_opportunity)

    # Publish
    await bus.publish(OpportunityPayload(...))

    # Start the dispatch loop
    await bus.run()
    """

    def __init__(self, queue_maxsize: int = 500) -> None:
        self._subscribers: Dict[MessageType, List[SubscriberCallback]] = defaultdict(list)
        self._queues: Dict[MessageType, MessageQueue] = {
            mt: MessageQueue(maxsize=queue_maxsize) for mt in MessageType
        }
        self._running = False
        self._dispatch_tasks: List[asyncio.Task] = []

    # ── Subscription ─────────────────────────────────────────────────────────

    def subscribe(
        self,
        topic: MessageType,
        callback: SubscriberCallback,
    ) -> None:
        """Register `callback` to be called whenever a `topic` message arrives."""
        self._subscribers[topic].append(callback)
        log.debug("Subscribed %s to topic %s", callback.__qualname__, topic)

    def unsubscribe(
        self,
        topic: MessageType,
        callback: SubscriberCallback,
    ) -> None:
        try:
            self._subscribers[topic].remove(callback)
        except ValueError:
            pass

    # ── Publishing ────────────────────────────────────────────────────────────

    async def publish(self, message: BaseMessage) -> None:
        """Put a message onto its topic queue."""
        queue = self._queues[message.message_type]
        await queue.put(message)  # type: ignore[arg-type]
        log.debug(
            "Published %s [priority=%s] from %s",
            message.message_type,
            message.priority,
            message.source,
        )

    def publish_nowait(self, message: BaseMessage) -> None:
        """Non-blocking publish; drops message if queue is full."""
        queue = self._queues[message.message_type]
        try:
            queue.put_nowait(message)  # type: ignore[arg-type]
        except asyncio.QueueFull:
            log.warning(
                "Queue full for topic %s — message %s dropped",
                message.message_type,
                message.message_id,
            )

    # ── Dispatch loop ─────────────────────────────────────────────────────────

    async def _dispatch_topic(self, topic: MessageType) -> None:
        """Continuously drain one topic queue and call subscribers."""
        queue = self._queues[topic]
        while self._running:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                log.error("Queue error for topic %s: %s", topic, exc)
                continue

            subscribers = self._subscribers.get(topic, [])
            if not subscribers:
                queue.task_done()
                continue

            for cb in subscribers:
                try:
                    await cb(message)
                except Exception as exc:
                    log.error(
                        "Subscriber %s raised on message %s: %s",
                        cb.__qualname__,
                        message.message_id,
                        exc,
                        exc_info=True,
                    )
            queue.task_done()

    async def run(self) -> None:
        """Start per-topic dispatch tasks and run until stop() is called."""
        self._running = True
        log.info("PayloadCommunicator started — %d topics active", len(MessageType))
        self._dispatch_tasks = [
            asyncio.create_task(self._dispatch_topic(topic), name=f"dispatch-{topic}")
            for topic in MessageType
        ]
        try:
            await asyncio.gather(*self._dispatch_tasks)
        except asyncio.CancelledError:
            pass

    def stop(self) -> None:
        """Signal the dispatch loop to stop."""
        self._running = False
        for task in self._dispatch_tasks:
            task.cancel()
        log.info("PayloadCommunicator stopped.")

    # ── Stats ─────────────────────────────────────────────────────────────────

    def queue_sizes(self) -> Dict[str, int]:
        return {topic.value: q.size for topic, q in self._queues.items()}
