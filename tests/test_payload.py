"""
Tests for the payload communication system.
"""

from __future__ import annotations

import asyncio
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from src.payload.communicator import PayloadCommunicator
from src.payload.protocol import (
    BaseMessage,
    ExecutionResultPayload,
    MessageType,
    OpportunityPayload,
    Priority,
    SwapHop,
    SystemEventPayload,
)
from src.payload.queue import MessageQueue


# ── Protocol ─────────────────────────────────────────────────────────────────

class TestProtocol:
    def test_base_message_defaults(self) -> None:
        msg = BaseMessage()
        assert msg.message_id != ""
        assert msg.message_type == MessageType.SYSTEM_EVENT
        assert msg.priority == Priority.MEDIUM
        assert msg.timestamp > 0

    def test_to_dict_round_trip(self) -> None:
        msg  = BaseMessage(source="test", metadata={"key": "value"})
        d    = msg.to_dict()
        msg2 = BaseMessage.from_dict(d)
        assert msg2.message_id == msg.message_id
        assert msg2.source == "test"
        assert msg2.metadata == {"key": "value"}

    def test_opportunity_payload_type(self) -> None:
        opp = OpportunityPayload(asset="0xUSDC", loan_amount_wei=1000)
        assert opp.message_type == MessageType.OPPORTUNITY
        assert opp.priority == Priority.HIGH

    def test_execution_result_payload(self) -> None:
        result = ExecutionResultPayload(
            opportunity_id="abc123",
            success=True,
            tx_hash="0xdeadbeef",
            actual_profit_usd=12.50,
        )
        assert result.success is True
        assert result.message_type == MessageType.EXECUTION_RESULT

    def test_priority_ordering(self) -> None:
        assert Priority.CRITICAL < Priority.HIGH < Priority.MEDIUM < Priority.LOW

    def test_swap_hop_creation(self) -> None:
        hop = SwapHop(
            dex_name="uniswap_v3",
            router_address="0xRouter",
            token_in="0xUSDC",
            token_out="0xWETH",
            fee_bps=30,
            is_v3=True,
        )
        assert hop.is_v3 is True
        assert hop.fee_bps == 30


# ── MessageQueue ──────────────────────────────────────────────────────────────

class TestMessageQueue:
    @pytest.mark.asyncio
    async def test_put_and_get(self) -> None:
        q   = MessageQueue()
        msg = BaseMessage(priority=Priority.HIGH)
        await q.put(msg)
        out = await q.get()
        assert out.message_id == msg.message_id

    @pytest.mark.asyncio
    async def test_priority_ordering(self) -> None:
        q    = MessageQueue()
        low  = BaseMessage(priority=Priority.LOW)
        high = BaseMessage(priority=Priority.CRITICAL)
        mid  = BaseMessage(priority=Priority.MEDIUM)

        await q.put(low)
        await q.put(mid)
        await q.put(high)

        first  = await q.get()
        second = await q.get()
        third  = await q.get()

        assert first.priority  == Priority.CRITICAL
        assert second.priority == Priority.MEDIUM
        assert third.priority  == Priority.LOW

    @pytest.mark.asyncio
    async def test_size_tracking(self) -> None:
        q = MessageQueue()
        assert q.size == 0
        await q.put(BaseMessage())
        assert q.size == 1

    def test_get_nowait_empty(self) -> None:
        q   = MessageQueue()
        msg = q.get_nowait()
        assert msg is None

    def test_put_nowait_and_get_nowait(self) -> None:
        q   = MessageQueue()
        msg = BaseMessage()
        q.put_nowait(msg)
        out = q.get_nowait()
        assert out is not None
        assert out.message_id == msg.message_id


# ── PayloadCommunicator ───────────────────────────────────────────────────────

class TestPayloadCommunicator:
    @pytest.mark.asyncio
    async def test_subscribe_and_receive(self) -> None:
        bus      = PayloadCommunicator()
        received: List[BaseMessage] = []

        async def on_system(msg: BaseMessage) -> None:
            received.append(msg)

        bus.subscribe(MessageType.SYSTEM_EVENT, on_system)

        msg = SystemEventPayload(event_name="test_event")
        await bus.publish(msg)

        # Run bus briefly to dispatch
        task = asyncio.create_task(bus.run())
        await asyncio.sleep(0.1)
        bus.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(received) == 1
        assert received[0].message_id == msg.message_id

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self) -> None:
        bus = PayloadCommunicator()
        calls: List[str] = []

        async def sub1(msg: BaseMessage) -> None:
            calls.append("sub1")

        async def sub2(msg: BaseMessage) -> None:
            calls.append("sub2")

        bus.subscribe(MessageType.OPPORTUNITY, sub1)
        bus.subscribe(MessageType.OPPORTUNITY, sub2)

        await bus.publish(OpportunityPayload())

        task = asyncio.create_task(bus.run())
        await asyncio.sleep(0.1)
        bus.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "sub1" in calls
        assert "sub2" in calls

    @pytest.mark.asyncio
    async def test_unsubscribe(self) -> None:
        bus   = PayloadCommunicator()
        calls: List[str] = []

        async def cb(msg: BaseMessage) -> None:
            calls.append("called")

        bus.subscribe(MessageType.SYSTEM_EVENT, cb)
        bus.unsubscribe(MessageType.SYSTEM_EVENT, cb)
        await bus.publish(SystemEventPayload(event_name="test"))

        task = asyncio.create_task(bus.run())
        await asyncio.sleep(0.1)
        bus.stop()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert len(calls) == 0

    def test_queue_sizes(self) -> None:
        bus = PayloadCommunicator()
        sizes = bus.queue_sizes()
        assert set(sizes.keys()) == {mt.value for mt in MessageType}

    @pytest.mark.asyncio
    async def test_publish_nowait_no_crash(self) -> None:
        bus = PayloadCommunicator()
        msg = SystemEventPayload(event_name="nowait_test")
        bus.publish_nowait(msg)   # Should not raise
        assert bus.queue_sizes()[MessageType.SYSTEM_EVENT.value] == 1
