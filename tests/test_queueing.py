from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from llmkit_lite.graphs import WorkflowIdentity
from llmkit_lite.operations import OperationRequest
from llmkit_lite.queueing import (
    InMemoryOperationQueue,
    OperationJob,
    QueueDelivery,
    QueueError,
)


class FrozenClock:
    def __init__(self, current: datetime) -> None:
        self.current = current

    def __call__(self) -> datetime:
        return self.current

    def advance(self, duration: timedelta) -> None:
        self.current += duration


class TokenFactory:
    def __init__(self) -> None:
        self.sequence = 0

    def __call__(self) -> str:
        self.sequence += 1
        return f"token-{self.sequence}"


@pytest.fixture
def frozen_clock() -> FrozenClock:
    return FrozenClock(datetime(2026, 9, 15, 12, 0, tzinfo=UTC))


@pytest.fixture
def operation_job() -> OperationJob:
    return OperationJob(
        "operation-1",
        WorkflowIdentity("thread-1", "recovery"),
        "request-key-1",
        OperationRequest("reboot", {"device": "router-1"}),
    )


@pytest.fixture
def operation_queue(frozen_clock: FrozenClock) -> InMemoryOperationQueue:
    return InMemoryOperationQueue(
        visibility_timeout=timedelta(seconds=30),
        clock=frozen_clock,
        token_factory=TokenFactory(),
    )


async def test_given_published_job_when_receiving_then_delivery_is_temporarily_hidden(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
    frozen_clock: FrozenClock,
) -> None:
    await operation_queue.publish(operation_job, message_id="message-1")

    delivery = await operation_queue.receive()
    hidden = await operation_queue.receive()

    assert delivery == QueueDelivery(
        message_id="message-1",
        job=operation_job,
        receipt_token="token-1:1",
        attempt=1,
        received_at=frozen_clock.current,
        visible_at=frozen_clock.current + timedelta(seconds=30),
    )
    assert hidden is None


async def test_given_unacknowledged_job_when_visibility_expires_then_it_is_redelivered(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
    frozen_clock: FrozenClock,
) -> None:
    await operation_queue.publish(operation_job, message_id="message-1")
    first = await operation_queue.receive()
    assert first is not None
    frozen_clock.current = first.visible_at

    redelivered = await operation_queue.receive()

    assert redelivered is not None
    assert redelivered.message_id == first.message_id
    assert redelivered.job is first.job
    assert redelivered.attempt == 2
    assert redelivered.receipt_token != first.receipt_token


async def test_given_acknowledged_job_when_receiving_again_then_queue_is_empty(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
    frozen_clock: FrozenClock,
) -> None:
    await operation_queue.publish(operation_job, message_id="message-1")
    delivery = await operation_queue.receive()
    assert delivery is not None

    await operation_queue.acknowledge(delivery)
    frozen_clock.advance(timedelta(minutes=1))

    assert await operation_queue.receive() is None


async def test_given_redelivery_when_old_receipt_acknowledges_then_job_is_preserved(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
    frozen_clock: FrozenClock,
) -> None:
    await operation_queue.publish(operation_job, message_id="message-1")
    first = await operation_queue.receive()
    assert first is not None
    frozen_clock.current = first.visible_at
    second = await operation_queue.receive()
    assert second is not None

    with pytest.raises(QueueError) as exc_info:
        await operation_queue.acknowledge(first)

    assert exc_info.value.code == "queue_stale_delivery"
    await operation_queue.acknowledge(second)
    assert await operation_queue.receive() is None


async def test_given_expired_visibility_when_acknowledging_then_delivery_is_stale(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
    frozen_clock: FrozenClock,
) -> None:
    await operation_queue.publish(operation_job, message_id="message-1")
    delivery = await operation_queue.receive()
    assert delivery is not None
    frozen_clock.current = delivery.visible_at

    with pytest.raises(QueueError) as exc_info:
        await operation_queue.acknowledge(delivery)

    assert exc_info.value.code == "queue_stale_delivery"
    assert await operation_queue.receive() is not None


async def test_given_heartbeat_when_extending_visibility_then_old_deadline_is_replaced(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
    frozen_clock: FrozenClock,
) -> None:
    await operation_queue.publish(operation_job, message_id="message-1")
    delivery = await operation_queue.receive()
    assert delivery is not None
    frozen_clock.advance(timedelta(seconds=20))

    extended = await operation_queue.extend_visibility(delivery)
    frozen_clock.current = delivery.visible_at

    assert extended.receipt_token == delivery.receipt_token
    assert extended.visible_at == delivery.visible_at + timedelta(seconds=20)
    assert await operation_queue.receive() is None
    frozen_clock.current = extended.visible_at
    assert await operation_queue.receive() is not None


async def test_given_one_visible_job_when_workers_receive_then_exactly_one_gets_it(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
) -> None:
    await operation_queue.publish(operation_job, message_id="message-1")

    deliveries = await asyncio.gather(*(operation_queue.receive() for _ in range(20)))

    assert len([item for item in deliveries if item is not None]) == 1


async def test_given_multiple_jobs_when_receiving_then_publish_order_is_preserved(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
) -> None:
    second_job = OperationJob(
        "operation-2",
        operation_job.identity,
        "request-key-2",
        operation_job.request,
    )
    await operation_queue.publish(operation_job, message_id="message-1")
    await operation_queue.publish(second_job, message_id="message-2")

    first = await operation_queue.receive()
    second = await operation_queue.receive()

    assert first is not None and first.message_id == "message-1"
    assert second is not None and second.message_id == "message-2"


async def test_given_duplicate_message_id_when_publishing_then_safe_error_is_raised(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
) -> None:
    await operation_queue.publish(operation_job, message_id="private-message-id")

    with pytest.raises(QueueError) as exc_info:
        await operation_queue.publish(operation_job, message_id="private-message-id")

    assert exc_info.value.code == "queue_message_exists"
    assert "private-message-id" not in str(exc_info.value)


@pytest.mark.parametrize(
    "duration",
    [timedelta(0), timedelta(seconds=-1), "30 seconds", None],
)
def test_given_invalid_visibility_when_constructing_queue_then_validation_fails(
    duration,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        InMemoryOperationQueue(visibility_timeout=duration)


def test_given_invalid_job_fields_when_constructing_job_then_validation_fails(
    operation_job: OperationJob,
) -> None:
    with pytest.raises(ValueError):
        OperationJob(
            " ",
            operation_job.identity,
            operation_job.idempotency_key,
            operation_job.request,
        )
    with pytest.raises(TypeError):
        OperationJob(
            operation_job.operation_id,
            object(),  # type: ignore[arg-type]
            operation_job.idempotency_key,
            operation_job.request,
        )
