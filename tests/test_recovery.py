from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from opentelemetry.trace import StatusCode

from llmkit_lite.graphs import WorkflowIdentity
from llmkit_lite.leases import InMemoryLeaseStore, LeaseManager
from llmkit_lite.operations import (
    DurableOperationExecutor,
    OperationLifecycle,
    OperationObservation,
    OperationOutcome,
    OperationRequest,
)
from llmkit_lite.queueing import (
    InMemoryOperationQueue,
    OperationJob,
    QueueDelivery,
    QueueError,
)
from llmkit_lite.recovery import (
    DurableOperationHandler,
    RecoveryContext,
    RecoveryDisposition,
    RecoveryError,
    RecoveryWorker,
)
from llmkit_lite.state import InMemoryStateStore, OperationState, OperationStatus


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


class ScriptedAdapter:
    def __init__(
        self,
        *,
        dispatch_results: list[OperationObservation | BaseException] | None = None,
        inspect_results: list[OperationObservation | BaseException] | None = None,
    ) -> None:
        self.dispatch_results = list(dispatch_results or [])
        self.inspect_results = list(inspect_results or [])
        self.dispatch_calls: list[OperationState] = []
        self.inspect_calls: list[OperationState] = []

    async def dispatch(self, operation: OperationState) -> OperationObservation:
        self.dispatch_calls.append(operation)
        return self._next(self.dispatch_results)

    async def inspect(self, operation: OperationState) -> OperationObservation:
        self.inspect_calls.append(operation)
        return self._next(self.inspect_results)

    @staticmethod
    def _next(
        results: list[OperationObservation | BaseException],
    ) -> OperationObservation:
        if not results:
            raise AssertionError("scripted adapter ran out of results")
        result = results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


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


@pytest.fixture
def lease_manager(frozen_clock: FrozenClock) -> LeaseManager:
    return LeaseManager(
        InMemoryLeaseStore(),
        lease_duration=timedelta(seconds=30),
        clock=frozen_clock,
    )


def terminal_state(
    job: OperationJob,
    status: OperationStatus = OperationStatus.COMPLETED,
) -> OperationState:
    return OperationState(
        operation_id=job.operation_id,
        identity=job.identity,
        idempotency_key=job.idempotency_key,
        request_fingerprint=job.request.fingerprint,
        status=status,
        revision=1,
        values={},
    )


def durable_handler(
    state_store: InMemoryStateStore,
    adapter: ScriptedAdapter,
) -> DurableOperationHandler:
    return DurableOperationHandler(
        DurableOperationExecutor(OperationLifecycle(state_store), adapter)
    )


async def test_given_empty_queue_when_worker_runs_then_empty_result_is_returned(
    operation_queue: InMemoryOperationQueue,
    lease_manager: LeaseManager,
) -> None:
    async def handler(job: OperationJob, context: RecoveryContext) -> OperationState:
        raise AssertionError("handler should not run")

    worker = RecoveryWorker(
        operation_queue,
        lease_manager,
        handler,
        worker_id="worker-1",
    )

    result = await worker.run_once()

    assert result.disposition is RecoveryDisposition.EMPTY
    assert result.delivery_attempt is None
    assert result.state is None


@pytest.mark.parametrize(
    "outcome,expected_status",
    [
        (OperationOutcome.COMPLETED, OperationStatus.COMPLETED),
        (OperationOutcome.FAILED, OperationStatus.FAILED),
    ],
)
async def test_given_terminal_execution_when_worker_runs_then_message_is_acknowledged(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
    lease_manager: LeaseManager,
    outcome: OperationOutcome,
    expected_status: OperationStatus,
) -> None:
    await operation_queue.publish(operation_job, message_id="message-1")
    adapter = ScriptedAdapter(
        dispatch_results=[OperationObservation(outcome, {"result": "done"})]
    )
    worker = RecoveryWorker(
        operation_queue,
        lease_manager,
        durable_handler(InMemoryStateStore(), adapter),
        worker_id="worker-1",
    )

    result = await worker.run_once()

    assert result.disposition is RecoveryDisposition.ACKNOWLEDGED
    assert result.delivery_attempt == 1
    assert result.state is not None and result.state.status is expected_status
    assert len(adapter.dispatch_calls) == 1
    assert await operation_queue.receive() is None


async def test_given_active_lease_when_other_worker_runs_then_delivery_is_deferred(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    await operation_queue.publish(operation_job, message_id="message-1")
    await lease_manager.acquire(operation_job.operation_id, "worker-1")
    handler_calls = 0

    async def handler(job: OperationJob, context: RecoveryContext) -> OperationState:
        nonlocal handler_calls
        handler_calls += 1
        return terminal_state(job)

    worker = RecoveryWorker(
        operation_queue,
        lease_manager,
        handler,
        worker_id="worker-2",
    )

    deferred = await worker.run_once()
    frozen_clock.advance(timedelta(seconds=30))
    recovered = await worker.run_once()

    assert deferred.disposition is RecoveryDisposition.DEFERRED
    assert deferred.delivery_attempt == 1
    assert recovered.disposition is RecoveryDisposition.ACKNOWLEDGED
    assert recovered.delivery_attempt == 2
    assert handler_calls == 1


async def test_given_uncertain_dispatch_when_redelivered_then_recovery_inspects_once(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    await operation_queue.publish(operation_job, message_id="message-1")
    state_store = InMemoryStateStore()
    first_adapter = ScriptedAdapter(
        dispatch_results=[TimeoutError("private downstream timeout")]
    )
    first_worker = RecoveryWorker(
        operation_queue,
        lease_manager,
        durable_handler(state_store, first_adapter),
        worker_id="worker-1",
    )

    first = await first_worker.run_once()
    frozen_clock.advance(timedelta(seconds=30))
    recovery_adapter = ScriptedAdapter(
        inspect_results=[
            OperationObservation(OperationOutcome.COMPLETED, {"result": "online"})
        ]
    )
    recovery_worker = RecoveryWorker(
        operation_queue,
        lease_manager,
        durable_handler(state_store, recovery_adapter),
        worker_id="worker-2",
    )
    recovered = await recovery_worker.run_once()

    assert first.disposition is RecoveryDisposition.RETRY
    assert recovered.disposition is RecoveryDisposition.ACKNOWLEDGED
    assert recovered.delivery_attempt == 2
    assert recovered.state is not None
    assert recovered.state.status is OperationStatus.COMPLETED
    assert len(first_adapter.dispatch_calls) == 1
    assert recovery_adapter.dispatch_calls == []
    assert len(recovery_adapter.inspect_calls) == 1
    assert await operation_queue.receive() is None


async def test_given_nonterminal_result_when_worker_runs_then_message_is_retried(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    await operation_queue.publish(operation_job, message_id="message-1")
    adapter = ScriptedAdapter(
        dispatch_results=[OperationObservation(OperationOutcome.UNKNOWN)]
    )
    worker = RecoveryWorker(
        operation_queue,
        lease_manager,
        durable_handler(InMemoryStateStore(), adapter),
        worker_id="worker-1",
    )

    result = await worker.run_once()

    assert result.disposition is RecoveryDisposition.RETRY
    assert result.state is not None
    assert result.state.status is OperationStatus.DISPATCHING
    assert await operation_queue.receive() is None
    frozen_clock.advance(timedelta(seconds=30))
    assert await operation_queue.receive() is not None


async def test_given_cancelled_handler_when_worker_runs_then_message_is_redelivered(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    await operation_queue.publish(operation_job, message_id="message-1")

    async def handler(job: OperationJob, context: RecoveryContext) -> OperationState:
        raise asyncio.CancelledError

    worker = RecoveryWorker(
        operation_queue,
        lease_manager,
        handler,
        worker_id="worker-1",
    )

    with pytest.raises(asyncio.CancelledError):
        await worker.run_once()
    frozen_clock.advance(timedelta(seconds=30))

    redelivered = await operation_queue.receive()
    assert redelivered is not None and redelivered.attempt == 2


async def test_given_handler_failure_when_worker_runs_then_safe_error_preserves_message(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    await operation_queue.publish(operation_job, message_id="private-message-id")

    async def handler(job: OperationJob, context: RecoveryContext) -> OperationState:
        raise RuntimeError("private handler failure")

    worker = RecoveryWorker(
        operation_queue,
        lease_manager,
        handler,
        worker_id="private-worker-id",
    )

    with pytest.raises(RecoveryError) as exc_info:
        await worker.run_once()
    frozen_clock.advance(timedelta(seconds=30))

    assert exc_info.value.code == "recovery_handler_failed"
    assert "private" not in str(exc_info.value)
    assert await operation_queue.receive() is not None


async def test_given_handler_heartbeat_when_processing_then_both_deadlines_extend(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    await operation_queue.publish(operation_job, message_id="message-1")
    snapshots: list[tuple[datetime, datetime, int]] = []

    async def handler(job: OperationJob, context: RecoveryContext) -> OperationState:
        original_expiry = context.lease.expires_at
        frozen_clock.advance(timedelta(seconds=20))
        renewed = await context.heartbeat()
        snapshots.append(
            (
                original_expiry,
                context.delivery.visible_at,
                renewed.revision,
            )
        )
        return terminal_state(job)

    worker = RecoveryWorker(
        operation_queue,
        lease_manager,
        handler,
        worker_id="worker-1",
    )

    result = await worker.run_once()

    original_expiry, queue_expiry, revision = snapshots[0]
    assert result.disposition is RecoveryDisposition.ACKNOWLEDGED
    assert queue_expiry == frozen_clock.current + timedelta(seconds=30)
    assert queue_expiry > original_expiry
    assert revision == 2
    stored = await lease_manager.lease_store.get_lease(operation_job.operation_id)
    assert stored is not None and stored.expires_at == queue_expiry


async def test_given_visibility_failure_when_heartbeating_then_lease_renews_first(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    class FailingVisibilityQueue:
        async def receive(self) -> QueueDelivery | None:
            return await operation_queue.receive()

        async def acknowledge(self, delivery: QueueDelivery) -> None:
            await operation_queue.acknowledge(delivery)

        async def extend_visibility(
            self,
            delivery: QueueDelivery,
            *,
            visibility_timeout: timedelta | None = None,
        ) -> QueueDelivery:
            raise QueueError("queue_stale_delivery")

    await operation_queue.publish(operation_job, message_id="message-1")

    async def handler(job: OperationJob, context: RecoveryContext) -> OperationState:
        frozen_clock.advance(timedelta(seconds=20))
        await context.heartbeat()
        raise AssertionError("failed heartbeat should stop processing")

    worker = RecoveryWorker(
        FailingVisibilityQueue(),  # type: ignore[arg-type]
        lease_manager,
        handler,
        worker_id="worker-1",
    )

    result = await worker.run_once()

    assert result.disposition is RecoveryDisposition.RETRY
    stored = await lease_manager.lease_store.get_lease(operation_job.operation_id)
    assert stored is not None
    assert stored.revision == 2
    assert stored.expires_at == frozen_clock.current + timedelta(seconds=30)


async def test_given_lease_takeover_before_ack_when_worker_finishes_then_it_cannot_ack(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    await operation_queue.publish(operation_job, message_id="message-1")

    async def handler(job: OperationJob, context: RecoveryContext) -> OperationState:
        frozen_clock.current = context.lease.expires_at
        takeover = await lease_manager.acquire(job.operation_id, "worker-2")
        assert takeover.lease.fencing_token == context.fencing_token + 1
        return terminal_state(job)

    worker = RecoveryWorker(
        operation_queue,
        lease_manager,
        handler,
        worker_id="worker-1",
    )

    result = await worker.run_once()
    redelivered = await operation_queue.receive()

    assert result.disposition is RecoveryDisposition.RETRY
    assert result.state is not None
    assert redelivered is not None and redelivered.attempt == 2


async def test_given_mismatched_result_when_worker_runs_then_safe_error_is_raised(
    operation_queue: InMemoryOperationQueue,
    operation_job: OperationJob,
    lease_manager: LeaseManager,
) -> None:
    await operation_queue.publish(operation_job, message_id="message-1")

    async def handler(job: OperationJob, context: RecoveryContext) -> OperationState:
        mismatched = OperationJob(
            job.operation_id,
            job.identity,
            "different-key",
            job.request,
        )
        return terminal_state(mismatched)

    worker = RecoveryWorker(
        operation_queue,
        lease_manager,
        handler,
        worker_id="worker-1",
    )

    with pytest.raises(RecoveryError) as exc_info:
        await worker.run_once()

    assert exc_info.value.code == "recovery_invalid_result"


async def test_given_successful_recovery_when_tracing_then_identifiers_are_excluded(
    frozen_clock: FrozenClock,
    lease_manager: LeaseManager,
    in_memory_tracing,
) -> None:
    queue = InMemoryOperationQueue(
        visibility_timeout=timedelta(seconds=30),
        clock=frozen_clock,
        token_factory=TokenFactory(),
    )
    job = OperationJob(
        "private-operation-id",
        WorkflowIdentity("private-thread", "private-namespace"),
        "private-idempotency-key",
        OperationRequest("private-action", {"secret": "private-payload"}),
    )
    await queue.publish(job, message_id="private-message-id")

    async def handler(job: OperationJob, context: RecoveryContext) -> OperationState:
        return terminal_state(job)

    worker = RecoveryWorker(
        queue,
        lease_manager,
        handler,
        worker_id="private-worker-id",
    )

    result = await worker.run_once()

    span = next(
        item
        for item in in_memory_tracing.get_finished_spans()
        if item.name == "recovery.run_once"
    )
    assert span.status.status_code is StatusCode.UNSET
    assert span.attributes == {
        "queue.delivery_attempt": 1,
        "lease.fencing_token": 1,
        "lease.revision": 1,
        "recovery.outcome": "acknowledged",
        "operation.status": "completed",
        "operation.revision": result.state.revision,
    }
    serialized = repr(span)
    for secret in (
        "private-operation-id",
        "private-thread",
        "private-namespace",
        "private-idempotency-key",
        "private-message-id",
        "private-worker-id",
        "private-payload",
    ):
        assert secret not in serialized
