"""Queue redelivery coordination for leased durable operation recovery."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from llmkit_lite.leases import (
    LeaseAcquisitionOutcome,
    LeaseError,
    LeaseManager,
    WorkerLease,
)
from llmkit_lite.observability import set_span_error, trace_span
from llmkit_lite.operations import (
    DurableOperationExecutor,
    OperationExecutionError,
    OperationLifecycleError,
)
from llmkit_lite.queueing import (
    OperationJob,
    OperationQueue,
    QueueDelivery,
    QueueError,
)
from llmkit_lite.state import OperationState, OperationStatus, StateStoreError

_ERROR_DETAILS = {
    "recovery_queue_failed": "recovery queue operation failed",
    "recovery_handler_failed": "recovery job handler failed",
    "recovery_invalid_delivery": "recovery queue returned an invalid delivery",
    "recovery_invalid_result": "recovery job handler returned an invalid result",
}

_TERMINAL_STATUSES = frozenset(
    {
        OperationStatus.COMPLETED,
        OperationStatus.FAILED,
    }
)


def _normalize_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


class RecoveryError(Exception):
    """Safe recovery failure suitable for application error mapping."""

    def __init__(self, code: str) -> None:
        if code not in _ERROR_DETAILS:
            raise ValueError("unsupported recovery error code")
        detail = _ERROR_DETAILS[code]
        super().__init__(detail)
        self.code = code
        self.detail = detail


class RecoveryDisposition(StrEnum):
    """What happened to one attempted queue delivery."""

    EMPTY = "empty"
    DEFERRED = "deferred"
    RETRY = "retry"
    ACKNOWLEDGED = "acknowledged"


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    """Outcome of one recovery-worker iteration."""

    disposition: RecoveryDisposition
    delivery_attempt: int | None = None
    state: OperationState | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, RecoveryDisposition):
            raise TypeError("recovery disposition must be a RecoveryDisposition")
        if self.delivery_attempt is not None and (
            isinstance(self.delivery_attempt, bool)
            or not isinstance(self.delivery_attempt, int)
            or self.delivery_attempt < 1
        ):
            raise ValueError("recovery delivery attempt must be a positive integer")
        if self.state is not None and not isinstance(self.state, OperationState):
            raise TypeError("recovery state must be an OperationState")
        if self.disposition is RecoveryDisposition.EMPTY and (
            self.delivery_attempt is not None or self.state is not None
        ):
            raise ValueError("empty recovery result cannot include delivery state")
        if (
            self.disposition is not RecoveryDisposition.EMPTY
            and self.delivery_attempt is None
        ):
            raise ValueError("non-empty recovery result requires a delivery attempt")


class RecoveryHandler(Protocol):
    """Process one operation job while carrying its current lease context."""

    async def __call__(
        self,
        job: OperationJob,
        context: RecoveryContext,
    ) -> OperationState:
        """Return the operation state reached during this delivery."""


class RecoveryContext:
    """Mutable lease and delivery context available to a recovery handler."""

    def __init__(
        self,
        lease_manager: LeaseManager,
        queue: OperationQueue,
        delivery: QueueDelivery,
        lease: WorkerLease,
    ) -> None:
        if not isinstance(lease_manager, LeaseManager):
            raise TypeError("lease manager must be a LeaseManager")
        for method_name in ("acknowledge", "extend_visibility"):
            if not callable(getattr(queue, method_name, None)):
                raise TypeError(f"operation queue must implement {method_name}")
        if not isinstance(delivery, QueueDelivery):
            raise TypeError("recovery delivery must be a QueueDelivery")
        if not isinstance(lease, WorkerLease):
            raise TypeError("recovery lease must be a WorkerLease")
        if lease.operation_id != delivery.job.operation_id:
            raise ValueError("recovery lease must match the delivery operation")
        self.lease_manager = lease_manager
        self.queue = queue
        self._delivery = delivery
        self._lease = lease

    @property
    def delivery(self) -> QueueDelivery:
        """Return the most recent queue-delivery snapshot."""

        return self._delivery

    @property
    def lease(self) -> WorkerLease:
        """Return the most recently renewed lease snapshot."""

        return self._lease

    @property
    def fencing_token(self) -> int:
        """Return the ownership epoch to pass to protected resource writes."""

        return self._lease.fencing_token

    async def heartbeat(self) -> WorkerLease:
        """Renew the lease first, then extend this delivery's visibility."""

        with trace_span(
            "recovery.heartbeat",
            attributes={"queue.delivery_attempt": self._delivery.attempt},
        ) as span:
            try:
                renewed = await self.lease_manager.renew(
                    self._lease.operation_id,
                    self._lease.owner_id,
                    self._lease.fencing_token,
                )
                self._lease = renewed
                delivery = await self.queue.extend_visibility(self._delivery)
                self._delivery = _require_extended_delivery(
                    delivery,
                    previous=self._delivery,
                )
            except asyncio.CancelledError:
                _record_failure(span, "recovery_cancelled")
                raise
            except (LeaseError, QueueError, RecoveryError) as exc:
                _record_failure(span, exc.code)
                raise
            except Exception as exc:
                error = RecoveryError("recovery_queue_failed")
                _record_failure(span, error.code)
                raise error from exc

            if span is not None:
                span.set_attribute("recovery.outcome", "renewed")
                span.set_attribute("lease.revision", renewed.revision)
                span.set_attribute("lease.fencing_token", renewed.fencing_token)
            return renewed

    async def assert_owned(self) -> WorkerLease:
        """Verify that this worker's current ownership epoch remains valid."""

        return await self.lease_manager.assert_owned(
            self._lease.operation_id,
            self._lease.owner_id,
            self._lease.fencing_token,
        )


class DurableOperationHandler:
    """Adapt DurableOperationExecutor to the recovery-handler contract."""

    def __init__(self, executor: DurableOperationExecutor) -> None:
        if not isinstance(executor, DurableOperationExecutor):
            raise TypeError("executor must be a DurableOperationExecutor")
        self.executor = executor

    async def __call__(
        self,
        job: OperationJob,
        context: RecoveryContext,
    ) -> OperationState:
        if not isinstance(job, OperationJob):
            raise TypeError("recovery job must be an OperationJob")
        if not isinstance(context, RecoveryContext):
            raise TypeError("recovery context must be a RecoveryContext")
        await context.assert_owned()
        return await self.executor.execute(
            job.operation_id,
            job.identity,
            job.idempotency_key,
            job.request,
        )


class RecoveryWorker:
    """Process at most one visible job using leases and durable operation state."""

    def __init__(
        self,
        queue: OperationQueue,
        lease_manager: LeaseManager,
        handler: RecoveryHandler,
        *,
        worker_id: str,
    ) -> None:
        for method_name in ("receive", "acknowledge", "extend_visibility"):
            if not callable(getattr(queue, method_name, None)):
                raise TypeError(f"operation queue must implement {method_name}")
        if not isinstance(lease_manager, LeaseManager):
            raise TypeError("lease manager must be a LeaseManager")
        if not callable(handler):
            raise TypeError("recovery handler must be callable")
        self.queue = queue
        self.lease_manager = lease_manager
        self.handler = handler
        self.worker_id = _normalize_identifier(worker_id, "worker ID")

    async def run_once(self) -> RecoveryResult:
        """Process one visible delivery, acknowledging only a terminal result."""

        with trace_span("recovery.run_once") as span:
            delivery = await self._receive(span)
            if delivery is None:
                result = RecoveryResult(RecoveryDisposition.EMPTY)
                _record_result(span, result)
                return result
            if span is not None:
                span.set_attribute("queue.delivery_attempt", delivery.attempt)

            try:
                acquisition = await self.lease_manager.acquire(
                    delivery.job.operation_id,
                    self.worker_id,
                )
            except asyncio.CancelledError:
                _record_failure(span, "recovery_cancelled")
                raise
            except LeaseError as exc:
                result = RecoveryResult(
                    RecoveryDisposition.RETRY,
                    delivery_attempt=delivery.attempt,
                )
                _record_result(span, result, reason=exc.code)
                return result

            if span is not None:
                span.set_attribute(
                    "lease.fencing_token",
                    acquisition.lease.fencing_token,
                )
                span.set_attribute("lease.revision", acquisition.lease.revision)

            if acquisition.outcome is LeaseAcquisitionOutcome.HELD_BY_OTHER:
                result = RecoveryResult(
                    RecoveryDisposition.DEFERRED,
                    delivery_attempt=delivery.attempt,
                )
                _record_result(span, result, reason="lease_held_by_other")
                return result

            context = RecoveryContext(
                self.lease_manager,
                self.queue,
                delivery,
                acquisition.lease,
            )
            state = await self._handle(delivery.job, context, span)
            if state is None:
                result = RecoveryResult(
                    RecoveryDisposition.RETRY,
                    delivery_attempt=delivery.attempt,
                )
                _record_result(span, result, reason="operation_retryable")
                return result

            try:
                _require_operation_result(state, delivery.job)
            except RecoveryError as exc:
                _record_failure(span, exc.code)
                raise

            if state.status not in _TERMINAL_STATUSES:
                result = RecoveryResult(
                    RecoveryDisposition.RETRY,
                    delivery_attempt=delivery.attempt,
                    state=state,
                )
                _record_result(span, result, reason="operation_non_terminal")
                return result

            try:
                await context.assert_owned()
                await self.queue.acknowledge(context.delivery)
            except asyncio.CancelledError:
                _record_failure(span, "recovery_cancelled")
                raise
            except (LeaseError, QueueError) as exc:
                result = RecoveryResult(
                    RecoveryDisposition.RETRY,
                    delivery_attempt=delivery.attempt,
                    state=state,
                )
                _record_result(span, result, reason=exc.code)
                return result
            except Exception as exc:
                error = RecoveryError("recovery_queue_failed")
                _record_failure(span, error.code)
                raise error from exc

            result = RecoveryResult(
                RecoveryDisposition.ACKNOWLEDGED,
                delivery_attempt=delivery.attempt,
                state=state,
            )
            _record_result(span, result)
            return result

    async def _receive(self, span) -> QueueDelivery | None:
        try:
            delivery = await self.queue.receive()
        except asyncio.CancelledError:
            _record_failure(span, "recovery_cancelled")
            raise
        except Exception as exc:
            error = RecoveryError("recovery_queue_failed")
            _record_failure(span, error.code)
            raise error from exc
        if delivery is not None and not isinstance(delivery, QueueDelivery):
            error = RecoveryError("recovery_invalid_delivery")
            _record_failure(span, error.code)
            raise error
        return delivery

    async def _handle(
        self,
        job: OperationJob,
        context: RecoveryContext,
        span,
    ) -> OperationState | None:
        try:
            return await self.handler(job, context)
        except asyncio.CancelledError:
            _record_failure(span, "recovery_cancelled")
            raise
        except (
            LeaseError,
            QueueError,
            OperationExecutionError,
            OperationLifecycleError,
            StateStoreError,
        ):
            return None
        except RecoveryError as exc:
            _record_failure(span, exc.code)
            raise
        except Exception as exc:
            error = RecoveryError("recovery_handler_failed")
            _record_failure(span, error.code)
            raise error from exc


def _require_extended_delivery(
    value: Any,
    *,
    previous: QueueDelivery,
) -> QueueDelivery:
    if not isinstance(value, QueueDelivery) or (
        value.message_id != previous.message_id
        or value.job != previous.job
        or value.attempt != previous.attempt
        or value.visible_at <= previous.received_at
    ):
        raise RecoveryError("recovery_invalid_delivery")
    return value


def _require_operation_result(state: Any, job: OperationJob) -> OperationState:
    if not isinstance(state, OperationState) or (
        state.identity != job.identity
        or state.idempotency_key != job.idempotency_key
        or state.request_fingerprint != job.request.fingerprint
    ):
        raise RecoveryError("recovery_invalid_result")
    return state


def _record_result(
    span,
    result: RecoveryResult,
    *,
    reason: str | None = None,
) -> None:
    if span is None:
        return
    span.set_attribute("recovery.outcome", result.disposition.value)
    if result.delivery_attempt is not None:
        span.set_attribute("queue.delivery_attempt", result.delivery_attempt)
    if result.state is not None:
        span.set_attribute("operation.status", result.state.status.value)
        span.set_attribute("operation.revision", result.state.revision)
    if reason is not None:
        span.set_attribute("recovery.reason", reason)


def _record_failure(span, code: str) -> None:
    if span is not None:
        span.set_attribute("recovery.outcome", "failed")
        span.set_attribute("error.type", code)
    set_span_error(span, code)
