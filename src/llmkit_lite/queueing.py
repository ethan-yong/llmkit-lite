"""Visibility-based queue contracts for durable operation delivery."""

from __future__ import annotations

import asyncio
import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

from llmkit_lite.graphs import WorkflowIdentity
from llmkit_lite.operations import OperationRequest

_ERROR_DETAILS = {
    "queue_message_exists": "queue message already exists",
    "queue_stale_delivery": "queue delivery is no longer current",
}


def _normalize_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _require_datetime(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _require_duration(value: timedelta, field_name: str) -> timedelta:
    if not isinstance(value, timedelta):
        raise TypeError(f"{field_name} must be a timedelta")
    seconds = value.total_seconds()
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return value


def _require_attempt(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("queue delivery attempt must be an integer")
    if value < 1:
        raise ValueError("queue delivery attempt must be greater than zero")
    return value


class QueueError(Exception):
    """Safe queue failure suitable for application error mapping."""

    def __init__(self, code: str) -> None:
        if code not in _ERROR_DETAILS:
            raise ValueError("unsupported queue error code")
        detail = _ERROR_DETAILS[code]
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class OperationJob:
    """Validated operation request stored in a work queue."""

    operation_id: str
    identity: WorkflowIdentity
    idempotency_key: str
    request: OperationRequest

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_id",
            _normalize_identifier(self.operation_id, "operation ID"),
        )
        if not isinstance(self.identity, WorkflowIdentity):
            raise TypeError("operation job identity must be a WorkflowIdentity")
        object.__setattr__(
            self,
            "idempotency_key",
            _normalize_identifier(self.idempotency_key, "idempotency key"),
        )
        if not isinstance(self.request, OperationRequest):
            raise TypeError("operation job request must be an OperationRequest")


@dataclass(frozen=True, slots=True)
class QueueDelivery:
    """One visibility-limited delivery of an operation job."""

    message_id: str
    job: OperationJob
    receipt_token: str
    attempt: int
    received_at: datetime
    visible_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "message_id",
            _normalize_identifier(self.message_id, "queue message ID"),
        )
        if not isinstance(self.job, OperationJob):
            raise TypeError("queue delivery job must be an OperationJob")
        object.__setattr__(
            self,
            "receipt_token",
            _normalize_identifier(self.receipt_token, "queue receipt token"),
        )
        object.__setattr__(self, "attempt", _require_attempt(self.attempt))
        received_at = _require_datetime(self.received_at, "queue receive time")
        visible_at = _require_datetime(self.visible_at, "queue visibility time")
        if visible_at <= received_at:
            raise ValueError("queue visibility time must follow receive time")


class OperationQueue(Protocol):
    """Queue boundary with visibility timeout and receipt fencing semantics."""

    async def publish(
        self,
        job: OperationJob,
        *,
        message_id: str | None = None,
    ) -> str:
        """Add one operation job and return its stable message ID."""

    async def receive(self) -> QueueDelivery | None:
        """Return and temporarily hide the next visible delivery."""

    async def acknowledge(self, delivery: QueueDelivery) -> None:
        """Permanently remove a current delivery."""

    async def extend_visibility(
        self,
        delivery: QueueDelivery,
        *,
        visibility_timeout: timedelta | None = None,
    ) -> QueueDelivery:
        """Extend a current delivery's invisibility period."""


@dataclass(slots=True)
class _QueuedMessage:
    message_id: str
    job: OperationJob
    visible_at: datetime
    attempt: int = 0
    receipt_token: str | None = None
    received_at: datetime | None = None


class InMemoryOperationQueue:
    """Concurrency-safe visibility queue for tests and local development."""

    def __init__(
        self,
        *,
        visibility_timeout: timedelta = timedelta(seconds=30),
        clock: Callable[[], datetime] | None = None,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        if clock is not None and not callable(clock):
            raise TypeError("queue clock must be callable")
        if token_factory is not None and not callable(token_factory):
            raise TypeError("queue token factory must be callable")
        self.visibility_timeout = _require_duration(
            visibility_timeout,
            "queue visibility timeout",
        )
        self.clock = clock or _utc_now
        self.token_factory = token_factory or _new_token
        self._messages: dict[str, _QueuedMessage] = {}
        self._receipt_sequence = 0
        self._lock = asyncio.Lock()

    async def publish(
        self,
        job: OperationJob,
        *,
        message_id: str | None = None,
    ) -> str:
        if not isinstance(job, OperationJob):
            raise TypeError("queue job must be an OperationJob")
        resolved_message_id = _normalize_identifier(
            self.token_factory() if message_id is None else message_id,
            "queue message ID",
        )
        published_at = _require_datetime(self.clock(), "queue clock value")

        async with self._lock:
            if resolved_message_id in self._messages:
                raise QueueError("queue_message_exists")
            self._messages[resolved_message_id] = _QueuedMessage(
                message_id=resolved_message_id,
                job=job,
                visible_at=published_at,
            )
        return resolved_message_id

    async def receive(self) -> QueueDelivery | None:
        received_at = _require_datetime(self.clock(), "queue clock value")

        async with self._lock:
            record = next(
                (
                    item
                    for item in self._messages.values()
                    if item.visible_at <= received_at
                ),
                None,
            )
            if record is None:
                return None

            self._receipt_sequence += 1
            token = _normalize_identifier(
                self.token_factory(),
                "queue receipt token",
            )
            record.attempt += 1
            record.receipt_token = f"{token}:{self._receipt_sequence}"
            record.received_at = received_at
            record.visible_at = received_at + self.visibility_timeout
            return _delivery_from_record(record)

    async def acknowledge(self, delivery: QueueDelivery) -> None:
        resolved_delivery = _require_delivery(delivery)
        acknowledged_at = _require_datetime(self.clock(), "queue clock value")

        async with self._lock:
            record = self._require_current_delivery(
                resolved_delivery,
                at=acknowledged_at,
            )
            del self._messages[record.message_id]

    async def extend_visibility(
        self,
        delivery: QueueDelivery,
        *,
        visibility_timeout: timedelta | None = None,
    ) -> QueueDelivery:
        resolved_delivery = _require_delivery(delivery)
        resolved_timeout = (
            self.visibility_timeout
            if visibility_timeout is None
            else _require_duration(visibility_timeout, "queue visibility timeout")
        )
        extended_at = _require_datetime(self.clock(), "queue clock value")

        async with self._lock:
            record = self._require_current_delivery(
                resolved_delivery,
                at=extended_at,
            )
            record.visible_at = extended_at + resolved_timeout
            return _delivery_from_record(record)

    def _require_current_delivery(
        self,
        delivery: QueueDelivery,
        *,
        at: datetime,
    ) -> _QueuedMessage:
        record = self._messages.get(delivery.message_id)
        if (
            record is None
            or record.receipt_token != delivery.receipt_token
            or record.visible_at <= at
        ):
            raise QueueError("queue_stale_delivery")
        return record


def _require_delivery(value: QueueDelivery) -> QueueDelivery:
    if not isinstance(value, QueueDelivery):
        raise TypeError("queue delivery must be a QueueDelivery")
    return value


def _delivery_from_record(record: _QueuedMessage) -> QueueDelivery:
    if record.receipt_token is None or record.received_at is None:
        raise RuntimeError("queued message is not currently delivered")
    return QueueDelivery(
        message_id=record.message_id,
        job=record.job,
        receipt_token=record.receipt_token,
        attempt=record.attempt,
        received_at=record.received_at,
        visible_at=record.visible_at,
    )


def _new_token() -> str:
    return uuid.uuid4().hex


def _utc_now() -> datetime:
    return datetime.now(UTC)
