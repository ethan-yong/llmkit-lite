"""Atomic worker lease acquisition for durable operations."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol

from llmkit_lite.observability import set_span_error, trace_span

_ERROR_DETAILS = {
    "lease_store_failed": "worker lease could not be persisted",
    "lease_invalid_record": "lease store returned an invalid record",
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


def _require_revision(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("lease revision must be an integer")
    if value < 1:
        raise ValueError("lease revision must be greater than zero")
    return value


def _require_duration(value: timedelta) -> timedelta:
    if not isinstance(value, timedelta):
        raise TypeError("lease duration must be a timedelta")
    seconds = value.total_seconds()
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("lease duration must be greater than zero")
    return value


class LeaseError(Exception):
    """Safe lease failure suitable for application error mapping."""

    def __init__(self, code: str) -> None:
        if code not in _ERROR_DETAILS:
            raise ValueError("unsupported lease error code")
        detail = _ERROR_DETAILS[code]
        super().__init__(detail)
        self.code = code
        self.detail = detail


class LeaseAcquisitionOutcome(StrEnum):
    """Result of atomically attempting to acquire an operation lease."""

    ACQUIRED = "acquired"
    ALREADY_OWNED = "already_owned"
    HELD_BY_OTHER = "held_by_other"


@dataclass(frozen=True, slots=True)
class WorkerLease:
    """Immutable ownership record for one durable operation."""

    operation_id: str
    owner_id: str
    acquired_at: datetime
    expires_at: datetime
    revision: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_id",
            _normalize_identifier(self.operation_id, "operation ID"),
        )
        object.__setattr__(
            self,
            "owner_id",
            _normalize_identifier(self.owner_id, "lease owner ID"),
        )
        acquired_at = _require_datetime(self.acquired_at, "lease acquisition time")
        expires_at = _require_datetime(self.expires_at, "lease expiration time")
        if expires_at <= acquired_at:
            raise ValueError("lease expiration time must follow acquisition time")
        object.__setattr__(self, "revision", _require_revision(self.revision))

    def is_expired(self, at: datetime) -> bool:
        """Return whether the lease has expired at the supplied instant."""

        return _require_datetime(at, "lease comparison time") >= self.expires_at

    def to_dict(self) -> dict[str, str | int]:
        """Return a detached, JSON-compatible representation."""

        return {
            "operation_id": self.operation_id,
            "owner_id": self.owner_id,
            "acquired_at": self.acquired_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "revision": self.revision,
        }


@dataclass(frozen=True, slots=True)
class LeaseAcquisition:
    """The current lease and outcome of an atomic acquisition attempt."""

    lease: WorkerLease
    outcome: LeaseAcquisitionOutcome

    def __post_init__(self) -> None:
        if not isinstance(self.lease, WorkerLease):
            raise TypeError("acquisition lease must be a WorkerLease")
        if not isinstance(self.outcome, LeaseAcquisitionOutcome):
            raise TypeError("acquisition outcome must be a LeaseAcquisitionOutcome")

    @property
    def owns_lease(self) -> bool:
        """Return whether the requesting worker owns the resulting lease."""

        return self.outcome is not LeaseAcquisitionOutcome.HELD_BY_OTHER


class LeaseStore(Protocol):
    """Persistence boundary that must decide lease acquisition atomically."""

    async def get_lease(self, operation_id: str) -> WorkerLease | None:
        """Return the current lease for an operation, if one exists."""

    async def acquire_lease(
        self,
        operation_id: str,
        owner_id: str,
        *,
        acquired_at: datetime,
        expires_at: datetime,
    ) -> LeaseAcquisition:
        """Create, reuse, or take over a lease in one atomic store operation."""


class InMemoryLeaseStore:
    """Concurrency-safe reference lease store for tests and local development."""

    def __init__(self) -> None:
        self._leases: dict[str, WorkerLease] = {}
        self._lock = asyncio.Lock()

    async def get_lease(self, operation_id: str) -> WorkerLease | None:
        resolved_operation_id = _normalize_identifier(operation_id, "operation ID")
        async with self._lock:
            return self._leases.get(resolved_operation_id)

    async def acquire_lease(
        self,
        operation_id: str,
        owner_id: str,
        *,
        acquired_at: datetime,
        expires_at: datetime,
    ) -> LeaseAcquisition:
        resolved_operation_id = _normalize_identifier(operation_id, "operation ID")
        resolved_owner_id = _normalize_identifier(owner_id, "lease owner ID")
        resolved_acquired_at = _require_datetime(
            acquired_at,
            "lease acquisition time",
        )
        resolved_expires_at = _require_datetime(expires_at, "lease expiration time")
        if resolved_expires_at <= resolved_acquired_at:
            raise ValueError("lease expiration time must follow acquisition time")

        async with self._lock:
            current = self._leases.get(resolved_operation_id)
            if current is not None and not current.is_expired(resolved_acquired_at):
                outcome = (
                    LeaseAcquisitionOutcome.ALREADY_OWNED
                    if current.owner_id == resolved_owner_id
                    else LeaseAcquisitionOutcome.HELD_BY_OTHER
                )
                return LeaseAcquisition(current, outcome)

            lease = WorkerLease(
                operation_id=resolved_operation_id,
                owner_id=resolved_owner_id,
                acquired_at=resolved_acquired_at,
                expires_at=resolved_expires_at,
                revision=1 if current is None else current.revision + 1,
            )
            self._leases[resolved_operation_id] = lease
            return LeaseAcquisition(lease, LeaseAcquisitionOutcome.ACQUIRED)


class LeaseManager:
    """Acquire operation leases through an injected atomic store."""

    def __init__(
        self,
        lease_store: LeaseStore,
        *,
        lease_duration: timedelta = timedelta(seconds=30),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        for method_name in ("get_lease", "acquire_lease"):
            if not callable(getattr(lease_store, method_name, None)):
                raise TypeError(f"lease store must implement {method_name}")
        if clock is not None and not callable(clock):
            raise TypeError("lease clock must be callable")
        self.lease_store = lease_store
        self.lease_duration = _require_duration(lease_duration)
        self.clock = clock or _utc_now

    async def acquire(self, operation_id: str, owner_id: str) -> LeaseAcquisition:
        """Atomically acquire an absent or expired lease for one operation."""

        resolved_operation_id = _normalize_identifier(operation_id, "operation ID")
        resolved_owner_id = _normalize_identifier(owner_id, "lease owner ID")
        acquired_at = _require_datetime(self.clock(), "lease clock value")
        expires_at = acquired_at + self.lease_duration

        with trace_span(
            "lease.acquire",
            attributes={
                "lease.duration_seconds": self.lease_duration.total_seconds(),
            },
        ) as span:
            try:
                result = await self.lease_store.acquire_lease(
                    resolved_operation_id,
                    resolved_owner_id,
                    acquired_at=acquired_at,
                    expires_at=expires_at,
                )
                _require_acquisition(
                    result,
                    operation_id=resolved_operation_id,
                    owner_id=resolved_owner_id,
                )
            except asyncio.CancelledError:
                _record_failure(span, "lease_cancelled")
                raise
            except LeaseError as exc:
                _record_failure(span, exc.code)
                raise
            except Exception as exc:
                error = LeaseError("lease_store_failed")
                _record_failure(span, error.code)
                raise error from exc

            if span is not None:
                span.set_attribute("lease.outcome", result.outcome.value)
                span.set_attribute("lease.revision", result.lease.revision)
            return result


def _require_acquisition(
    result: LeaseAcquisition,
    *,
    operation_id: str,
    owner_id: str,
) -> None:
    if not isinstance(result, LeaseAcquisition):
        raise LeaseError("lease_invalid_record")
    if result.lease.operation_id != operation_id:
        raise LeaseError("lease_invalid_record")
    if (
        result.outcome is not LeaseAcquisitionOutcome.HELD_BY_OTHER
        and result.lease.owner_id != owner_id
    ):
        raise LeaseError("lease_invalid_record")
    if (
        result.outcome is LeaseAcquisitionOutcome.HELD_BY_OTHER
        and result.lease.owner_id == owner_id
    ):
        raise LeaseError("lease_invalid_record")


def _record_failure(span, code: str) -> None:
    if span is not None:
        span.set_attribute("lease.outcome", "failed")
        span.set_attribute("error.type", code)
    set_span_error(span, code)


def _utc_now() -> datetime:
    return datetime.now(UTC)
