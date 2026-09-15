"""Renewable, fenced worker leases for durable operations."""

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
    "lease_not_found": "worker lease was not found",
    "lease_expired": "worker lease has expired",
    "lease_not_owned": "worker lease ownership is no longer current",
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


def _require_fencing_token(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("lease fencing token must be an integer")
    if value < 1:
        raise ValueError("lease fencing token must be greater than zero")
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
    fencing_token: int

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
        object.__setattr__(
            self,
            "fencing_token",
            _require_fencing_token(self.fencing_token),
        )

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
            "fencing_token": self.fencing_token,
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
    """Persistence boundary for atomic acquisition and ownership checks."""

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

    async def renew_lease(
        self,
        operation_id: str,
        owner_id: str,
        fencing_token: int,
        *,
        renewed_at: datetime,
        expires_at: datetime,
    ) -> WorkerLease:
        """Extend a current, unexpired lease in one atomic store operation."""

    async def validate_lease(
        self,
        operation_id: str,
        owner_id: str,
        fencing_token: int,
        *,
        checked_at: datetime,
    ) -> WorkerLease:
        """Return a current lease only when its owner and fence still match."""


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
                fencing_token=1 if current is None else current.fencing_token + 1,
            )
            self._leases[resolved_operation_id] = lease
            return LeaseAcquisition(lease, LeaseAcquisitionOutcome.ACQUIRED)

    async def renew_lease(
        self,
        operation_id: str,
        owner_id: str,
        fencing_token: int,
        *,
        renewed_at: datetime,
        expires_at: datetime,
    ) -> WorkerLease:
        resolved_operation_id = _normalize_identifier(operation_id, "operation ID")
        resolved_owner_id = _normalize_identifier(owner_id, "lease owner ID")
        resolved_token = _require_fencing_token(fencing_token)
        resolved_renewed_at = _require_datetime(renewed_at, "lease renewal time")
        resolved_expires_at = _require_datetime(expires_at, "lease expiration time")
        if resolved_expires_at <= resolved_renewed_at:
            raise ValueError("lease expiration time must follow renewal time")

        async with self._lock:
            current = self._require_owned_lease(
                resolved_operation_id,
                resolved_owner_id,
                resolved_token,
                at=resolved_renewed_at,
            )
            renewed = WorkerLease(
                operation_id=current.operation_id,
                owner_id=current.owner_id,
                acquired_at=current.acquired_at,
                expires_at=resolved_expires_at,
                revision=current.revision + 1,
                fencing_token=current.fencing_token,
            )
            self._leases[resolved_operation_id] = renewed
            return renewed

    async def validate_lease(
        self,
        operation_id: str,
        owner_id: str,
        fencing_token: int,
        *,
        checked_at: datetime,
    ) -> WorkerLease:
        resolved_operation_id = _normalize_identifier(operation_id, "operation ID")
        resolved_owner_id = _normalize_identifier(owner_id, "lease owner ID")
        resolved_token = _require_fencing_token(fencing_token)
        resolved_checked_at = _require_datetime(checked_at, "lease check time")

        async with self._lock:
            return self._require_owned_lease(
                resolved_operation_id,
                resolved_owner_id,
                resolved_token,
                at=resolved_checked_at,
            )

    def _require_owned_lease(
        self,
        operation_id: str,
        owner_id: str,
        fencing_token: int,
        *,
        at: datetime,
    ) -> WorkerLease:
        current = self._leases.get(operation_id)
        if current is None:
            raise LeaseError("lease_not_found")
        if current.is_expired(at):
            raise LeaseError("lease_expired")
        if current.owner_id != owner_id or current.fencing_token != fencing_token:
            raise LeaseError("lease_not_owned")
        return current


class LeaseManager:
    """Acquire, renew, and validate leases through an injected atomic store."""

    def __init__(
        self,
        lease_store: LeaseStore,
        *,
        lease_duration: timedelta = timedelta(seconds=30),
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        for method_name in (
            "get_lease",
            "acquire_lease",
            "renew_lease",
            "validate_lease",
        ):
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
                    acquired_at=acquired_at,
                    expires_at=expires_at,
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
                span.set_attribute(
                    "lease.fencing_token",
                    result.lease.fencing_token,
                )
            return result

    async def renew(
        self,
        operation_id: str,
        owner_id: str,
        fencing_token: int,
    ) -> WorkerLease:
        """Atomically extend a lease while its ownership epoch is still current."""

        resolved_operation_id = _normalize_identifier(operation_id, "operation ID")
        resolved_owner_id = _normalize_identifier(owner_id, "lease owner ID")
        resolved_token = _require_fencing_token(fencing_token)
        renewed_at = _require_datetime(self.clock(), "lease clock value")
        expires_at = renewed_at + self.lease_duration

        with trace_span(
            "lease.renew",
            attributes={
                "lease.duration_seconds": self.lease_duration.total_seconds(),
            },
        ) as span:
            try:
                lease = await self.lease_store.renew_lease(
                    resolved_operation_id,
                    resolved_owner_id,
                    resolved_token,
                    renewed_at=renewed_at,
                    expires_at=expires_at,
                )
                _require_owned_result(
                    lease,
                    operation_id=resolved_operation_id,
                    owner_id=resolved_owner_id,
                    fencing_token=resolved_token,
                    checked_at=renewed_at,
                )
                if lease.expires_at != expires_at:
                    raise LeaseError("lease_invalid_record")
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
                span.set_attribute("lease.outcome", "renewed")
                span.set_attribute("lease.revision", lease.revision)
                span.set_attribute("lease.fencing_token", lease.fencing_token)
            return lease

    async def assert_owned(
        self,
        operation_id: str,
        owner_id: str,
        fencing_token: int,
    ) -> WorkerLease:
        """Reject work when the lease is missing, expired, or superseded."""

        resolved_operation_id = _normalize_identifier(operation_id, "operation ID")
        resolved_owner_id = _normalize_identifier(owner_id, "lease owner ID")
        resolved_token = _require_fencing_token(fencing_token)
        checked_at = _require_datetime(self.clock(), "lease clock value")

        with trace_span("lease.validate") as span:
            try:
                lease = await self.lease_store.validate_lease(
                    resolved_operation_id,
                    resolved_owner_id,
                    resolved_token,
                    checked_at=checked_at,
                )
                _require_owned_result(
                    lease,
                    operation_id=resolved_operation_id,
                    owner_id=resolved_owner_id,
                    fencing_token=resolved_token,
                    checked_at=checked_at,
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
                span.set_attribute("lease.outcome", "owned")
                span.set_attribute("lease.revision", lease.revision)
                span.set_attribute("lease.fencing_token", lease.fencing_token)
            return lease


def _require_acquisition(
    result: LeaseAcquisition,
    *,
    operation_id: str,
    owner_id: str,
    acquired_at: datetime,
    expires_at: datetime,
) -> None:
    if not isinstance(result, LeaseAcquisition):
        raise LeaseError("lease_invalid_record")
    if result.lease.operation_id != operation_id:
        raise LeaseError("lease_invalid_record")
    if result.lease.is_expired(acquired_at):
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
    if result.outcome is LeaseAcquisitionOutcome.ACQUIRED and (
        result.lease.acquired_at != acquired_at or result.lease.expires_at != expires_at
    ):
        raise LeaseError("lease_invalid_record")


def _require_owned_result(
    lease: WorkerLease,
    *,
    operation_id: str,
    owner_id: str,
    fencing_token: int,
    checked_at: datetime,
) -> None:
    if not isinstance(lease, WorkerLease):
        raise LeaseError("lease_invalid_record")
    if (
        lease.operation_id != operation_id
        or lease.owner_id != owner_id
        or lease.fencing_token != fencing_token
        or lease.is_expired(checked_at)
    ):
        raise LeaseError("lease_invalid_record")


def _record_failure(span, code: str) -> None:
    if span is not None:
        span.set_attribute("lease.outcome", "failed")
        span.set_attribute("error.type", code)
    set_span_error(span, code)


def _utc_now() -> datetime:
    return datetime.now(UTC)
