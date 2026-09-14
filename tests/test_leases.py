from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from opentelemetry.trace import StatusCode

from llmkit_lite.leases import (
    InMemoryLeaseStore,
    LeaseAcquisition,
    LeaseAcquisitionOutcome,
    LeaseError,
    LeaseManager,
    WorkerLease,
)


class FrozenClock:
    def __init__(self, current: datetime) -> None:
        self.current = current

    def __call__(self) -> datetime:
        return self.current

    def advance(self, duration: timedelta) -> None:
        self.current += duration


@pytest.fixture
def frozen_clock() -> FrozenClock:
    return FrozenClock(datetime(2026, 9, 14, 12, 0, tzinfo=UTC))


@pytest.fixture
def lease_manager(frozen_clock: FrozenClock) -> LeaseManager:
    return LeaseManager(
        InMemoryLeaseStore(),
        lease_duration=timedelta(seconds=30),
        clock=frozen_clock,
    )


async def test_given_no_lease_when_acquiring_then_worker_becomes_owner(
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    result = await lease_manager.acquire("operation-1", "worker-1")

    assert result.outcome is LeaseAcquisitionOutcome.ACQUIRED
    assert result.owns_lease is True
    assert result.lease == WorkerLease(
        operation_id="operation-1",
        owner_id="worker-1",
        acquired_at=frozen_clock.current,
        expires_at=frozen_clock.current + timedelta(seconds=30),
        revision=1,
    )


async def test_given_valid_owned_lease_when_reacquiring_then_expiry_is_unchanged(
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    first = await lease_manager.acquire("operation-1", "worker-1")
    frozen_clock.advance(timedelta(seconds=10))

    repeated = await lease_manager.acquire("operation-1", "worker-1")

    assert repeated.outcome is LeaseAcquisitionOutcome.ALREADY_OWNED
    assert repeated.owns_lease is True
    assert repeated.lease is first.lease
    assert repeated.lease.expires_at == first.lease.expires_at
    assert repeated.lease.revision == 1


async def test_given_valid_lease_when_other_worker_acquires_then_owner_is_preserved(
    lease_manager: LeaseManager,
) -> None:
    first = await lease_manager.acquire("operation-1", "worker-1")

    rejected = await lease_manager.acquire("operation-1", "worker-2")

    assert rejected.outcome is LeaseAcquisitionOutcome.HELD_BY_OTHER
    assert rejected.owns_lease is False
    assert rejected.lease is first.lease


async def test_given_lease_at_expiry_when_other_worker_acquires_then_takeover_succeeds(
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    first = await lease_manager.acquire("operation-1", "worker-1")
    frozen_clock.current = first.lease.expires_at

    takeover = await lease_manager.acquire("operation-1", "worker-2")

    assert takeover.outcome is LeaseAcquisitionOutcome.ACQUIRED
    assert takeover.lease.owner_id == "worker-2"
    assert takeover.lease.acquired_at == first.lease.expires_at
    assert takeover.lease.revision == 2


async def test_given_many_workers_when_acquiring_concurrently_then_exactly_one_wins(
    lease_manager: LeaseManager,
) -> None:
    results = await asyncio.gather(
        *(
            lease_manager.acquire("operation-1", f"worker-{index}")
            for index in range(20)
        )
    )

    acquired = [
        result
        for result in results
        if result.outcome is LeaseAcquisitionOutcome.ACQUIRED
    ]
    rejected = [
        result
        for result in results
        if result.outcome is LeaseAcquisitionOutcome.HELD_BY_OTHER
    ]
    assert len(acquired) == 1
    assert len(rejected) == 19
    assert {result.lease for result in results} == {acquired[0].lease}


async def test_given_expired_lease_when_workers_take_over_then_one_revision_is_added(
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    original = await lease_manager.acquire("operation-1", "worker-original")
    frozen_clock.current = original.lease.expires_at

    results = await asyncio.gather(
        *(
            lease_manager.acquire("operation-1", f"worker-{index}")
            for index in range(10)
        )
    )

    acquired = [
        result
        for result in results
        if result.outcome is LeaseAcquisitionOutcome.ACQUIRED
    ]
    assert len(acquired) == 1
    assert all(result.lease.revision == 2 for result in results)
    assert {result.lease.owner_id for result in results} == {acquired[0].lease.owner_id}


async def test_given_different_operations_when_acquiring_then_leases_are_independent(
    lease_manager: LeaseManager,
) -> None:
    first, second = await asyncio.gather(
        lease_manager.acquire("operation-1", "worker-1"),
        lease_manager.acquire("operation-2", "worker-2"),
    )

    assert first.outcome is LeaseAcquisitionOutcome.ACQUIRED
    assert second.outcome is LeaseAcquisitionOutcome.ACQUIRED
    assert first.lease.operation_id != second.lease.operation_id


def test_given_lease_when_serializing_then_values_are_json_compatible(
    frozen_clock: FrozenClock,
) -> None:
    lease = WorkerLease(
        " operation-1 ",
        " worker-1 ",
        frozen_clock.current,
        frozen_clock.current + timedelta(seconds=30),
        1,
    )

    assert lease.to_dict() == {
        "operation_id": "operation-1",
        "owner_id": "worker-1",
        "acquired_at": "2026-09-14T12:00:00+00:00",
        "expires_at": "2026-09-14T12:00:30+00:00",
        "revision": 1,
    }


@pytest.mark.parametrize("identifier", ["", "   ", 123, None])
async def test_given_invalid_identifiers_when_acquiring_then_validation_fails(
    lease_manager: LeaseManager,
    identifier,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        await lease_manager.acquire(identifier, "worker-1")
    with pytest.raises((TypeError, ValueError)):
        await lease_manager.acquire("operation-1", identifier)


@pytest.mark.parametrize(
    "duration",
    [timedelta(0), timedelta(seconds=-1), "30 seconds", None],
)
def test_given_invalid_duration_when_constructing_manager_then_validation_fails(
    duration,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        LeaseManager(InMemoryLeaseStore(), lease_duration=duration)


def test_given_naive_timestamps_when_constructing_lease_then_validation_fails() -> None:
    naive = datetime(2026, 9, 14, 12, 0)

    with pytest.raises(ValueError, match="timezone-aware"):
        WorkerLease(
            "operation-1",
            "worker-1",
            naive,
            naive + timedelta(seconds=30),
            1,
        )


async def test_given_unexpected_store_failure_when_acquiring_then_error_is_safe(
    frozen_clock: FrozenClock,
) -> None:
    class BrokenStore:
        async def get_lease(self, operation_id: str) -> None:
            raise RuntimeError("private database failure")

        async def acquire_lease(self, *args, **kwargs) -> LeaseAcquisition:
            raise RuntimeError("private database failure")

    manager = LeaseManager(BrokenStore(), clock=frozen_clock)  # type: ignore[arg-type]

    with pytest.raises(LeaseError) as exc_info:
        await manager.acquire("private-operation-id", "private-worker-id")

    assert exc_info.value.code == "lease_store_failed"
    assert "private" not in str(exc_info.value)
    assert "database" not in str(exc_info.value)


async def test_given_store_validation_failure_when_acquiring_then_error_is_safe(
    frozen_clock: FrozenClock,
) -> None:
    class BrokenStore:
        async def get_lease(self, operation_id: str) -> None:
            raise ValueError("private adapter validation detail")

        async def acquire_lease(self, *args, **kwargs) -> LeaseAcquisition:
            raise ValueError("private adapter validation detail")

    manager = LeaseManager(BrokenStore(), clock=frozen_clock)  # type: ignore[arg-type]

    with pytest.raises(LeaseError) as exc_info:
        await manager.acquire("private-operation-id", "private-worker-id")

    assert exc_info.value.code == "lease_store_failed"
    assert "private" not in str(exc_info.value)


async def test_given_invalid_store_result_when_acquiring_then_error_is_safe(
    frozen_clock: FrozenClock,
) -> None:
    class InvalidStore:
        async def get_lease(self, operation_id: str) -> None:
            return None

        async def acquire_lease(self, *args, **kwargs) -> object:
            return object()

    manager = LeaseManager(InvalidStore(), clock=frozen_clock)  # type: ignore[arg-type]

    with pytest.raises(LeaseError) as exc_info:
        await manager.acquire("private-operation-id", "private-worker-id")

    assert exc_info.value.code == "lease_invalid_record"
    assert "private" not in str(exc_info.value)


async def test_given_acquisition_when_tracing_then_identifiers_are_excluded(
    frozen_clock: FrozenClock,
    in_memory_tracing,
) -> None:
    manager = LeaseManager(
        InMemoryLeaseStore(),
        lease_duration=timedelta(seconds=30),
        clock=frozen_clock,
    )

    result = await manager.acquire("private-operation-id", "private-worker-id")

    span = next(
        item
        for item in in_memory_tracing.get_finished_spans()
        if item.name == "lease.acquire"
    )
    assert span.status.status_code is StatusCode.UNSET
    assert span.attributes == {
        "lease.duration_seconds": 30.0,
        "lease.outcome": "acquired",
        "lease.revision": result.lease.revision,
    }
    serialized = repr(span)
    assert "private-operation-id" not in serialized
    assert "private-worker-id" not in serialized
