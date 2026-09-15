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
        fencing_token=1,
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
    assert takeover.lease.fencing_token == 2


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
    assert acquired[0].lease.fencing_token == 1


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
    assert all(result.lease.fencing_token == 2 for result in results)
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
        1,
    )

    assert lease.to_dict() == {
        "operation_id": "operation-1",
        "owner_id": "worker-1",
        "acquired_at": "2026-09-14T12:00:00+00:00",
        "expires_at": "2026-09-14T12:00:30+00:00",
        "revision": 1,
        "fencing_token": 1,
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
            1,
        )


async def test_given_unexpected_store_failure_when_using_leases_then_error_is_safe(
    frozen_clock: FrozenClock,
) -> None:
    class BrokenStore:
        async def get_lease(self, operation_id: str) -> None:
            raise RuntimeError("private database failure")

        async def acquire_lease(self, *args, **kwargs) -> LeaseAcquisition:
            raise RuntimeError("private database failure")

        async def renew_lease(self, *args, **kwargs) -> WorkerLease:
            raise RuntimeError("private database failure")

        async def validate_lease(self, *args, **kwargs) -> WorkerLease:
            raise RuntimeError("private database failure")

    manager = LeaseManager(BrokenStore(), clock=frozen_clock)  # type: ignore[arg-type]

    with pytest.raises(LeaseError) as acquire_info:
        await manager.acquire("private-operation-id", "private-worker-id")
    with pytest.raises(LeaseError) as renew_info:
        await manager.renew("private-operation-id", "private-worker-id", 1)
    with pytest.raises(LeaseError) as validation_info:
        await manager.assert_owned("private-operation-id", "private-worker-id", 1)

    for error in (acquire_info.value, renew_info.value, validation_info.value):
        assert error.code == "lease_store_failed"
        assert "private" not in str(error)
        assert "database" not in str(error)


async def test_given_store_validation_failure_when_acquiring_then_error_is_safe(
    frozen_clock: FrozenClock,
) -> None:
    class BrokenStore:
        async def get_lease(self, operation_id: str) -> None:
            raise ValueError("private adapter validation detail")

        async def acquire_lease(self, *args, **kwargs) -> LeaseAcquisition:
            raise ValueError("private adapter validation detail")

        async def renew_lease(self, *args, **kwargs) -> WorkerLease:
            raise ValueError("private adapter validation detail")

        async def validate_lease(self, *args, **kwargs) -> WorkerLease:
            raise ValueError("private adapter validation detail")

    manager = LeaseManager(BrokenStore(), clock=frozen_clock)  # type: ignore[arg-type]

    with pytest.raises(LeaseError) as exc_info:
        await manager.acquire("private-operation-id", "private-worker-id")

    assert exc_info.value.code == "lease_store_failed"
    assert "private" not in str(exc_info.value)


async def test_given_invalid_store_result_when_using_leases_then_error_is_safe(
    frozen_clock: FrozenClock,
) -> None:
    class InvalidStore:
        async def get_lease(self, operation_id: str) -> None:
            return None

        async def acquire_lease(self, *args, **kwargs) -> object:
            return object()

        async def renew_lease(self, *args, **kwargs) -> object:
            return object()

        async def validate_lease(self, *args, **kwargs) -> object:
            return object()

    manager = LeaseManager(InvalidStore(), clock=frozen_clock)  # type: ignore[arg-type]

    with pytest.raises(LeaseError) as acquire_info:
        await manager.acquire("private-operation-id", "private-worker-id")
    with pytest.raises(LeaseError) as renew_info:
        await manager.renew("private-operation-id", "private-worker-id", 1)
    with pytest.raises(LeaseError) as validation_info:
        await manager.assert_owned("private-operation-id", "private-worker-id", 1)

    for error in (acquire_info.value, renew_info.value, validation_info.value):
        assert error.code == "lease_invalid_record"
        assert "private" not in str(error)


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
        "lease.fencing_token": result.lease.fencing_token,
    }
    serialized = repr(span)
    assert "private-operation-id" not in serialized
    assert "private-worker-id" not in serialized


async def test_given_owned_lease_when_renewing_then_expiry_and_revision_advance(
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    acquired = await lease_manager.acquire("operation-1", "worker-1")
    frozen_clock.advance(timedelta(seconds=20))

    renewed = await lease_manager.renew(
        "operation-1",
        "worker-1",
        acquired.lease.fencing_token,
    )

    assert renewed.acquired_at == acquired.lease.acquired_at
    assert renewed.expires_at == frozen_clock.current + timedelta(seconds=30)
    assert renewed.expires_at > acquired.lease.expires_at
    assert renewed.revision == acquired.lease.revision + 1
    assert renewed.fencing_token == acquired.lease.fencing_token


async def test_given_repeated_heartbeats_when_renewing_then_fence_stays_constant(
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    acquired = await lease_manager.acquire("operation-1", "worker-1")
    frozen_clock.advance(timedelta(seconds=10))
    first_renewal = await lease_manager.renew(
        "operation-1",
        "worker-1",
        acquired.lease.fencing_token,
    )
    frozen_clock.advance(timedelta(seconds=10))

    second_renewal = await lease_manager.renew(
        "operation-1",
        "worker-1",
        acquired.lease.fencing_token,
    )

    assert second_renewal.revision == first_renewal.revision + 1
    assert second_renewal.expires_at > first_renewal.expires_at
    assert second_renewal.fencing_token == acquired.lease.fencing_token


async def test_given_renewed_lease_when_old_expiry_arrives_then_takeover_is_rejected(
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    acquired = await lease_manager.acquire("operation-1", "worker-1")
    frozen_clock.advance(timedelta(seconds=20))
    renewed = await lease_manager.renew(
        "operation-1",
        "worker-1",
        acquired.lease.fencing_token,
    )
    frozen_clock.current = acquired.lease.expires_at

    contender = await lease_manager.acquire("operation-1", "worker-2")

    assert contender.outcome is LeaseAcquisitionOutcome.HELD_BY_OTHER
    assert contender.lease is renewed


@pytest.mark.parametrize(
    "owner_id,fencing_token",
    [("worker-2", 1), ("worker-1", 2)],
)
async def test_given_wrong_owner_or_fence_when_renewing_then_lease_is_rejected(
    lease_manager: LeaseManager,
    owner_id: str,
    fencing_token: int,
) -> None:
    acquired = await lease_manager.acquire("operation-1", "worker-1")

    with pytest.raises(LeaseError) as exc_info:
        await lease_manager.renew("operation-1", owner_id, fencing_token)

    assert exc_info.value.code == "lease_not_owned"
    assert (
        await lease_manager.assert_owned(
            "operation-1",
            "worker-1",
            acquired.lease.fencing_token,
        )
        is acquired.lease
    )


async def test_given_expired_lease_when_renewing_then_lease_is_rejected(
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    acquired = await lease_manager.acquire("operation-1", "worker-1")
    frozen_clock.current = acquired.lease.expires_at

    with pytest.raises(LeaseError) as exc_info:
        await lease_manager.renew(
            "operation-1",
            "worker-1",
            acquired.lease.fencing_token,
        )

    assert exc_info.value.code == "lease_expired"


async def test_given_missing_lease_when_renewing_then_lease_is_rejected(
    lease_manager: LeaseManager,
) -> None:
    with pytest.raises(LeaseError) as exc_info:
        await lease_manager.renew("operation-missing", "worker-1", 1)

    assert exc_info.value.code == "lease_not_found"


async def test_given_current_fence_when_validating_then_current_lease_is_returned(
    lease_manager: LeaseManager,
) -> None:
    acquired = await lease_manager.acquire("operation-1", "worker-1")

    validated = await lease_manager.assert_owned(
        "operation-1",
        "worker-1",
        acquired.lease.fencing_token,
    )

    assert validated is acquired.lease


async def test_given_expired_lease_when_validating_then_work_is_rejected(
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    acquired = await lease_manager.acquire("operation-1", "worker-1")
    frozen_clock.current = acquired.lease.expires_at + timedelta(seconds=1)

    with pytest.raises(LeaseError) as exc_info:
        await lease_manager.assert_owned(
            "operation-1",
            "worker-1",
            acquired.lease.fencing_token,
        )

    assert exc_info.value.code == "lease_expired"


@pytest.mark.parametrize(
    "owner_id,fencing_token",
    [("worker-2", 1), ("worker-1", 2)],
)
async def test_given_wrong_owner_or_fence_when_validating_then_work_is_rejected(
    lease_manager: LeaseManager,
    owner_id: str,
    fencing_token: int,
) -> None:
    await lease_manager.acquire("operation-1", "worker-1")

    with pytest.raises(LeaseError) as exc_info:
        await lease_manager.assert_owned(
            "operation-1",
            owner_id,
            fencing_token,
        )

    assert exc_info.value.code == "lease_not_owned"


async def test_given_takeover_when_old_worker_continues_then_old_fence_is_rejected(
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    original = await lease_manager.acquire("operation-1", "worker-1")
    frozen_clock.current = original.lease.expires_at
    takeover = await lease_manager.acquire("operation-1", "worker-2")

    with pytest.raises(LeaseError) as renew_info:
        await lease_manager.renew(
            "operation-1",
            "worker-1",
            original.lease.fencing_token,
        )
    with pytest.raises(LeaseError) as validation_info:
        await lease_manager.assert_owned(
            "operation-1",
            "worker-1",
            original.lease.fencing_token,
        )

    assert takeover.lease.fencing_token == original.lease.fencing_token + 1
    assert renew_info.value.code == "lease_not_owned"
    assert validation_info.value.code == "lease_not_owned"


async def test_given_expiry_race_when_renewing_and_taking_over_then_old_owner_loses(
    lease_manager: LeaseManager,
    frozen_clock: FrozenClock,
) -> None:
    original = await lease_manager.acquire("operation-1", "worker-1")
    frozen_clock.current = original.lease.expires_at

    async def attempt_renewal() -> WorkerLease | LeaseError:
        try:
            return await lease_manager.renew(
                "operation-1",
                "worker-1",
                original.lease.fencing_token,
            )
        except LeaseError as exc:
            return exc

    renewal, takeover = await asyncio.gather(
        attempt_renewal(),
        lease_manager.acquire("operation-1", "worker-2"),
    )

    assert isinstance(renewal, LeaseError)
    assert renewal.code in {"lease_expired", "lease_not_owned"}
    assert takeover.outcome is LeaseAcquisitionOutcome.ACQUIRED
    assert takeover.lease.owner_id == "worker-2"
    assert takeover.lease.fencing_token == original.lease.fencing_token + 1


@pytest.mark.parametrize("fencing_token", [0, -1, True, 1.5, "1", None])
async def test_given_invalid_fence_when_renewing_then_validation_fails(
    lease_manager: LeaseManager,
    fencing_token,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        await lease_manager.renew("operation-1", "worker-1", fencing_token)
    with pytest.raises((TypeError, ValueError)):
        await lease_manager.assert_owned("operation-1", "worker-1", fencing_token)


async def test_given_cancelled_renewal_when_renewing_then_cancellation_propagates(
    frozen_clock: FrozenClock,
) -> None:
    class CancellingStore:
        async def get_lease(self, operation_id: str) -> None:
            return None

        async def acquire_lease(self, *args, **kwargs) -> LeaseAcquisition:
            raise AssertionError("acquisition should not be called")

        async def renew_lease(self, *args, **kwargs) -> WorkerLease:
            raise asyncio.CancelledError

        async def validate_lease(self, *args, **kwargs) -> WorkerLease:
            raise AssertionError("validation should not be called")

    manager = LeaseManager(
        CancellingStore(),  # type: ignore[arg-type]
        clock=frozen_clock,
    )

    with pytest.raises(asyncio.CancelledError):
        await manager.renew("operation-1", "worker-1", 1)


async def test_given_renewal_when_tracing_then_identifiers_are_excluded(
    frozen_clock: FrozenClock,
    in_memory_tracing,
) -> None:
    manager = LeaseManager(InMemoryLeaseStore(), clock=frozen_clock)
    acquired = await manager.acquire("private-operation-id", "private-worker-id")

    renewed = await manager.renew(
        "private-operation-id",
        "private-worker-id",
        acquired.lease.fencing_token,
    )
    await manager.assert_owned(
        "private-operation-id",
        "private-worker-id",
        renewed.fencing_token,
    )

    spans = {
        item.name: item
        for item in in_memory_tracing.get_finished_spans()
        if item.name in {"lease.renew", "lease.validate"}
    }
    assert spans["lease.renew"].attributes == {
        "lease.duration_seconds": 30.0,
        "lease.outcome": "renewed",
        "lease.revision": renewed.revision,
        "lease.fencing_token": renewed.fencing_token,
    }
    assert spans["lease.validate"].attributes == {
        "lease.outcome": "owned",
        "lease.revision": renewed.revision,
        "lease.fencing_token": renewed.fencing_token,
    }
    serialized = repr(spans)
    assert "private-operation-id" not in serialized
    assert "private-worker-id" not in serialized
