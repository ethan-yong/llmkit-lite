from __future__ import annotations

import asyncio
from collections.abc import Awaitable

import pytest
from opentelemetry.trace import StatusCode

from llmkit_lite.admission import (
    AdmissionError,
    AdmissionKey,
    ConcurrencyOutcome,
    ConcurrencyPolicy,
    ConcurrencySaturationMode,
    InMemoryConcurrencyLimiter,
    InMemoryTokenBucketLimiter,
    RateLimitOutcome,
    TokenBucketPolicy,
)


class FrozenClock:
    def __init__(self, current: float = 0) -> None:
        self.current = current

    def __call__(self) -> float:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += seconds


class TokenSequence:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self) -> str:
        self.value += 1
        return f"lease-{self.value}"


class AdvancingTimeoutRunner:
    def __init__(self, clock: FrozenClock) -> None:
        self.clock = clock
        self.calls: list[float] = []

    async def __call__(
        self,
        seconds: float,
        operation: Awaitable[None],
    ) -> None:
        self.calls.append(seconds)
        close = getattr(operation, "close", None)
        if callable(close):
            close()
        self.clock.advance(seconds)
        raise TimeoutError


@pytest.fixture
def admission_key() -> AdmissionKey:
    return AdmissionKey("tenant-1", "model-large", "requests")


@pytest.fixture
def frozen_clock() -> FrozenClock:
    return FrozenClock()


def _reject_policy(
    *,
    max_leases: int = 1,
    lease_ttl_seconds: float = 30,
) -> ConcurrencyPolicy:
    return ConcurrencyPolicy(
        max_leases=max_leases,
        lease_ttl_seconds=lease_ttl_seconds,
    )


def _wait_policy(
    *,
    max_leases: int = 1,
    lease_ttl_seconds: float = 30,
    wait_timeout_seconds: float = 10,
    max_waiters: int = 3,
) -> ConcurrencyPolicy:
    return ConcurrencyPolicy(
        max_leases=max_leases,
        lease_ttl_seconds=lease_ttl_seconds,
        saturation_mode=ConcurrencySaturationMode.WAIT,
        wait_timeout_seconds=wait_timeout_seconds,
        max_waiters=max_waiters,
    )


def _concurrency_limiter(
    policy: ConcurrencyPolicy,
    clock: FrozenClock,
    *,
    timeout_runner=None,
) -> InMemoryConcurrencyLimiter:
    return InMemoryConcurrencyLimiter(
        policy,
        clock=clock,
        token_factory=TokenSequence(),
        _timeout_runner=timeout_runner,
    )


async def test_given_fresh_bucket_when_acquiring_weight_then_capacity_is_consumed(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    limiter = InMemoryTokenBucketLimiter(
        TokenBucketPolicy(capacity=10, refill_per_second=2),
        clock=frozen_clock,
    )

    decision = await limiter.acquire(admission_key, cost=4)

    assert decision.allowed is True
    assert decision.outcome is RateLimitOutcome.ALLOWED
    assert decision.remaining == 6
    assert decision.retry_after_seconds is None


async def test_given_empty_bucket_when_requesting_weight_then_retry_is_calculated(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    limiter = InMemoryTokenBucketLimiter(
        TokenBucketPolicy(capacity=10, refill_per_second=2),
        clock=frozen_clock,
    )
    await limiter.acquire(admission_key, cost=8)

    decision = await limiter.acquire(admission_key, cost=6)

    assert decision.allowed is False
    assert decision.outcome is RateLimitOutcome.EXCEEDED
    assert decision.remaining == 2
    assert decision.retry_after_seconds == 2


async def test_given_elapsed_time_when_acquiring_then_bucket_refills_and_caps(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    limiter = InMemoryTokenBucketLimiter(
        TokenBucketPolicy(capacity=10, refill_per_second=2),
        clock=frozen_clock,
    )
    await limiter.acquire(admission_key, cost=8)
    frozen_clock.advance(1)

    after_refill = await limiter.acquire(admission_key, cost=4)
    frozen_clock.advance(100)
    after_long_idle = await limiter.acquire(admission_key, cost=1)

    assert after_refill.remaining == 0
    assert after_long_idle.remaining == 9


async def test_given_cost_above_capacity_when_acquiring_then_denial_is_permanent(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    limiter = InMemoryTokenBucketLimiter(
        TokenBucketPolicy(capacity=10, refill_per_second=1),
        clock=frozen_clock,
    )

    rejected = await limiter.acquire(admission_key, cost=11)
    allowed = await limiter.acquire(admission_key, cost=10)

    assert rejected.outcome is RateLimitOutcome.COST_EXCEEDS_CAPACITY
    assert rejected.retry_after_seconds is None
    assert rejected.remaining == 10
    assert allowed.allowed is True
    assert allowed.remaining == 0


async def test_given_distinct_keys_when_acquiring_then_buckets_are_isolated(
    frozen_clock: FrozenClock,
) -> None:
    limiter = InMemoryTokenBucketLimiter(
        TokenBucketPolicy(capacity=2, refill_per_second=1),
        clock=frozen_clock,
    )
    tenant_one = AdmissionKey("tenant-1", "model", "requests")
    tenant_two = AdmissionKey("tenant-2", "model", "requests")
    token_limit = AdmissionKey("tenant-1", "model", "tokens")

    decisions = await asyncio.gather(
        limiter.acquire(tenant_one, cost=2),
        limiter.acquire(tenant_two, cost=2),
        limiter.acquire(token_limit, cost=2),
    )

    assert all(decision.allowed for decision in decisions)


async def test_given_one_bucket_when_acquiring_concurrently_then_capacity_is_atomic(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    limiter = InMemoryTokenBucketLimiter(
        TokenBucketPolicy(capacity=5, refill_per_second=1),
        clock=frozen_clock,
    )

    decisions = await asyncio.gather(
        *(limiter.acquire(admission_key) for _ in range(20))
    )

    assert sum(decision.allowed for decision in decisions) == 5
    assert (
        sum(decision.outcome is RateLimitOutcome.EXCEEDED for decision in decisions)
        == 15
    )


@pytest.mark.parametrize(
    ("capacity", "refill_rate"),
    [
        (0, 1),
        (-1, 1),
        (True, 1),
        (float("inf"), 1),
        (1, 0),
        (1, -1),
        (1, float("nan")),
    ],
)
def test_given_invalid_values_when_creating_bucket_policy_then_validation_fails(
    capacity,
    refill_rate,
) -> None:
    with pytest.raises(ValueError, match="finite number greater than zero"):
        TokenBucketPolicy(capacity, refill_rate)


@pytest.mark.parametrize("cost", [0, -1, True, float("inf"), float("nan")])
async def test_given_invalid_cost_when_acquiring_then_validation_fails(
    admission_key: AdmissionKey,
    cost,
) -> None:
    limiter = InMemoryTokenBucketLimiter(TokenBucketPolicy(1, 1))

    with pytest.raises(ValueError, match="finite number greater than zero"):
        await limiter.acquire(admission_key, cost=cost)


@pytest.mark.parametrize("value", ["", "   ", 123, None])
def test_given_invalid_identifier_when_creating_key_then_validation_fails(
    value,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        AdmissionKey(value, "model", "requests")
    with pytest.raises((TypeError, ValueError)):
        AdmissionKey("tenant", value, "requests")
    with pytest.raises((TypeError, ValueError)):
        AdmissionKey("tenant", "model", value)


async def test_given_invalid_or_regressed_clock_when_acquiring_then_rejected(
    admission_key: AdmissionKey,
) -> None:
    clock = FrozenClock(2)
    limiter = InMemoryTokenBucketLimiter(TokenBucketPolicy(1, 1), clock=clock)
    await limiter.acquire(admission_key)
    clock.current = 1

    with pytest.raises(ValueError, match="must not move backwards"):
        await limiter.acquire(admission_key)

    invalid = InMemoryTokenBucketLimiter(
        TokenBucketPolicy(1, 1),
        clock=lambda: float("nan"),
    )
    with pytest.raises(ValueError, match="finite non-negative"):
        await invalid.acquire(admission_key)


async def test_given_capacity_when_acquiring_leases_then_excess_is_rejected(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    limiter = _concurrency_limiter(_reject_policy(max_leases=2), frozen_clock)

    first = await limiter.acquire(admission_key)
    second = await limiter.acquire(admission_key)
    rejected = await limiter.acquire(admission_key)

    assert first.allowed is True
    assert second.allowed is True
    assert rejected.outcome is ConcurrencyOutcome.LIMIT_REACHED
    assert rejected.retry_after_seconds == 30


async def test_given_concurrent_acquisitions_then_active_leases_never_exceed_limit(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    limiter = _concurrency_limiter(
        _reject_policy(max_leases=3),
        frozen_clock,
    )

    decisions = await asyncio.gather(
        *(limiter.acquire(admission_key) for _ in range(20))
    )

    assert sum(decision.allowed for decision in decisions) == 3
    assert (
        sum(
            decision.outcome is ConcurrencyOutcome.LIMIT_REACHED
            for decision in decisions
        )
        == 17
    )


async def test_given_distinct_keys_when_acquiring_leases_then_limits_are_isolated(
    frozen_clock: FrozenClock,
) -> None:
    limiter = _concurrency_limiter(_reject_policy(), frozen_clock)
    first_key = AdmissionKey("tenant-1", "model", "concurrency")
    second_key = AdmissionKey("tenant-2", "model", "concurrency")

    first, second = await asyncio.gather(
        limiter.acquire(first_key),
        limiter.acquire(second_key),
    )

    assert first.allowed is True
    assert second.allowed is True


async def test_given_fifo_waiters_when_releasing_then_oldest_acquires_first(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    limiter = _concurrency_limiter(_wait_policy(), frozen_clock)
    initial = await limiter.acquire(admission_key)
    assert initial.lease is not None
    first_waiter = asyncio.create_task(limiter.acquire(admission_key))
    await asyncio.sleep(0)
    second_waiter = asyncio.create_task(limiter.acquire(admission_key))
    await asyncio.sleep(0)

    await limiter.release(initial.lease)
    first = await first_waiter

    assert first.allowed is True
    assert second_waiter.done() is False
    assert first.lease is not None
    await limiter.release(first.lease)
    second = await second_waiter
    assert second.allowed is True


async def test_given_full_wait_queue_when_acquiring_then_request_is_rejected(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    limiter = _concurrency_limiter(
        _wait_policy(max_waiters=1),
        frozen_clock,
    )
    initial = await limiter.acquire(admission_key)
    waiter = asyncio.create_task(limiter.acquire(admission_key))
    await asyncio.sleep(0)

    rejected = await limiter.acquire(admission_key)

    assert rejected.outcome is ConcurrencyOutcome.QUEUE_FULL
    assert rejected.retry_after_seconds == 30
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert initial.lease is not None
    await limiter.release(initial.lease)


async def test_given_wait_timeout_when_capacity_stays_full_then_waiter_is_removed(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    timeout_runner = AdvancingTimeoutRunner(frozen_clock)
    limiter = _concurrency_limiter(
        _wait_policy(wait_timeout_seconds=5),
        frozen_clock,
        timeout_runner=timeout_runner,
    )
    initial = await limiter.acquire(admission_key)

    timed_out = await limiter.acquire(admission_key)

    assert timed_out.outcome is ConcurrencyOutcome.WAIT_TIMEOUT
    assert timed_out.retry_after_seconds == 25
    assert timeout_runner.calls == [5]
    assert initial.lease is not None
    await limiter.release(initial.lease)
    replacement = await limiter.acquire(admission_key)
    assert replacement.allowed is True


async def test_given_lease_expiry_while_waiting_then_oldest_waiter_acquires(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    timeout_runner = AdvancingTimeoutRunner(frozen_clock)
    limiter = _concurrency_limiter(
        _wait_policy(lease_ttl_seconds=5, wait_timeout_seconds=10),
        frozen_clock,
        timeout_runner=timeout_runner,
    )
    original = await limiter.acquire(admission_key)

    replacement = await limiter.acquire(admission_key)

    assert original.lease is not None
    assert replacement.allowed is True
    assert replacement.lease is not None
    assert replacement.lease.token != original.lease.token
    assert replacement.lease.acquired_at == 5
    assert timeout_runner.calls == [5]


async def test_given_cancelled_waiter_when_releasing_then_next_waiter_acquires(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    limiter = _concurrency_limiter(_wait_policy(), frozen_clock)
    initial = await limiter.acquire(admission_key)
    cancelled_waiter = asyncio.create_task(limiter.acquire(admission_key))
    await asyncio.sleep(0)
    next_waiter = asyncio.create_task(limiter.acquire(admission_key))
    await asyncio.sleep(0)

    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter
    assert initial.lease is not None
    await limiter.release(initial.lease)

    acquired = await next_waiter
    assert acquired.allowed is True


async def test_given_assigned_waiter_when_cancelled_then_lease_is_not_leaked(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    limiter = _concurrency_limiter(_wait_policy(), frozen_clock)
    initial = await limiter.acquire(admission_key)
    waiter = asyncio.create_task(limiter.acquire(admission_key))
    await asyncio.sleep(0)
    assert initial.lease is not None

    await limiter.release(initial.lease)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    replacement = await limiter.acquire(admission_key)
    assert replacement.allowed is True


async def test_given_expired_waiter_when_releasing_then_next_waiter_acquires(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    limiter = _concurrency_limiter(
        _wait_policy(wait_timeout_seconds=5),
        frozen_clock,
    )
    initial = await limiter.acquire(admission_key)
    first_waiter = asyncio.create_task(limiter.acquire(admission_key))
    await asyncio.sleep(0)
    frozen_clock.advance(1)
    second_waiter = asyncio.create_task(limiter.acquire(admission_key))
    await asyncio.sleep(0)
    frozen_clock.advance(4)

    assert initial.lease is not None
    await limiter.release(initial.lease)
    first, second = await asyncio.gather(first_waiter, second_waiter)

    assert first.outcome is ConcurrencyOutcome.WAIT_TIMEOUT
    assert second.allowed is True


async def test_given_active_lease_when_renewing_then_expiration_is_extended(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    limiter = _concurrency_limiter(
        _reject_policy(lease_ttl_seconds=10),
        frozen_clock,
    )
    acquired = await limiter.acquire(admission_key)
    assert acquired.lease is not None
    frozen_clock.advance(5)

    renewed = await limiter.renew(acquired.lease)

    assert renewed.token == acquired.lease.token
    assert renewed.acquired_at == acquired.lease.acquired_at
    assert renewed.expires_at == 15


async def test_given_expired_replaced_lease_when_releasing_then_replacement_is_safe(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    limiter = _concurrency_limiter(
        _reject_policy(lease_ttl_seconds=5),
        frozen_clock,
    )
    original = await limiter.acquire(admission_key)
    assert original.lease is not None
    frozen_clock.advance(5)
    replacement = await limiter.acquire(admission_key)
    assert replacement.lease is not None

    with pytest.raises(AdmissionError) as exc_info:
        await limiter.release(original.lease)

    assert exc_info.value.code == "concurrency_lease_not_active"
    still_full = await limiter.acquire(admission_key)
    assert still_full.outcome is ConcurrencyOutcome.LIMIT_REACHED
    await limiter.release(replacement.lease)


async def test_given_expired_lease_when_renewing_then_safe_error_is_raised(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    limiter = _concurrency_limiter(
        _reject_policy(lease_ttl_seconds=5),
        frozen_clock,
    )
    acquired = await limiter.acquire(admission_key)
    assert acquired.lease is not None
    frozen_clock.advance(5)

    with pytest.raises(AdmissionError) as exc_info:
        await limiter.renew(acquired.lease)

    assert exc_info.value.code == "concurrency_lease_not_active"
    assert "tenant-1" not in str(exc_info.value)


async def test_given_regressed_concurrency_clock_when_acquiring_then_rejected(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    frozen_clock.current = 2
    limiter = _concurrency_limiter(_reject_policy(), frozen_clock)
    await limiter.acquire(admission_key)
    frozen_clock.current = 1

    with pytest.raises(ValueError, match="must not move backwards"):
        await limiter.acquire(admission_key)

    invalid = InMemoryConcurrencyLimiter(
        _reject_policy(),
        clock=lambda: float("nan"),
    )
    with pytest.raises(ValueError, match="finite non-negative"):
        await invalid.acquire(admission_key)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ConcurrencyPolicy(max_leases=0),
        lambda: ConcurrencyPolicy(max_leases=True),
        lambda: ConcurrencyPolicy(max_leases=1, lease_ttl_seconds=0),
        lambda: ConcurrencyPolicy(max_leases=1, saturation_mode="unknown"),
        lambda: ConcurrencyPolicy(
            max_leases=1,
            saturation_mode="reject",
            wait_timeout_seconds=1,
        ),
        lambda: ConcurrencyPolicy(
            max_leases=1,
            saturation_mode="wait",
            max_waiters=1,
        ),
        lambda: ConcurrencyPolicy(
            max_leases=1,
            saturation_mode="wait",
            wait_timeout_seconds=1,
            max_waiters=0,
        ),
    ],
)
def test_given_invalid_values_when_creating_concurrency_policy_then_rejected(
    factory,
) -> None:
    with pytest.raises(ValueError):
        factory()


async def test_given_duplicate_token_when_acquiring_then_safe_error_is_raised(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    limiter = InMemoryConcurrencyLimiter(
        _reject_policy(max_leases=2),
        clock=frozen_clock,
        token_factory=lambda: "duplicate",
    )
    await limiter.acquire(admission_key)

    with pytest.raises(AdmissionError) as exc_info:
        await limiter.acquire(admission_key)

    assert exc_info.value.code == "concurrency_token_conflict"


async def test_given_reused_expired_token_when_acquiring_then_stale_release_is_safe(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    limiter = InMemoryConcurrencyLimiter(
        _reject_policy(lease_ttl_seconds=5),
        clock=frozen_clock,
        token_factory=lambda: "duplicate",
    )
    original = await limiter.acquire(admission_key)
    assert original.lease is not None
    frozen_clock.advance(5)

    with pytest.raises(AdmissionError) as exc_info:
        await limiter.acquire(admission_key)

    assert exc_info.value.code == "concurrency_token_conflict"
    with pytest.raises(AdmissionError) as release_info:
        await limiter.release(original.lease)
    assert release_info.value.code == "concurrency_lease_not_active"


async def test_given_waiter_failure_when_waiting_then_safe_error_and_cleanup(
    admission_key: AdmissionKey,
    frozen_clock: FrozenClock,
) -> None:
    async def broken_timeout_runner(
        seconds: float,
        operation: Awaitable[None],
    ) -> None:
        close = getattr(operation, "close", None)
        if callable(close):
            close()
        raise RuntimeError("private waiter failure")

    limiter = _concurrency_limiter(
        _wait_policy(),
        frozen_clock,
        timeout_runner=broken_timeout_runner,
    )
    initial = await limiter.acquire(admission_key)

    with pytest.raises(AdmissionError) as exc_info:
        await limiter.acquire(admission_key)

    assert exc_info.value.code == "concurrency_wait_failed"
    assert "private" not in str(exc_info.value)
    assert initial.lease is not None
    await limiter.release(initial.lease)
    replacement = await limiter.acquire(admission_key)
    assert replacement.allowed is True


async def test_given_private_keys_when_tracing_then_identifiers_are_excluded(
    frozen_clock: FrozenClock,
    in_memory_tracing,
) -> None:
    private_key = AdmissionKey(
        "private-tenant",
        "private-model",
        "private-limit",
    )
    rate_limiter = InMemoryTokenBucketLimiter(
        TokenBucketPolicy(2, 1),
        clock=frozen_clock,
    )
    concurrency_limiter = InMemoryConcurrencyLimiter(
        _reject_policy(),
        clock=frozen_clock,
        token_factory=lambda: "private-token",
    )

    await rate_limiter.acquire(private_key)
    acquired = await concurrency_limiter.acquire(private_key)
    assert acquired.lease is not None
    await concurrency_limiter.release(acquired.lease)

    spans = [
        span
        for span in in_memory_tracing.get_finished_spans()
        if span.name.startswith("admission.")
    ]
    assert len(spans) == 3
    assert all(span.status.status_code is StatusCode.UNSET for span in spans)
    serialized = repr(spans)
    assert "private-tenant" not in serialized
    assert "private-model" not in serialized
    assert "private-limit" not in serialized
    assert "private-token" not in serialized
