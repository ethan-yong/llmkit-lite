"""Process-local admission controls for rate and concurrency limits."""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from llmkit_lite.observability import set_span_error, trace_span

Clock = Callable[[], float]
TokenFactory = Callable[[], str]
TimeoutRunner = Callable[[float, Awaitable[None]], Awaitable[None]]

_ERROR_DETAILS = {
    "concurrency_lease_not_active": "concurrency lease is no longer active",
    "concurrency_token_conflict": "concurrency lease token is already active",
    "concurrency_wait_failed": "concurrency wait could not be completed",
}


def _normalize_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _require_positive_number(value: float, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a finite number greater than zero")
    return float(value)


def _require_non_negative_time(value: float, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    return float(value)


def _require_positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be an integer greater than zero")
    return value


def _require_non_negative_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


class AdmissionError(Exception):
    """Safe admission failure suitable for application error mapping."""

    def __init__(self, code: str) -> None:
        if code not in _ERROR_DETAILS:
            raise ValueError("unsupported admission error code")
        detail = _ERROR_DETAILS[code]
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class AdmissionKey:
    """One tenant, protected resource, and independently limited dimension."""

    tenant_id: str
    resource: str
    limit_name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            _normalize_identifier(self.tenant_id, "admission tenant ID"),
        )
        object.__setattr__(
            self,
            "resource",
            _normalize_identifier(self.resource, "admission resource"),
        )
        object.__setattr__(
            self,
            "limit_name",
            _normalize_identifier(self.limit_name, "admission limit name"),
        )


@dataclass(frozen=True, slots=True)
class TokenBucketPolicy:
    """Capacity and continuous refill rate for one token bucket limiter."""

    capacity: float
    refill_per_second: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capacity",
            _require_positive_number(self.capacity, "token bucket capacity"),
        )
        object.__setattr__(
            self,
            "refill_per_second",
            _require_positive_number(
                self.refill_per_second,
                "token bucket refill rate",
            ),
        )


class RateLimitOutcome(StrEnum):
    """Stable outcome of one weighted token-bucket acquisition."""

    ALLOWED = "allowed"
    EXCEEDED = "rate_limit_exceeded"
    COST_EXCEEDS_CAPACITY = "cost_exceeds_capacity"


@dataclass(frozen=True, slots=True)
class RateLimitDecision:
    """Result of attempting to consume capacity from a token bucket."""

    outcome: RateLimitOutcome
    remaining: float
    retry_after_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, RateLimitOutcome):
            raise TypeError("rate limit outcome must be a RateLimitOutcome")
        object.__setattr__(
            self,
            "remaining",
            _require_non_negative_time(self.remaining, "remaining capacity"),
        )
        if self.outcome is RateLimitOutcome.EXCEEDED:
            object.__setattr__(
                self,
                "retry_after_seconds",
                _require_positive_number(
                    self.retry_after_seconds,
                    "rate limit retry delay",
                ),
            )
        elif self.retry_after_seconds is not None:
            raise ValueError("only an exceeded rate limit has a retry delay")

    @property
    def allowed(self) -> bool:
        """Return whether the requested capacity was consumed."""

        return self.outcome is RateLimitOutcome.ALLOWED


class RateLimiter(Protocol):
    """Boundary for atomically consuming weighted rate-limit capacity."""

    async def acquire(
        self,
        key: AdmissionKey,
        *,
        cost: float = 1,
    ) -> RateLimitDecision:
        """Consume capacity or return a stable rejection decision."""


@dataclass(slots=True)
class _TokenBucket:
    tokens: float
    updated_at: float


class InMemoryTokenBucketLimiter:
    """Concurrency-safe token buckets for tests and local development."""

    def __init__(
        self,
        policy: TokenBucketPolicy,
        *,
        clock: Clock = time.monotonic,
    ) -> None:
        if not isinstance(policy, TokenBucketPolicy):
            raise TypeError("rate limit policy must be a TokenBucketPolicy")
        if not callable(clock):
            raise TypeError("rate limit clock must be callable")
        self.policy = policy
        self.clock = clock
        self._buckets: dict[AdmissionKey, _TokenBucket] = {}
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        key: AdmissionKey,
        *,
        cost: float = 1,
    ) -> RateLimitDecision:
        """Atomically consume weighted capacity from one independent bucket."""

        if not isinstance(key, AdmissionKey):
            raise TypeError("rate limit key must be an AdmissionKey")
        normalized_cost = _require_positive_number(cost, "rate limit cost")

        with trace_span(
            "admission.rate_limit",
            attributes={
                "admission.capacity": self.policy.capacity,
                "admission.refill_per_second": self.policy.refill_per_second,
                "admission.cost": normalized_cost,
            },
        ) as span:
            async with self._lock:
                now = _require_non_negative_time(
                    self.clock(),
                    "rate limit clock value",
                )
                bucket = self._buckets.get(key)
                if bucket is None:
                    bucket = _TokenBucket(self.policy.capacity, now)
                    self._buckets[key] = bucket
                elif now < bucket.updated_at:
                    raise ValueError("rate limit clock must not move backwards")

                elapsed = now - bucket.updated_at
                bucket.tokens = min(
                    self.policy.capacity,
                    bucket.tokens + elapsed * self.policy.refill_per_second,
                )
                bucket.updated_at = now

                if normalized_cost > self.policy.capacity:
                    decision = RateLimitDecision(
                        RateLimitOutcome.COST_EXCEEDS_CAPACITY,
                        bucket.tokens,
                    )
                elif normalized_cost <= bucket.tokens:
                    bucket.tokens -= normalized_cost
                    decision = RateLimitDecision(
                        RateLimitOutcome.ALLOWED,
                        bucket.tokens,
                    )
                else:
                    retry_after = (
                        normalized_cost - bucket.tokens
                    ) / self.policy.refill_per_second
                    decision = RateLimitDecision(
                        RateLimitOutcome.EXCEEDED,
                        bucket.tokens,
                        retry_after,
                    )

            if span is not None:
                span.set_attribute("admission.outcome", decision.outcome.value)
                span.set_attribute("admission.remaining", decision.remaining)
                if decision.retry_after_seconds is not None:
                    span.set_attribute(
                        "admission.retry_after_seconds",
                        decision.retry_after_seconds,
                    )
            return decision


class ConcurrencySaturationMode(StrEnum):
    """Behavior when every concurrency lease is occupied."""

    REJECT = "reject"
    WAIT = "wait"


@dataclass(frozen=True, slots=True)
class ConcurrencyPolicy:
    """Concurrency capacity, lease lifetime, and saturation behavior."""

    max_leases: int
    lease_ttl_seconds: float = 30
    saturation_mode: ConcurrencySaturationMode | str = ConcurrencySaturationMode.REJECT
    wait_timeout_seconds: float | None = None
    max_waiters: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_leases",
            _require_positive_int(self.max_leases, "maximum concurrency leases"),
        )
        object.__setattr__(
            self,
            "lease_ttl_seconds",
            _require_positive_number(
                self.lease_ttl_seconds,
                "concurrency lease lifetime",
            ),
        )
        try:
            mode = ConcurrencySaturationMode(self.saturation_mode)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"unsupported concurrency saturation mode: {self.saturation_mode!r}"
            ) from exc
        object.__setattr__(self, "saturation_mode", mode)
        waiters = _require_non_negative_int(
            self.max_waiters,
            "maximum concurrency waiters",
        )
        object.__setattr__(self, "max_waiters", waiters)

        if mode is ConcurrencySaturationMode.REJECT:
            if self.wait_timeout_seconds is not None or waiters != 0:
                raise ValueError(
                    "reject mode cannot configure a wait timeout or waiters"
                )
            return
        if self.wait_timeout_seconds is None:
            raise ValueError("wait mode requires a wait timeout")
        object.__setattr__(
            self,
            "wait_timeout_seconds",
            _require_positive_number(
                self.wait_timeout_seconds,
                "concurrency wait timeout",
            ),
        )
        if waiters < 1:
            raise ValueError("wait mode requires at least one waiter")


class ConcurrencyOutcome(StrEnum):
    """Stable outcome of one concurrency acquisition attempt."""

    ACQUIRED = "acquired"
    LIMIT_REACHED = "concurrency_limit_reached"
    QUEUE_FULL = "concurrency_queue_full"
    WAIT_TIMEOUT = "concurrency_wait_timeout"


@dataclass(frozen=True, slots=True)
class ConcurrencyLease:
    """Token-protected ownership of one concurrency slot."""

    key: AdmissionKey
    token: str
    acquired_at: float
    expires_at: float

    def __post_init__(self) -> None:
        if not isinstance(self.key, AdmissionKey):
            raise TypeError("concurrency lease key must be an AdmissionKey")
        object.__setattr__(
            self,
            "token",
            _normalize_identifier(self.token, "concurrency lease token"),
        )
        acquired_at = _require_non_negative_time(
            self.acquired_at,
            "concurrency lease acquisition time",
        )
        expires_at = _require_non_negative_time(
            self.expires_at,
            "concurrency lease expiration time",
        )
        if expires_at <= acquired_at:
            raise ValueError(
                "concurrency lease expiration must follow acquisition time"
            )
        object.__setattr__(self, "acquired_at", acquired_at)
        object.__setattr__(self, "expires_at", expires_at)

    def is_expired(self, at: float) -> bool:
        """Return whether the lease has expired at the supplied time."""

        return (
            _require_non_negative_time(
                at,
                "concurrency lease comparison time",
            )
            >= self.expires_at
        )


@dataclass(frozen=True, slots=True)
class ConcurrencyDecision:
    """Result of acquiring or being denied one concurrency lease."""

    outcome: ConcurrencyOutcome
    lease: ConcurrencyLease | None = None
    retry_after_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, ConcurrencyOutcome):
            raise TypeError("concurrency outcome must be a ConcurrencyOutcome")
        if self.outcome is ConcurrencyOutcome.ACQUIRED:
            if not isinstance(self.lease, ConcurrencyLease):
                raise TypeError("acquired concurrency decision requires a lease")
            if self.retry_after_seconds is not None:
                raise ValueError("acquired concurrency decision cannot retry")
            return
        if self.lease is not None:
            raise ValueError("rejected concurrency decision cannot contain a lease")
        if self.retry_after_seconds is not None:
            object.__setattr__(
                self,
                "retry_after_seconds",
                _require_positive_number(
                    self.retry_after_seconds,
                    "concurrency retry delay",
                ),
            )

    @property
    def allowed(self) -> bool:
        """Return whether a concurrency lease was acquired."""

        return self.outcome is ConcurrencyOutcome.ACQUIRED


class ConcurrencyLimiter(Protocol):
    """Boundary for acquiring, renewing, and releasing concurrency leases."""

    async def acquire(self, key: AdmissionKey) -> ConcurrencyDecision:
        """Acquire a lease or return a stable saturation decision."""

    async def renew(self, lease: ConcurrencyLease) -> ConcurrencyLease:
        """Extend one currently active concurrency lease."""

    async def release(self, lease: ConcurrencyLease) -> None:
        """Release one currently active concurrency lease."""


@dataclass(slots=True)
class _ConcurrencyWaiter:
    deadline: float
    event: asyncio.Event
    lease: ConcurrencyLease | None = None


class InMemoryConcurrencyLimiter:
    """FIFO concurrency leases for tests and local development."""

    def __init__(
        self,
        policy: ConcurrencyPolicy,
        *,
        clock: Clock = time.monotonic,
        token_factory: TokenFactory | None = None,
        _timeout_runner: TimeoutRunner | None = None,
    ) -> None:
        if not isinstance(policy, ConcurrencyPolicy):
            raise TypeError("concurrency policy must be a ConcurrencyPolicy")
        if not callable(clock):
            raise TypeError("concurrency clock must be callable")
        resolved_factory = token_factory or _new_token
        if not callable(resolved_factory):
            raise TypeError("concurrency token factory must be callable")
        resolved_timeout_runner = _timeout_runner or _run_with_timeout
        if not callable(resolved_timeout_runner):
            raise TypeError("concurrency timeout runner must be callable")

        self.policy = policy
        self.clock = clock
        self.token_factory = resolved_factory
        self._timeout_runner = resolved_timeout_runner
        self._active: dict[AdmissionKey, dict[str, ConcurrencyLease]] = {}
        self._waiters: dict[AdmissionKey, deque[_ConcurrencyWaiter]] = {}
        self._last_seen: dict[AdmissionKey, float] = {}
        self._issued_tokens: set[str] = set()
        self._lock = asyncio.Lock()

    async def acquire(self, key: AdmissionKey) -> ConcurrencyDecision:
        """Acquire immediately, reject, or wait in FIFO order within bounds."""

        if not isinstance(key, AdmissionKey):
            raise TypeError("concurrency key must be an AdmissionKey")

        with trace_span(
            "admission.concurrency.acquire",
            attributes={
                "admission.max_leases": self.policy.max_leases,
                "admission.lease_ttl_seconds": self.policy.lease_ttl_seconds,
                "admission.saturation_mode": self.policy.saturation_mode.value,
                "admission.max_waiters": self.policy.max_waiters,
            },
        ) as span:
            try:
                decision, waiter = await self._begin_acquire(key)
                if waiter is not None:
                    decision = await self._await_waiter(key, waiter)
            except asyncio.CancelledError:
                _record_failure(span, "concurrency_cancelled")
                raise
            except AdmissionError as exc:
                _record_failure(span, exc.code)
                raise

            _record_concurrency_decision(span, decision)
            return decision

    async def renew(self, lease: ConcurrencyLease) -> ConcurrencyLease:
        """Extend an active lease without changing its ownership token."""

        if not isinstance(lease, ConcurrencyLease):
            raise TypeError("lease must be a ConcurrencyLease")

        with trace_span(
            "admission.concurrency.renew",
            attributes={
                "admission.lease_ttl_seconds": self.policy.lease_ttl_seconds,
            },
        ) as span:
            async with self._lock:
                now = self._now_locked(lease.key)
                self._expire_locked(lease.key, now)
                self._service_waiters_locked(lease.key, now)
                active = self._active.get(lease.key, {})
                current = active.get(lease.token)
                if current is None:
                    error = AdmissionError("concurrency_lease_not_active")
                    _record_failure(span, error.code)
                    raise error
                renewed = ConcurrencyLease(
                    key=current.key,
                    token=current.token,
                    acquired_at=current.acquired_at,
                    expires_at=now + self.policy.lease_ttl_seconds,
                )
                active[current.token] = renewed

            if span is not None:
                span.set_attribute("admission.outcome", "renewed")
            return renewed

    async def release(self, lease: ConcurrencyLease) -> None:
        """Release an active lease and hand capacity to the oldest waiter."""

        if not isinstance(lease, ConcurrencyLease):
            raise TypeError("lease must be a ConcurrencyLease")

        with trace_span("admission.concurrency.release") as span:
            async with self._lock:
                now = self._now_locked(lease.key)
                self._expire_locked(lease.key, now)
                self._service_waiters_locked(lease.key, now)
                active = self._active.get(lease.key, {})
                current = active.get(lease.token)
                if current is None:
                    error = AdmissionError("concurrency_lease_not_active")
                    _record_failure(span, error.code)
                    raise error
                del active[lease.token]
                self._service_waiters_locked(lease.key, now)
                self._clean_empty_locked(lease.key)

            if span is not None:
                span.set_attribute("admission.outcome", "released")

    async def _begin_acquire(
        self,
        key: AdmissionKey,
    ) -> tuple[ConcurrencyDecision, _ConcurrencyWaiter | None]:
        async with self._lock:
            now = self._now_locked(key)
            self._expire_locked(key, now)
            self._service_waiters_locked(key, now)
            active = self._active.setdefault(key, {})
            queue = self._waiters.setdefault(key, deque())

            if len(active) < self.policy.max_leases and not queue:
                lease = self._issue_lease_locked(key, now)
                return ConcurrencyDecision(
                    ConcurrencyOutcome.ACQUIRED,
                    lease,
                ), None

            retry_after = self._retry_after_locked(key, now)
            if self.policy.saturation_mode is ConcurrencySaturationMode.REJECT:
                self._clean_empty_locked(key)
                return ConcurrencyDecision(
                    ConcurrencyOutcome.LIMIT_REACHED,
                    retry_after_seconds=retry_after,
                ), None

            if len(queue) >= self.policy.max_waiters:
                return ConcurrencyDecision(
                    ConcurrencyOutcome.QUEUE_FULL,
                    retry_after_seconds=retry_after,
                ), None

            assert self.policy.wait_timeout_seconds is not None
            waiter = _ConcurrencyWaiter(
                deadline=now + self.policy.wait_timeout_seconds,
                event=asyncio.Event(),
            )
            queue.append(waiter)
            return ConcurrencyDecision(
                ConcurrencyOutcome.LIMIT_REACHED,
                retry_after_seconds=retry_after,
            ), waiter

    async def _await_waiter(
        self,
        key: AdmissionKey,
        waiter: _ConcurrencyWaiter,
    ) -> ConcurrencyDecision:
        try:
            while True:
                async with self._lock:
                    now = self._now_locked(key)
                    self._expire_locked(key, now)
                    self._service_waiters_locked(key, now)
                    if waiter.lease is not None:
                        return ConcurrencyDecision(
                            ConcurrencyOutcome.ACQUIRED,
                            waiter.lease,
                        )
                    if now >= waiter.deadline:
                        self._remove_waiter_locked(key, waiter)
                        retry_after = self._retry_after_locked(key, now)
                        self._clean_empty_locked(key)
                        return ConcurrencyDecision(
                            ConcurrencyOutcome.WAIT_TIMEOUT,
                            retry_after_seconds=retry_after,
                        )

                    wait_seconds = waiter.deadline - now
                    active = self._active.get(key, {})
                    if active:
                        next_expiry = min(lease.expires_at for lease in active.values())
                        wait_seconds = min(wait_seconds, next_expiry - now)

                try:
                    await self._timeout_runner(wait_seconds, waiter.event.wait())
                except TimeoutError:
                    continue
        except asyncio.CancelledError:
            await self._abandon_waiter(key, waiter)
            raise
        except AdmissionError:
            await self._abandon_waiter(key, waiter)
            raise
        except Exception as exc:
            await self._abandon_waiter(key, waiter)
            raise AdmissionError("concurrency_wait_failed") from exc

    async def _abandon_waiter(
        self,
        key: AdmissionKey,
        waiter: _ConcurrencyWaiter,
    ) -> None:
        async with self._lock:
            now = self._last_seen.get(key, 0.0)
            try:
                now = self._now_locked(key)
            except (TypeError, ValueError):
                pass
            self._remove_waiter_locked(key, waiter)
            if waiter.lease is not None:
                active = self._active.get(key, {})
                active.pop(waiter.lease.token, None)
                waiter.lease = None
            self._expire_locked(key, now)
            self._service_waiters_locked(key, now)
            self._clean_empty_locked(key)

    def _now_locked(self, key: AdmissionKey) -> float:
        now = _require_non_negative_time(
            self.clock(),
            "concurrency clock value",
        )
        previous = self._last_seen.get(key)
        if previous is not None and now < previous:
            raise ValueError("concurrency clock must not move backwards")
        self._last_seen[key] = now
        return now

    def _issue_lease_locked(
        self,
        key: AdmissionKey,
        now: float,
    ) -> ConcurrencyLease:
        token = _normalize_identifier(
            self.token_factory(),
            "concurrency lease token",
        )
        active = self._active.setdefault(key, {})
        if token in self._issued_tokens:
            raise AdmissionError("concurrency_token_conflict")
        self._issued_tokens.add(token)
        lease = ConcurrencyLease(
            key=key,
            token=token,
            acquired_at=now,
            expires_at=now + self.policy.lease_ttl_seconds,
        )
        active[token] = lease
        return lease

    def _expire_locked(self, key: AdmissionKey, now: float) -> None:
        active = self._active.get(key, {})
        expired_tokens = [
            token for token, lease in active.items() if lease.is_expired(now)
        ]
        for token in expired_tokens:
            del active[token]

    def _service_waiters_locked(self, key: AdmissionKey, now: float) -> None:
        active = self._active.setdefault(key, {})
        queue = self._waiters.setdefault(key, deque())
        while len(active) < self.policy.max_leases and queue:
            waiter = queue[0]
            if waiter.deadline <= now:
                queue.popleft()
                waiter.event.set()
                continue
            lease = self._issue_lease_locked(key, now)
            queue.popleft()
            waiter.lease = lease
            waiter.event.set()

    def _remove_waiter_locked(
        self,
        key: AdmissionKey,
        waiter: _ConcurrencyWaiter,
    ) -> None:
        queue = self._waiters.get(key)
        if queue is None:
            return
        try:
            queue.remove(waiter)
        except ValueError:
            pass

    def _retry_after_locked(
        self,
        key: AdmissionKey,
        now: float,
    ) -> float | None:
        active = self._active.get(key, {})
        if not active:
            return None
        return min(lease.expires_at for lease in active.values()) - now

    def _clean_empty_locked(self, key: AdmissionKey) -> None:
        if not self._active.get(key):
            self._active.pop(key, None)
        if not self._waiters.get(key):
            self._waiters.pop(key, None)


async def _run_with_timeout(
    timeout_seconds: float,
    operation: Awaitable[None],
) -> None:
    async with asyncio.timeout(timeout_seconds):
        await operation


def _new_token() -> str:
    return uuid.uuid4().hex


def _record_concurrency_decision(span, decision: ConcurrencyDecision) -> None:
    if span is None:
        return
    span.set_attribute("admission.outcome", decision.outcome.value)
    if decision.retry_after_seconds is not None:
        span.set_attribute(
            "admission.retry_after_seconds",
            decision.retry_after_seconds,
        )


def _record_failure(span, code: str) -> None:
    if span is not None:
        span.set_attribute("admission.outcome", "failed")
        span.set_attribute("error.type", code)
    set_span_error(span, code)
