"""Deterministic routing and resilience for LLM provider adapters."""

from __future__ import annotations

import asyncio
import logging
import random as random_module
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import httpx

from llmkit_lite.admission import (
    AdmissionKey,
    ConcurrencyLease,
    ConcurrencyLimiter,
    RateLimiter,
)
from llmkit_lite.budgets import BudgetKey, BudgetLedger, BudgetReservation
from llmkit_lite.llm import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    LlmEndpointConfig,
    LlmGatewayError,
    LlmProviderAdapter,
    OpenAICompatibleAdapter,
    TokenUsage,
    missing_capabilities,
)
from llmkit_lite.observability import set_span_error, trace_span

logger = logging.getLogger("llmkit_lite.routing")

RoutePredicate = Callable[[ChatCompletionRequest], bool]
Clock = Callable[[], float]
Sleep = Callable[[float], Awaitable[None]]
Random = Callable[[], float]
CircuitStatus = Literal["closed", "open", "half_open"]

_ADMISSION_ERROR_DETAILS = {
    "rate_limit_exceeded": "request rate limit was exceeded",
    "cost_exceeds_capacity": "request cost exceeds rate-limit capacity",
    "concurrency_limit_reached": "concurrency limit was reached",
    "concurrency_queue_full": "concurrency wait queue is full",
    "concurrency_wait_timeout": "concurrency wait timed out",
    "budget_token_limit": "token budget was exceeded",
    "budget_cost_limit": "cost budget was exceeded",
    "concurrency_lease_renewal_failed": "concurrency lease renewal failed",
    "admission_cleanup_failed": "admission cleanup failed",
}


def _normalized_name(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _require_positive_int(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be an integer greater than zero")


def _require_non_negative(value: float, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _optional_normalized_name(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    return _normalized_name(value, field_name)


class LlmAdmissionError(Exception):
    """Safe admission denial that is never treated as a provider failure."""

    def __init__(
        self,
        code: str,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        try:
            detail = _ADMISSION_ERROR_DETAILS[code]
        except KeyError:
            raise ValueError("unsupported LLM admission error code") from None
        if retry_after_seconds is not None:
            if (
                isinstance(retry_after_seconds, bool)
                or not isinstance(retry_after_seconds, (int, float))
                or retry_after_seconds <= 0
            ):
                raise ValueError("admission retry delay must be greater than zero")
            retry_after_seconds = float(retry_after_seconds)
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True, slots=True)
class LlmAdmissionContext:
    """Caller-supplied tenant identity and accounting inputs."""

    tenant_id: str
    estimated_input_tokens: int
    token_period_id: str | None = None
    cost_period_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tenant_id",
            _normalized_name(self.tenant_id, "admission tenant ID"),
        )
        if (
            isinstance(self.estimated_input_tokens, bool)
            or not isinstance(self.estimated_input_tokens, int)
            or self.estimated_input_tokens < 0
        ):
            raise ValueError(
                "estimated input tokens must be a non-negative integer"
            )
        object.__setattr__(
            self,
            "token_period_id",
            _optional_normalized_name(
                self.token_period_id,
                "admission token period ID",
            ),
        )
        object.__setattr__(
            self,
            "cost_period_id",
            _optional_normalized_name(
                self.cost_period_id,
                "admission cost period ID",
            ),
        )


@dataclass(frozen=True, slots=True)
class LlmAdmissionControls:
    """Optional admission dependencies applied to each provider attempt."""

    request_rate_limiter: RateLimiter | None = None
    token_rate_limiter: RateLimiter | None = None
    concurrency_limiter: ConcurrencyLimiter | None = None
    budget_ledger: BudgetLedger | None = None

    def __post_init__(self) -> None:
        requirements = {
            "request rate limiter": (self.request_rate_limiter, ("acquire",)),
            "token rate limiter": (self.token_rate_limiter, ("acquire",)),
            "concurrency limiter": (
                self.concurrency_limiter,
                ("acquire", "renew", "release"),
            ),
            "budget ledger": (
                self.budget_ledger,
                ("reserve", "mark_attempted", "reconcile", "cancel"),
            ),
        }
        for field_name, (value, methods) in requirements.items():
            if value is not None and not all(
                callable(getattr(value, method, None)) for method in methods
            ):
                raise TypeError(f"{field_name} does not implement its protocol")

    @property
    def enabled(self) -> bool:
        """Return whether at least one admission control is configured."""

        return any(
            control is not None
            for control in (
                self.request_rate_limiter,
                self.token_rate_limiter,
                self.concurrency_limiter,
                self.budget_ledger,
            )
        )


@dataclass(slots=True)
class _AttemptAdmission:
    lease: ConcurrencyLease | None = None
    reservation: BudgetReservation | None = None
    budget_attempted: bool = False


@dataclass(frozen=True)
class LlmResiliencePolicy:
    """Retry and circuit-breaker settings shared by a router's routes."""

    max_attempts_per_route: int = 2
    initial_backoff_seconds: float = 0.25
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 2.0
    jitter_ratio: float = 0.2
    circuit_failure_threshold: int = 3
    circuit_recovery_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        _require_positive_int(
            self.max_attempts_per_route, "max_attempts_per_route"
        )
        _require_non_negative(
            self.initial_backoff_seconds, "initial_backoff_seconds"
        )
        if (
            isinstance(self.backoff_multiplier, bool)
            or not isinstance(self.backoff_multiplier, (int, float))
            or self.backoff_multiplier < 1
        ):
            raise ValueError("backoff_multiplier must be at least one")
        _require_non_negative(self.max_backoff_seconds, "max_backoff_seconds")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError(
                "max_backoff_seconds must be at least initial_backoff_seconds"
            )
        if (
            isinstance(self.jitter_ratio, bool)
            or not isinstance(self.jitter_ratio, (int, float))
            or not 0 <= self.jitter_ratio <= 1
        ):
            raise ValueError("jitter_ratio must be between zero and one")
        _require_positive_int(
            self.circuit_failure_threshold, "circuit_failure_threshold"
        )
        if (
            isinstance(self.circuit_recovery_timeout_seconds, bool)
            or not isinstance(self.circuit_recovery_timeout_seconds, (int, float))
            or self.circuit_recovery_timeout_seconds <= 0
        ):
            raise ValueError(
                "circuit_recovery_timeout_seconds must be greater than zero"
            )


@dataclass(frozen=True)
class LlmRoute:
    """One named provider endpoint and its ordered fallback routes."""

    name: str
    endpoint: LlmEndpointConfig
    adapter: LlmProviderAdapter = field(default_factory=OpenAICompatibleAdapter)
    fallback_routes: Sequence[str] = ()

    def __post_init__(self) -> None:
        normalized_name = _normalized_name(self.name, "route name")
        normalized_fallbacks = tuple(
            _normalized_name(name, "fallback route name")
            for name in self.fallback_routes
        )
        if normalized_name in normalized_fallbacks:
            raise ValueError(f"route {normalized_name!r} cannot fall back to itself")
        if len(set(normalized_fallbacks)) != len(normalized_fallbacks):
            raise ValueError(
                f"route {normalized_name!r} has duplicate fallback routes"
            )
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "fallback_routes", normalized_fallbacks)


@dataclass(frozen=True)
class LlmRouteRule:
    """An ordered predicate that selects a named route."""

    name: str
    route_name: str
    predicate: RoutePredicate

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalized_name(self.name, "rule name"))
        object.__setattr__(
            self,
            "route_name",
            _normalized_name(self.route_name, "rule route name"),
        )
        if not callable(self.predicate):
            raise TypeError("rule predicate must be callable")


@dataclass
class _CircuitBreaker:
    status: CircuitStatus = "closed"
    consecutive_failures: int = 0
    opened_at: float | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def acquire(
        self,
        *,
        now: float,
        recovery_timeout_seconds: float,
    ) -> tuple[bool, bool]:
        """Return whether a call is allowed and whether it is a half-open probe."""

        async with self.lock:
            if self.status == "closed":
                return True, False
            if self.status == "open":
                assert self.opened_at is not None
                if now - self.opened_at < recovery_timeout_seconds:
                    return False, False
                self.status = "half_open"
                return True, True
            return False, False

    async def record_success(self) -> bool:
        """Close the circuit and report whether its state changed."""

        async with self.lock:
            changed = self.status != "closed"
            self.status = "closed"
            self.consecutive_failures = 0
            self.opened_at = None
            return changed

    async def record_failure(self, *, now: float, threshold: int) -> bool:
        """Record one exhausted route call and report whether the circuit opened."""

        async with self.lock:
            if self.status == "open":
                return False
            if self.status == "half_open":
                self.status = "open"
                self.consecutive_failures = threshold
                self.opened_at = now
                return True

            self.consecutive_failures += 1
            if self.consecutive_failures < threshold:
                return False
            self.status = "open"
            self.opened_at = now
            return True

    async def abandon_probe(self, *, now: float) -> None:
        """Release a probe that ended without a gateway success or failure."""

        async with self.lock:
            if self.status == "half_open":
                self.status = "open"
                self.opened_at = now


class LlmRouter:
    """Select an LLM route and execute it with retries and fallback."""

    def __init__(
        self,
        routes: Sequence[LlmRoute],
        *,
        default_route: str,
        rules: Sequence[LlmRouteRule] = (),
        resilience_policy: LlmResiliencePolicy | None = None,
        admission_controls: LlmAdmissionControls | None = None,
        _clock: Clock = time.monotonic,
        _sleep: Sleep = asyncio.sleep,
        _lease_sleep: Sleep = asyncio.sleep,
        _random: Random = random_module.random,
    ) -> None:
        routes_by_name: dict[str, LlmRoute] = {}
        for route in routes:
            if route.name in routes_by_name:
                raise ValueError(f"duplicate route name: {route.name!r}")
            routes_by_name[route.name] = route
        if not routes_by_name:
            raise ValueError("at least one route is required")

        normalized_default = _normalized_name(default_route, "default route")
        if normalized_default not in routes_by_name:
            raise ValueError(f"default route does not exist: {normalized_default!r}")

        normalized_rules = tuple(rules)
        rule_names: set[str] = set()
        for rule in normalized_rules:
            if rule.name in rule_names:
                raise ValueError(f"duplicate rule name: {rule.name!r}")
            rule_names.add(rule.name)
            if rule.route_name not in routes_by_name:
                raise ValueError(
                    f"rule {rule.name!r} targets unknown route: {rule.route_name!r}"
                )

        for route in routes_by_name.values():
            unknown_fallbacks = [
                name for name in route.fallback_routes if name not in routes_by_name
            ]
            if unknown_fallbacks:
                raise ValueError(
                    f"route {route.name!r} targets unknown fallback route: "
                    f"{unknown_fallbacks[0]!r}"
                )

        self._routes = MappingProxyType(routes_by_name)
        self._default_route = normalized_default
        self._rules = normalized_rules
        self._policy = resilience_policy or LlmResiliencePolicy()
        if admission_controls is not None and not isinstance(
            admission_controls,
            LlmAdmissionControls,
        ):
            raise TypeError("admission controls must be LlmAdmissionControls or None")
        if not callable(_lease_sleep):
            raise TypeError("lease sleep must be callable")
        self._admission_controls = admission_controls
        self._clock = _clock
        self._sleep = _sleep
        self._lease_sleep = _lease_sleep
        self._random = _random
        self._circuits = {name: _CircuitBreaker() for name in routes_by_name}

    def resolve(self, request: ChatCompletionRequest) -> LlmRoute:
        """Resolve the first matching rule or return the default route."""

        for rule in self._rules:
            try:
                matched = rule.predicate(request)
            except Exception as exc:
                raise LlmGatewayError(
                    "llm_route_selection_failed",
                    f"route rule {rule.name!r} failed",
                ) from exc
            if matched:
                return self._routes[rule.route_name]
        return self._routes[self._default_route]

    def _retry_delay(self, failed_attempt: int) -> float:
        nominal = (
            self._policy.initial_backoff_seconds
            * self._policy.backoff_multiplier ** (failed_attempt - 1)
        )
        random_value = min(1.0, max(0.0, self._random()))
        jitter = 1 + self._policy.jitter_ratio * (2 * random_value - 1)
        return min(self._policy.max_backoff_seconds, nominal * jitter)

    def _validate_admission_context(
        self,
        context: LlmAdmissionContext | None,
    ) -> None:
        controls = self._admission_controls
        if controls is None or not controls.enabled:
            return
        if not isinstance(context, LlmAdmissionContext):
            raise TypeError(
                "admission context must be supplied when controls are enabled"
            )
        if controls.budget_ledger is not None and (
            context.token_period_id is None or context.cost_period_id is None
        ):
            raise ValueError(
                "token and cost period IDs are required when budgets are enabled"
            )

    def _admission_error(
        self,
        *,
        control: str,
        code: str,
        retry_after_seconds: float | None,
        span: Any,
    ) -> LlmAdmissionError:
        error = LlmAdmissionError(
            code,
            retry_after_seconds=retry_after_seconds,
        )
        attributes: dict[str, str | float] = {
            "llmkit.admission.control": control,
            "admission.outcome": code,
            "error.type": code,
        }
        if retry_after_seconds is not None:
            attributes["admission.retry_after_seconds"] = retry_after_seconds
        _add_event(span, "llm.admission_denied", attributes)
        if span is not None:
            span.set_attribute("error.type", code)
            set_span_error(span, code)
        return error

    async def _cleanup_admission(
        self,
        state: _AttemptAdmission,
        *,
        usage: TokenUsage | None,
        span: Any,
    ) -> None:
        controls = self._admission_controls
        assert controls is not None
        failures: list[BaseException] = []

        if state.reservation is not None:
            try:
                if state.budget_attempted:
                    assert controls.budget_ledger is not None
                    await controls.budget_ledger.reconcile(state.reservation, usage)
                else:
                    assert controls.budget_ledger is not None
                    await controls.budget_ledger.cancel(state.reservation)
            except BaseException as exc:
                failures.append(exc)
            else:
                state.reservation = None

        if state.lease is not None:
            try:
                assert controls.concurrency_limiter is not None
                await controls.concurrency_limiter.release(state.lease)
            except BaseException as exc:
                failures.append(exc)
            else:
                state.lease = None

        if failures:
            error = self._admission_error(
                control="cleanup",
                code="admission_cleanup_failed",
                retry_after_seconds=None,
                span=span,
            )
            raise error from failures[0]

    async def _admit_attempt(
        self,
        route: LlmRoute,
        request: ChatCompletionRequest,
        context: LlmAdmissionContext,
        *,
        span: Any,
    ) -> _AttemptAdmission:
        controls = self._admission_controls
        assert controls is not None and controls.enabled
        state = _AttemptAdmission()
        resource = f"{route.endpoint.provider}:{route.endpoint.model_name}"
        estimated_tokens = context.estimated_input_tokens + request.max_tokens

        try:
            if controls.concurrency_limiter is not None:
                decision = await controls.concurrency_limiter.acquire(
                    AdmissionKey(context.tenant_id, resource, "concurrency")
                )
                if not decision.allowed:
                    raise self._admission_error(
                        control="concurrency",
                        code=decision.outcome.value,
                        retry_after_seconds=decision.retry_after_seconds,
                        span=span,
                    )
                assert decision.lease is not None
                state.lease = decision.lease

            if controls.budget_ledger is not None:
                assert context.token_period_id is not None
                assert context.cost_period_id is not None
                decision = await controls.budget_ledger.reserve(
                    BudgetKey(
                        context.tenant_id,
                        context.token_period_id,
                        context.cost_period_id,
                    ),
                    provider=route.endpoint.provider,
                    model=route.endpoint.model_name,
                    estimated_input_tokens=context.estimated_input_tokens,
                    max_output_tokens=request.max_tokens,
                )
                if not decision.allowed:
                    raise self._admission_error(
                        control="budget",
                        code=decision.outcome.value,
                        retry_after_seconds=None,
                        span=span,
                    )
                assert decision.reservation is not None
                state.reservation = decision.reservation

            if controls.request_rate_limiter is not None:
                decision = await controls.request_rate_limiter.acquire(
                    AdmissionKey(context.tenant_id, resource, "requests")
                )
                if not decision.allowed:
                    raise self._admission_error(
                        control="request_rate",
                        code=decision.outcome.value,
                        retry_after_seconds=decision.retry_after_seconds,
                        span=span,
                    )

            if controls.token_rate_limiter is not None:
                decision = await controls.token_rate_limiter.acquire(
                    AdmissionKey(context.tenant_id, resource, "tokens"),
                    cost=estimated_tokens,
                )
                if not decision.allowed:
                    raise self._admission_error(
                        control="token_rate",
                        code=decision.outcome.value,
                        retry_after_seconds=decision.retry_after_seconds,
                        span=span,
                    )
        except BaseException as exc:
            try:
                await self._cleanup_admission(state, usage=None, span=span)
            except LlmAdmissionError as cleanup_error:
                raise cleanup_error from exc
            raise

        _add_event(
            span,
            "llm.admission_allowed",
            {
                "admission.outcome": "allowed",
                "admission.estimated_tokens": estimated_tokens,
            },
        )
        return state

    async def _renew_lease_until_complete(
        self,
        state: _AttemptAdmission,
        provider_task: asyncio.Task[ChatCompletionResponse],
    ) -> None:
        controls = self._admission_controls
        assert controls is not None
        assert controls.concurrency_limiter is not None

        while not provider_task.done():
            assert state.lease is not None
            remaining = state.lease.expires_at - self._clock()
            await self._lease_sleep(max(0.0, remaining / 2))
            if provider_task.done():
                return
            state.lease = await controls.concurrency_limiter.renew(state.lease)

    async def _invoke_provider(
        self,
        route: LlmRoute,
        request: ChatCompletionRequest,
        *,
        http_client: httpx.AsyncClient,
        state: _AttemptAdmission,
        span: Any,
    ) -> ChatCompletionResponse:
        provider_task = asyncio.create_task(
            route.adapter.complete(
                request,
                cfg=route.endpoint,
                http_client=http_client,
            )
        )
        renewal_task: asyncio.Task[None] | None = None
        if state.lease is not None:
            renewal_task = asyncio.create_task(
                self._renew_lease_until_complete(state, provider_task)
            )

        try:
            if renewal_task is None:
                return await provider_task

            await asyncio.wait(
                (provider_task, renewal_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if renewal_task.done():
                renewal_error = renewal_task.exception()
                if renewal_error is not None:
                    provider_task.cancel()
                    await asyncio.gather(provider_task, return_exceptions=True)
                    error = self._admission_error(
                        control="concurrency",
                        code="concurrency_lease_renewal_failed",
                        retry_after_seconds=None,
                        span=span,
                    )
                    raise error from renewal_error
            return await provider_task
        finally:
            for task in (provider_task, renewal_task):
                if task is not None and not task.done():
                    task.cancel()
            pending = tuple(
                task
                for task in (provider_task, renewal_task)
                if task is not None
            )
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def _complete_attempt(
        self,
        route: LlmRoute,
        request: ChatCompletionRequest,
        *,
        http_client: httpx.AsyncClient,
        span: Any,
        admission_context: LlmAdmissionContext | None,
    ) -> ChatCompletionResponse:
        controls = self._admission_controls
        if controls is None or not controls.enabled:
            return await route.adapter.complete(
                request,
                cfg=route.endpoint,
                http_client=http_client,
            )

        assert admission_context is not None
        state = await self._admit_attempt(
            route,
            request,
            admission_context,
            span=span,
        )
        try:
            if state.reservation is not None:
                assert controls.budget_ledger is not None
                await controls.budget_ledger.mark_attempted(state.reservation)
                state.budget_attempted = True
            response = await self._invoke_provider(
                route,
                request,
                http_client=http_client,
                state=state,
                span=span,
            )
            if not isinstance(response, ChatCompletionResponse):
                raise TypeError(
                    "LLM provider adapters must return ChatCompletionResponse"
                )
        except BaseException as exc:
            try:
                await self._cleanup_admission(state, usage=None, span=span)
            except LlmAdmissionError as cleanup_error:
                raise cleanup_error from exc
            raise

        await self._cleanup_admission(state, usage=response.usage, span=span)
        return response

    async def _complete_route(
        self,
        route: LlmRoute,
        request: ChatCompletionRequest,
        *,
        http_client: httpx.AsyncClient,
        span: Any,
        admission_context: LlmAdmissionContext | None,
    ) -> ChatCompletionResponse:
        for attempt in range(1, self._policy.max_attempts_per_route + 1):
            _add_event(
                span,
                "llm.route_attempt",
                _route_attributes(route, attempt=attempt),
            )
            try:
                response = await self._complete_attempt(
                    route,
                    request,
                    http_client=http_client,
                    span=span,
                    admission_context=admission_context,
                )
                if not isinstance(response, ChatCompletionResponse):
                    raise TypeError(
                        "LLM provider adapters must return ChatCompletionResponse"
                    )
                return response
            except LlmGatewayError as exc:
                _add_event(
                    span,
                    "llm.route_attempt_failed",
                    {
                        **_route_attributes(route, attempt=attempt),
                        "error.type": exc.code,
                    },
                )
                if attempt == self._policy.max_attempts_per_route:
                    raise
                delay = self._retry_delay(attempt)
                logger.warning(
                    "Retrying LLM route %s after %s (attempt=%d delay=%.3fs)",
                    route.name,
                    exc.code,
                    attempt + 1,
                    delay,
                )
                _add_event(
                    span,
                    "llm.route_retry",
                    {
                        **_route_attributes(route, attempt=attempt + 1),
                        "error.type": exc.code,
                        "llmkit.retry.delay_seconds": delay,
                    },
                )
                await self._sleep(delay)
        raise AssertionError("retry loop completed without returning or raising")

    async def complete_response(
        self,
        request: ChatCompletionRequest,
        *,
        http_client: httpx.AsyncClient,
        admission_context: LlmAdmissionContext | None = None,
    ) -> ChatCompletionResponse:
        """Resolve a route and return its normalized provider response."""

        self._validate_admission_context(admission_context)
        primary = self.resolve(request)
        candidate_names = (primary.name, *primary.fallback_routes)
        attributes: dict[str, str | int] = {
            "llmkit.route.primary": primary.name,
            "llmkit.route.candidate_count": len(candidate_names),
        }
        with trace_span("llm.router.complete", attributes=attributes) as span:
            _add_event(span, "llm.route_selected", _route_attributes(primary))
            last_error: LlmGatewayError | None = None
            attempted_any = False
            compatible_any = False
            previous_route = primary.name
            previous_error_type = "llm_circuit_open"

            for index, route_name in enumerate(candidate_names):
                route = self._routes[route_name]
                missing = missing_capabilities(request, route.endpoint)
                if missing:
                    missing_names = ",".join(
                        sorted(capability.value for capability in missing)
                    )
                    logger.info(
                        "Skipping incompatible LLM route %s (missing=%s)",
                        route.name,
                        missing_names,
                    )
                    _add_event(
                        span,
                        "llm.route_incompatible",
                        {
                            **_route_attributes(route),
                            "llmkit.capability.missing": missing_names,
                        },
                    )
                    previous_route = route.name
                    previous_error_type = "llm_capability_unsupported"
                    continue

                compatible_any = True
                circuit = self._circuits[route_name]
                allowed, is_probe = await circuit.acquire(
                    now=self._clock(),
                    recovery_timeout_seconds=(
                        self._policy.circuit_recovery_timeout_seconds
                    ),
                )
                if not allowed:
                    logger.warning("Skipping open LLM route %s", route.name)
                    _add_event(
                        span,
                        "llm.circuit_skipped",
                        _route_attributes(route),
                    )
                    previous_route = route.name
                    previous_error_type = "llm_circuit_open"
                    continue
                if is_probe:
                    _add_event(
                        span,
                        "llm.circuit_half_open",
                        _route_attributes(route),
                    )

                if index > 0:
                    fallback_attributes = {
                        "llmkit.route.from": previous_route,
                        "llmkit.route.to": route.name,
                        "error.type": previous_error_type,
                    }
                    _add_event(span, "llm.route_fallback", fallback_attributes)

                attempted_any = True
                try:
                    result = await self._complete_route(
                        route,
                        request,
                        http_client=http_client,
                        span=span,
                        admission_context=admission_context,
                    )
                except LlmGatewayError as exc:
                    last_error = exc
                    opened = await circuit.record_failure(
                        now=self._clock(),
                        threshold=self._policy.circuit_failure_threshold,
                    )
                    if opened:
                        logger.warning("Opened circuit for LLM route %s", route.name)
                        _add_event(
                            span,
                            "llm.circuit_opened",
                            {
                                **_route_attributes(route),
                                "error.type": exc.code,
                            },
                        )
                    previous_route = route.name
                    previous_error_type = exc.code
                    continue
                except BaseException:
                    if is_probe:
                        await circuit.abandon_probe(now=self._clock())
                    raise

                closed = await circuit.record_success()
                if closed:
                    _add_event(
                        span,
                        "llm.circuit_closed",
                        _route_attributes(route),
                    )
                return result

            if last_error is not None:
                error = LlmGatewayError(
                    "llm_routes_exhausted",
                    "all eligible LLM routes failed",
                )
                if span is not None:
                    span.set_attribute("error.type", error.code)
                    set_span_error(span, error.code)
                raise error from last_error

            assert not attempted_any
            if compatible_any:
                error = LlmGatewayError(
                    "llm_routes_unavailable",
                    "all compatible LLM route circuits are open",
                )
            else:
                error = LlmGatewayError(
                    "llm_capabilities_unavailable",
                    "no configured LLM route supports the required capabilities",
                )
            if span is not None:
                span.set_attribute("error.type", error.code)
                set_span_error(span, error.code)
            raise error

    async def complete(
        self,
        request: ChatCompletionRequest,
        *,
        http_client: httpx.AsyncClient,
        admission_context: LlmAdmissionContext | None = None,
    ) -> str:
        """Resolve a route and return text for compatibility-oriented callers."""

        response = await self.complete_response(
            request,
            http_client=http_client,
            admission_context=admission_context,
        )
        if response.text is None:
            raise LlmGatewayError(
                "llm_text_response_required",
                "LLM response did not contain text content",
            )
        return response.text


def _route_attributes(
    route: LlmRoute,
    *,
    attempt: int | None = None,
) -> dict[str, str | int]:
    attributes: dict[str, str | int] = {
        "llmkit.route.name": route.name,
        "gen_ai.provider.name": route.endpoint.provider,
        "gen_ai.request.model": route.endpoint.model_name,
    }
    if attempt is not None:
        attributes["llmkit.retry.attempt"] = attempt
    return attributes


def _add_event(
    span: Any,
    name: str,
    attributes: dict[str, str | int | float],
) -> None:
    if span is not None:
        span.add_event(name, attributes)
