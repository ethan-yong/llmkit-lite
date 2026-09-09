import asyncio

import httpx
import pytest

from llmkit_lite.llm import (
    ChatCompletionRequest,
    LlmEndpointConfig,
    LlmGatewayError,
)
from llmkit_lite.routing import (
    LlmResiliencePolicy,
    LlmRoute,
    LlmRouter,
    LlmRouteRule,
)


class RecordingAdapter:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[
            tuple[ChatCompletionRequest, LlmEndpointConfig, httpx.AsyncClient]
        ] = []

    async def complete(
        self,
        request: ChatCompletionRequest,
        *,
        cfg: LlmEndpointConfig,
        http_client: httpx.AsyncClient,
    ) -> str:
        self.calls.append((request, cfg, http_client))
        return self.result


class ScriptedAdapter(RecordingAdapter):
    def __init__(self, outcomes: list[str | BaseException]) -> None:
        super().__init__("")
        self.outcomes = outcomes.copy()

    async def complete(
        self,
        request: ChatCompletionRequest,
        *,
        cfg: LlmEndpointConfig,
        http_client: httpx.AsyncClient,
    ) -> str:
        self.calls.append((request, cfg, http_client))
        if not self.outcomes:
            raise AssertionError("scripted adapter has no remaining outcome")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class BlockingProbeAdapter(RecordingAdapter):
    def __init__(self) -> None:
        super().__init__("recovered")
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(
        self,
        request: ChatCompletionRequest,
        *,
        cfg: LlmEndpointConfig,
        http_client: httpx.AsyncClient,
    ) -> str:
        self.calls.append((request, cfg, http_client))
        if len(self.calls) == 1:
            raise LlmGatewayError("llm_timeout", "first call failed")
        self.started.set()
        await self.release.wait()
        return self.result


def _endpoint(name: str) -> LlmEndpointConfig:
    return LlmEndpointConfig(
        provider=name,
        base_url=f"https://{name}.example.com",
        model_name=f"{name}-model",
    )


def _request(**metadata: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "route this"}],
        max_tokens=100,
        timeout_seconds=5,
        routing_metadata=metadata,
    )


async def test_first_matching_rule_selects_exactly_one_adapter() -> None:
    premium_adapter = RecordingAdapter("premium response")
    secondary_adapter = RecordingAdapter("secondary response")
    default_adapter = RecordingAdapter("default response")
    premium_endpoint = _endpoint("premium")
    router = LlmRouter(
        routes=(
            LlmRoute("premium", premium_endpoint, premium_adapter),
            LlmRoute("secondary", _endpoint("secondary"), secondary_adapter),
            LlmRoute("default", _endpoint("default"), default_adapter),
        ),
        default_route="default",
        rules=(
            LlmRouteRule(
                "premium workload",
                "premium",
                lambda request: request.routing_metadata.get("tier") == "premium",
            ),
            LlmRouteRule("catch all", "secondary", lambda request: True),
        ),
    )
    request = _request(tier="premium")

    async with httpx.AsyncClient() as client:
        result = await router.complete(request, http_client=client)

    assert result == "premium response"
    assert premium_adapter.calls == [(request, premium_endpoint, client)]
    assert secondary_adapter.calls == []
    assert default_adapter.calls == []


async def test_no_matching_rule_uses_default_route() -> None:
    selected_adapter = RecordingAdapter("selected")
    skipped_adapter = RecordingAdapter("skipped")
    default_endpoint = _endpoint("default")
    router = LlmRouter(
        routes=(
            LlmRoute("special", _endpoint("special"), skipped_adapter),
            LlmRoute("default", default_endpoint, selected_adapter),
        ),
        default_route="default",
        rules=(
            LlmRouteRule(
                "special workload",
                "special",
                lambda request: request.routing_metadata.get("workload") == "special",
            ),
        ),
    )
    request = _request(workload="ordinary")

    async with httpx.AsyncClient() as client:
        result = await router.complete(request, http_client=client)

    assert result == "selected"
    assert selected_adapter.calls == [(request, default_endpoint, client)]
    assert skipped_adapter.calls == []


@pytest.mark.parametrize("name", ["", "   "])
def test_route_rejects_empty_names(name: str) -> None:
    with pytest.raises(ValueError, match="route name must be non-empty"):
        LlmRoute(name, _endpoint("default"))


def test_router_rejects_empty_routes() -> None:
    with pytest.raises(ValueError, match="at least one route is required"):
        LlmRouter((), default_route="default")


def test_router_rejects_duplicate_route_names() -> None:
    with pytest.raises(ValueError, match="duplicate route name"):
        LlmRouter(
            (
                LlmRoute("same", _endpoint("first")),
                LlmRoute("same", _endpoint("second")),
            ),
            default_route="same",
        )


def test_router_rejects_missing_default_route() -> None:
    with pytest.raises(ValueError, match="default route does not exist"):
        LlmRouter(
            (LlmRoute("available", _endpoint("available")),),
            default_route="missing",
        )


def test_router_rejects_rule_with_unknown_route() -> None:
    with pytest.raises(ValueError, match="targets unknown route"):
        LlmRouter(
            (LlmRoute("default", _endpoint("default")),),
            default_route="default",
            rules=(LlmRouteRule("bad target", "missing", lambda request: True),),
        )


def test_router_rejects_duplicate_rule_names() -> None:
    with pytest.raises(ValueError, match="duplicate rule name"):
        LlmRouter(
            (LlmRoute("default", _endpoint("default")),),
            default_route="default",
            rules=(
                LlmRouteRule("same", "default", lambda request: False),
                LlmRouteRule("same", "default", lambda request: True),
            ),
        )


async def test_predicate_failure_is_safe_and_does_not_call_an_adapter() -> None:
    adapter = RecordingAdapter("must not run")

    def fail_selection(request: ChatCompletionRequest) -> bool:
        raise RuntimeError(f"do not expose {request.routing_metadata['secret']}")

    router = LlmRouter(
        (LlmRoute("default", _endpoint("default"), adapter),),
        default_route="default",
        rules=(LlmRouteRule("unsafe predicate", "default", fail_selection),),
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(LlmGatewayError) as exc_info:
            await router.complete(_request(secret="private-value"), http_client=client)

    assert exc_info.value.code == "llm_route_selection_failed"
    assert exc_info.value.detail == "route rule 'unsafe predicate' failed"
    assert "private-value" not in str(exc_info.value)
    assert adapter.calls == []


def test_resilience_policy_has_balanced_defaults() -> None:
    assert LlmResiliencePolicy() == LlmResiliencePolicy(
        max_attempts_per_route=2,
        initial_backoff_seconds=0.25,
        backoff_multiplier=2,
        max_backoff_seconds=2,
        jitter_ratio=0.2,
        circuit_failure_threshold=3,
        circuit_recovery_timeout_seconds=30,
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_attempts_per_route": 0},
        {"initial_backoff_seconds": -1},
        {"backoff_multiplier": 0.5},
        {"max_backoff_seconds": 0.1},
        {"jitter_ratio": 1.1},
        {"circuit_failure_threshold": 0},
        {"circuit_recovery_timeout_seconds": 0},
    ],
)
def test_resilience_policy_rejects_invalid_values(overrides: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        LlmResiliencePolicy(**overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "error_code",
    [
        "llm_timeout",
        "llm_fetch_failed",
        "llm_http_400",
        "llm_http_429",
        "llm_http_500",
        "llm_invalid_response_json",
    ],
)
async def test_every_gateway_error_is_retried(error_code: str) -> None:
    adapter = ScriptedAdapter(
        [LlmGatewayError(error_code, "provider failed"), "recovered"]
    )
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    router = LlmRouter(
        (LlmRoute("default", _endpoint("default"), adapter),),
        default_route="default",
        _sleep=record_sleep,
        _random=lambda: 0.5,
    )

    async with httpx.AsyncClient() as client:
        result = await router.complete(_request(), http_client=client)

    assert result == "recovered"
    assert len(adapter.calls) == 2
    assert delays == [0.25]


async def test_retries_use_exponential_backoff_before_success() -> None:
    adapter = ScriptedAdapter(
        [
            LlmGatewayError("llm_timeout", "one"),
            LlmGatewayError("llm_timeout", "two"),
            "recovered",
        ]
    )
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    router = LlmRouter(
        (LlmRoute("default", _endpoint("default"), adapter),),
        default_route="default",
        resilience_policy=LlmResiliencePolicy(max_attempts_per_route=3),
        _sleep=record_sleep,
        _random=lambda: 0.5,
    )

    async with httpx.AsyncClient() as client:
        result = await router.complete(_request(), http_client=client)

    assert result == "recovered"
    assert delays == [0.25, 0.5]


async def test_retry_jitter_stays_within_ratio_and_maximum_delay() -> None:
    adapter = ScriptedAdapter(
        [
            LlmGatewayError("llm_timeout", "one"),
            LlmGatewayError("llm_timeout", "two"),
            "recovered",
        ]
    )
    delays: list[float] = []
    random_values = iter((0.0, 1.0))

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    router = LlmRouter(
        (LlmRoute("default", _endpoint("default"), adapter),),
        default_route="default",
        resilience_policy=LlmResiliencePolicy(
            max_attempts_per_route=3,
            initial_backoff_seconds=1,
            backoff_multiplier=4,
            max_backoff_seconds=2,
        ),
        _sleep=record_sleep,
        _random=lambda: next(random_values),
    )

    async with httpx.AsyncClient() as client:
        result = await router.complete(_request(), http_client=client)

    assert result == "recovered"
    assert delays == [0.8, 2]


async def test_exhausted_primary_uses_its_fallback_once() -> None:
    primary = ScriptedAdapter(
        [
            LlmGatewayError("llm_http_400", "one"),
            LlmGatewayError("llm_invalid_response_json", "two"),
        ]
    )
    fallback = ScriptedAdapter(["fallback response"])
    router = LlmRouter(
        (
            LlmRoute(
                "primary",
                _endpoint("primary"),
                primary,
                fallback_routes=("fallback",),
            ),
            LlmRoute("fallback", _endpoint("fallback"), fallback),
        ),
        default_route="primary",
        _sleep=_no_sleep,
        _random=lambda: 0.5,
    )

    async with httpx.AsyncClient() as client:
        result = await router.complete(_request(), http_client=client)

    assert result == "fallback response"
    assert len(primary.calls) == 2
    assert len(fallback.calls) == 1


async def test_multiple_fallbacks_are_attempted_in_declared_order() -> None:
    call_order: list[str] = []

    class OrderedAdapter(RecordingAdapter):
        def __init__(self, name: str, outcome: str | LlmGatewayError) -> None:
            super().__init__("")
            self.name = name
            self.outcome = outcome

        async def complete(
            self,
            request: ChatCompletionRequest,
            *,
            cfg: LlmEndpointConfig,
            http_client: httpx.AsyncClient,
        ) -> str:
            call_order.append(self.name)
            if isinstance(self.outcome, LlmGatewayError):
                raise self.outcome
            return self.outcome

    primary = OrderedAdapter("primary", LlmGatewayError("llm_timeout", "one"))
    first = OrderedAdapter("first", LlmGatewayError("llm_http_500", "two"))
    second = OrderedAdapter("second", "recovered")
    router = LlmRouter(
        (
            LlmRoute(
                "primary",
                _endpoint("primary"),
                primary,
                fallback_routes=("first", "second"),
            ),
            LlmRoute("first", _endpoint("first"), first),
            LlmRoute("second", _endpoint("second"), second),
        ),
        default_route="primary",
        resilience_policy=LlmResiliencePolicy(max_attempts_per_route=1),
    )

    async with httpx.AsyncClient() as client:
        result = await router.complete(_request(), http_client=client)

    assert result == "recovered"
    assert call_order == ["primary", "first", "second"]


async def test_fallback_chains_are_flat_not_recursive() -> None:
    def failure(name: str) -> LlmGatewayError:
        return LlmGatewayError("llm_timeout", name)

    primary = ScriptedAdapter([failure("p1"), failure("p2")])
    fallback = ScriptedAdapter([failure("f1"), failure("f2")])
    nested = RecordingAdapter("must not run")
    router = LlmRouter(
        (
            LlmRoute(
                "primary",
                _endpoint("primary"),
                primary,
                fallback_routes=("fallback",),
            ),
            LlmRoute(
                "fallback",
                _endpoint("fallback"),
                fallback,
                fallback_routes=("nested",),
            ),
            LlmRoute("nested", _endpoint("nested"), nested),
        ),
        default_route="primary",
        _sleep=_no_sleep,
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(LlmGatewayError) as exc_info:
            await router.complete(_request(), http_client=client)

    assert exc_info.value.code == "llm_routes_exhausted"
    assert nested.calls == []


def test_route_rejects_duplicate_and_self_fallbacks() -> None:
    with pytest.raises(ValueError, match="duplicate fallback"):
        LlmRoute(
            "primary",
            _endpoint("primary"),
            fallback_routes=("backup", "backup"),
        )
    with pytest.raises(ValueError, match="cannot fall back to itself"):
        LlmRoute(
            "primary",
            _endpoint("primary"),
            fallback_routes=("primary",),
        )


def test_router_rejects_unknown_fallback_route() -> None:
    with pytest.raises(ValueError, match="unknown fallback route"):
        LlmRouter(
            (
                LlmRoute(
                    "primary",
                    _endpoint("primary"),
                    fallback_routes=("missing",),
                ),
            ),
            default_route="primary",
        )


async def test_all_failed_routes_raise_safe_exhausted_error() -> None:
    primary = ScriptedAdapter(
        [
            LlmGatewayError("llm_timeout", "private primary one"),
            LlmGatewayError("llm_timeout", "private primary two"),
        ]
    )
    final_error = LlmGatewayError("llm_http_500", "private fallback")
    fallback = ScriptedAdapter(
        [LlmGatewayError("llm_http_500", "first"), final_error]
    )
    router = LlmRouter(
        (
            LlmRoute(
                "primary",
                _endpoint("primary"),
                primary,
                fallback_routes=("fallback",),
            ),
            LlmRoute("fallback", _endpoint("fallback"), fallback),
        ),
        default_route="primary",
        _sleep=_no_sleep,
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(LlmGatewayError) as exc_info:
            await router.complete(_request(secret="private-value"), http_client=client)

    assert exc_info.value.code == "llm_routes_exhausted"
    assert exc_info.value.detail == "all eligible LLM routes failed"
    assert exc_info.value.__cause__ is final_error
    assert "private" not in str(exc_info.value)


async def test_circuit_opens_after_three_exhausted_route_calls() -> None:
    errors = [LlmGatewayError("llm_timeout", str(index)) for index in range(6)]
    adapter = ScriptedAdapter(errors)
    router = LlmRouter(
        (LlmRoute("default", _endpoint("default"), adapter),),
        default_route="default",
        _sleep=_no_sleep,
    )

    async with httpx.AsyncClient() as client:
        for _ in range(3):
            with pytest.raises(LlmGatewayError) as exc_info:
                await router.complete(_request(), http_client=client)
            assert exc_info.value.code == "llm_routes_exhausted"

        with pytest.raises(LlmGatewayError) as exc_info:
            await router.complete(_request(), http_client=client)

    assert exc_info.value.code == "llm_routes_unavailable"
    assert len(adapter.calls) == 6


async def test_open_primary_circuit_is_skipped_for_fallback() -> None:
    clock = FakeClock()
    primary = ScriptedAdapter([LlmGatewayError("llm_timeout", "failed")])
    fallback = RecordingAdapter("fallback response")
    router = LlmRouter(
        (
            LlmRoute(
                "primary",
                _endpoint("primary"),
                primary,
                fallback_routes=("fallback",),
            ),
            LlmRoute("fallback", _endpoint("fallback"), fallback),
        ),
        default_route="primary",
        resilience_policy=LlmResiliencePolicy(
            max_attempts_per_route=1,
            circuit_failure_threshold=1,
        ),
        _clock=clock,
    )

    async with httpx.AsyncClient() as client:
        first = await router.complete(_request(), http_client=client)
        second = await router.complete(_request(), http_client=client)

    assert first == second == "fallback response"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 2


async def test_success_resets_consecutive_circuit_failures() -> None:
    def fail(value: str) -> LlmGatewayError:
        return LlmGatewayError("llm_timeout", value)

    adapter = ScriptedAdapter(
        [fail("one"), "first recovery", fail("two"), "second recovery"]
    )
    router = LlmRouter(
        (LlmRoute("default", _endpoint("default"), adapter),),
        default_route="default",
        resilience_policy=LlmResiliencePolicy(
            max_attempts_per_route=1,
            circuit_failure_threshold=2,
        ),
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(LlmGatewayError):
            await router.complete(_request(), http_client=client)
        assert await router.complete(_request(), http_client=client) == "first recovery"
        with pytest.raises(LlmGatewayError):
            await router.complete(_request(), http_client=client)
        result = await router.complete(_request(), http_client=client)
        assert result == "second recovery"


async def test_failed_half_open_probe_reopens_then_success_closes_circuit() -> None:
    clock = FakeClock()

    def fail(value: str) -> LlmGatewayError:
        return LlmGatewayError("llm_timeout", value)

    adapter = ScriptedAdapter([fail("open"), fail("probe"), "recovered", "healthy"])
    router = LlmRouter(
        (LlmRoute("default", _endpoint("default"), adapter),),
        default_route="default",
        resilience_policy=LlmResiliencePolicy(
            max_attempts_per_route=1,
            circuit_failure_threshold=1,
        ),
        _clock=clock,
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(LlmGatewayError):
            await router.complete(_request(), http_client=client)
        clock.advance(30)
        with pytest.raises(LlmGatewayError):
            await router.complete(_request(), http_client=client)
        with pytest.raises(LlmGatewayError) as exc_info:
            await router.complete(_request(), http_client=client)
        assert exc_info.value.code == "llm_routes_unavailable"

        clock.advance(30)
        assert await router.complete(_request(), http_client=client) == "recovered"
        assert await router.complete(_request(), http_client=client) == "healthy"


async def test_only_one_concurrent_half_open_probe_is_allowed() -> None:
    clock = FakeClock()
    adapter = BlockingProbeAdapter()
    router = LlmRouter(
        (LlmRoute("default", _endpoint("default"), adapter),),
        default_route="default",
        resilience_policy=LlmResiliencePolicy(
            max_attempts_per_route=1,
            circuit_failure_threshold=1,
        ),
        _clock=clock,
    )

    async with httpx.AsyncClient() as client:
        with pytest.raises(LlmGatewayError):
            await router.complete(_request(), http_client=client)
        clock.advance(30)
        probe = asyncio.create_task(router.complete(_request(), http_client=client))
        await adapter.started.wait()

        with pytest.raises(LlmGatewayError) as exc_info:
            await router.complete(_request(), http_client=client)
        assert exc_info.value.code == "llm_routes_unavailable"

        adapter.release.set()
        assert await probe == "recovered"

    assert len(adapter.calls) == 2


async def test_unexpected_errors_and_cancellation_are_not_retried() -> None:
    for unexpected in (RuntimeError("bug"), asyncio.CancelledError()):
        adapter = ScriptedAdapter([unexpected])
        fallback = RecordingAdapter("must not run")
        router = LlmRouter(
            (
                LlmRoute(
                    "default",
                    _endpoint("default"),
                    adapter,
                    fallback_routes=("fallback",),
                ),
                LlmRoute("fallback", _endpoint("fallback"), fallback),
            ),
            default_route="default",
            _sleep=_no_sleep,
        )

        async with httpx.AsyncClient() as client:
            with pytest.raises(type(unexpected)):
                await router.complete(_request(), http_client=client)

        assert len(adapter.calls) == 1
        assert fallback.calls == []


async def test_response_format_compatibility_retry_is_not_a_router_retry(
    in_memory_tracing,
) -> None:
    http_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        if http_calls == 1:
            return httpx.Response(400, json={"error": "unsupported"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
        )

    router = LlmRouter(
        (LlmRoute("default", _endpoint("default")),),
        default_route="default",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await router.complete(_request(), http_client=client)

    router_span = next(
        span
        for span in in_memory_tracing.get_finished_spans()
        if span.name == "llm.router.complete"
    )
    assert result == "ok"
    assert http_calls == 2
    assert "llm.route_retry" not in [event.name for event in router_span.events]


async def test_routing_telemetry_excludes_sensitive_values(in_memory_tracing) -> None:
    primary = ScriptedAdapter(
        [
            LlmGatewayError("llm_http_400", "private failure"),
            LlmGatewayError("llm_http_400", "private failure"),
        ]
    )
    fallback = RecordingAdapter("private response")
    router = LlmRouter(
        (
            LlmRoute(
                "primary",
                _endpoint("primary"),
                primary,
                fallback_routes=("fallback",),
            ),
            LlmRoute("fallback", _endpoint("fallback"), fallback),
        ),
        default_route="primary",
        _sleep=_no_sleep,
        _random=lambda: 0.5,
    )
    request = ChatCompletionRequest(
        messages=[{"role": "user", "content": "private prompt"}],
        max_tokens=100,
        timeout_seconds=5,
        routing_metadata={"tenant": "private metadata"},
    )

    async with httpx.AsyncClient() as client:
        assert await router.complete(request, http_client=client) == "private response"

    span = next(
        span
        for span in in_memory_tracing.get_finished_spans()
        if span.name == "llm.router.complete"
    )
    exported = str(span.attributes) + str(span.events)
    assert "private prompt" not in exported
    assert "private response" not in exported
    assert "private metadata" not in exported
    assert "private failure" not in exported
    assert [event.name for event in span.events] == [
        "llm.route_selected",
        "llm.route_attempt",
        "llm.route_attempt_failed",
        "llm.route_retry",
        "llm.route_attempt",
        "llm.route_attempt_failed",
        "llm.route_fallback",
        "llm.route_attempt",
    ]


async def _no_sleep(delay: float) -> None:
    pass
