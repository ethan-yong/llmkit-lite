import httpx
import pytest

from llmkit_lite.llm import (
    ChatCompletionRequest,
    LlmEndpointConfig,
    LlmGatewayError,
)
from llmkit_lite.routing import LlmRoute, LlmRouter, LlmRouteRule


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
