"""Deterministic routing for LLM provider adapters."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import httpx

from llmkit_lite.llm import (
    ChatCompletionRequest,
    LlmEndpointConfig,
    LlmGatewayError,
    LlmProviderAdapter,
    OpenAICompatibleAdapter,
)

RoutePredicate = Callable[[ChatCompletionRequest], bool]


def _normalized_name(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


@dataclass(frozen=True)
class LlmRoute:
    """One named provider endpoint available to a router."""

    name: str
    endpoint: LlmEndpointConfig
    adapter: LlmProviderAdapter = field(default_factory=OpenAICompatibleAdapter)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalized_name(self.name, "route name"))


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


class LlmRouter:
    """Select exactly one LLM route using ordered predicates."""

    def __init__(
        self,
        routes: Sequence[LlmRoute],
        *,
        default_route: str,
        rules: Sequence[LlmRouteRule] = (),
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

        self._routes = MappingProxyType(routes_by_name)
        self._default_route = normalized_default
        self._rules = normalized_rules

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

    async def complete(
        self,
        request: ChatCompletionRequest,
        *,
        http_client: httpx.AsyncClient,
    ) -> str:
        """Resolve a route and execute its provider adapter once."""

        route = self.resolve(request)
        return await route.adapter.complete(
            request,
            cfg=route.endpoint,
            http_client=http_client,
        )
