"""Authorization-enforced execution boundary for application tools."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence, Set
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from llmkit_lite.authorization import (
    AuthorizationError,
    AuthorizationPolicy,
    AuthorizationRequest,
    Principal,
    ScopeAuthorizationPolicy,
    get_principal,
    require_authorization,
)
from llmkit_lite.observability import set_span_error, trace_span

ToolHandler = Callable[[Mapping[str, Any]], Any | Awaitable[Any]]

_logger = logging.getLogger(__name__)


def _normalize_name(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("tool name must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError("tool name must not be empty")
    return normalized


def _normalize_scopes(values: Set[str]) -> frozenset[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Set):
        raise TypeError("tool required scopes must be a set of strings")

    normalized: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("tool required scope must be a string")
        scope = value.strip()
        if not scope:
            raise ValueError("tool required scope must not be empty")
        normalized.add(scope)
    return frozenset(normalized)


def _exception_type(exc: BaseException) -> str:
    return f"{type(exc).__module__}.{type(exc).__qualname__}"


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A named callable and the scopes required to invoke it."""

    name: str
    handler: ToolHandler
    required_scopes: Set[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_name(self.name))
        if not callable(self.handler):
            raise TypeError("tool handler must be callable")
        object.__setattr__(
            self,
            "required_scopes",
            _normalize_scopes(self.required_scopes),
        )


class ToolExecutionError(Exception):
    """Safe tool-boundary failure suitable for application error mapping."""

    def __init__(self, code: str) -> None:
        if code != "tool_not_found":
            raise ValueError("unsupported tool execution error code")
        detail = "tool is not registered"
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True, init=False)
class AuthorizedToolExecutor:
    """Immutable registry that authorizes every tool at invocation time."""

    _tools: Mapping[str, ToolDefinition]
    policy: AuthorizationPolicy

    def __init__(
        self,
        tools: Sequence[ToolDefinition],
        *,
        policy: AuthorizationPolicy | None = None,
    ) -> None:
        if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
            raise TypeError("tools must be a sequence of ToolDefinition values")

        registry: dict[str, ToolDefinition] = {}
        for definition in tools:
            if not isinstance(definition, ToolDefinition):
                raise TypeError("tools must contain only ToolDefinition values")
            if definition.name in registry:
                raise ValueError(f"duplicate tool registration: {definition.name}")
            registry[definition.name] = definition

        resolved_policy = policy if policy is not None else ScopeAuthorizationPolicy()
        if not callable(getattr(resolved_policy, "authorize", None)):
            raise TypeError("policy must implement authorize")

        object.__setattr__(self, "_tools", MappingProxyType(registry))
        object.__setattr__(self, "policy", resolved_policy)

    @property
    def tools(self) -> tuple[ToolDefinition, ...]:
        """Return the registered definitions in declaration order."""

        return tuple(self._tools.values())

    async def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        principal: Principal | None = None,
    ) -> Any:
        """Authorize and invoke one registered tool exactly once."""

        try:
            normalized_name = _normalize_name(tool_name)
        except (TypeError, ValueError) as exc:
            raise ToolExecutionError("tool_not_found") from exc

        definition = self._tools.get(normalized_name)
        if definition is None:
            raise ToolExecutionError("tool_not_found")
        if not isinstance(arguments, Mapping):
            raise TypeError("tool arguments must be a mapping")

        active_principal = principal if principal is not None else get_principal()
        request = AuthorizationRequest(
            action="tools.execute",
            resource=f"tool:{definition.name}",
            required_scopes=definition.required_scopes,
        )

        with trace_span(
            "tool.execute",
            attributes={"tool.name": definition.name},
        ) as span:
            try:
                decision = await require_authorization(
                    self.policy,
                    request,
                    principal=active_principal,
                )
            except AuthorizationError as exc:
                if span is not None:
                    span.set_attribute("tool.outcome", "denied")
                    span.set_attribute("tool.reason_code", exc.reason_code)
                    span.set_attribute("exception.type", _exception_type(exc))
                set_span_error(span, _exception_type(exc))
                _logger.warning(
                    "tool execution denied",
                    extra={
                        "tool_name": definition.name,
                        "tool_outcome": "denied",
                        "tool_reason_code": exc.reason_code,
                        "tool_exception_type": _exception_type(exc),
                    },
                )
                raise
            except BaseException as exc:
                exception_type = _exception_type(exc)
                if span is not None:
                    span.set_attribute("tool.outcome", "failed")
                    span.set_attribute("tool.reason_code", "policy_failure")
                    span.set_attribute("exception.type", exception_type)
                set_span_error(span, exception_type)
                _logger.warning(
                    "tool authorization failed",
                    extra={
                        "tool_name": definition.name,
                        "tool_outcome": "failed",
                        "tool_reason_code": "policy_failure",
                        "tool_exception_type": exception_type,
                    },
                )
                raise

            if span is not None:
                span.set_attribute("tool.outcome", "allowed")
                span.set_attribute("tool.reason_code", decision.reason_code)
            _logger.info(
                "tool execution allowed",
                extra={
                    "tool_name": definition.name,
                    "tool_outcome": "allowed",
                    "tool_reason_code": decision.reason_code,
                },
            )

            copied_arguments = dict(arguments)
            try:
                result = definition.handler(copied_arguments)
                if inspect.isawaitable(result):
                    result = await result
            except BaseException as exc:
                exception_type = _exception_type(exc)
                if span is not None:
                    span.set_attribute("tool.outcome", "failed")
                    span.set_attribute("tool.reason_code", "handler_failure")
                    span.set_attribute("exception.type", exception_type)
                set_span_error(span, exception_type)
                _logger.warning(
                    "tool execution failed",
                    extra={
                        "tool_name": definition.name,
                        "tool_outcome": "failed",
                        "tool_reason_code": "handler_failure",
                        "tool_exception_type": exception_type,
                    },
                )
                raise

            if span is not None:
                span.set_attribute("tool.outcome", "succeeded")
                span.set_attribute("tool.reason_code", "handler_completed")
            _logger.info(
                "tool execution succeeded",
                extra={
                    "tool_name": definition.name,
                    "tool_outcome": "succeeded",
                    "tool_reason_code": "handler_completed",
                },
            )
            return result
