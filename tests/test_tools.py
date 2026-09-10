import asyncio
import logging
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from opentelemetry.trace import StatusCode

from llmkit_lite.authorization import (
    AuthorizationDecision,
    AuthorizationError,
    AuthorizationRequest,
    Principal,
    get_principal,
    principal_context,
)
from llmkit_lite.tools import (
    AuthorizedToolExecutor,
    ToolDefinition,
    ToolExecutionError,
)


def _definition(handler, *, scopes=None) -> ToolDefinition:
    return ToolDefinition(
        " weather ",
        handler,
        required_scopes=scopes or {" tools:weather:execute "},
    )


def _principal(subject: str = "user") -> Principal:
    return Principal(subject, scopes={"tools:weather:execute"})


def test_tool_definition_normalizes_and_freezes_values() -> None:
    scopes = {" tools:weather:execute "}
    definition = _definition(lambda arguments: arguments, scopes=scopes)
    scopes.add("admin")

    assert definition.name == "weather"
    assert definition.required_scopes == frozenset({"tools:weather:execute"})
    with pytest.raises(FrozenInstanceError):
        definition.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory,expected_error",
    [
        (lambda: ToolDefinition("", lambda arguments: None), ValueError),
        (lambda: ToolDefinition(1, lambda arguments: None), TypeError),
        (lambda: ToolDefinition("weather", object()), TypeError),
        (
            lambda: ToolDefinition(
                "weather", lambda arguments: None, required_scopes=["scope"]
            ),
            TypeError,
        ),
        (
            lambda: ToolDefinition(
                "weather", lambda arguments: None, required_scopes={" "}
            ),
            ValueError,
        ),
        (
            lambda: ToolDefinition(
                "weather", lambda arguments: None, required_scopes={1}
            ),
            TypeError,
        ),
    ],
)
def test_tool_definition_rejects_invalid_values(factory, expected_error) -> None:
    with pytest.raises(expected_error):
        factory()


def test_executor_validates_registry_and_policy() -> None:
    definition = _definition(lambda arguments: None)

    with pytest.raises(ValueError, match="duplicate tool registration"):
        AuthorizedToolExecutor((definition, definition))
    with pytest.raises(TypeError, match="ToolDefinition"):
        AuthorizedToolExecutor((object(),))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sequence"):
        AuthorizedToolExecutor("weather")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="implement authorize"):
        AuthorizedToolExecutor((definition,), policy=object())  # type: ignore[arg-type]


async def test_allowed_sync_tool_executes_once_with_shallow_copy() -> None:
    calls: list[Mapping[str, Any]] = []

    def handler(arguments: Mapping[str, Any]) -> str:
        calls.append(arguments)
        arguments["city"] = "changed"  # type: ignore[index]
        return "sunny"

    original = {"city": "Kuala Lumpur", "nested": {"unit": "c"}}
    executor = AuthorizedToolExecutor((_definition(handler),))

    result = await executor.execute(" weather ", original, principal=_principal())

    assert result == "sunny"
    assert len(calls) == 1
    assert calls[0] is not original
    assert calls[0]["nested"] is original["nested"]
    assert original["city"] == "Kuala Lumpur"


async def test_allowed_async_tool_executes_once() -> None:
    calls = 0

    async def handler(arguments: Mapping[str, Any]) -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return str(arguments["city"])

    executor = AuthorizedToolExecutor((_definition(handler),))

    assert (
        await executor.execute(
            "weather",
            {"city": "Penang"},
            principal=_principal(),
        )
        == "Penang"
    )
    assert calls == 1


@pytest.mark.parametrize(
    "principal,scopes",
    [
        (None, {"tools:weather:execute"}),
        (Principal("user", scopes={"other"}), {"tools:weather:execute"}),
        (Principal("user", scopes={"tools:weather:execute"}), set()),
    ],
)
async def test_default_policy_denies_without_invoking_handler(
    principal: Principal | None,
    scopes: set[str],
) -> None:
    calls = 0

    def handler(arguments: Mapping[str, Any]) -> None:
        nonlocal calls
        calls += 1

    definition = ToolDefinition("weather", handler, required_scopes=scopes)
    executor = AuthorizedToolExecutor((definition,))

    with pytest.raises(AuthorizationError):
        await executor.execute("weather", {"credential": "secret"}, principal=principal)

    assert calls == 0


async def test_uses_active_principal_and_explicit_principal_takes_precedence() -> None:
    observed: list[str] = []

    class RecordingPolicy:
        async def authorize(
            self,
            principal: Principal | None,
            request: AuthorizationRequest,
        ) -> AuthorizationDecision:
            assert principal is not None
            observed.append(principal.subject)
            return AuthorizationDecision.allow("test_allow")

    executor = AuthorizedToolExecutor(
        (_definition(lambda arguments: "ok"),),
        policy=RecordingPolicy(),
    )

    with principal_context(Principal("context-user")):
        await executor.execute("weather", {})
        await executor.execute(
            "weather",
            {},
            principal=Principal("explicit-user"),
        )

    assert observed == ["context-user", "explicit-user"]


async def test_explicit_principal_is_bound_during_async_handler() -> None:
    observed: list[str | None] = []

    async def handler(arguments: Mapping[str, Any]) -> str:
        active = get_principal()
        observed.append(active.subject if active is not None else None)
        await asyncio.sleep(0)
        active = get_principal()
        observed.append(active.subject if active is not None else None)
        return "ok"

    executor = AuthorizedToolExecutor((_definition(handler),))
    outer = Principal("outer")
    explicit = _principal("explicit")

    with principal_context(outer):
        assert await executor.execute("weather", {}, principal=explicit) == "ok"
        assert get_principal() is outer

    assert observed == ["explicit", "explicit"]


async def test_policy_receives_only_authorization_metadata() -> None:
    recorded: list[tuple[Principal | None, AuthorizationRequest]] = []

    class RecordingPolicy:
        async def authorize(
            self,
            principal: Principal | None,
            request: AuthorizationRequest,
        ) -> AuthorizationDecision:
            recorded.append((principal, request))
            return AuthorizationDecision.allow("policy_allow")

    principal = Principal("private-subject")
    executor = AuthorizedToolExecutor(
        (_definition(lambda arguments: "private-result"),),
        policy=RecordingPolicy(),
    )

    await executor.execute(
        "weather",
        {"prompt": "private-prompt", "credential": "private-key"},
        principal=principal,
    )

    assert recorded == [
        (
            principal,
            AuthorizationRequest(
                action="tools.execute",
                resource="tool:weather",
                required_scopes={"tools:weather:execute"},
            ),
        )
    ]
    assert not hasattr(recorded[0][1], "arguments")


async def test_unknown_tool_error_is_safe_and_emits_no_span(
    in_memory_tracing,
) -> None:
    executor = AuthorizedToolExecutor((_definition(lambda arguments: None),))

    with pytest.raises(ToolExecutionError) as exc_info:
        await executor.execute("private-unknown-name", {"private": "value"})

    assert exc_info.value.code == "tool_not_found"
    assert exc_info.value.detail == "tool is not registered"
    assert "private" not in str(exc_info.value)
    assert in_memory_tracing.get_finished_spans() == ()


async def test_invalid_arguments_are_rejected_before_policy_or_handler() -> None:
    calls = 0

    class RecordingPolicy:
        async def authorize(
            self,
            principal: Principal | None,
            request: AuthorizationRequest,
        ) -> AuthorizationDecision:
            nonlocal calls
            calls += 1
            return AuthorizationDecision.allow()

    executor = AuthorizedToolExecutor(
        (_definition(lambda arguments: None),),
        policy=RecordingPolicy(),
    )

    with pytest.raises(TypeError, match="arguments must be a mapping"):
        await executor.execute("weather", ["not", "a", "mapping"])  # type: ignore[arg-type]

    assert calls == 0


async def test_policy_failures_and_invalid_results_never_invoke_handler() -> None:
    handler_calls = 0

    def handler(arguments: Mapping[str, Any]) -> None:
        nonlocal handler_calls
        handler_calls += 1

    class BrokenPolicy:
        async def authorize(
            self,
            principal: Principal | None,
            request: AuthorizationRequest,
        ) -> AuthorizationDecision:
            raise RuntimeError("private policy failure")

    class InvalidPolicy:
        async def authorize(
            self,
            principal: Principal | None,
            request: AuthorizationRequest,
        ) -> object:
            return object()

    for policy, expected in (
        (BrokenPolicy(), RuntimeError),
        (InvalidPolicy(), TypeError),
    ):
        executor = AuthorizedToolExecutor(
            (_definition(handler),),
            policy=policy,  # type: ignore[arg-type]
        )
        with pytest.raises(expected):
            await executor.execute("weather", {}, principal=_principal())

    assert handler_calls == 0


async def test_handler_failure_and_cancellation_propagate_unchanged() -> None:
    error = RuntimeError("private handler message")
    cancelled = asyncio.CancelledError()

    for raised in (error, cancelled):
        calls = 0

        def handler(
            arguments: Mapping[str, Any],
            error_to_raise: BaseException = raised,
        ) -> None:
            nonlocal calls
            calls += 1
            raise error_to_raise

        executor = AuthorizedToolExecutor((_definition(handler),))

        with pytest.raises(type(raised)) as exc_info:
            await executor.execute("weather", {}, principal=_principal())

        assert exc_info.value is raised
        assert calls == 1


async def test_concurrent_calls_keep_context_principals_isolated() -> None:
    observed: list[tuple[str, str]] = []
    ready = asyncio.Event()
    entered = 0

    class RecordingPolicy:
        async def authorize(
            self,
            principal: Principal | None,
            request: AuthorizationRequest,
        ) -> AuthorizationDecision:
            nonlocal entered
            assert principal is not None
            entered += 1
            if entered == 2:
                ready.set()
            await ready.wait()
            observed.append((principal.subject, request.resource))
            return AuthorizationDecision.allow("concurrent_allow")

    executor = AuthorizedToolExecutor(
        (_definition(lambda arguments: arguments["value"]),),
        policy=RecordingPolicy(),
    )

    async def run(subject: str) -> str:
        with principal_context(Principal(subject)):
            return await executor.execute("weather", {"value": subject})

    assert set(await asyncio.gather(run("first"), run("second"))) == {
        "first",
        "second",
    }
    assert set(observed) == {
        ("first", "tool:weather"),
        ("second", "tool:weather"),
    }


async def test_telemetry_and_logs_exclude_sensitive_values(
    in_memory_tracing,
    caplog,
) -> None:
    class DenyPolicy:
        async def authorize(
            self,
            principal: Principal | None,
            request: AuthorizationRequest,
        ) -> AuthorizationDecision:
            return AuthorizationDecision.deny("stable_denial")

    secrets = {
        "private-subject",
        "private-scope",
        "private-prompt",
        "private-result",
        "private-key",
    }
    definition = ToolDefinition(
        "weather",
        lambda arguments: "private-result",
        required_scopes={"private-scope"},
    )
    executor = AuthorizedToolExecutor((definition,), policy=DenyPolicy())

    with caplog.at_level(logging.INFO, logger="llmkit_lite.tools"):
        with pytest.raises(AuthorizationError):
            await executor.execute(
                "weather",
                {"prompt": "private-prompt", "credential": "private-key"},
                principal=Principal("private-subject"),
            )

    spans = in_memory_tracing.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "tool.execute"
    assert spans[0].status.status_code is StatusCode.ERROR
    assert spans[0].attributes == {
        "tool.name": "weather",
        "tool.outcome": "denied",
        "tool.reason_code": "stable_denial",
        "exception.type": "llmkit_lite.authorization.AuthorizationError",
    }
    serialized = repr(spans) + caplog.text
    for secret in secrets:
        assert secret not in serialized


async def test_success_telemetry_contains_only_safe_audit_fields(
    in_memory_tracing,
) -> None:
    executor = AuthorizedToolExecutor((_definition(lambda arguments: "secret"),))

    await executor.execute(
        "weather",
        {"prompt": "secret"},
        principal=_principal(),
    )

    span = in_memory_tracing.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.UNSET
    assert span.attributes == {
        "tool.name": "weather",
        "tool.outcome": "succeeded",
        "tool.reason_code": "handler_completed",
    }
    assert "secret" not in repr(span)
