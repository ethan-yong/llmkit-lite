import asyncio

import pytest

from llmkit_lite.authorization import (
    AuthorizationDecision,
    AuthorizationError,
    AuthorizationRequest,
    Principal,
    ScopeAuthorizationPolicy,
    get_principal,
    principal_context,
    require_authorization,
)


def _request(*scopes: str) -> AuthorizationRequest:
    return AuthorizationRequest(
        action="tools.execute",
        resource="tool:weather",
        required_scopes=set(scopes),
    )


def test_principal_normalizes_and_freezes_identity_grants() -> None:
    roles = {" operator ", "support"}
    scopes = {" tools:weather:execute ", "profile:read"}

    principal = Principal(" user-123 ", roles=roles, scopes=scopes)
    roles.add("admin")
    scopes.add("admin:all")

    assert principal.subject == "user-123"
    assert principal.roles == frozenset({"operator", "support"})
    assert principal.scopes == frozenset(
        {"tools:weather:execute", "profile:read"}
    )


def test_authorization_request_normalizes_and_freezes_values() -> None:
    required_scopes = {" tools:weather:execute "}

    request = AuthorizationRequest(
        action=" tools.execute ",
        resource=" tool:weather ",
        required_scopes=required_scopes,
    )
    required_scopes.add("other")

    assert request.action == "tools.execute"
    assert request.resource == "tool:weather"
    assert request.required_scopes == frozenset({"tools:weather:execute"})


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Principal(""),
        lambda: Principal("user", roles={" "}),
        lambda: Principal("user", scopes={" "}),
        lambda: AuthorizationRequest("", "resource"),
        lambda: AuthorizationRequest("action", " "),
        lambda: AuthorizationRequest("action", "resource", {""}),
        lambda: AuthorizationDecision.allow(" "),
    ],
)
def test_authorization_models_reject_empty_identifiers(factory) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        factory()


def test_authorization_models_reject_wrong_types() -> None:
    with pytest.raises(TypeError, match="principal subject must be a string"):
        Principal(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a set of strings"):
        Principal("user", scopes="scope")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a set of strings"):
        Principal("user", roles=["operator"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a boolean"):
        AuthorizationDecision(allowed=1, reason_code="invalid")  # type: ignore[arg-type]


def test_principal_context_defaults_to_none_and_restores_nested_values() -> None:
    outer = Principal("outer")
    inner = Principal("inner")

    assert get_principal() is None
    with principal_context(outer):
        assert get_principal() is outer
        with principal_context(inner):
            assert get_principal() is inner
        assert get_principal() is outer
    assert get_principal() is None


async def test_principal_context_isolates_concurrent_tasks() -> None:
    all_entered = asyncio.Event()
    entered = 0
    observed: list[Principal | None] = []

    async def capture(principal: Principal) -> None:
        nonlocal entered
        with principal_context(principal):
            entered += 1
            if entered == 2:
                all_entered.set()
            await all_entered.wait()
            observed.append(get_principal())

    first = Principal("first")
    second = Principal("second")
    await asyncio.gather(capture(first), capture(second))

    assert set(observed) == {first, second}
    assert get_principal() is None


async def test_scope_policy_requires_an_authenticated_principal() -> None:
    decision = await ScopeAuthorizationPolicy().authorize(
        None,
        _request("tools:weather:execute"),
    )

    assert decision == AuthorizationDecision.deny("authentication_required")


async def test_scope_policy_denies_missing_scope_requirement() -> None:
    decision = await ScopeAuthorizationPolicy().authorize(
        Principal("user", scopes={"tools:weather:execute"}),
        _request(),
    )

    assert decision == AuthorizationDecision.deny("scope_requirement_missing")


async def test_scope_policy_requires_every_declared_scope() -> None:
    policy = ScopeAuthorizationPolicy()
    request = _request("tools:weather:execute", "location:read")

    denied = await policy.authorize(
        Principal("user", scopes={"tools:weather:execute"}),
        request,
    )
    allowed = await policy.authorize(
        Principal(
            "user",
            scopes={"tools:weather:execute", "location:read", "profile:read"},
        ),
        request,
    )

    assert denied == AuthorizationDecision.deny("missing_required_scope")
    assert allowed == AuthorizationDecision.allow("required_scopes_present")


async def test_require_authorization_returns_allowed_custom_policy_decision() -> None:
    principal = Principal("user")
    request = _request("custom")

    class RecordingPolicy:
        def __init__(self) -> None:
            self.calls: list[tuple[Principal | None, AuthorizationRequest]] = []

        async def authorize(
            self,
            received_principal: Principal | None,
            received_request: AuthorizationRequest,
        ) -> AuthorizationDecision:
            self.calls.append((received_principal, received_request))
            return AuthorizationDecision.allow("custom_allow")

    policy = RecordingPolicy()

    decision = await require_authorization(
        policy,
        request,
        principal=principal,
    )

    assert decision == AuthorizationDecision.allow("custom_allow")
    assert policy.calls == [(principal, request)]


@pytest.mark.parametrize(
    "principal,expected_code",
    [
        (None, "authentication_required"),
        (Principal("private-user"), "authorization_denied"),
    ],
)
async def test_require_authorization_raises_safe_denial_errors(
    principal: Principal | None,
    expected_code: str,
) -> None:
    class DenyPolicy:
        async def authorize(
            self,
            principal: Principal | None,
            request: AuthorizationRequest,
        ) -> AuthorizationDecision:
            return AuthorizationDecision.deny("private-policy-reason")

    with pytest.raises(AuthorizationError) as exc_info:
        await require_authorization(
            DenyPolicy(),
            AuthorizationRequest(
                "private-action",
                "private-resource",
                {"private-scope"},
            ),
            principal=principal,
        )

    assert exc_info.value.code == expected_code
    assert exc_info.value.reason_code == "private-policy-reason"
    assert "private" not in str(exc_info.value)


async def test_require_authorization_rejects_invalid_policy_results() -> None:
    class InvalidPolicy:
        async def authorize(
            self,
            principal: Principal | None,
            request: AuthorizationRequest,
        ) -> object:
            return object()

    with pytest.raises(
        TypeError,
        match="authorization policy must return AuthorizationDecision",
    ):
        await require_authorization(
            InvalidPolicy(),  # type: ignore[arg-type]
            _request("scope"),
            principal=Principal("user"),
        )


async def test_require_authorization_propagates_policy_failures() -> None:
    class BrokenPolicy:
        async def authorize(
            self,
            principal: Principal | None,
            request: AuthorizationRequest,
        ) -> AuthorizationDecision:
            raise RuntimeError("policy unavailable")

    with pytest.raises(RuntimeError, match="policy unavailable"):
        await require_authorization(
            BrokenPolicy(),
            _request("scope"),
            principal=Principal("user"),
        )
