"""Authorization context and policy primitives for protected operations."""

from __future__ import annotations

import contextvars
from collections.abc import Iterator, Set
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Protocol


def _normalize_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _normalize_identifiers(
    values: Set[str],
    field_name: str,
) -> frozenset[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Set):
        raise TypeError(f"{field_name} must be a set of strings")
    return frozenset(
        _normalize_identifier(value, f"{field_name} item") for value in values
    )


@dataclass(frozen=True, slots=True)
class Principal:
    """Authenticated identity and its application-level grants."""

    subject: str
    roles: Set[str] = field(default_factory=frozenset)
    scopes: Set[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "subject",
            _normalize_identifier(self.subject, "principal subject"),
        )
        object.__setattr__(
            self,
            "roles",
            _normalize_identifiers(self.roles, "principal roles"),
        )
        object.__setattr__(
            self,
            "scopes",
            _normalize_identifiers(self.scopes, "principal scopes"),
        )


@dataclass(frozen=True, slots=True)
class AuthorizationRequest:
    """Permission required to perform one protected operation."""

    action: str
    resource: str
    required_scopes: Set[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action",
            _normalize_identifier(self.action, "authorization action"),
        )
        object.__setattr__(
            self,
            "resource",
            _normalize_identifier(self.resource, "authorization resource"),
        )
        object.__setattr__(
            self,
            "required_scopes",
            _normalize_identifiers(
                self.required_scopes,
                "authorization required scopes",
            ),
        )


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    """Stable result returned by an authorization policy."""

    allowed: bool
    reason_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.allowed, bool):
            raise TypeError("authorization allowed must be a boolean")
        object.__setattr__(
            self,
            "reason_code",
            _normalize_identifier(self.reason_code, "authorization reason code"),
        )

    @classmethod
    def allow(cls, reason_code: str = "allowed") -> AuthorizationDecision:
        """Build an allowed decision."""

        return cls(allowed=True, reason_code=reason_code)

    @classmethod
    def deny(cls, reason_code: str = "policy_denied") -> AuthorizationDecision:
        """Build a denied decision."""

        return cls(allowed=False, reason_code=reason_code)


class AuthorizationPolicy(Protocol):
    """Asynchronous policy boundary for authorization decisions."""

    async def authorize(
        self,
        principal: Principal | None,
        request: AuthorizationRequest,
    ) -> AuthorizationDecision:
        """Return an allow or deny decision for one operation."""


class ScopeAuthorizationPolicy:
    """Require an authenticated principal to hold every declared scope."""

    async def authorize(
        self,
        principal: Principal | None,
        request: AuthorizationRequest,
    ) -> AuthorizationDecision:
        if principal is None:
            return AuthorizationDecision.deny("authentication_required")
        if not request.required_scopes:
            return AuthorizationDecision.deny("scope_requirement_missing")
        if request.required_scopes.issubset(principal.scopes):
            return AuthorizationDecision.allow("required_scopes_present")
        return AuthorizationDecision.deny("missing_required_scope")


class AuthorizationError(Exception):
    """Safe authorization failure suitable for boundary error mapping."""

    def __init__(self, code: str, reason_code: str) -> None:
        normalized_code = _normalize_identifier(code, "authorization error code")
        normalized_reason = _normalize_identifier(
            reason_code,
            "authorization error reason code",
        )
        detail = (
            "authentication is required"
            if normalized_code == "authentication_required"
            else "authorization was denied"
        )
        super().__init__(detail)
        self.code = normalized_code
        self.detail = detail
        self.reason_code = normalized_reason


_principal: contextvars.ContextVar[Principal | None] = contextvars.ContextVar(
    "llmkit_principal",
    default=None,
)


def get_principal() -> Principal | None:
    """Return the principal bound to the current execution, if present."""

    return _principal.get()


@contextmanager
def principal_context(principal: Principal) -> Iterator[Principal]:
    """Bind a principal for one synchronous or asynchronous execution scope."""

    if not isinstance(principal, Principal):
        raise TypeError("principal must be a Principal")
    token = _principal.set(principal)
    try:
        yield principal
    finally:
        _principal.reset(token)


async def require_authorization(
    policy: AuthorizationPolicy,
    request: AuthorizationRequest,
    *,
    principal: Principal | None,
) -> AuthorizationDecision:
    """Require an allowed policy decision or raise a safe authorization error."""

    decision = await policy.authorize(principal, request)
    if not isinstance(decision, AuthorizationDecision):
        raise TypeError("authorization policy must return AuthorizationDecision")
    if decision.allowed:
        return decision
    code = "authentication_required" if principal is None else "authorization_denied"
    raise AuthorizationError(code, decision.reason_code)
