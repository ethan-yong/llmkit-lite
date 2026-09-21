"""Process-local token and monetary budget reservations."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal, localcontext
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from llmkit_lite.llm import TokenUsage
from llmkit_lite.observability import set_span_error, trace_span

_MILLION = Decimal(1_000_000)
_ERROR_DETAILS = {
    "budget_price_unavailable": "model price is not configured",
    "budget_currency_mismatch": "model price currency does not match budget",
    "budget_reservation_not_active": "budget reservation is no longer active",
    "budget_attempt_already_started": "budget attempt was already started",
    "budget_attempt_not_started": "budget attempt has not started",
    "budget_cancel_forbidden": "attempted call cannot be cancelled",
    "budget_token_conflict": "budget reservation token is already used",
}


def _identifier(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def _currency(value: str) -> str:
    normalized = _identifier(value, "budget currency").upper()
    if len(normalized) != 3 or not normalized.isascii() or not normalized.isalpha():
        raise ValueError("budget currency must be a three-letter ASCII code")
    return normalized


def _count(value: int, field: str, *, positive: bool = False) -> int:
    lower_bound = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < lower_bound:
        requirement = "greater than zero" if positive else "non-negative"
        raise ValueError(f"{field} must be an integer {requirement}")
    return value


def _money(value: Decimal, field: str, *, positive: bool = False) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise TypeError(f"{field} must be a finite Decimal")
    if value < 0 or (positive and value == 0):
        requirement = "greater than zero" if positive else "non-negative"
        raise ValueError(f"{field} must be {requirement}")
    return value


def _price_for_tokens(
    input_tokens: int,
    output_tokens: int,
    price: ModelPrice,
) -> Decimal:
    # Precision is local, so callers' Decimal settings cannot round a charge.
    rate_digits = max(
        len(price.input_per_million.as_tuple().digits),
        len(price.output_per_million.as_tuple().digits),
    )
    token_digits = len(str(max(input_tokens, output_tokens, 1)))
    with localcontext() as context:
        context.prec = max(28, rate_digits + token_digits + 4)
        return (
            input_tokens * price.input_per_million
            + output_tokens * price.output_per_million
        ) / _MILLION


def _money_sum(left: Decimal, right: Decimal) -> Decimal:
    """Add monetary values without ambient Decimal precision loss."""

    common_exponent = min(left.as_tuple().exponent, right.as_tuple().exponent)
    highest_place = max(left.adjusted(), right.adjusted(), 0)
    with localcontext() as context:
        context.prec = max(28, highest_place - common_exponent + 3)
        return left + right


class BudgetError(Exception):
    """Safe budget failure suitable for application error mapping."""

    def __init__(self, code: str) -> None:
        if code not in _ERROR_DETAILS:
            raise ValueError("unsupported budget error code")
        detail = _ERROR_DETAILS[code]
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """Shared per-tenant token and monetary quota limits."""

    token_limit: int
    cost_limit: Decimal
    currency: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "token_limit",
            _count(self.token_limit, "token limit", positive=True),
        )
        object.__setattr__(
            self,
            "cost_limit",
            _money(self.cost_limit, "cost limit", positive=True),
        )
        object.__setattr__(self, "currency", _currency(self.currency))


@dataclass(frozen=True, slots=True)
class BudgetKey:
    """Tenant and caller-supplied token/cost accounting periods."""

    tenant_id: str
    token_period_id: str
    cost_period_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _identifier(self.tenant_id, "tenant ID"))
        object.__setattr__(
            self,
            "token_period_id",
            _identifier(self.token_period_id, "token period ID"),
        )
        object.__setattr__(
            self,
            "cost_period_id",
            _identifier(self.cost_period_id, "cost period ID"),
        )


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Configured price per million input and output tokens."""

    provider: str
    model: str
    input_per_million: Decimal
    output_per_million: Decimal
    currency: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _identifier(self.provider, "provider"))
        object.__setattr__(self, "model", _identifier(self.model, "model"))
        object.__setattr__(
            self,
            "input_per_million",
            _money(self.input_per_million, "input token price"),
        )
        object.__setattr__(
            self,
            "output_per_million",
            _money(self.output_per_million, "output token price"),
        )
        object.__setattr__(self, "currency", _currency(self.currency))


class ModelPriceTable:
    """Immutable lookup of application-configured model prices."""

    def __init__(self, prices: Sequence[ModelPrice]) -> None:
        if isinstance(prices, (str, bytes)) or not isinstance(prices, Sequence):
            raise TypeError("model prices must be a sequence")
        entries: dict[tuple[str, str], ModelPrice] = {}
        for price in prices:
            if not isinstance(price, ModelPrice):
                raise TypeError("model prices must contain ModelPrice values")
            key = (price.provider, price.model)
            if key in entries:
                raise ValueError("duplicate model price")
            entries[key] = price
        self._prices = MappingProxyType(entries)

    def get(self, provider: str, model: str) -> ModelPrice:
        """Return one configured price or a stable safe error."""

        key = (_identifier(provider, "provider"), _identifier(model, "model"))
        try:
            return self._prices[key]
        except KeyError:
            raise BudgetError("budget_price_unavailable") from None


class BudgetOutcome(StrEnum):
    """Stable outcome of an atomic dual-quota reservation."""

    RESERVED = "reserved"
    TOKEN_LIMIT = "budget_token_limit"
    COST_LIMIT = "budget_cost_limit"


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    """Opaque token and immutable estimate for one provider attempt."""

    token: str
    key: BudgetKey
    provider: str
    model: str
    reserved_tokens: int
    reserved_cost: Decimal
    _ledger_id: str = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "token", _identifier(self.token, "reservation token"))
        if not isinstance(self.key, BudgetKey):
            raise TypeError("reservation key must be a BudgetKey")
        object.__setattr__(self, "provider", _identifier(self.provider, "provider"))
        object.__setattr__(self, "model", _identifier(self.model, "model"))
        object.__setattr__(
            self,
            "reserved_tokens",
            _count(self.reserved_tokens, "reserved tokens", positive=True),
        )
        object.__setattr__(
            self,
            "reserved_cost",
            _money(self.reserved_cost, "reserved cost"),
        )
        object.__setattr__(
            self,
            "_ledger_id",
            _identifier(self._ledger_id, "budget ledger ID"),
        )


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """A reservation or stable token/cost quota denial."""

    outcome: BudgetOutcome
    reservation: BudgetReservation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, BudgetOutcome):
            raise TypeError("budget outcome must be a BudgetOutcome")
        if (self.outcome is BudgetOutcome.RESERVED) != isinstance(
            self.reservation, BudgetReservation
        ):
            raise ValueError("only a reserved decision may contain a reservation")

    @property
    def allowed(self) -> bool:
        """Return whether both quotas were reserved."""

        return self.outcome is BudgetOutcome.RESERVED


@dataclass(frozen=True, slots=True)
class BudgetSettlement:
    """Actual or conservative fallback charge for one attempted call."""

    reservation: BudgetReservation
    charged_tokens: int
    charged_cost: Decimal
    usage_reported: bool

    def __post_init__(self) -> None:
        if not isinstance(self.reservation, BudgetReservation):
            raise TypeError("settlement reservation must be a BudgetReservation")
        object.__setattr__(
            self,
            "charged_tokens",
            _count(self.charged_tokens, "charged tokens"),
        )
        object.__setattr__(
            self,
            "charged_cost",
            _money(self.charged_cost, "charged cost"),
        )
        if not isinstance(self.usage_reported, bool):
            raise TypeError("usage reported must be a boolean")


class BudgetLedger(Protocol):
    """Atomic reservation and settlement boundary for token/cost quotas."""

    async def reserve(
        self,
        key: BudgetKey,
        *,
        provider: str,
        model: str,
        estimated_input_tokens: int,
        max_output_tokens: int,
    ) -> BudgetDecision:
        """Reserve estimated tokens and cost against both periods."""

    async def mark_attempted(self, reservation: BudgetReservation) -> None:
        """Mark that a provider call has started and can no longer be cancelled."""

    async def reconcile(
        self,
        reservation: BudgetReservation,
        usage: TokenUsage | None,
    ) -> BudgetSettlement:
        """Finalize a provider attempt with actual or reserved usage."""

    async def cancel(self, reservation: BudgetReservation) -> None:
        """Refund a reservation only if no provider call was attempted."""


@dataclass(slots=True)
class _PendingReservation:
    reservation: BudgetReservation
    price: ModelPrice
    attempted: bool = False


class InMemoryBudgetLedger:
    """Atomic process-local budget ledger for tests and local development."""

    def __init__(
        self,
        policy: BudgetPolicy,
        prices: ModelPriceTable,
        *,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(policy, BudgetPolicy):
            raise TypeError("budget policy must be a BudgetPolicy")
        if not isinstance(prices, ModelPriceTable):
            raise TypeError("model prices must be a ModelPriceTable")
        resolved_factory = token_factory if token_factory is not None else _new_token
        if not callable(resolved_factory):
            raise TypeError("budget token factory must be callable")
        self.policy = policy
        self.prices = prices
        self.token_factory = resolved_factory
        self._ledger_id = uuid.uuid4().hex
        self._token_usage: dict[tuple[str, str], int] = {}
        self._cost_usage: dict[tuple[str, str], Decimal] = {}
        self._pending: dict[str, _PendingReservation] = {}
        self._issued_tokens: set[str] = set()
        self._lock = asyncio.Lock()

    async def reserve(
        self,
        key: BudgetKey,
        *,
        provider: str,
        model: str,
        estimated_input_tokens: int,
        max_output_tokens: int,
    ) -> BudgetDecision:
        """Atomically reserve both quotas before an individual provider call."""

        if not isinstance(key, BudgetKey):
            raise TypeError("budget key must be a BudgetKey")
        input_count = _count(estimated_input_tokens, "estimated input tokens")
        output_count = _count(max_output_tokens, "maximum output tokens")
        if input_count + output_count == 0:
            raise ValueError("estimated token count must be greater than zero")
        price = self.prices.get(provider, model)
        if price.currency != self.policy.currency:
            raise BudgetError("budget_currency_mismatch")
        reserved_tokens = input_count + output_count
        reserved_cost = _price_for_tokens(input_count, output_count, price)
        token_key = (key.tenant_id, key.token_period_id)
        cost_key = (key.tenant_id, key.cost_period_id)

        with trace_span(
            "budget.reserve",
            attributes={"budget.estimated_tokens": reserved_tokens},
        ) as span:
            async with self._lock:
                token_total = self._token_usage.get(token_key, 0)
                cost_total = self._cost_usage.get(cost_key, Decimal(0))
                if token_total + reserved_tokens > self.policy.token_limit:
                    decision = BudgetDecision(BudgetOutcome.TOKEN_LIMIT)
                elif _money_sum(cost_total, reserved_cost) > self.policy.cost_limit:
                    decision = BudgetDecision(BudgetOutcome.COST_LIMIT)
                else:
                    token = _identifier(
                        self.token_factory(), "budget reservation token"
                    )
                    if token in self._issued_tokens:
                        error = BudgetError("budget_token_conflict")
                        _record_failure(span, error.code)
                        raise error
                    reservation = BudgetReservation(
                        token=token,
                        key=key,
                        provider=price.provider,
                        model=price.model,
                        reserved_tokens=reserved_tokens,
                        reserved_cost=reserved_cost,
                        _ledger_id=self._ledger_id,
                    )
                    self._issued_tokens.add(token)
                    self._pending[token] = _PendingReservation(reservation, price)
                    self._token_usage[token_key] = token_total + reserved_tokens
                    self._cost_usage[cost_key] = _money_sum(cost_total, reserved_cost)
                    decision = BudgetDecision(BudgetOutcome.RESERVED, reservation)

            if span is not None:
                span.set_attribute("budget.outcome", decision.outcome.value)
            return decision

    async def mark_attempted(self, reservation: BudgetReservation) -> None:
        """Prevent refund once the provider attempt begins."""

        with trace_span("budget.mark_attempted") as span:
            async with self._lock:
                pending = self._require_pending(reservation, span)
                if pending.attempted:
                    error = BudgetError("budget_attempt_already_started")
                    _record_failure(span, error.code)
                    raise error
                pending.attempted = True
            if span is not None:
                span.set_attribute("budget.outcome", "attempted")

    async def reconcile(
        self,
        reservation: BudgetReservation,
        usage: TokenUsage | None,
    ) -> BudgetSettlement:
        """Charge actual usage, or reserved usage when none was returned."""

        if usage is not None and not isinstance(usage, TokenUsage):
            raise TypeError("budget usage must be TokenUsage or None")
        with trace_span("budget.reconcile") as span:
            async with self._lock:
                pending = self._require_pending(reservation, span)
                if not pending.attempted:
                    error = BudgetError("budget_attempt_not_started")
                    _record_failure(span, error.code)
                    raise error
                if usage is None:
                    charged_tokens = reservation.reserved_tokens
                    charged_cost = reservation.reserved_cost
                else:
                    charged_tokens = usage.total_tokens
                    charged_cost = _price_for_tokens(
                        usage.input_tokens,
                        usage.output_tokens,
                        pending.price,
                    )
                token_key = (
                    reservation.key.tenant_id,
                    reservation.key.token_period_id,
                )
                cost_key = (
                    reservation.key.tenant_id,
                    reservation.key.cost_period_id,
                )
                self._token_usage[token_key] += (
                    charged_tokens - reservation.reserved_tokens
                )
                self._cost_usage[cost_key] = _money_sum(
                    self._cost_usage[cost_key],
                    _money_sum(charged_cost, -reservation.reserved_cost),
                )
                del self._pending[reservation.token]
                settlement = BudgetSettlement(
                    reservation=reservation,
                    charged_tokens=charged_tokens,
                    charged_cost=charged_cost,
                    usage_reported=usage is not None,
                )
            if span is not None:
                span.set_attribute("budget.outcome", "reconciled")
                span.set_attribute("budget.charged_tokens", charged_tokens)
                span.set_attribute("budget.usage_reported", usage is not None)
            return settlement

    async def cancel(self, reservation: BudgetReservation) -> None:
        """Release capacity only for a provider call not yet attempted."""

        with trace_span("budget.cancel") as span:
            async with self._lock:
                pending = self._require_pending(reservation, span)
                if pending.attempted:
                    error = BudgetError("budget_cancel_forbidden")
                    _record_failure(span, error.code)
                    raise error
                token_key = (
                    reservation.key.tenant_id,
                    reservation.key.token_period_id,
                )
                cost_key = (
                    reservation.key.tenant_id,
                    reservation.key.cost_period_id,
                )
                self._token_usage[token_key] -= reservation.reserved_tokens
                self._cost_usage[cost_key] = _money_sum(
                    self._cost_usage[cost_key],
                    -reservation.reserved_cost,
                )
                del self._pending[reservation.token]
            if span is not None:
                span.set_attribute("budget.outcome", "cancelled")

    def _require_pending(
        self,
        reservation: BudgetReservation,
        span,
    ) -> _PendingReservation:
        if not isinstance(reservation, BudgetReservation):
            raise TypeError("reservation must be a BudgetReservation")
        pending = self._pending.get(reservation.token)
        if (
            reservation._ledger_id != self._ledger_id
            or pending is None
            or pending.reservation != reservation
        ):
            error = BudgetError("budget_reservation_not_active")
            _record_failure(span, error.code)
            raise error
        return pending


def _new_token() -> str:
    return uuid.uuid4().hex


def _record_failure(span, code: str) -> None:
    if span is not None:
        span.set_attribute("budget.outcome", "failed")
        span.set_attribute("error.type", code)
    set_span_error(span, code)
