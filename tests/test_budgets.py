from __future__ import annotations

import asyncio
from dataclasses import replace
from decimal import Decimal, localcontext

import pytest
from opentelemetry.trace import StatusCode

from llmkit_lite.budgets import (
    BudgetError,
    BudgetKey,
    BudgetOutcome,
    BudgetPolicy,
    InMemoryBudgetLedger,
    ModelPrice,
    ModelPriceTable,
)
from llmkit_lite.llm import TokenUsage


def _key(
    tenant: str = "tenant-1",
    token_period: str = "2026-09-20",
    cost_period: str = "2026-09",
) -> BudgetKey:
    return BudgetKey(tenant, token_period, cost_period)


def _prices() -> ModelPriceTable:
    return ModelPriceTable(
        (
            ModelPrice(
                "provider-a",
                "model-a",
                Decimal("2.50"),
                Decimal("7.50"),
                "MYR",
            ),
            ModelPrice(
                "provider-b",
                "model-b",
                Decimal("1.00"),
                Decimal("3.00"),
                "MYR",
            ),
        )
    )


def _ledger(
    *,
    token_limit: int = 100,
    cost_limit: str = "1.00",
    token_factory=None,
) -> InMemoryBudgetLedger:
    return InMemoryBudgetLedger(
        BudgetPolicy(token_limit, Decimal(cost_limit), "MYR"),
        _prices(),
        token_factory=token_factory,
    )


async def _reserve(
    ledger: InMemoryBudgetLedger,
    key: BudgetKey | None = None,
    *,
    input_tokens: int = 10,
    output_tokens: int = 10,
    provider: str = "provider-a",
    model: str = "model-a",
):
    return await ledger.reserve(
        key or _key(),
        provider=provider,
        model=model,
        estimated_input_tokens=input_tokens,
        max_output_tokens=output_tokens,
    )


async def test_given_estimate_when_reserving_then_tokens_and_cost_are_exact() -> None:
    ledger = _ledger()

    decision = await _reserve(ledger, input_tokens=3, output_tokens=7)

    assert decision.outcome is BudgetOutcome.RESERVED
    assert decision.allowed is True
    assert decision.reservation is not None
    assert decision.reservation.reserved_tokens == 10
    assert decision.reservation.reserved_cost == Decimal("0.000060")


async def test_given_token_limit_when_reserving_then_cost_is_unchanged() -> None:
    ledger = _ledger(token_limit=10, cost_limit="0.0001")
    first = await _reserve(ledger, input_tokens=5, output_tokens=5)

    denied = await _reserve(ledger, input_tokens=5, output_tokens=5)

    assert first.allowed is True
    assert denied.outcome is BudgetOutcome.TOKEN_LIMIT
    assert denied.reservation is None
    assert first.reservation is not None
    await ledger.cancel(first.reservation)
    assert (await _reserve(ledger, input_tokens=5, output_tokens=5)).allowed


async def test_given_cost_limit_when_reserving_then_tokens_are_unchanged() -> None:
    ledger = _ledger(token_limit=100, cost_limit="0.000045")

    denied = await _reserve(ledger, input_tokens=5, output_tokens=5)
    allowed = await _reserve(ledger, input_tokens=5, output_tokens=4)

    assert denied.outcome is BudgetOutcome.COST_LIMIT
    assert allowed.allowed is True
    assert allowed.reservation is not None
    assert allowed.reservation.reserved_tokens == 9


async def test_given_concurrent_reservations_then_quotas_are_atomic() -> None:
    ledger = _ledger(token_limit=30, cost_limit="0.0002")

    decisions = await asyncio.gather(*(_reserve(ledger) for _ in range(20)))

    assert sum(decision.allowed for decision in decisions) == 1
    assert (
        sum(decision.outcome is BudgetOutcome.TOKEN_LIMIT for decision in decisions)
        == 19
    )


async def test_given_cost_pressure_then_concurrent_reservations_are_atomic() -> None:
    ledger = _ledger(token_limit=100, cost_limit="0.00011")

    decisions = await asyncio.gather(*(_reserve(ledger) for _ in range(20)))

    assert sum(decision.allowed for decision in decisions) == 1
    assert (
        sum(decision.outcome is BudgetOutcome.COST_LIMIT for decision in decisions)
        == 19
    )


async def test_given_distinct_tenants_and_periods_when_reserving_then_independent() -> (
    None
):
    ledger = _ledger(token_limit=20, cost_limit="0.0001")
    keys = (
        _key(),
        _key("tenant-2"),
        _key(token_period="2026-09-21", cost_period="2026-10"),
    )

    decisions = await asyncio.gather(*(_reserve(ledger, key) for key in keys))

    assert all(decision.allowed for decision in decisions)


async def test_given_new_token_period_then_cost_period_stays_shared() -> None:
    ledger = _ledger(token_limit=20, cost_limit="0.00011")
    first = await _reserve(ledger)

    second = await _reserve(ledger, _key(token_period="2026-09-21"))

    assert first.allowed is True
    assert second.outcome is BudgetOutcome.COST_LIMIT


async def test_given_new_cost_period_then_token_period_stays_shared() -> None:
    ledger = _ledger(token_limit=20)
    first = await _reserve(ledger)

    second = await _reserve(ledger, _key(cost_period="2026-10"))

    assert first.allowed is True
    assert second.outcome is BudgetOutcome.TOKEN_LIMIT


async def test_given_reported_usage_then_unused_capacity_is_refunded() -> None:
    ledger = _ledger(token_limit=20, cost_limit="0.0001")
    first = await _reserve(ledger)
    assert first.reservation is not None
    await ledger.mark_attempted(first.reservation)

    settled = await ledger.reconcile(
        first.reservation,
        TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5),
    )
    second = await _reserve(ledger, input_tokens=10, output_tokens=5)

    assert settled.charged_tokens == 5
    assert settled.charged_cost == Decimal("0.0000275")
    assert settled.usage_reported is True
    assert second.allowed is True


async def test_given_actual_overrun_then_future_requests_are_denied() -> None:
    ledger = _ledger(token_limit=20, cost_limit="0.0001")
    first = await _reserve(ledger, input_tokens=5, output_tokens=5)
    assert first.reservation is not None
    await ledger.mark_attempted(first.reservation)

    settled = await ledger.reconcile(
        first.reservation,
        TokenUsage(input_tokens=15, output_tokens=10, total_tokens=25),
    )
    denied = await _reserve(ledger, input_tokens=1, output_tokens=0)

    assert settled.charged_tokens == 25
    assert settled.charged_cost == Decimal("0.0001125")
    assert denied.outcome is BudgetOutcome.TOKEN_LIMIT


async def test_given_cost_overrun_when_reconciling_then_future_cost_is_denied() -> None:
    ledger = _ledger(token_limit=100, cost_limit="0.00005")
    first = await _reserve(ledger, input_tokens=10, output_tokens=0)
    assert first.reservation is not None
    await ledger.mark_attempted(first.reservation)

    await ledger.reconcile(
        first.reservation,
        TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
    )
    denied = await _reserve(ledger, input_tokens=1, output_tokens=0)

    assert denied.outcome is BudgetOutcome.COST_LIMIT


async def test_given_missing_usage_when_reconciling_then_reservation_is_charged() -> (
    None
):
    ledger = _ledger(token_limit=20, cost_limit="0.0001")
    decision = await _reserve(ledger)
    assert decision.reservation is not None
    await ledger.mark_attempted(decision.reservation)

    settled = await ledger.reconcile(decision.reservation, None)
    denied = await _reserve(ledger)

    assert settled.charged_tokens == decision.reservation.reserved_tokens
    assert settled.charged_cost == decision.reservation.reserved_cost
    assert settled.usage_reported is False
    assert denied.outcome is BudgetOutcome.TOKEN_LIMIT


async def test_given_zero_input_estimate_when_output_is_positive_then_allowed() -> None:
    ledger = _ledger()

    decision = await _reserve(ledger, input_tokens=0, output_tokens=5)

    assert decision.allowed is True
    assert decision.reservation is not None
    assert decision.reservation.reserved_tokens == 5


async def test_given_zero_total_estimate_when_reserving_then_rejected() -> None:
    ledger = _ledger()

    with pytest.raises(ValueError, match="estimated token count"):
        await _reserve(ledger, input_tokens=0, output_tokens=0)


async def test_given_unattempted_call_then_both_quotas_are_refunded() -> None:
    ledger = _ledger(token_limit=20, cost_limit="0.0001")
    decision = await _reserve(ledger)
    assert decision.reservation is not None

    await ledger.cancel(decision.reservation)

    assert (await _reserve(ledger)).allowed


async def test_given_attempted_call_when_cancelling_then_refund_is_forbidden() -> None:
    ledger = _ledger()
    decision = await _reserve(ledger)
    assert decision.reservation is not None
    await ledger.mark_attempted(decision.reservation)

    with pytest.raises(BudgetError) as exc_info:
        await ledger.cancel(decision.reservation)

    assert exc_info.value.code == "budget_cancel_forbidden"
    await ledger.reconcile(decision.reservation, None)


async def test_given_unattempted_call_when_reconciling_then_rejected() -> None:
    ledger = _ledger()
    decision = await _reserve(ledger)
    assert decision.reservation is not None

    with pytest.raises(BudgetError) as exc_info:
        await ledger.reconcile(decision.reservation, None)

    assert exc_info.value.code == "budget_attempt_not_started"


async def test_given_duplicate_attempt_or_settlement_when_repeating_then_rejected() -> (
    None
):
    ledger = _ledger()
    decision = await _reserve(ledger)
    assert decision.reservation is not None
    await ledger.mark_attempted(decision.reservation)

    with pytest.raises(BudgetError) as attempt_info:
        await ledger.mark_attempted(decision.reservation)
    await ledger.reconcile(decision.reservation, None)
    with pytest.raises(BudgetError) as settle_info:
        await ledger.reconcile(decision.reservation, None)

    assert attempt_info.value.code == "budget_attempt_already_started"
    assert settle_info.value.code == "budget_reservation_not_active"


async def test_given_foreign_or_modified_reservation_when_using_then_rejected() -> None:
    source = _ledger(token_factory=lambda: "same-token")
    target = _ledger(token_factory=lambda: "same-token")
    source_decision = await _reserve(source)
    target_decision = await _reserve(target)
    assert source_decision.reservation is not None
    assert target_decision.reservation is not None

    with pytest.raises(BudgetError) as foreign_info:
        await target.mark_attempted(source_decision.reservation)
    with pytest.raises(BudgetError) as modified_info:
        await source.cancel(replace(source_decision.reservation, reserved_tokens=1))

    assert foreign_info.value.code == "budget_reservation_not_active"
    assert modified_info.value.code == "budget_reservation_not_active"


async def test_given_duplicate_token_when_reserving_then_safe_error_is_raised() -> None:
    ledger = _ledger(token_factory=lambda: "same-token")
    first = await _reserve(ledger)
    assert first.reservation is not None
    await ledger.cancel(first.reservation)

    with pytest.raises(BudgetError) as exc_info:
        await _reserve(ledger)

    assert exc_info.value.code == "budget_token_conflict"


async def test_given_new_period_then_old_attempt_settles_original_period() -> None:
    ledger = _ledger(token_limit=20, cost_limit="0.0001")
    old = await _reserve(ledger)
    assert old.reservation is not None
    await ledger.mark_attempted(old.reservation)
    new = await _reserve(
        ledger,
        _key(token_period="2026-09-21", cost_period="2026-10"),
    )

    settled = await ledger.reconcile(
        old.reservation,
        TokenUsage(input_tokens=10, output_tokens=10, total_tokens=20),
    )

    assert new.allowed is True
    assert settled.reservation.key == _key()


async def test_given_second_model_when_reserving_then_configured_price_is_used() -> (
    None
):
    ledger = _ledger()

    decision = await _reserve(
        ledger,
        input_tokens=3,
        output_tokens=7,
        provider="provider-b",
        model="model-b",
    )

    assert decision.reservation is not None
    assert decision.reservation.reserved_cost == Decimal("0.000024")


async def test_given_low_decimal_precision_then_reservation_price_stays_exact() -> None:
    ledger = _ledger()

    with localcontext() as context:
        context.prec = 3
        decision = await _reserve(ledger, input_tokens=3, output_tokens=7)

    assert decision.reservation is not None
    assert decision.reservation.reserved_cost == Decimal("0.000060")


async def test_given_missing_price_or_currency_mismatch_then_safe_error() -> None:
    ledger = _ledger()
    with pytest.raises(BudgetError) as missing_info:
        await _reserve(ledger, provider="unknown", model="model")

    mismatched = InMemoryBudgetLedger(
        BudgetPolicy(100, Decimal("1"), "USD"),
        _prices(),
    )
    with pytest.raises(BudgetError) as currency_info:
        await _reserve(mismatched)

    assert missing_info.value.code == "budget_price_unavailable"
    assert currency_info.value.code == "budget_currency_mismatch"


@pytest.mark.parametrize(
    "value",
    [Decimal("NaN"), Decimal("Infinity"), Decimal("-1"), 1.0, "1.0"],
)
def test_given_invalid_money_when_configuring_then_rejected(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        BudgetPolicy(10, value, "MYR")
    with pytest.raises((TypeError, ValueError)):
        ModelPrice("provider", "model", value, Decimal("1"), "MYR")


@pytest.mark.parametrize("value", ["", "  ", "US", "USDD", "€€€", 1])
def test_given_invalid_currency_when_configuring_then_rejected(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        BudgetPolicy(10, Decimal("1"), value)


@pytest.mark.parametrize("value", ["", "  ", 1, None])
def test_given_invalid_period_or_tenant_when_creating_key_then_rejected(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        BudgetKey(value, "day", "month")
    with pytest.raises((TypeError, ValueError)):
        BudgetKey("tenant", value, "month")
    with pytest.raises((TypeError, ValueError)):
        BudgetKey("tenant", "day", value)


@pytest.mark.parametrize("value", [-1, True, 1.5, "1", None])
async def test_given_invalid_token_counts_when_reserving_then_rejected(value) -> None:
    ledger = _ledger()
    with pytest.raises(ValueError):
        await _reserve(ledger, input_tokens=value)
    with pytest.raises(ValueError):
        await _reserve(ledger, output_tokens=value)


def test_given_duplicate_or_invalid_prices_when_configuring_then_rejected() -> None:
    price = ModelPrice("provider", "model", Decimal("0"), Decimal("1"), "MYR")
    with pytest.raises(ValueError, match="duplicate model price"):
        ModelPriceTable((price, price))
    with pytest.raises(TypeError, match="ModelPrice"):
        ModelPriceTable((object(),))


def test_given_invalid_budget_policy_when_configuring_then_rejected() -> None:
    with pytest.raises(ValueError, match="token limit"):
        BudgetPolicy(0, Decimal("1"), "MYR")
    with pytest.raises(ValueError, match="cost limit"):
        BudgetPolicy(10, Decimal("0"), "MYR")


async def test_given_error_when_tracing_then_code_is_safe(in_memory_tracing) -> None:
    ledger = _ledger()
    decision = await _reserve(ledger)
    assert decision.reservation is not None
    await ledger.mark_attempted(decision.reservation)

    with pytest.raises(BudgetError):
        await ledger.cancel(decision.reservation)

    span = next(
        span
        for span in in_memory_tracing.get_finished_spans()
        if span.name == "budget.cancel"
    )
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == "budget_cancel_forbidden"
    assert span.attributes["budget.outcome"] == "failed"
    assert "tenant-1" not in repr(span)


async def test_given_private_values_when_tracing_then_identifiers_and_prices_absent(
    in_memory_tracing,
) -> None:
    ledger = InMemoryBudgetLedger(
        BudgetPolicy(20, Decimal("0.0001"), "MYR"),
        ModelPriceTable(
            (
                ModelPrice(
                    "private-provider",
                    "private-model",
                    Decimal("2.50"),
                    Decimal("7.50"),
                    "MYR",
                ),
            )
        ),
        token_factory=lambda: "private-token",
    )
    key = BudgetKey("private-tenant", "private-day", "private-month")
    decision = await ledger.reserve(
        key,
        provider="private-provider",
        model="private-model",
        estimated_input_tokens=10,
        max_output_tokens=10,
    )
    assert decision.reservation is not None
    await ledger.mark_attempted(decision.reservation)
    await ledger.reconcile(decision.reservation, None)

    spans = [
        span
        for span in in_memory_tracing.get_finished_spans()
        if span.name.startswith("budget.")
    ]
    assert len(spans) == 3
    assert all(span.status.status_code is StatusCode.UNSET for span in spans)
    serialized = repr(spans)
    for private in (
        "private-tenant",
        "private-day",
        "private-month",
        "private-provider",
        "private-model",
        "private-token",
        "2.50",
        "7.50",
    ):
        assert private not in serialized
