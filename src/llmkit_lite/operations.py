"""Durable lifecycle rules for external operations."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from llmkit_lite.graphs import WorkflowIdentity
from llmkit_lite.observability import set_span_error, trace_span
from llmkit_lite.state import (
    OperationState,
    OperationStatus,
    StateStore,
    StateStoreError,
)

_ERROR_DETAILS = {
    "operation_not_found": "operation was not found",
    "operation_invalid_transition": "operation status transition is not allowed",
    "operation_terminal": "operation is already terminal",
    "operation_store_failed": "operation state could not be persisted",
    "operation_invalid_record": "operation store returned an invalid record",
}

_ALLOWED_TRANSITIONS = {
    OperationStatus.PENDING: frozenset(
        {
            OperationStatus.DISPATCHING,
            OperationStatus.FAILED,
        }
    ),
    OperationStatus.DISPATCHING: frozenset(
        {
            OperationStatus.IN_PROGRESS,
            OperationStatus.COMPLETED,
            OperationStatus.FAILED,
        }
    ),
    OperationStatus.IN_PROGRESS: frozenset(
        {
            OperationStatus.COMPLETED,
            OperationStatus.FAILED,
        }
    ),
    OperationStatus.COMPLETED: frozenset(),
    OperationStatus.FAILED: frozenset(),
}

_TERMINAL_STATUSES = frozenset(
    {
        OperationStatus.COMPLETED,
        OperationStatus.FAILED,
    }
)


class OperationLifecycleError(Exception):
    """Safe operation lifecycle failure for application error mapping."""

    def __init__(self, code: str) -> None:
        if code not in _ERROR_DETAILS:
            raise ValueError("unsupported operation lifecycle error code")
        detail = _ERROR_DETAILS[code]
        super().__init__(detail)
        self.code = code
        self.detail = detail


class OperationLifecycle:
    """Create and transition operation records through durable state storage."""

    def __init__(self, state_store: StateStore) -> None:
        for method_name in ("get_operation", "save_operation"):
            if not callable(getattr(state_store, method_name, None)):
                raise TypeError(f"state store must implement {method_name}")
        self.state_store = state_store

    async def create(
        self,
        operation_id: str,
        identity: WorkflowIdentity,
        idempotency_key: str,
        values: Mapping[str, Any] | None = None,
    ) -> OperationState:
        """Persist a new operation before any downstream dispatch begins."""

        resolved_id = _normalize_identifier(operation_id, "operation ID")
        resolved_identity = _require_identity(identity)
        resolved_key = _normalize_identifier(idempotency_key, "idempotency key")
        with trace_span(
            "operation.lifecycle.create",
            attributes={"operation.to_status": OperationStatus.PENDING.value},
        ) as span:
            try:
                state = await self.state_store.save_operation(
                    resolved_id,
                    resolved_identity,
                    resolved_key,
                    OperationStatus.PENDING,
                    {} if values is None else values,
                )
                _require_persisted_operation(
                    state,
                    operation_id=resolved_id,
                    identity=resolved_identity,
                    idempotency_key=resolved_key,
                    status=OperationStatus.PENDING,
                    revision=1,
                )
            except asyncio.CancelledError:
                _record_failure(span, "cancelled", "operation_cancelled")
                raise
            except (OperationLifecycleError, StateStoreError) as exc:
                _record_failure(span, "failed", exc.code)
                raise
            except (TypeError, ValueError):
                raise
            except Exception as exc:
                error = OperationLifecycleError("operation_store_failed")
                _record_failure(span, "failed", error.code)
                raise error from exc

            if span is not None:
                span.set_attribute("operation.outcome", "created")
                span.set_attribute("operation.revision", state.revision)
            return state

    async def transition(
        self,
        operation_id: str,
        target_status: OperationStatus,
        *,
        expected_revision: int,
        values: Mapping[str, Any] | None = None,
    ) -> OperationState:
        """Compare-and-set an operation through one allowed status transition."""

        if not isinstance(target_status, OperationStatus):
            raise TypeError("target status must be an OperationStatus")
        resolved_id = _normalize_identifier(operation_id, "operation ID")
        _require_revision(expected_revision)

        with trace_span(
            "operation.lifecycle.transition",
            attributes={
                "operation.to_status": target_status.value,
                "operation.expected_revision": expected_revision,
            },
        ) as span:
            try:
                current = await self.state_store.get_operation(resolved_id)
                if current is None:
                    raise OperationLifecycleError("operation_not_found")
                if not isinstance(current, OperationState):
                    raise OperationLifecycleError("operation_invalid_record")
                if current.operation_id != resolved_id:
                    raise OperationLifecycleError("operation_invalid_record")
                if span is not None:
                    span.set_attribute(
                        "operation.from_status",
                        current.status.value,
                    )
                if current.revision != expected_revision:
                    raise StateStoreError("state_revision_conflict")
                if current.status in _TERMINAL_STATUSES:
                    raise OperationLifecycleError("operation_terminal")
                if target_status not in _ALLOWED_TRANSITIONS[current.status]:
                    raise OperationLifecycleError("operation_invalid_transition")

                state = await self.state_store.save_operation(
                    current.operation_id,
                    current.identity,
                    current.idempotency_key,
                    target_status,
                    current.values if values is None else values,
                    expected_revision=expected_revision,
                )
                _require_persisted_operation(
                    state,
                    operation_id=current.operation_id,
                    identity=current.identity,
                    idempotency_key=current.idempotency_key,
                    status=target_status,
                    revision=current.revision + 1,
                )
            except asyncio.CancelledError:
                _record_failure(span, "cancelled", "operation_cancelled")
                raise
            except (OperationLifecycleError, StateStoreError) as exc:
                _record_failure(span, "failed", exc.code)
                raise
            except (TypeError, ValueError):
                raise
            except Exception as exc:
                error = OperationLifecycleError("operation_store_failed")
                _record_failure(span, "failed", error.code)
                raise error from exc

            if span is not None:
                span.set_attribute("operation.outcome", "transitioned")
                span.set_attribute("operation.revision", state.revision)
            return state


def _require_persisted_operation(
    value: Any,
    *,
    operation_id: str,
    identity: WorkflowIdentity,
    idempotency_key: str,
    status: OperationStatus,
    revision: int,
) -> OperationState:
    if not isinstance(value, OperationState) or (
        value.operation_id != operation_id
        or value.identity != identity
        or value.idempotency_key != idempotency_key
        or value.status is not status
        or value.revision != revision
    ):
        raise OperationLifecycleError("operation_invalid_record")
    return value


def _normalize_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _require_identity(identity: WorkflowIdentity) -> WorkflowIdentity:
    if not isinstance(identity, WorkflowIdentity):
        raise TypeError("operation identity must be a WorkflowIdentity")
    return identity


def _require_revision(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("expected operation revision must be an integer")
    if value < 1:
        raise ValueError("expected operation revision must be greater than zero")
    return value


def _record_failure(span: Any, outcome: str, error_code: str) -> None:
    if span is not None:
        span.set_attribute("operation.outcome", outcome)
        span.set_attribute("error.type", error_code)
    set_span_error(span, error_code)
