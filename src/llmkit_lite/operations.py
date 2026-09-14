"""Durable lifecycle rules for external operations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol

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
    "operation_idempotency_conflict": "idempotency key was reused inconsistently",
}

_EXECUTION_ERROR_DETAILS = {
    "operation_outcome_unknown": "operation outcome is unknown",
    "operation_reconciliation_failed": "operation outcome could not be reconciled",
    "operation_adapter_invalid_response": (
        "operation adapter returned an invalid response"
    ),
    "operation_execution_failed": "operation execution failed",
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


@dataclass(frozen=True, slots=True)
class OperationReservation:
    """An idempotent operation lookup and whether this caller created it."""

    state: OperationState
    created: bool

    def __post_init__(self) -> None:
        if not isinstance(self.state, OperationState):
            raise TypeError("reservation state must be an OperationState")
        if not isinstance(self.created, bool):
            raise TypeError("reservation created flag must be a boolean")


@dataclass(frozen=True, slots=True)
class OperationRequest:
    """Canonical, persistence-safe input for one idempotent operation."""

    action: str
    values: Mapping[str, Any]
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action",
            _normalize_identifier(self.action, "operation action"),
        )
        frozen_values = _freeze_json_mapping(self.values, "operation request values")
        object.__setattr__(self, "values", frozen_values)
        canonical = json.dumps(
            {
                "action": self.action,
                "values": _thaw_json(frozen_values),
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        object.__setattr__(self, "_fingerprint", hashlib.sha256(canonical).hexdigest())

    @property
    def fingerprint(self) -> str:
        """Return the deterministic SHA-256 request fingerprint."""

        return self._fingerprint

    def to_dict(self) -> dict[str, Any]:
        """Return detached values suitable for persistence or dispatch."""

        return {
            "action": self.action,
            "values": _thaw_json(self.values),
        }


class OperationOutcome(StrEnum):
    """A downstream observation used to advance durable operation state."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class OperationObservation:
    """Validated result of dispatching or inspecting an operation."""

    outcome: OperationOutcome
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, OperationOutcome):
            raise TypeError("operation outcome must be an OperationOutcome")
        object.__setattr__(
            self,
            "values",
            _freeze_json_mapping(self.values, "operation observation values"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return detached observation values suitable for persistence."""

        return {
            "outcome": self.outcome.value,
            "values": _thaw_json(self.values),
        }


class OperationAdapter(Protocol):
    """Dispatch side effects and inspect their downstream status."""

    async def dispatch(self, operation: OperationState) -> OperationObservation:
        """Send a new operation to the downstream service."""

    async def inspect(self, operation: OperationState) -> OperationObservation:
        """Read the downstream status without repeating the side effect."""


class OperationExecutionError(Exception):
    """Safe execution or reconciliation failure for application boundaries."""

    def __init__(self, code: str) -> None:
        if code not in _EXECUTION_ERROR_DETAILS:
            raise ValueError("unsupported operation execution error code")
        detail = _EXECUTION_ERROR_DETAILS[code]
        super().__init__(detail)
        self.code = code
        self.detail = detail


class OperationLifecycle:
    """Create and transition operation records through durable state storage."""

    def __init__(self, state_store: StateStore) -> None:
        for method_name in (
            "get_operation",
            "get_operation_by_idempotency_key",
            "save_operation",
        ):
            if not callable(getattr(state_store, method_name, None)):
                raise TypeError(f"state store must implement {method_name}")
        self.state_store = state_store

    async def create(
        self,
        operation_id: str,
        identity: WorkflowIdentity,
        idempotency_key: str,
        request_fingerprint: str,
        values: Mapping[str, Any] | None = None,
    ) -> OperationState:
        """Persist a new operation before any downstream dispatch begins."""

        resolved_id = _normalize_identifier(operation_id, "operation ID")
        resolved_identity = _require_identity(identity)
        resolved_key = _normalize_identifier(idempotency_key, "idempotency key")
        resolved_fingerprint = _normalize_identifier(
            request_fingerprint,
            "request fingerprint",
        )
        with trace_span(
            "operation.lifecycle.create",
            attributes={"operation.to_status": OperationStatus.PENDING.value},
        ) as span:
            try:
                state = await self.state_store.save_operation(
                    resolved_id,
                    resolved_identity,
                    resolved_key,
                    resolved_fingerprint,
                    OperationStatus.PENDING,
                    {} if values is None else values,
                )
                _require_persisted_operation(
                    state,
                    operation_id=resolved_id,
                    identity=resolved_identity,
                    idempotency_key=resolved_key,
                    request_fingerprint=resolved_fingerprint,
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

    async def get_or_create(
        self,
        operation_id: str,
        identity: WorkflowIdentity,
        idempotency_key: str,
        request: OperationRequest,
    ) -> OperationReservation:
        """Return a matching operation or atomically create its pending record."""

        resolved_id = _normalize_identifier(operation_id, "operation ID")
        resolved_identity = _require_identity(identity)
        resolved_key = _normalize_identifier(idempotency_key, "idempotency key")
        if not isinstance(request, OperationRequest):
            raise TypeError("operation request must be an OperationRequest")

        with trace_span("operation.lifecycle.reserve") as span:
            try:
                existing = await self.state_store.get_operation_by_idempotency_key(
                    resolved_identity,
                    resolved_key,
                )
                if existing is not None:
                    reservation = _reuse_operation(
                        existing,
                        identity=resolved_identity,
                        idempotency_key=resolved_key,
                        request=request,
                    )
                else:
                    try:
                        created = await self.create(
                            resolved_id,
                            resolved_identity,
                            resolved_key,
                            request.fingerprint,
                            {"request": request.to_dict()},
                        )
                    except StateStoreError as exc:
                        if exc.code not in {
                            "state_already_exists",
                            "state_idempotency_conflict",
                        }:
                            raise
                        existing = (
                            await self.state_store.get_operation_by_idempotency_key(
                                resolved_identity,
                                resolved_key,
                            )
                        )
                        if existing is None:
                            existing = await self.state_store.get_operation(resolved_id)
                        if existing is None:
                            raise OperationLifecycleError(
                                "operation_invalid_record"
                            ) from exc
                        reservation = _reuse_operation(
                            existing,
                            identity=resolved_identity,
                            idempotency_key=resolved_key,
                            request=request,
                        )
                    else:
                        reservation = OperationReservation(created, True)
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
                span.set_attribute(
                    "operation.outcome",
                    "created" if reservation.created else "reused",
                )
                span.set_attribute("operation.revision", reservation.state.revision)
                span.set_attribute("operation.status", reservation.state.status.value)
            return reservation

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
                    current.request_fingerprint,
                    target_status,
                    current.values if values is None else values,
                    expected_revision=expected_revision,
                )
                _require_persisted_operation(
                    state,
                    operation_id=current.operation_id,
                    identity=current.identity,
                    idempotency_key=current.idempotency_key,
                    request_fingerprint=current.request_fingerprint,
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


class DurableOperationExecutor:
    """Execute once and reconcile repeated requests through durable state."""

    def __init__(
        self,
        lifecycle: OperationLifecycle,
        adapter: OperationAdapter,
    ) -> None:
        if not isinstance(lifecycle, OperationLifecycle):
            raise TypeError("lifecycle must be an OperationLifecycle")
        for method_name in ("dispatch", "inspect"):
            if not callable(getattr(adapter, method_name, None)):
                raise TypeError(f"operation adapter must implement {method_name}")
        self.lifecycle = lifecycle
        self.adapter = adapter

    async def execute(
        self,
        operation_id: str,
        identity: WorkflowIdentity,
        idempotency_key: str,
        request: OperationRequest,
    ) -> OperationState:
        """Dispatch a new request or reconcile its existing durable operation."""

        if not isinstance(request, OperationRequest):
            raise TypeError("operation request must be an OperationRequest")
        with trace_span("operation.execute") as span:
            try:
                reservation = await self.lifecycle.get_or_create(
                    operation_id,
                    identity,
                    idempotency_key,
                    request,
                )
                state = reservation.state
                if span is not None:
                    span.set_attribute("operation.reused", not reservation.created)

                if state.status in _TERMINAL_STATUSES:
                    result = state
                    outcome = "reused"
                elif state.status is OperationStatus.PENDING:
                    try:
                        dispatching = await self.lifecycle.transition(
                            state.operation_id,
                            OperationStatus.DISPATCHING,
                            expected_revision=state.revision,
                        )
                    except StateStoreError as exc:
                        if exc.code != "state_revision_conflict":
                            raise
                        dispatching = await self._reload(state.operation_id)
                        if dispatching.status is OperationStatus.PENDING:
                            raise OperationExecutionError(
                                "operation_execution_failed"
                            ) from exc
                        result = await self._inspect(dispatching)
                        outcome = "reconciled"
                    else:
                        result = await self._dispatch(dispatching)
                        outcome = "dispatched"
                else:
                    result = await self._inspect(state)
                    outcome = "reconciled"
            except asyncio.CancelledError:
                _record_failure(span, "cancelled", "operation_cancelled")
                raise
            except (
                OperationExecutionError,
                OperationLifecycleError,
                StateStoreError,
            ) as exc:
                _record_failure(span, "failed", exc.code)
                raise
            except (TypeError, ValueError):
                raise
            except Exception as exc:
                error = OperationExecutionError("operation_execution_failed")
                _record_failure(span, "failed", error.code)
                raise error from exc

            if span is not None:
                span.set_attribute("operation.outcome", outcome)
                span.set_attribute("operation.status", result.status.value)
                span.set_attribute("operation.revision", result.revision)
            return result

    async def _dispatch(self, state: OperationState) -> OperationState:
        try:
            observation = await self.adapter.dispatch(state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise OperationExecutionError("operation_outcome_unknown") from exc
        return await self._apply_observation(state, observation)

    async def _inspect(self, state: OperationState) -> OperationState:
        try:
            observation = await self.adapter.inspect(state)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise OperationExecutionError("operation_reconciliation_failed") from exc
        return await self._apply_observation(state, observation)

    async def _apply_observation(
        self,
        state: OperationState,
        observation: OperationObservation,
    ) -> OperationState:
        if not isinstance(observation, OperationObservation):
            raise OperationExecutionError("operation_adapter_invalid_response")
        if observation.outcome is OperationOutcome.UNKNOWN:
            return state
        target_status = OperationStatus(observation.outcome.value)
        if target_status is state.status:
            return state
        values = _thaw_json(state.values)
        values["observation"] = observation.to_dict()
        try:
            return await self.lifecycle.transition(
                state.operation_id,
                target_status,
                expected_revision=state.revision,
                values=values,
            )
        except StateStoreError as exc:
            if exc.code != "state_revision_conflict":
                raise
            return await self._reload(state.operation_id)

    async def _reload(self, operation_id: str) -> OperationState:
        state = await self.lifecycle.state_store.get_operation(operation_id)
        if not isinstance(state, OperationState):
            raise OperationLifecycleError("operation_invalid_record")
        return state


def _require_persisted_operation(
    value: Any,
    *,
    operation_id: str,
    identity: WorkflowIdentity,
    idempotency_key: str,
    request_fingerprint: str,
    status: OperationStatus,
    revision: int,
) -> OperationState:
    if not isinstance(value, OperationState) or (
        value.operation_id != operation_id
        or value.identity != identity
        or value.idempotency_key != idempotency_key
        or value.request_fingerprint != request_fingerprint
        or value.status is not status
        or value.revision != revision
    ):
        raise OperationLifecycleError("operation_invalid_record")
    return value


def _reuse_operation(
    value: Any,
    *,
    identity: WorkflowIdentity,
    idempotency_key: str,
    request: OperationRequest,
) -> OperationReservation:
    if not isinstance(value, OperationState) or (
        value.identity != identity or value.idempotency_key != idempotency_key
    ):
        raise OperationLifecycleError("operation_invalid_record")
    if value.request_fingerprint != request.fingerprint:
        raise OperationLifecycleError("operation_idempotency_conflict")
    return OperationReservation(value, False)


def _freeze_json_mapping(
    values: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    return _freeze_json_value(values, field_name)


def _freeze_json_value(value: Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} keys must be strings")
            frozen[key] = _freeze_json_value(item, field_name)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item, field_name) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"{field_name} must contain only JSON values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
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
