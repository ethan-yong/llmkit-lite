"""Versioned persistence contracts for conversation and operation state."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol

from llmkit_lite.graphs import WorkflowIdentity

_ERROR_DETAILS = {
    "state_already_exists": "state record already exists",
    "state_idempotency_conflict": "idempotency key is already registered",
    "state_revision_conflict": "state record revision does not match",
    "state_record_not_found": "state record was not found",
    "state_invalid_record": "state record conflicts with persisted identity",
}


def _normalize_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _normalize_revision(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value < 1:
        raise ValueError(f"{field_name} must be greater than zero")
    return value


def _normalize_expected_revision(value: int | None) -> int | None:
    if value is None:
        return None
    return _normalize_revision(value, "expected state revision")


def _freeze_json(value: Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} keys must be strings")
            frozen[key] = _freeze_json(item, field_name)
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item, field_name) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"{field_name} must contain only JSON values")


def _freeze_values(values: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError("state values must be a mapping")
    return _freeze_json(values, "state values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


class StateStoreError(Exception):
    """Safe persistence failure suitable for application error mapping."""

    def __init__(self, code: str) -> None:
        if code not in _ERROR_DETAILS:
            raise ValueError("unsupported state store error code")
        detail = _ERROR_DETAILS[code]
        super().__init__(detail)
        self.code = code
        self.detail = detail


class OperationStatus(StrEnum):
    """Persisted lifecycle status for an external operation."""

    PENDING = "pending"
    DISPATCHING = "dispatching"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ConversationState:
    """Immutable, versioned snapshot of one conversation's runtime values."""

    identity: WorkflowIdentity
    revision: int
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, WorkflowIdentity):
            raise TypeError("conversation identity must be a WorkflowIdentity")
        object.__setattr__(
            self,
            "revision",
            _normalize_revision(self.revision, "conversation state revision"),
        )
        object.__setattr__(self, "values", _freeze_values(self.values))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible representation."""

        return {
            "thread_id": self.identity.thread_id,
            "checkpoint_namespace": self.identity.checkpoint_namespace,
            "revision": self.revision,
            "values": _thaw_json(self.values),
        }


@dataclass(frozen=True, slots=True)
class OperationState:
    """Immutable, versioned snapshot of one external operation's values."""

    operation_id: str
    identity: WorkflowIdentity
    idempotency_key: str
    request_fingerprint: str
    status: OperationStatus
    revision: int
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "operation_id",
            _normalize_identifier(self.operation_id, "operation ID"),
        )
        if not isinstance(self.identity, WorkflowIdentity):
            raise TypeError("operation identity must be a WorkflowIdentity")
        object.__setattr__(
            self,
            "idempotency_key",
            _normalize_identifier(self.idempotency_key, "idempotency key"),
        )
        object.__setattr__(
            self,
            "request_fingerprint",
            _normalize_identifier(self.request_fingerprint, "request fingerprint"),
        )
        if not isinstance(self.status, OperationStatus):
            raise TypeError("operation status must be an OperationStatus")
        object.__setattr__(
            self,
            "revision",
            _normalize_revision(self.revision, "operation state revision"),
        )
        object.__setattr__(self, "values", _freeze_values(self.values))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible representation."""

        return {
            "operation_id": self.operation_id,
            "thread_id": self.identity.thread_id,
            "checkpoint_namespace": self.identity.checkpoint_namespace,
            "idempotency_key": self.idempotency_key,
            "request_fingerprint": self.request_fingerprint,
            "status": self.status.value,
            "revision": self.revision,
            "values": _thaw_json(self.values),
        }


class StateStore(Protocol):
    """Persistence boundary for versioned conversation and operation snapshots."""

    async def get_conversation(
        self,
        identity: WorkflowIdentity,
    ) -> ConversationState | None:
        """Return the latest conversation snapshot, if one exists."""

    async def save_conversation(
        self,
        identity: WorkflowIdentity,
        values: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> ConversationState:
        """Create or compare-and-set one conversation snapshot."""

    async def get_operation(self, operation_id: str) -> OperationState | None:
        """Return the latest operation snapshot, if one exists."""

    async def get_operation_by_idempotency_key(
        self,
        identity: WorkflowIdentity,
        idempotency_key: str,
    ) -> OperationState | None:
        """Return the operation registered for one conversation-scoped key."""

    async def save_operation(
        self,
        operation_id: str,
        identity: WorkflowIdentity,
        idempotency_key: str,
        request_fingerprint: str,
        status: OperationStatus,
        values: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> OperationState:
        """Create or compare-and-set one operation snapshot."""


class InMemoryStateStore:
    """Concurrency-safe reference store that does not survive process restarts."""

    def __init__(self) -> None:
        self._conversations: dict[WorkflowIdentity, ConversationState] = {}
        self._operations: dict[str, OperationState] = {}
        self._operation_keys: dict[tuple[WorkflowIdentity, str], str] = {}
        self._lock = asyncio.Lock()

    async def get_conversation(
        self,
        identity: WorkflowIdentity,
    ) -> ConversationState | None:
        resolved_identity = _require_identity(identity, "conversation identity")
        async with self._lock:
            return self._conversations.get(resolved_identity)

    async def save_conversation(
        self,
        identity: WorkflowIdentity,
        values: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> ConversationState:
        resolved_identity = _require_identity(identity, "conversation identity")
        resolved_expected = _normalize_expected_revision(expected_revision)
        frozen_values = _freeze_values(values)

        async with self._lock:
            current = self._conversations.get(resolved_identity)
            revision = _next_revision(current, resolved_expected)
            state = ConversationState(
                identity=resolved_identity,
                revision=revision,
                values=frozen_values,
            )
            self._conversations[resolved_identity] = state
            return state

    async def get_operation(self, operation_id: str) -> OperationState | None:
        resolved_id = _normalize_identifier(operation_id, "operation ID")
        async with self._lock:
            return self._operations.get(resolved_id)

    async def get_operation_by_idempotency_key(
        self,
        identity: WorkflowIdentity,
        idempotency_key: str,
    ) -> OperationState | None:
        resolved_identity = _require_identity(identity, "operation identity")
        resolved_key = _normalize_identifier(idempotency_key, "idempotency key")
        async with self._lock:
            operation_id = self._operation_keys.get((resolved_identity, resolved_key))
            return None if operation_id is None else self._operations.get(operation_id)

    async def save_operation(
        self,
        operation_id: str,
        identity: WorkflowIdentity,
        idempotency_key: str,
        request_fingerprint: str,
        status: OperationStatus,
        values: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> OperationState:
        resolved_id = _normalize_identifier(operation_id, "operation ID")
        resolved_identity = _require_identity(identity, "operation identity")
        resolved_key = _normalize_identifier(idempotency_key, "idempotency key")
        resolved_fingerprint = _normalize_identifier(
            request_fingerprint,
            "request fingerprint",
        )
        if not isinstance(status, OperationStatus):
            raise TypeError("operation status must be an OperationStatus")
        resolved_expected = _normalize_expected_revision(expected_revision)
        frozen_values = _freeze_values(values)

        async with self._lock:
            current = self._operations.get(resolved_id)
            indexed_id = self._operation_keys.get((resolved_identity, resolved_key))
            if current is None and indexed_id is not None and indexed_id != resolved_id:
                raise StateStoreError("state_idempotency_conflict")
            revision = _next_revision(current, resolved_expected)
            if current is not None and (
                current.identity != resolved_identity
                or current.idempotency_key != resolved_key
                or current.request_fingerprint != resolved_fingerprint
            ):
                raise StateStoreError("state_invalid_record")
            state = OperationState(
                operation_id=resolved_id,
                identity=resolved_identity,
                idempotency_key=resolved_key,
                request_fingerprint=resolved_fingerprint,
                status=status,
                revision=revision,
                values=frozen_values,
            )
            self._operations[resolved_id] = state
            self._operation_keys[(resolved_identity, resolved_key)] = resolved_id
            return state


def _require_identity(
    identity: WorkflowIdentity,
    field_name: str,
) -> WorkflowIdentity:
    if not isinstance(identity, WorkflowIdentity):
        raise TypeError(f"{field_name} must be a WorkflowIdentity")
    return identity


def _next_revision(
    current: ConversationState | OperationState | None,
    expected_revision: int | None,
) -> int:
    if current is None:
        if expected_revision is not None:
            raise StateStoreError("state_record_not_found")
        return 1
    if expected_revision is None:
        raise StateStoreError("state_already_exists")
    if current.revision != expected_revision:
        raise StateStoreError("state_revision_conflict")
    return current.revision + 1
