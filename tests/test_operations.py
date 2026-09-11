from __future__ import annotations

import asyncio
from typing import Any

import pytest
from opentelemetry.trace import StatusCode

from llmkit_lite.graphs import WorkflowIdentity
from llmkit_lite.operations import OperationLifecycle, OperationLifecycleError
from llmkit_lite.state import (
    InMemoryStateStore,
    OperationState,
    OperationStatus,
    StateStoreError,
)


async def _create_operation(
    lifecycle: OperationLifecycle,
    *,
    values: dict[str, Any] | None = None,
) -> OperationState:
    return await lifecycle.create(
        "operation-123",
        WorkflowIdentity("thread-123", "support"),
        "key-123",
        values,
    )


def test_lifecycle_validates_state_store() -> None:
    with pytest.raises(TypeError, match="get_operation"):
        OperationLifecycle(object())  # type: ignore[arg-type]

    incomplete = type(
        "IncompleteStore",
        (),
        {"get_operation": lambda self, operation_id: None},
    )()
    with pytest.raises(TypeError, match="save_operation"):
        OperationLifecycle(incomplete)  # type: ignore[arg-type]


async def test_create_persists_pending_operation() -> None:
    store = InMemoryStateStore()
    lifecycle = OperationLifecycle(store)

    created = await _create_operation(
        lifecycle,
        values={"command": {"device": "router-1"}},
    )

    assert created.status is OperationStatus.PENDING
    assert created.revision == 1
    assert created.values == {"command": {"device": "router-1"}}
    assert await store.get_operation("operation-123") is created


async def test_lifecycle_supports_normal_operation_sequence() -> None:
    lifecycle = OperationLifecycle(InMemoryStateStore())
    pending = await _create_operation(lifecycle, values={"stage": "created"})

    dispatching = await lifecycle.transition(
        pending.operation_id,
        OperationStatus.DISPATCHING,
        expected_revision=pending.revision,
        values={"stage": "command_sent"},
    )
    in_progress = await lifecycle.transition(
        dispatching.operation_id,
        OperationStatus.IN_PROGRESS,
        expected_revision=dispatching.revision,
        values={"downstream_id": "command-456"},
    )
    completed = await lifecycle.transition(
        in_progress.operation_id,
        OperationStatus.COMPLETED,
        expected_revision=in_progress.revision,
        values={"result": "online"},
    )

    assert [
        pending.status,
        dispatching.status,
        in_progress.status,
        completed.status,
    ] == [
        OperationStatus.PENDING,
        OperationStatus.DISPATCHING,
        OperationStatus.IN_PROGRESS,
        OperationStatus.COMPLETED,
    ]
    assert completed.revision == 4
    assert completed.values == {"result": "online"}


async def test_transition_preserves_values_when_no_replacement_is_given() -> None:
    lifecycle = OperationLifecycle(InMemoryStateStore())
    pending = await _create_operation(lifecycle, values={"device": "router-1"})

    dispatching = await lifecycle.transition(
        pending.operation_id,
        OperationStatus.DISPATCHING,
        expected_revision=pending.revision,
    )

    assert dispatching.values == {"device": "router-1"}
    assert dispatching.values is not pending.values


@pytest.mark.parametrize(
    "starting_status,target_status",
    [
        (OperationStatus.PENDING, OperationStatus.FAILED),
        (OperationStatus.DISPATCHING, OperationStatus.COMPLETED),
        (OperationStatus.DISPATCHING, OperationStatus.FAILED),
        (OperationStatus.IN_PROGRESS, OperationStatus.FAILED),
    ],
)
async def test_lifecycle_supports_confirmed_terminal_transitions(
    starting_status: OperationStatus,
    target_status: OperationStatus,
) -> None:
    lifecycle = OperationLifecycle(InMemoryStateStore())
    state = await _create_operation(lifecycle)
    if starting_status is OperationStatus.DISPATCHING:
        state = await lifecycle.transition(
            state.operation_id,
            OperationStatus.DISPATCHING,
            expected_revision=state.revision,
        )
    elif starting_status is OperationStatus.IN_PROGRESS:
        state = await lifecycle.transition(
            state.operation_id,
            OperationStatus.DISPATCHING,
            expected_revision=state.revision,
        )
        state = await lifecycle.transition(
            state.operation_id,
            OperationStatus.IN_PROGRESS,
            expected_revision=state.revision,
        )

    terminal = await lifecycle.transition(
        state.operation_id,
        target_status,
        expected_revision=state.revision,
    )

    assert terminal.status is target_status


@pytest.mark.parametrize(
    "target_status",
    [
        OperationStatus.PENDING,
        OperationStatus.COMPLETED,
        OperationStatus.IN_PROGRESS,
    ],
)
async def test_invalid_transition_does_not_change_persisted_state(
    target_status: OperationStatus,
) -> None:
    store = InMemoryStateStore()
    lifecycle = OperationLifecycle(store)
    pending = await _create_operation(lifecycle)

    with pytest.raises(OperationLifecycleError) as exc_info:
        await lifecycle.transition(
            pending.operation_id,
            target_status,
            expected_revision=pending.revision,
        )

    assert exc_info.value.code == "operation_invalid_transition"
    assert await store.get_operation(pending.operation_id) is pending


async def test_lifecycle_rejects_backward_transition() -> None:
    store = InMemoryStateStore()
    lifecycle = OperationLifecycle(store)
    pending = await _create_operation(lifecycle)
    dispatching = await lifecycle.transition(
        pending.operation_id,
        OperationStatus.DISPATCHING,
        expected_revision=pending.revision,
    )
    in_progress = await lifecycle.transition(
        dispatching.operation_id,
        OperationStatus.IN_PROGRESS,
        expected_revision=dispatching.revision,
    )

    with pytest.raises(OperationLifecycleError) as exc_info:
        await lifecycle.transition(
            in_progress.operation_id,
            OperationStatus.DISPATCHING,
            expected_revision=in_progress.revision,
        )

    assert exc_info.value.code == "operation_invalid_transition"
    assert await store.get_operation(in_progress.operation_id) is in_progress


@pytest.mark.parametrize(
    "terminal_status",
    [OperationStatus.COMPLETED, OperationStatus.FAILED],
)
async def test_terminal_operations_reject_further_transitions(
    terminal_status: OperationStatus,
) -> None:
    store = InMemoryStateStore()
    lifecycle = OperationLifecycle(store)
    pending = await _create_operation(lifecycle)
    dispatching = await lifecycle.transition(
        pending.operation_id,
        OperationStatus.DISPATCHING,
        expected_revision=pending.revision,
    )
    terminal = await lifecycle.transition(
        dispatching.operation_id,
        terminal_status,
        expected_revision=dispatching.revision,
    )

    with pytest.raises(OperationLifecycleError) as exc_info:
        await lifecycle.transition(
            terminal.operation_id,
            OperationStatus.FAILED,
            expected_revision=terminal.revision,
        )

    assert exc_info.value.code == "operation_terminal"
    assert await store.get_operation(terminal.operation_id) is terminal


async def test_missing_operation_raises_safe_error() -> None:
    lifecycle = OperationLifecycle(InMemoryStateStore())

    with pytest.raises(OperationLifecycleError) as exc_info:
        await lifecycle.transition(
            "private-operation-id",
            OperationStatus.DISPATCHING,
            expected_revision=1,
        )

    assert exc_info.value.code == "operation_not_found"
    assert "private-operation-id" not in str(exc_info.value)


async def test_stale_revision_is_rejected_before_transition() -> None:
    store = InMemoryStateStore()
    lifecycle = OperationLifecycle(store)
    pending = await _create_operation(lifecycle)

    with pytest.raises(StateStoreError) as exc_info:
        await lifecycle.transition(
            pending.operation_id,
            OperationStatus.DISPATCHING,
            expected_revision=pending.revision + 1,
        )

    assert exc_info.value.code == "state_revision_conflict"
    assert await store.get_operation(pending.operation_id) is pending


async def test_only_one_concurrent_transition_wins() -> None:
    lifecycle = OperationLifecycle(InMemoryStateStore())
    pending = await _create_operation(lifecycle)
    dispatching = await lifecycle.transition(
        pending.operation_id,
        OperationStatus.DISPATCHING,
        expected_revision=pending.revision,
    )

    async def transition(
        target: OperationStatus,
    ) -> OperationState | StateStoreError:
        try:
            return await lifecycle.transition(
                dispatching.operation_id,
                target,
                expected_revision=dispatching.revision,
            )
        except StateStoreError as exc:
            return exc

    outcomes = await asyncio.gather(
        transition(OperationStatus.COMPLETED),
        transition(OperationStatus.FAILED),
    )

    successes = [item for item in outcomes if isinstance(item, OperationState)]
    conflicts = [item for item in outcomes if isinstance(item, StateStoreError)]
    assert len(successes) == 1
    assert successes[0].status in {OperationStatus.COMPLETED, OperationStatus.FAILED}
    assert len(conflicts) == 1
    assert conflicts[0].code == "state_revision_conflict"


async def test_unknown_outcome_remains_dispatching_until_reconciled() -> None:
    store = InMemoryStateStore()
    lifecycle = OperationLifecycle(store)
    pending = await _create_operation(lifecycle)
    dispatching = await lifecycle.transition(
        pending.operation_id,
        OperationStatus.DISPATCHING,
        expected_revision=pending.revision,
        values={"outcome": "unknown"},
    )

    loaded = await store.get_operation(dispatching.operation_id)

    assert loaded is dispatching
    assert loaded.status is OperationStatus.DISPATCHING
    assert loaded.values == {"outcome": "unknown"}


@pytest.mark.parametrize("revision", [0, -1, True, 1.5])
async def test_transition_validates_expected_revision(revision) -> None:
    lifecycle = OperationLifecycle(InMemoryStateStore())

    with pytest.raises((TypeError, ValueError)):
        await lifecycle.transition(
            "operation-123",
            OperationStatus.DISPATCHING,
            expected_revision=revision,
        )


async def test_lifecycle_maps_unexpected_store_failure_safely() -> None:
    class BrokenStore:
        async def get_operation(self, operation_id: str) -> None:
            raise RuntimeError("private database failure")

        async def save_operation(self, *args, **kwargs) -> OperationState:
            raise RuntimeError("private database failure")

    lifecycle = OperationLifecycle(BrokenStore())  # type: ignore[arg-type]

    with pytest.raises(OperationLifecycleError) as create_info:
        await _create_operation(lifecycle, values={"private": "payload"})
    assert create_info.value.code == "operation_store_failed"

    with pytest.raises(OperationLifecycleError) as transition_info:
        await lifecycle.transition(
            "private-operation",
            OperationStatus.DISPATCHING,
            expected_revision=1,
        )
    assert transition_info.value.code == "operation_store_failed"

    for error in (create_info.value, transition_info.value):
        assert "private" not in str(error)
        assert "database" not in str(error)


async def test_lifecycle_rejects_invalid_store_record() -> None:
    class InvalidStore:
        async def get_operation(self, operation_id: str) -> object:
            return object()

        async def save_operation(self, *args, **kwargs) -> object:
            return object()

    lifecycle = OperationLifecycle(InvalidStore())  # type: ignore[arg-type]

    with pytest.raises(OperationLifecycleError) as create_info:
        await _create_operation(lifecycle)
    assert create_info.value.code == "operation_invalid_record"

    with pytest.raises(OperationLifecycleError) as transition_info:
        await lifecycle.transition(
            "operation-123",
            OperationStatus.DISPATCHING,
            expected_revision=1,
        )
    assert transition_info.value.code == "operation_invalid_record"


async def test_transition_telemetry_excludes_sensitive_state(
    in_memory_tracing,
) -> None:
    lifecycle = OperationLifecycle(InMemoryStateStore())
    pending = await lifecycle.create(
        "private-operation-id",
        WorkflowIdentity("private-thread", "private-namespace"),
        "private-idempotency-key",
        {"private": "operation payload"},
    )

    with pytest.raises(OperationLifecycleError):
        await lifecycle.transition(
            pending.operation_id,
            OperationStatus.COMPLETED,
            expected_revision=pending.revision,
        )

    span = next(
        item
        for item in in_memory_tracing.get_finished_spans()
        if item.name == "operation.lifecycle.transition"
    )
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes == {
        "operation.to_status": "completed",
        "operation.expected_revision": 1,
        "operation.from_status": "pending",
        "operation.outcome": "failed",
        "error.type": "operation_invalid_transition",
        "exception.type": "llmkit_lite.operations.OperationLifecycleError",
    }
    serialized = repr(span)
    for secret in (
        "private-operation-id",
        "private-thread",
        "private-namespace",
        "private-idempotency-key",
        "operation payload",
    ):
        assert secret not in serialized
