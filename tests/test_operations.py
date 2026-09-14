from __future__ import annotations

import asyncio
from typing import Any

import pytest
from opentelemetry.trace import StatusCode

from llmkit_lite.graphs import WorkflowIdentity
from llmkit_lite.operations import (
    DurableOperationExecutor,
    OperationExecutionError,
    OperationLifecycle,
    OperationLifecycleError,
    OperationObservation,
    OperationOutcome,
    OperationRequest,
)
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
        "fingerprint-123",
        values,
    )


class ScriptedAdapter:
    def __init__(
        self,
        *,
        dispatch_results: list[Any] | None = None,
        inspect_results: list[Any] | None = None,
    ) -> None:
        self.dispatch_results = list(dispatch_results or [])
        self.inspect_results = list(inspect_results or [])
        self.dispatch_calls: list[OperationState] = []
        self.inspect_calls: list[OperationState] = []

    async def dispatch(self, operation: OperationState) -> OperationObservation:
        self.dispatch_calls.append(operation)
        result = self.dispatch_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    async def inspect(self, operation: OperationState) -> OperationObservation:
        self.inspect_calls.append(operation)
        result = self.inspect_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def test_lifecycle_validates_state_store() -> None:
    with pytest.raises(TypeError, match="get_operation"):
        OperationLifecycle(object())  # type: ignore[arg-type]

    incomplete = type(
        "IncompleteStore",
        (),
        {
            "get_operation": lambda self, operation_id: None,
            "get_operation_by_idempotency_key": (
                lambda self, identity, idempotency_key: None
            ),
        },
    )()
    with pytest.raises(TypeError, match="save_operation"):
        OperationLifecycle(incomplete)  # type: ignore[arg-type]


def test_operation_request_fingerprint_is_canonical_and_values_are_frozen() -> None:
    first_values = {"device": "router-1", "options": {"force": True, "ports": [2, 1]}}
    second_values = {"options": {"ports": [2, 1], "force": True}, "device": "router-1"}

    first = OperationRequest(" reboot ", first_values)
    second = OperationRequest("reboot", second_values)
    first_values["device"] = "changed"

    assert first.action == "reboot"
    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64
    assert first.to_dict() == {
        "action": "reboot",
        "values": {
            "device": "router-1",
            "options": {"force": True, "ports": [2, 1]},
        },
    }
    with pytest.raises(TypeError):
        first.values["device"] = "changed"  # type: ignore[index]


def test_operation_request_fingerprint_changes_with_request() -> None:
    original = OperationRequest("reboot", {"device": "router-1"})

    assert (
        original.fingerprint
        != OperationRequest(
            "inspect",
            {"device": "router-1"},
        ).fingerprint
    )
    assert (
        original.fingerprint
        != OperationRequest(
            "reboot",
            {"device": "router-2"},
        ).fingerprint
    )


def test_operation_request_and_observation_validate_json_values() -> None:
    with pytest.raises(TypeError):
        OperationRequest("reboot", {"invalid": object()})
    with pytest.raises(TypeError):
        OperationRequest("reboot", {1: "invalid"})  # type: ignore[dict-item]
    with pytest.raises(TypeError):
        OperationObservation(OperationOutcome.COMPLETED, {"invalid": float("nan")})
    with pytest.raises(TypeError):
        OperationObservation("completed")  # type: ignore[arg-type]


async def test_get_or_create_reuses_matching_request() -> None:
    lifecycle = OperationLifecycle(InMemoryStateStore())
    identity = WorkflowIdentity("thread-123")
    request = OperationRequest("reboot", {"device": "router-1"})

    created = await lifecycle.get_or_create(
        "operation-1",
        identity,
        "request-key",
        request,
    )
    reused = await lifecycle.get_or_create(
        "operation-2",
        identity,
        "request-key",
        request,
    )

    assert created.created is True
    assert reused.created is False
    assert reused.state is created.state
    assert reused.state.operation_id == "operation-1"


async def test_get_or_create_rejects_inconsistent_idempotency_reuse() -> None:
    lifecycle = OperationLifecycle(InMemoryStateStore())
    identity = WorkflowIdentity("thread-123")
    await lifecycle.get_or_create(
        "operation-1",
        identity,
        "request-key",
        OperationRequest("reboot", {"device": "router-1"}),
    )

    with pytest.raises(OperationLifecycleError) as exc_info:
        await lifecycle.get_or_create(
            "operation-2",
            identity,
            "request-key",
            OperationRequest("reboot", {"device": "router-2"}),
        )

    assert exc_info.value.code == "operation_idempotency_conflict"


async def test_concurrent_reservations_create_one_operation() -> None:
    lifecycle = OperationLifecycle(InMemoryStateStore())
    identity = WorkflowIdentity("thread-123")
    request = OperationRequest("reboot", {"device": "router-1"})

    reservations = await asyncio.gather(
        lifecycle.get_or_create("operation-1", identity, "request-key", request),
        lifecycle.get_or_create("operation-2", identity, "request-key", request),
    )

    assert sum(item.created for item in reservations) == 1
    assert reservations[0].state is reservations[1].state


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

        async def get_operation_by_idempotency_key(
            self,
            identity: WorkflowIdentity,
            idempotency_key: str,
        ) -> None:
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

        async def get_operation_by_idempotency_key(
            self,
            identity: WorkflowIdentity,
            idempotency_key: str,
        ) -> object:
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
        "private-fingerprint",
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


@pytest.mark.parametrize(
    "outcome,expected_status",
    [
        (OperationOutcome.COMPLETED, OperationStatus.COMPLETED),
        (OperationOutcome.FAILED, OperationStatus.FAILED),
    ],
)
async def test_executor_dispatches_once_and_reuses_terminal_result(
    outcome: OperationOutcome,
    expected_status: OperationStatus,
) -> None:
    adapter = ScriptedAdapter(
        dispatch_results=[
            OperationObservation(outcome, {"downstream": "result"}),
        ]
    )
    executor = DurableOperationExecutor(
        OperationLifecycle(InMemoryStateStore()),
        adapter,
    )
    identity = WorkflowIdentity("thread-123")
    request = OperationRequest("reboot", {"device": "router-1"})

    first = await executor.execute("operation-1", identity, "request-key", request)
    second = await executor.execute("operation-2", identity, "request-key", request)

    assert first.status is expected_status
    assert second is first
    assert len(adapter.dispatch_calls) == 1
    assert adapter.inspect_calls == []


async def test_executor_treats_different_idempotency_keys_independently() -> None:
    adapter = ScriptedAdapter(
        dispatch_results=[
            OperationObservation(OperationOutcome.COMPLETED, {"result": 1}),
            OperationObservation(OperationOutcome.COMPLETED, {"result": 2}),
        ]
    )
    executor = DurableOperationExecutor(
        OperationLifecycle(InMemoryStateStore()),
        adapter,
    )
    identity = WorkflowIdentity("thread-123")
    request = OperationRequest("reboot", {"device": "router-1"})

    first = await executor.execute("operation-1", identity, "key-1", request)
    second = await executor.execute("operation-2", identity, "key-2", request)

    assert first.operation_id == "operation-1"
    assert second.operation_id == "operation-2"
    assert len(adapter.dispatch_calls) == 2


async def test_concurrent_duplicate_requests_dispatch_only_once() -> None:
    class BlockingAdapter:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.dispatch_calls = 0
            self.inspect_calls = 0

        async def dispatch(self, operation: OperationState) -> OperationObservation:
            self.dispatch_calls += 1
            self.started.set()
            await self.release.wait()
            return OperationObservation(OperationOutcome.COMPLETED)

        async def inspect(self, operation: OperationState) -> OperationObservation:
            self.inspect_calls += 1
            return OperationObservation(OperationOutcome.COMPLETED)

    adapter = BlockingAdapter()
    executor = DurableOperationExecutor(
        OperationLifecycle(InMemoryStateStore()),
        adapter,
    )
    identity = WorkflowIdentity("thread-123")
    request = OperationRequest("reboot", {"device": "router-1"})

    first_task = asyncio.create_task(
        executor.execute("operation-1", identity, "request-key", request)
    )
    await adapter.started.wait()
    duplicate = await executor.execute(
        "operation-2",
        identity,
        "request-key",
        request,
    )
    adapter.release.set()
    completed = await first_task

    assert adapter.dispatch_calls == 1
    assert adapter.inspect_calls == 1
    assert duplicate.status is OperationStatus.COMPLETED
    assert completed.status is OperationStatus.COMPLETED


async def test_new_executor_reconciles_uncertain_dispatch_after_restart() -> None:
    store = InMemoryStateStore()
    identity = WorkflowIdentity("thread-123")
    request = OperationRequest("reboot", {"device": "router-1"})
    first_adapter = ScriptedAdapter(dispatch_results=[TimeoutError("private timeout")])
    first_executor = DurableOperationExecutor(
        OperationLifecycle(store),
        first_adapter,
    )

    with pytest.raises(OperationExecutionError) as dispatch_info:
        await first_executor.execute(
            "operation-1",
            identity,
            "request-key",
            request,
        )
    assert dispatch_info.value.code == "operation_outcome_unknown"
    uncertain = await store.get_operation_by_idempotency_key(identity, "request-key")
    assert uncertain is not None
    assert uncertain.status is OperationStatus.DISPATCHING

    recovery_adapter = ScriptedAdapter(
        inspect_results=[
            OperationObservation(
                OperationOutcome.COMPLETED,
                {"result": "online"},
            )
        ]
    )
    recovered_executor = DurableOperationExecutor(
        OperationLifecycle(store),
        recovery_adapter,
    )
    recovered = await recovered_executor.execute(
        "operation-after-restart",
        identity,
        "request-key",
        request,
    )

    assert recovered.operation_id == "operation-1"
    assert recovered.status is OperationStatus.COMPLETED
    assert recovered.values["observation"] == {
        "outcome": "completed",
        "values": {"result": "online"},
    }
    assert recovery_adapter.dispatch_calls == []
    assert len(recovery_adapter.inspect_calls) == 1


async def test_reconciliation_can_record_and_preserve_in_progress_state() -> None:
    store = InMemoryStateStore()
    identity = WorkflowIdentity("thread-123")
    request = OperationRequest("reboot", {"device": "router-1"})
    dispatch_adapter = ScriptedAdapter(
        dispatch_results=[OperationObservation(OperationOutcome.UNKNOWN)]
    )
    dispatch_executor = DurableOperationExecutor(
        OperationLifecycle(store),
        dispatch_adapter,
    )
    dispatching = await dispatch_executor.execute(
        "operation-1",
        identity,
        "request-key",
        request,
    )

    inspect_adapter = ScriptedAdapter(
        inspect_results=[
            OperationObservation(
                OperationOutcome.IN_PROGRESS,
                {"downstream_id": "command-456"},
            ),
            OperationObservation(OperationOutcome.IN_PROGRESS),
        ]
    )
    inspect_executor = DurableOperationExecutor(
        OperationLifecycle(store),
        inspect_adapter,
    )
    in_progress = await inspect_executor.execute(
        "operation-2",
        identity,
        "request-key",
        request,
    )
    unchanged = await inspect_executor.execute(
        "operation-3",
        identity,
        "request-key",
        request,
    )

    assert dispatching.status is OperationStatus.DISPATCHING
    assert in_progress.status is OperationStatus.IN_PROGRESS
    assert unchanged is in_progress
    assert unchanged.revision == in_progress.revision
    assert len(inspect_adapter.inspect_calls) == 2


@pytest.mark.parametrize(
    "failure",
    [
        PermissionError("private permission failure"),
        ConnectionError("private connection failure"),
    ],
)
async def test_reconciliation_failure_never_marks_operation_failed(
    failure: Exception,
) -> None:
    store = InMemoryStateStore()
    lifecycle = OperationLifecycle(store)
    identity = WorkflowIdentity("thread-123")
    request = OperationRequest("reboot", {"device": "router-1"})
    reservation = await lifecycle.get_or_create(
        "operation-1",
        identity,
        "request-key",
        request,
    )
    dispatching = await lifecycle.transition(
        reservation.state.operation_id,
        OperationStatus.DISPATCHING,
        expected_revision=reservation.state.revision,
    )
    executor = DurableOperationExecutor(
        lifecycle,
        ScriptedAdapter(inspect_results=[failure]),
    )

    with pytest.raises(OperationExecutionError) as exc_info:
        await executor.execute(
            "operation-2",
            identity,
            "request-key",
            request,
        )

    assert exc_info.value.code == "operation_reconciliation_failed"
    assert "private" not in str(exc_info.value)
    assert await store.get_operation(dispatching.operation_id) is dispatching


async def test_invalid_adapter_response_preserves_dispatching_state() -> None:
    store = InMemoryStateStore()
    executor = DurableOperationExecutor(
        OperationLifecycle(store),
        ScriptedAdapter(dispatch_results=[{"invalid": True}]),
    )
    identity = WorkflowIdentity("thread-123")
    request = OperationRequest("reboot", {})

    with pytest.raises(OperationExecutionError) as exc_info:
        await executor.execute("operation-1", identity, "request-key", request)

    assert exc_info.value.code == "operation_adapter_invalid_response"
    state = await store.get_operation("operation-1")
    assert state is not None
    assert state.status is OperationStatus.DISPATCHING


async def test_dispatch_cancellation_preserves_recoverable_state() -> None:
    store = InMemoryStateStore()
    executor = DurableOperationExecutor(
        OperationLifecycle(store),
        ScriptedAdapter(dispatch_results=[asyncio.CancelledError()]),
    )
    identity = WorkflowIdentity("thread-123")
    request = OperationRequest("reboot", {})

    with pytest.raises(asyncio.CancelledError):
        await executor.execute("operation-1", identity, "request-key", request)

    state = await store.get_operation("operation-1")
    assert state is not None
    assert state.status is OperationStatus.DISPATCHING


def test_executor_validates_lifecycle_and_adapter() -> None:
    lifecycle = OperationLifecycle(InMemoryStateStore())

    with pytest.raises(TypeError, match="OperationLifecycle"):
        DurableOperationExecutor(object(), ScriptedAdapter())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="dispatch"):
        DurableOperationExecutor(lifecycle, object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="inspect"):
        DurableOperationExecutor(
            lifecycle,
            type("DispatchOnly", (), {"dispatch": lambda self, state: None})(),
        )


async def test_executor_telemetry_excludes_request_and_operation_values(
    in_memory_tracing,
) -> None:
    adapter = ScriptedAdapter(
        dispatch_results=[
            OperationObservation(
                OperationOutcome.COMPLETED,
                {"private_result": "router online"},
            )
        ]
    )
    executor = DurableOperationExecutor(
        OperationLifecycle(InMemoryStateStore()),
        adapter,
    )

    result = await executor.execute(
        "private-operation-id",
        WorkflowIdentity("private-thread", "private-namespace"),
        "private-idempotency-key",
        OperationRequest("private-action", {"private_input": "router secret"}),
    )

    span = next(
        item
        for item in in_memory_tracing.get_finished_spans()
        if item.name == "operation.execute"
    )
    assert span.status.status_code is StatusCode.UNSET
    assert span.attributes == {
        "operation.reused": False,
        "operation.outcome": "dispatched",
        "operation.status": "completed",
        "operation.revision": result.revision,
    }
    serialized = repr(span)
    for secret in (
        "private-operation-id",
        "private-thread",
        "private-namespace",
        "private-idempotency-key",
        "private-action",
        "router secret",
        "router online",
    ):
        assert secret not in serialized
