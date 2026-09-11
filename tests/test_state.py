from __future__ import annotations

import asyncio
import json
from dataclasses import FrozenInstanceError

import pytest

from llmkit_lite.graphs import WorkflowIdentity
from llmkit_lite.state import (
    ConversationState,
    InMemoryStateStore,
    OperationState,
    OperationStatus,
    StateStoreError,
)


def test_conversation_state_normalizes_and_deeply_freezes_values() -> None:
    values = {
        "messages": [{"role": "user", "content": "status"}],
        "finding": {"online": True, "latency": 1.5},
    }

    state = ConversationState(
        WorkflowIdentity("thread-123", "support"),
        1,
        values,
    )
    values["messages"][0]["content"] = "changed"

    assert state.values["messages"] == ({"role": "user", "content": "status"},)
    assert state.values["finding"] == {"online": True, "latency": 1.5}
    with pytest.raises(TypeError):
        state.values["other"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        state.values["finding"]["online"] = False  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        state.revision = 2  # type: ignore[misc]


def test_operation_state_normalizes_identifiers_and_freezes_values() -> None:
    state = OperationState(
        operation_id=" reboot-123 ",
        identity=WorkflowIdentity("thread-123"),
        idempotency_key=" reboot-key-123 ",
        status=OperationStatus.PENDING,
        revision=1,
        values={"request": {"device": "router-1"}},
    )

    assert state.operation_id == "reboot-123"
    assert state.idempotency_key == "reboot-key-123"
    assert state.values == {"request": {"device": "router-1"}}
    with pytest.raises(TypeError):
        state.values["request"]["device"] = "router-2"  # type: ignore[index]


def test_state_records_export_detached_json_compatible_values() -> None:
    identity = WorkflowIdentity("thread-123", "support")
    conversation = ConversationState(identity, 2, {"messages": ["hello"]})
    operation = OperationState(
        "operation-123",
        identity,
        "key-123",
        OperationStatus.COMPLETED,
        3,
        {"result": {"accepted": True}},
    )

    conversation_data = conversation.to_dict()
    operation_data = operation.to_dict()
    conversation_data["values"]["messages"].append("changed")
    operation_data["values"]["result"]["accepted"] = False

    assert json.loads(json.dumps(conversation.to_dict())) == {
        "thread_id": "thread-123",
        "checkpoint_namespace": "support",
        "revision": 2,
        "values": {"messages": ["hello"]},
    }
    assert json.loads(json.dumps(operation.to_dict())) == {
        "operation_id": "operation-123",
        "thread_id": "thread-123",
        "checkpoint_namespace": "support",
        "idempotency_key": "key-123",
        "status": "completed",
        "revision": 3,
        "values": {"result": {"accepted": True}},
    }


@pytest.mark.parametrize(
    "factory,expected_error",
    [
        (lambda: ConversationState(object(), 1, {}), TypeError),
        (lambda: ConversationState(WorkflowIdentity("thread"), 0, {}), ValueError),
        (lambda: ConversationState(WorkflowIdentity("thread"), True, {}), TypeError),
        (
            lambda: ConversationState(
                WorkflowIdentity("thread"),
                1,
                {1: "invalid"},
            ),
            TypeError,
        ),
        (
            lambda: ConversationState(
                WorkflowIdentity("thread"),
                1,
                {"invalid": object()},
            ),
            TypeError,
        ),
        (
            lambda: ConversationState(
                WorkflowIdentity("thread"),
                1,
                {"invalid": float("inf")},
            ),
            TypeError,
        ),
        (
            lambda: OperationState(
                " ",
                WorkflowIdentity("thread"),
                "key",
                OperationStatus.PENDING,
                1,
                {},
            ),
            ValueError,
        ),
        (
            lambda: OperationState(
                "operation",
                WorkflowIdentity("thread"),
                " ",
                OperationStatus.PENDING,
                1,
                {},
            ),
            ValueError,
        ),
        (
            lambda: OperationState(
                "operation",
                WorkflowIdentity("thread"),
                "key",
                "pending",
                1,
                {},
            ),
            TypeError,
        ),
    ],
)
def test_state_models_reject_invalid_records(factory, expected_error) -> None:
    with pytest.raises(expected_error):
        factory()


async def test_store_creates_loads_and_updates_conversation_state() -> None:
    store = InMemoryStateStore()
    identity = WorkflowIdentity("thread-123", "support")
    values = {"messages": ["hello"], "pending_approval": None}

    created = await store.save_conversation(identity, values)
    values["messages"].append("changed")
    loaded = await store.get_conversation(identity)

    assert created.revision == 1
    assert loaded is created
    assert loaded.values == {
        "messages": ("hello",),
        "pending_approval": None,
    }

    updated = await store.save_conversation(
        identity,
        {"messages": ["hello", "world"]},
        expected_revision=created.revision,
    )

    assert updated.revision == 2
    assert updated.values["messages"] == ("hello", "world")
    assert await store.get_conversation(identity) is updated


async def test_store_isolates_conversations_by_complete_identity() -> None:
    store = InMemoryStateStore()
    first = WorkflowIdentity("thread-123", "support")
    second = WorkflowIdentity("thread-123", "billing")

    await store.save_conversation(first, {"team": "support"})

    assert await store.get_conversation(first) is not None
    assert await store.get_conversation(second) is None


async def test_store_creates_loads_and_updates_operation_state() -> None:
    store = InMemoryStateStore()
    identity = WorkflowIdentity("thread-123")

    created = await store.save_operation(
        " reboot-123 ",
        identity,
        " reboot-key-123 ",
        OperationStatus.PENDING,
        {"command": {"device": "router-1"}},
    )
    updated = await store.save_operation(
        "reboot-123",
        identity,
        "reboot-key-123",
        OperationStatus.DISPATCHING,
        {"downstream_id": "command-456"},
        expected_revision=created.revision,
    )

    assert created.operation_id == "reboot-123"
    assert created.idempotency_key == "reboot-key-123"
    assert created.status is OperationStatus.PENDING
    assert updated.status is OperationStatus.DISPATCHING
    assert updated.revision == 2
    assert updated.values == {"downstream_id": "command-456"}
    assert await store.get_operation(" reboot-123 ") is updated


async def test_conversation_and_operation_records_use_separate_namespaces() -> None:
    store = InMemoryStateStore()
    identity = WorkflowIdentity("shared-id")

    conversation = await store.save_conversation(identity, {"kind": "conversation"})
    operation = await store.save_operation(
        "shared-id",
        identity,
        "operation-key",
        OperationStatus.PENDING,
        {"kind": "operation"},
    )

    assert conversation.values["kind"] == "conversation"
    assert operation.values["kind"] == "operation"


async def test_store_enforces_create_and_compare_and_set_semantics() -> None:
    store = InMemoryStateStore()
    identity = WorkflowIdentity("thread-123")
    created = await store.save_conversation(identity, {"version": 1})

    with pytest.raises(StateStoreError) as duplicate_info:
        await store.save_conversation(identity, {"private": "duplicate payload"})
    assert duplicate_info.value.code == "state_already_exists"

    with pytest.raises(StateStoreError) as stale_info:
        await store.save_conversation(
            identity,
            {"private": "stale payload"},
            expected_revision=created.revision + 1,
        )
    assert stale_info.value.code == "state_revision_conflict"

    missing = WorkflowIdentity("missing-thread")
    with pytest.raises(StateStoreError) as missing_info:
        await store.save_conversation(
            missing,
            {},
            expected_revision=1,
        )
    assert missing_info.value.code == "state_record_not_found"

    for error in (duplicate_info.value, stale_info.value, missing_info.value):
        assert "private" not in str(error)
        assert "payload" not in str(error)


async def test_operation_identity_and_idempotency_key_cannot_change() -> None:
    store = InMemoryStateStore()
    identity = WorkflowIdentity("thread-123")
    created = await store.save_operation(
        "operation-123",
        identity,
        "key-123",
        OperationStatus.PENDING,
        {},
    )

    with pytest.raises(StateStoreError) as identity_info:
        await store.save_operation(
            "operation-123",
            WorkflowIdentity("other-thread"),
            "key-123",
            OperationStatus.DISPATCHING,
            {"private": "identity conflict"},
            expected_revision=created.revision,
        )
    assert identity_info.value.code == "state_invalid_record"

    with pytest.raises(StateStoreError) as key_info:
        await store.save_operation(
            "operation-123",
            identity,
            "other-key",
            OperationStatus.DISPATCHING,
            {"private": "key conflict"},
            expected_revision=created.revision,
        )
    assert key_info.value.code == "state_invalid_record"

    assert "private" not in str(identity_info.value)
    assert "private" not in str(key_info.value)


async def test_store_allows_only_one_concurrent_revision_update() -> None:
    store = InMemoryStateStore()
    identity = WorkflowIdentity("thread-123")
    created = await store.save_conversation(identity, {"winner": None})

    async def update(candidate: str) -> ConversationState | StateStoreError:
        try:
            return await store.save_conversation(
                identity,
                {"winner": candidate},
                expected_revision=created.revision,
            )
        except StateStoreError as exc:
            return exc

    outcomes = await asyncio.gather(update("a"), update("b"))

    successes = [item for item in outcomes if isinstance(item, ConversationState)]
    conflicts = [item for item in outcomes if isinstance(item, StateStoreError)]
    assert len(successes) == 1
    assert successes[0].revision == 2
    assert len(conflicts) == 1
    assert conflicts[0].code == "state_revision_conflict"


@pytest.mark.parametrize("expected_revision", [0, -1, True, 1.5])
async def test_store_rejects_invalid_expected_revisions(expected_revision) -> None:
    store = InMemoryStateStore()

    with pytest.raises((TypeError, ValueError)):
        await store.save_conversation(
            WorkflowIdentity("thread-123"),
            {},
            expected_revision=expected_revision,
        )
