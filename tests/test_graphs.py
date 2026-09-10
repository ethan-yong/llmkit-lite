from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Annotated, Any, TypedDict

import pytest

from llmkit_lite.graphs import (
    CachedGraph,
    CheckpointConfig,
    CheckpointedGraph,
    checkpointed_graph_config,
    invoke_graph,
    langgraph_config,
    list_extend_reducer,
    run_cached_graph,
)


class FakeGraph:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.should_fail = should_fail
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []

    async def ainvoke(self, state, *, config):
        self.calls.append((state, config))
        if self.should_fail:
            raise RuntimeError("graph failed")
        return {"ok": True, "dependency": config["configurable"]["dependency"]}


class FakeCheckpointer:
    async def aget_tuple(self, config):
        return None


def test_cached_graph_builds_once_and_resets() -> None:
    calls = {"n": 0}

    def build() -> FakeGraph:
        calls["n"] += 1
        return FakeGraph()

    cached = CachedGraph(build)
    first = cached.get()
    second = cached.get()
    assert first is second
    assert calls["n"] == 1

    cached.reset()
    assert cached.get() is not first
    assert calls["n"] == 2


def test_checkpoint_config_normalizes_and_is_frozen() -> None:
    checkpoint = CheckpointConfig(
        " support-thread-123 ",
        " support-agent ",
    )

    assert checkpoint.thread_id == "support-thread-123"
    assert checkpoint.checkpoint_namespace == "support-agent"
    with pytest.raises(FrozenInstanceError):
        checkpoint.thread_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory,expected_error",
    [
        (lambda: CheckpointConfig(""), ValueError),
        (lambda: CheckpointConfig(" "), ValueError),
        (lambda: CheckpointConfig(1), TypeError),
        (lambda: CheckpointConfig("thread", " "), ValueError),
        (lambda: CheckpointConfig("thread", 1), TypeError),
    ],
)
def test_checkpoint_config_rejects_invalid_identifiers(
    factory,
    expected_error,
) -> None:
    with pytest.raises(expected_error):
        factory()


def test_checkpointed_graph_builds_once_with_injected_checkpointer() -> None:
    checkpointer = FakeCheckpointer()
    calls: list[FakeCheckpointer] = []

    def build(injected_checkpointer) -> FakeGraph:
        calls.append(injected_checkpointer)
        return FakeGraph()

    cached = CheckpointedGraph(build, checkpointer)

    first = cached.get()
    assert cached.get() is first
    assert calls == [checkpointer]

    cached.reset()
    assert cached.get() is not first
    assert calls == [checkpointer, checkpointer]


def test_checkpointed_graph_validates_dependencies() -> None:
    with pytest.raises(TypeError, match="builder must be callable"):
        CheckpointedGraph(object(), FakeCheckpointer())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="implement aget_tuple"):
        CheckpointedGraph(lambda checkpointer: FakeGraph(), object())  # type: ignore[arg-type]


def test_checkpointed_graph_instances_keep_independent_caches() -> None:
    first_checkpointer = FakeCheckpointer()
    second_checkpointer = FakeCheckpointer()
    first = CheckpointedGraph(lambda checkpointer: FakeGraph(), first_checkpointer)
    second = CheckpointedGraph(
        lambda checkpointer: FakeGraph(),
        second_checkpointer,
    )

    assert first.get() is not second.get()
    assert first.checkpointer is first_checkpointer
    assert second.checkpointer is second_checkpointer


def test_langgraph_config_wraps_dependencies() -> None:
    assert langgraph_config({"http_client": "client"}, recursion_limit=50) == {
        "configurable": {"http_client": "client"},
        "recursion_limit": 50,
    }


def test_checkpointed_graph_config_preserves_dependencies_and_extras() -> None:
    dependencies = {"http_client": "client", "values": [1]}

    config = checkpointed_graph_config(
        CheckpointConfig(" thread-123 ", " support "),
        dependencies,
        recursion_limit=50,
    )

    assert config == {
        "configurable": {
            "http_client": "client",
            "values": [1],
            "thread_id": "thread-123",
            "checkpoint_ns": "support",
        },
        "recursion_limit": 50,
    }
    assert dependencies == {"http_client": "client", "values": [1]}
    assert config["configurable"] is not dependencies


def test_checkpointed_graph_config_omits_unset_namespace() -> None:
    assert checkpointed_graph_config(CheckpointConfig("thread-123")) == {
        "configurable": {"thread_id": "thread-123"}
    }


def test_checkpointed_graph_config_accepts_matching_reserved_values() -> None:
    config = checkpointed_graph_config(
        CheckpointConfig("thread-123", "support"),
        {"thread_id": "thread-123", "checkpoint_ns": "support"},
    )

    assert config["configurable"] == {
        "thread_id": "thread-123",
        "checkpoint_ns": "support",
    }


@pytest.mark.parametrize(
    "configurable",
    [
        {"thread_id": "other-thread"},
        {"checkpoint_ns": "other-namespace"},
        {"checkpoint_ns": "unexpected"},
    ],
)
def test_checkpointed_graph_config_rejects_conflicting_reserved_values(
    configurable,
) -> None:
    checkpoint = CheckpointConfig("thread-123", "support")

    with pytest.raises(ValueError, match="conflicting checkpoint configuration"):
        checkpointed_graph_config(checkpoint, configurable)


def test_checkpointed_graph_config_validates_inputs() -> None:
    with pytest.raises(TypeError, match="CheckpointConfig"):
        checkpointed_graph_config(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a mapping"):
        checkpointed_graph_config(
            CheckpointConfig("thread-123"),
            ["invalid"],  # type: ignore[arg-type]
        )


async def test_invoke_graph_passes_configurable_dependencies() -> None:
    graph = FakeGraph()
    result = await invoke_graph(
        graph,
        {"input": 1},
        configurable={"dependency": "client"},
    )
    assert result == {"ok": True, "dependency": "client"}
    assert graph.calls[0][1]["configurable"] == {"dependency": "client"}


async def test_invoke_graph_uses_sync_fallback_on_failure() -> None:
    graph = FakeGraph(should_fail=True)

    result = await invoke_graph(
        graph,
        {"input": 1},
        fallback=lambda exc, state: {
            "error": str(exc),
            "input": state["input"],
        },
    )
    assert result == {"error": "graph failed", "input": 1}


async def test_invoke_graph_uses_async_fallback_on_failure() -> None:
    graph = FakeGraph(should_fail=True)

    async def fallback(exc: Exception, state: dict[str, int]):
        return {"error": str(exc), "input": state["input"]}

    result = await invoke_graph(graph, {"input": 2}, fallback=fallback)
    assert result == {"error": "graph failed", "input": 2}


async def test_run_cached_graph_invokes_cached_instance() -> None:
    graph = FakeGraph()
    cached = CachedGraph(lambda: graph)
    result = await run_cached_graph(
        cached,
        {"input": 1},
        configurable={"dependency": "client"},
    )
    assert result == {"ok": True, "dependency": "client"}
    assert len(graph.calls) == 1


def test_list_extend_reducer_handles_none() -> None:
    assert list_extend_reducer(None, ["a"]) == ["a"]
    assert list_extend_reducer(["a"], ["b"]) == ["a", "b"]


async def test_langgraph_fanout_state_merge_with_list_reducer() -> None:
    pytest.importorskip("langgraph")
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Send

    class State(TypedDict, total=False):
        agents: list[str]
        outputs: Annotated[list[str], list_extend_reducer]

    class WorkerState(TypedDict, total=False):
        agent: str

    def load(_: State) -> dict[str, list[str]]:
        return {"outputs": []}

    def route(state):
        return [Send("worker", {"agent": agent}) for agent in state["agents"]]

    async def worker(state):
        return {"outputs": [state["agent"]]}

    graph_builder = StateGraph(State)
    graph_builder.add_node("load", load)
    graph_builder.add_node("worker", worker)
    graph_builder.add_edge(START, "load")
    graph_builder.add_conditional_edges("load", route, ["worker", END])
    graph_builder.add_edge("worker", END)
    graph = graph_builder.compile()

    result = await graph.ainvoke({"agents": ["a", "b", "c"]})
    assert sorted(result["outputs"]) == ["a", "b", "c"]
