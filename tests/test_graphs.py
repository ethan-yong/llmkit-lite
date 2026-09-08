from __future__ import annotations

from typing import Annotated, Any, TypedDict

import pytest

from llmkit_lite.graphs import (
    CachedGraph,
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


def test_langgraph_config_wraps_dependencies() -> None:
    assert langgraph_config({"http_client": "client"}, recursion_limit=50) == {
        "configurable": {"http_client": "client"},
        "recursion_limit": 50,
    }


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
