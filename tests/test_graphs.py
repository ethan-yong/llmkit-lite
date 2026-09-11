from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from typing import Annotated, Any, TypedDict

import pytest
from opentelemetry.trace import StatusCode

from llmkit_lite.graphs import (
    CachedGraph,
    CheckpointConfig,
    CheckpointedGraph,
    CheckpointedWorkflowRunner,
    WorkflowExecutionError,
    WorkflowExecutionPolicy,
    WorkflowIdentity,
    WorkflowInterrupt,
    WorkflowRunResult,
    checkpointed_graph_config,
    invoke_graph,
    langgraph_config,
    list_extend_reducer,
    run_cached_graph,
    workflow_run_config,
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


class ScriptedWorkflowGraph:
    def __init__(self, *results: Any) -> None:
        self.results = list(results)
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    async def ainvoke(self, graph_input: Any, *, config: dict[str, Any]) -> Any:
        self.calls.append((graph_input, config))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def _workflow_runner(
    graph: ScriptedWorkflowGraph,
    *,
    policy: WorkflowExecutionPolicy | None = None,
    timeout_runner: Any | None = None,
    resume_factory: Any | None = None,
) -> CheckpointedWorkflowRunner[ScriptedWorkflowGraph]:
    checkpointed = CheckpointedGraph(lambda checkpointer: graph, FakeCheckpointer())
    kwargs: dict[str, Any] = {"execution_policy": policy}
    if timeout_runner is not None:
        kwargs["_timeout_runner"] = timeout_runner
    if resume_factory is not None:
        kwargs["_resume_factory"] = resume_factory
    return CheckpointedWorkflowRunner(checkpointed, **kwargs)


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


def test_workflow_identity_normalizes_and_is_frozen() -> None:
    identity = WorkflowIdentity(
        " support-thread-123 ",
        " support-agent ",
    )

    assert identity.thread_id == "support-thread-123"
    assert identity.checkpoint_namespace == "support-agent"
    with pytest.raises(FrozenInstanceError):
        identity.thread_id = "changed"  # type: ignore[misc]


def test_checkpoint_config_remains_a_compatibility_alias() -> None:
    identity = CheckpointConfig("thread-123", "support")

    assert isinstance(identity, WorkflowIdentity)
    assert identity == WorkflowIdentity("thread-123", "support")


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


def test_workflow_identity_rejects_invalid_identifiers() -> None:
    with pytest.raises(ValueError):
        WorkflowIdentity(" ")
    with pytest.raises(TypeError):
        WorkflowIdentity(1)  # type: ignore[arg-type]


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


def test_workflow_run_config_separates_identity_from_runtime_state() -> None:
    state = {"messages": [{"role": "user", "content": "status"}]}
    dependencies = {"http_client": "client"}

    config = workflow_run_config(
        WorkflowIdentity("thread-123", "support"),
        dependencies,
        recursion_limit=25,
    )

    assert config == {
        "configurable": {
            "http_client": "client",
            "thread_id": "thread-123",
            "checkpoint_ns": "support",
        },
        "recursion_limit": 25,
    }
    assert "messages" not in config
    assert state == {"messages": [{"role": "user", "content": "status"}]}
    assert dependencies == {"http_client": "client"}


def test_workflow_run_config_validates_identity() -> None:
    with pytest.raises(TypeError, match="WorkflowIdentity"):
        workflow_run_config(object())  # type: ignore[arg-type]


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


def test_workflow_execution_models_validate_and_freeze_values() -> None:
    policy = WorkflowExecutionPolicy()
    interrupt = WorkflowInterrupt(" approval-1 ", {"question": "approve?"})
    result = WorkflowRunResult(output={"pending": True}, interrupts=(interrupt,))

    assert policy.deadline_seconds == 30
    assert interrupt.id == "approval-1"
    assert result.interrupted is True
    with pytest.raises(FrozenInstanceError):
        policy.deadline_seconds = 5  # type: ignore[misc]
    with pytest.raises(TypeError):
        WorkflowRunResult(output={}, interrupts=[interrupt])  # type: ignore[arg-type]

    for value in (0, -1, float("inf"), True):
        with pytest.raises((TypeError, ValueError)):
            WorkflowExecutionPolicy(value)  # type: ignore[arg-type]


def test_workflow_runner_validates_dependencies() -> None:
    graph = CheckpointedGraph(
        lambda checkpointer: ScriptedWorkflowGraph({}),
        FakeCheckpointer(),
    )

    with pytest.raises(TypeError, match="CheckpointedGraph"):
        CheckpointedWorkflowRunner(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="WorkflowExecutionPolicy"):
        CheckpointedWorkflowRunner(graph, execution_policy=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="timeout runner"):
        CheckpointedWorkflowRunner(graph, _timeout_runner=object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="resume factory"):
        CheckpointedWorkflowRunner(graph, _resume_factory=object())  # type: ignore[arg-type]


async def test_workflow_start_passes_state_config_and_overall_deadline() -> None:
    graph = ScriptedWorkflowGraph({"answer": "done"})
    deadlines: list[float] = []

    async def timeout_runner(seconds: float, operation) -> Any:
        deadlines.append(seconds)
        return await operation()

    runner = _workflow_runner(
        graph,
        policy=WorkflowExecutionPolicy(25),
        timeout_runner=timeout_runner,
    )
    dependencies = {"http_client": "client"}
    state = {"question": "status"}

    result = await runner.start(
        state,
        CheckpointConfig("thread-123", "support"),
        configurable=dependencies,
        deadline_seconds=10,
        recursion_limit=20,
    )

    assert result.output == {"answer": "done"}
    assert result.interrupted is False
    assert deadlines == [10.0]
    assert graph.calls == [
        (
            state,
            {
                "configurable": {
                    "http_client": "client",
                    "thread_id": "thread-123",
                    "checkpoint_ns": "support",
                },
                "recursion_limit": 20,
            },
        )
    ]
    assert dependencies == {"http_client": "client"}
    assert "question" not in graph.calls[0][1]
    assert "question" not in graph.calls[0][1]["configurable"]


async def test_workflow_runner_accepts_legacy_checkpoint_keyword() -> None:
    graph = ScriptedWorkflowGraph({"answer": "done"})
    runner = _workflow_runner(graph)

    await runner.start({}, checkpoint=CheckpointConfig("thread-123"))

    assert graph.calls[0][1]["configurable"]["thread_id"] == "thread-123"


async def test_workflow_runner_requires_exactly_one_identity() -> None:
    runner = _workflow_runner(ScriptedWorkflowGraph({"answer": "done"}))
    identity = WorkflowIdentity("thread-123")

    with pytest.raises(TypeError, match="WorkflowIdentity"):
        await runner.start({})
    with pytest.raises(TypeError, match="either identity or checkpoint"):
        await runner.start({}, identity, checkpoint=identity)


async def test_workflow_start_returns_pending_interruptions() -> None:
    raw_interrupt = type(
        "FakeInterrupt",
        (),
        {"id": "approval-1", "value": {"question": "approve?"}},
    )()
    output = {
        "status": "waiting",
        "__interrupt__": (
            raw_interrupt,
            {"id": "approval-2", "value": {"question": "confirm?"}},
        ),
    }
    runner = _workflow_runner(ScriptedWorkflowGraph(output))

    result = await runner.start({}, CheckpointConfig("thread-123"))

    assert result.output is output
    assert result.interrupted is True
    assert [(item.id, item.value) for item in result.interrupts] == [
        ("approval-1", {"question": "approve?"}),
        ("approval-2", {"question": "confirm?"}),
    ]


async def test_workflow_resume_reuses_thread_and_builds_resume_command() -> None:
    graph = ScriptedWorkflowGraph(
        {"__interrupt__": ({"id": "approval-1", "value": "approve"},)},
        {"approved": True},
    )
    resume_values: list[Any] = []

    def resume_factory(value: Any) -> dict[str, Any]:
        resume_values.append(value)
        return {"resume": value}

    runner = _workflow_runner(graph, resume_factory=resume_factory)
    checkpoint = CheckpointConfig("thread-123", "support")

    started = await runner.start({"request": "reboot"}, checkpoint)
    resumed = await runner.resume({"approved": True}, checkpoint)

    assert started.interrupted is True
    assert resumed.output == {"approved": True}
    assert resume_values == [{"approved": True}]
    assert graph.calls[1][0] == {"resume": {"approved": True}}
    assert graph.calls[0][1] == graph.calls[1][1]
    assert graph.calls[1][1]["configurable"]["thread_id"] == "thread-123"


@pytest.mark.parametrize(
    "output",
    [
        {"__interrupt__": None},
        {"__interrupt__": "invalid"},
        {"__interrupt__": ({"id": "", "value": "private"},)},
        {"__interrupt__": (object(),)},
    ],
)
async def test_workflow_rejects_malformed_interruptions(output) -> None:
    runner = _workflow_runner(ScriptedWorkflowGraph(output))

    with pytest.raises(WorkflowExecutionError) as exc_info:
        await runner.start({}, CheckpointConfig("thread-123"))

    assert exc_info.value.code == "workflow_invalid_response"
    assert "private" not in str(exc_info.value)


async def test_workflow_maps_start_and_resume_failures_safely() -> None:
    start_runner = _workflow_runner(
        ScriptedWorkflowGraph(RuntimeError("private start failure"))
    )
    with pytest.raises(WorkflowExecutionError) as start_info:
        await start_runner.start({}, CheckpointConfig("thread-123"))
    assert start_info.value.code == "workflow_execution_failed"
    assert "private" not in str(start_info.value)

    resume_runner = _workflow_runner(
        ScriptedWorkflowGraph(RuntimeError("private resume failure")),
        resume_factory=lambda value: {"resume": value},
    )
    with pytest.raises(WorkflowExecutionError) as resume_info:
        await resume_runner.resume(True, CheckpointConfig("thread-123"))
    assert resume_info.value.code == "workflow_invalid_resume"
    assert "private" not in str(resume_info.value)


async def test_workflow_timeout_is_safe_without_real_waiting() -> None:
    async def timeout_runner(seconds: float, operation) -> Any:
        assert seconds == 5
        raise TimeoutError("private timeout detail")

    graph = ScriptedWorkflowGraph({"unused": True})
    runner = _workflow_runner(graph, timeout_runner=timeout_runner)

    with pytest.raises(WorkflowExecutionError) as exc_info:
        await runner.start(
            {"private": "payload"},
            CheckpointConfig("thread-123"),
            deadline_seconds=5,
        )

    assert exc_info.value.code == "workflow_timed_out"
    assert "private" not in str(exc_info.value)
    assert graph.calls == []


async def test_workflow_cancellation_propagates_unchanged() -> None:
    cancelled = asyncio.CancelledError()
    runner = _workflow_runner(ScriptedWorkflowGraph(cancelled))

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await runner.start({}, CheckpointConfig("thread-123"))

    assert exc_info.value is cancelled


async def test_workflow_telemetry_excludes_state_resume_and_thread_values(
    in_memory_tracing,
) -> None:
    runner = _workflow_runner(
        ScriptedWorkflowGraph(RuntimeError("private exception message"))
    )

    with pytest.raises(WorkflowExecutionError):
        await runner.start(
            {"prompt": "private prompt"},
            CheckpointConfig("private thread", "private namespace"),
        )

    span = in_memory_tracing.get_finished_spans()[0]
    assert span.name == "graph.workflow.execute"
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes == {
        "workflow.operation": "start",
        "workflow.outcome": "failed",
        "error.type": "workflow_execution_failed",
        "exception.type": "llmkit_lite.graphs.WorkflowExecutionError",
    }
    serialized = repr(span)
    for secret in (
        "private prompt",
        "private thread",
        "private namespace",
        "private exception message",
    ):
        assert secret not in serialized


async def test_interrupted_workflow_telemetry_records_only_safe_counts(
    in_memory_tracing,
) -> None:
    runner = _workflow_runner(
        ScriptedWorkflowGraph(
            {
                "private_output": "secret result",
                "__interrupt__": (
                    {"id": "private interrupt", "value": "secret approval"},
                ),
            }
        )
    )

    await runner.start(
        {"prompt": "secret prompt"},
        CheckpointConfig("secret thread"),
    )

    span = in_memory_tracing.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.UNSET
    assert span.attributes == {
        "workflow.operation": "start",
        "workflow.outcome": "interrupted",
        "workflow.interrupt.count": 1,
    }
    serialized = repr(span)
    for secret in (
        "secret result",
        "secret approval",
        "secret prompt",
        "secret thread",
        "private interrupt",
    ):
        assert secret not in serialized


async def test_langgraph_interrupt_resumes_with_same_checkpoint() -> None:
    pytest.importorskip("langgraph")
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import interrupt

    class ApprovalState(TypedDict, total=False):
        request: str
        approved: bool

    def request_approval(state: ApprovalState) -> dict[str, bool]:
        approved = interrupt({"question": "Approve router reboot?"})
        return {"approved": bool(approved)}

    builder = StateGraph(ApprovalState)
    builder.add_node("approval", request_approval)
    builder.add_edge(START, "approval")
    builder.add_edge("approval", END)
    graph = CheckpointedGraph(
        lambda checkpointer: builder.compile(checkpointer=checkpointer),
        InMemorySaver(),
    )
    runner = CheckpointedWorkflowRunner(graph)
    checkpoint = CheckpointConfig("approval-thread")

    started = await runner.start({"request": "reboot"}, checkpoint)
    resumed = await runner.resume(True, checkpoint)

    assert started.interrupted is True
    assert started.interrupts[0].value == {"question": "Approve router reboot?"}
    assert resumed.interrupted is False
    assert resumed.output["approved"] is True


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
