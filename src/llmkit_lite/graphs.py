"""Small utilities for running LangGraph workflows in application services."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

from llmkit_lite.observability import set_span_error, trace_span

StateT = TypeVar("StateT")
OutputT = TypeVar("OutputT")
ItemT = TypeVar("ItemT")
GraphT = TypeVar("GraphT")

Fallback = Callable[[Exception, StateT], OutputT | Awaitable[OutputT]]

_WORKFLOW_ERROR_DETAILS = {
    "workflow_timed_out": "workflow execution timed out",
    "workflow_invalid_resume": "workflow could not be resumed",
    "workflow_execution_failed": "workflow execution failed",
    "workflow_invalid_response": "workflow returned an invalid response",
    "workflow_dependency_unavailable": "workflow support is not installed",
}


def _normalize_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _normalize_positive_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return normalized


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Stable workflow identity used by LangGraph checkpointers."""

    thread_id: str
    checkpoint_namespace: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "thread_id",
            _normalize_identifier(self.thread_id, "checkpoint thread ID"),
        )
        if self.checkpoint_namespace is not None:
            object.__setattr__(
                self,
                "checkpoint_namespace",
                _normalize_identifier(
                    self.checkpoint_namespace,
                    "checkpoint namespace",
                ),
            )


class Checkpointer(Protocol):
    """Minimum asynchronous lookup contract for an injected checkpointer."""

    async def aget_tuple(self, config: Mapping[str, Any]) -> Any:
        """Return the checkpoint associated with a LangGraph config."""


@dataclass
class CachedGraph(Generic[GraphT]):
    """Lazily compile and cache a graph-like object."""

    builder: Callable[[], GraphT]
    _compiled: GraphT | None = None

    def get(self) -> GraphT:
        if self._compiled is None:
            self._compiled = self.builder()
        return self._compiled

    def reset(self) -> None:
        self._compiled = None


@dataclass
class CheckpointedGraph(Generic[GraphT]):
    """Lazily build and cache a graph with an injected checkpointer."""

    builder: Callable[[Checkpointer], GraphT]
    checkpointer: Checkpointer
    _compiled: GraphT | None = None

    def __post_init__(self) -> None:
        if not callable(self.builder):
            raise TypeError("checkpointed graph builder must be callable")
        if not callable(getattr(self.checkpointer, "aget_tuple", None)):
            raise TypeError("checkpointer must implement aget_tuple")

    def get(self) -> GraphT:
        """Build once with the injected checkpointer and return the graph."""

        if self._compiled is None:
            self._compiled = self.builder(self.checkpointer)
        return self._compiled

    def reset(self) -> None:
        """Discard the compiled graph without mutating the checkpointer."""

        self._compiled = None


class WorkflowExecutionError(Exception):
    """Safe checkpointed-workflow failure for application error mapping."""

    def __init__(self, code: str) -> None:
        if code not in _WORKFLOW_ERROR_DETAILS:
            raise ValueError("unsupported workflow execution error code")
        detail = _WORKFLOW_ERROR_DETAILS[code]
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class WorkflowExecutionPolicy:
    """Overall deadline applied to each start or resume execution."""

    deadline_seconds: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "deadline_seconds",
            _normalize_positive_number(
                self.deadline_seconds,
                "workflow deadline",
            ),
        )


@dataclass(frozen=True, slots=True)
class WorkflowInterrupt:
    """One application-visible request for external workflow input."""

    id: str
    value: Any

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "id",
            _normalize_identifier(self.id, "workflow interrupt ID"),
        )


@dataclass(frozen=True, slots=True)
class WorkflowRunResult:
    """Completed output plus any pending human-input interruptions."""

    output: Any
    interrupts: tuple[WorkflowInterrupt, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.interrupts, tuple) or not all(
            isinstance(item, WorkflowInterrupt) for item in self.interrupts
        ):
            raise TypeError("workflow interrupts must be a tuple of WorkflowInterrupt")

    @property
    def interrupted(self) -> bool:
        """Return whether the workflow is waiting for external input."""

        return bool(self.interrupts)


WorkflowOperation = Callable[[], Awaitable[Any]]
WorkflowTimeoutRunner = Callable[[float, WorkflowOperation], Awaitable[Any]]
ResumeCommandFactory = Callable[[Any], Any]


async def _run_with_deadline(
    deadline_seconds: float,
    operation: WorkflowOperation,
) -> Any:
    try:
        async with asyncio.timeout(deadline_seconds):
            return await operation()
    except TimeoutError as exc:
        raise WorkflowExecutionError("workflow_timed_out") from exc


def _create_resume_command(value: Any) -> Any:
    try:
        from langgraph.types import Command
    except ImportError as exc:
        raise WorkflowExecutionError("workflow_dependency_unavailable") from exc
    return Command(resume=value)


class CheckpointedWorkflowRunner(Generic[GraphT]):
    """Start and resume a checkpointed graph within an overall deadline."""

    def __init__(
        self,
        graph: CheckpointedGraph[GraphT],
        *,
        execution_policy: WorkflowExecutionPolicy | None = None,
        _timeout_runner: WorkflowTimeoutRunner = _run_with_deadline,
        _resume_factory: ResumeCommandFactory = _create_resume_command,
    ) -> None:
        if not isinstance(graph, CheckpointedGraph):
            raise TypeError("graph must be a CheckpointedGraph")
        resolved_policy = (
            WorkflowExecutionPolicy()
            if execution_policy is None
            else execution_policy
        )
        if not isinstance(resolved_policy, WorkflowExecutionPolicy):
            raise TypeError("execution policy must be WorkflowExecutionPolicy")
        if not callable(_timeout_runner):
            raise TypeError("workflow timeout runner must be callable")
        if not callable(_resume_factory):
            raise TypeError("workflow resume factory must be callable")
        self.graph = graph
        self.execution_policy = resolved_policy
        self._timeout_runner = _timeout_runner
        self._resume_factory = _resume_factory

    async def start(
        self,
        state: StateT,
        checkpoint: CheckpointConfig,
        *,
        configurable: Mapping[str, Any] | None = None,
        deadline_seconds: float | None = None,
        **extra_config: Any,
    ) -> WorkflowRunResult:
        """Start a workflow under the supplied persistent identity."""

        return await self._execute(
            "start",
            lambda: state,
            checkpoint,
            configurable=configurable,
            deadline_seconds=deadline_seconds,
            extra_config=extra_config,
        )

    async def resume(
        self,
        value: Any,
        checkpoint: CheckpointConfig,
        *,
        configurable: Mapping[str, Any] | None = None,
        deadline_seconds: float | None = None,
        **extra_config: Any,
    ) -> WorkflowRunResult:
        """Resume the pending interruption for the supplied workflow identity."""

        return await self._execute(
            "resume",
            lambda: self._resume_factory(value),
            checkpoint,
            configurable=configurable,
            deadline_seconds=deadline_seconds,
            extra_config=extra_config,
        )

    async def _execute(
        self,
        operation_name: str,
        input_factory: Callable[[], Any],
        checkpoint: CheckpointConfig,
        *,
        configurable: Mapping[str, Any] | None,
        deadline_seconds: float | None,
        extra_config: Mapping[str, Any],
    ) -> WorkflowRunResult:
        deadline = (
            self.execution_policy.deadline_seconds
            if deadline_seconds is None
            else _normalize_positive_number(
                deadline_seconds,
                "workflow deadline",
            )
        )
        config = checkpointed_graph_config(
            checkpoint,
            configurable,
            **extra_config,
        )
        with trace_span(
            "graph.workflow.execute",
            attributes={"workflow.operation": operation_name},
        ) as span:
            try:
                graph_input = input_factory()
                graph = self.graph.get()
                ainvoke = getattr(graph, "ainvoke", None)
                if not callable(ainvoke):
                    raise TypeError("compiled graph must implement ainvoke")

                async def invoke() -> Any:
                    return await ainvoke(graph_input, config=config)

                output = await self._timeout_runner(deadline, invoke)
                result = _workflow_result(output)
            except asyncio.CancelledError:
                _record_workflow_failure(span, "cancelled", "workflow_cancelled")
                raise
            except WorkflowExecutionError as exc:
                _record_workflow_failure(span, "failed", exc.code)
                raise
            except TimeoutError as exc:
                error = WorkflowExecutionError("workflow_timed_out")
                _record_workflow_failure(span, "failed", error.code)
                raise error from exc
            except Exception as exc:
                code = (
                    "workflow_invalid_resume"
                    if operation_name == "resume"
                    else "workflow_execution_failed"
                )
                error = WorkflowExecutionError(code)
                _record_workflow_failure(span, "failed", error.code)
                raise error from exc

            if span is not None:
                span.set_attribute(
                    "workflow.outcome",
                    "interrupted" if result.interrupted else "completed",
                )
                span.set_attribute("workflow.interrupt.count", len(result.interrupts))
            return result


def _record_workflow_failure(span: Any, outcome: str, error_code: str) -> None:
    if span is not None:
        span.set_attribute("workflow.outcome", outcome)
        span.set_attribute("error.type", error_code)
    set_span_error(span, error_code)


def _workflow_result(output: Any) -> WorkflowRunResult:
    if not isinstance(output, Mapping) or "__interrupt__" not in output:
        return WorkflowRunResult(output=output)

    raw_interrupts = output["__interrupt__"]
    if isinstance(raw_interrupts, (str, bytes)) or not isinstance(
        raw_interrupts,
        Sequence,
    ):
        raise WorkflowExecutionError("workflow_invalid_response")

    interrupts: list[WorkflowInterrupt] = []
    for raw_interrupt in raw_interrupts:
        if isinstance(raw_interrupt, Mapping):
            interrupt_id = raw_interrupt.get("id")
            value = raw_interrupt.get("value")
        else:
            interrupt_id = getattr(raw_interrupt, "id", None)
            value = getattr(raw_interrupt, "value", None)
        try:
            interrupts.append(WorkflowInterrupt(interrupt_id, value))
        except (TypeError, ValueError) as exc:
            raise WorkflowExecutionError("workflow_invalid_response") from exc
    return WorkflowRunResult(output=output, interrupts=tuple(interrupts))


def langgraph_config(
    configurable: Mapping[str, Any] | None = None,
    **extra_config: Any,
) -> dict[str, Any]:
    """Build a LangGraph config payload with configurable dependencies."""

    config = dict(extra_config)
    config["configurable"] = dict(configurable or {})
    return config


def checkpointed_graph_config(
    checkpoint: CheckpointConfig,
    configurable: Mapping[str, Any] | None = None,
    **extra_config: Any,
) -> dict[str, Any]:
    """Build a LangGraph config with one authoritative checkpoint identity."""

    if not isinstance(checkpoint, CheckpointConfig):
        raise TypeError("checkpoint must be a CheckpointConfig")
    if configurable is not None and not isinstance(configurable, Mapping):
        raise TypeError("configurable dependencies must be a mapping or None")

    merged = dict(configurable or {})
    reserved = {"thread_id": checkpoint.thread_id}
    if checkpoint.checkpoint_namespace is not None:
        reserved["checkpoint_ns"] = checkpoint.checkpoint_namespace

    for key in ("thread_id", "checkpoint_ns"):
        if key not in merged:
            continue
        if key not in reserved or merged[key] != reserved[key]:
            raise ValueError(f"conflicting checkpoint configuration: {key}")

    merged.update(reserved)
    return langgraph_config(merged, **extra_config)


async def invoke_graph(
    graph: Any,
    state: StateT,
    *,
    configurable: Mapping[str, Any] | None = None,
    fallback: Fallback[StateT, OutputT] | None = None,
    **extra_config: Any,
) -> Any | OutputT:
    """Invoke a graph and optionally soft-fallback on failures."""

    try:
        return await graph.ainvoke(
            state,
            config=langgraph_config(configurable, **extra_config),
        )
    except Exception as exc:
        if fallback is None:
            raise
        maybe = fallback(exc, state)
        if inspect.isawaitable(maybe):
            return await maybe
        return maybe


async def run_cached_graph(
    cached: CachedGraph[GraphT],
    state: StateT,
    *,
    configurable: Mapping[str, Any] | None = None,
    fallback: Fallback[StateT, OutputT] | None = None,
    **extra_config: Any,
) -> Any | OutputT:
    """Invoke a lazily compiled graph."""

    return await invoke_graph(
        cached.get(),
        state,
        configurable=configurable,
        fallback=fallback,
        **extra_config,
    )


def list_extend_reducer(
    left: Sequence[ItemT] | None,
    right: Sequence[ItemT] | None,
) -> list[ItemT]:
    """Reducer for fan-out nodes that append list contributions."""

    return [*(left or []), *(right or [])]
