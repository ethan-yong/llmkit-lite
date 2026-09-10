"""Small utilities for running LangGraph workflows in application services."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

StateT = TypeVar("StateT")
OutputT = TypeVar("OutputT")
ItemT = TypeVar("ItemT")
GraphT = TypeVar("GraphT")

Fallback = Callable[[Exception, StateT], OutputT | Awaitable[OutputT]]


def _normalize_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
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
