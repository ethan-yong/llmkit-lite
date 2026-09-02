"""Small utilities for running LangGraph workflows in application services."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

StateT = TypeVar("StateT")
OutputT = TypeVar("OutputT")
ItemT = TypeVar("ItemT")
GraphT = TypeVar("GraphT")

Fallback = Callable[[Exception, StateT], OutputT | Awaitable[OutputT]]


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


def langgraph_config(
    configurable: Mapping[str, Any] | None = None,
    **extra_config: Any,
) -> dict[str, Any]:
    """Build a LangGraph config payload with configurable dependencies."""

    config = dict(extra_config)
    config["configurable"] = dict(configurable or {})
    return config


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
