"""Application dependency composition and managed resource lifecycle."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from llmkit_lite.observability import set_span_error, trace_span

_ERROR_DETAILS = {
    "runtime_not_started": "application runtime is not started",
    "runtime_dependency_not_found": "runtime dependency is not registered",
    "runtime_startup_failed": "application runtime startup failed",
    "runtime_shutdown_failed": "application runtime shutdown failed",
    "runtime_already_closed": "application runtime is already closed",
}

RuntimeFactory = Callable[[Mapping[str, Any]], Any | Awaitable[Any]]


def _normalize_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _normalize_requirements(
    requirements: Sequence[str],
    dependency_name: str,
) -> tuple[str, ...]:
    if isinstance(requirements, (str, bytes)) or not isinstance(
        requirements,
        Sequence,
    ):
        raise TypeError("runtime dependency requirements must be a sequence")
    normalized: list[str] = []
    seen: set[str] = set()
    for requirement in requirements:
        name = _normalize_identifier(requirement, "runtime dependency requirement")
        if name == dependency_name:
            raise ValueError("runtime dependency must not require itself")
        if name in seen:
            raise ValueError(f"duplicate runtime dependency requirement: {name}")
        seen.add(name)
        normalized.append(name)
    return tuple(normalized)


class RuntimeLifecycleError(Exception):
    """Safe runtime lifecycle failure for application error mapping."""

    def __init__(self, code: str) -> None:
        if code not in _ERROR_DETAILS:
            raise ValueError("unsupported runtime lifecycle error code")
        detail = _ERROR_DETAILS[code]
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class RuntimeDependency:
    """One named dependency factory and its initialization requirements."""

    name: str
    factory: RuntimeFactory
    requires: Sequence[str] = ()

    def __post_init__(self) -> None:
        normalized_name = _normalize_identifier(self.name, "runtime dependency name")
        if not callable(self.factory):
            raise TypeError("runtime dependency factory must be callable")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(
            self,
            "requires",
            _normalize_requirements(self.requires, normalized_name),
        )


class ApplicationRuntime:
    """Initialize, expose, and close an ordered dependency collection."""

    def __init__(self, dependencies: Sequence[RuntimeDependency]) -> None:
        if isinstance(dependencies, (str, bytes)) or not isinstance(
            dependencies,
            Sequence,
        ):
            raise TypeError("runtime dependencies must be a sequence")

        registered: list[RuntimeDependency] = []
        seen: set[str] = set()
        for dependency in dependencies:
            if not isinstance(dependency, RuntimeDependency):
                raise TypeError(
                    "runtime dependencies must contain RuntimeDependency values"
                )
            if dependency.name in seen:
                raise ValueError(
                    f"duplicate runtime dependency registration: {dependency.name}"
                )
            missing = [name for name in dependency.requires if name not in seen]
            if missing:
                raise ValueError(
                    f"runtime dependency requirement is not initialized: {missing[0]}"
                )
            seen.add(dependency.name)
            registered.append(dependency)

        self._registrations = tuple(registered)
        self._values: Mapping[str, Any] = MappingProxyType({})
        self._stack: AsyncExitStack | None = None
        self._state = "new"
        self._lock = asyncio.Lock()

    @property
    def registrations(self) -> tuple[RuntimeDependency, ...]:
        """Return immutable dependency registrations in startup order."""

        return self._registrations

    @property
    def dependencies(self) -> Mapping[str, Any]:
        """Return initialized dependencies through a read-only mapping."""

        self._require_started()
        return self._values

    @property
    def started(self) -> bool:
        """Return whether all dependencies are currently available."""

        return self._state == "started"

    def get(self, name: str) -> Any:
        """Return one initialized dependency by its registered name."""

        self._require_started()
        try:
            normalized_name = _normalize_identifier(name, "runtime dependency name")
        except (TypeError, ValueError) as exc:
            raise RuntimeLifecycleError("runtime_dependency_not_found") from exc
        if normalized_name not in self._values:
            raise RuntimeLifecycleError("runtime_dependency_not_found")
        return self._values[normalized_name]

    async def start(self) -> ApplicationRuntime:
        """Initialize every dependency exactly once in declaration order."""

        async with self._lock:
            if self._state == "started":
                return self
            if self._state == "closed":
                raise RuntimeLifecycleError("runtime_already_closed")

            stack = AsyncExitStack()
            values: dict[str, Any] = {}
            current_name: str | None = None
            with trace_span(
                "runtime.start",
                attributes={"runtime.dependency.count": len(self._registrations)},
            ) as span:
                self._state = "starting"
                try:
                    await stack.__aenter__()
                    for dependency in self._registrations:
                        current_name = dependency.name
                        available = MappingProxyType(dict(values))
                        resource = dependency.factory(available)
                        if inspect.isawaitable(resource):
                            resource = await resource
                        values[dependency.name] = await _enter_resource(
                            stack,
                            resource,
                        )
                except asyncio.CancelledError:
                    await _close_after_failed_start(stack)
                    self._state = "new"
                    _record_failure(span, "cancelled", "runtime_cancelled")
                    raise
                except Exception as exc:
                    await _close_after_failed_start(stack)
                    self._state = "new"
                    error = RuntimeLifecycleError("runtime_startup_failed")
                    _record_failure(
                        span,
                        "failed",
                        error.code,
                        dependency_name=current_name,
                    )
                    raise error from exc
                except BaseException:
                    await _close_after_failed_start(stack)
                    self._state = "new"
                    raise

                self._values = MappingProxyType(dict(values))
                self._stack = stack
                self._state = "started"
                if span is not None:
                    span.set_attribute("runtime.outcome", "started")
                return self

    async def close(self) -> None:
        """Close managed resources once in reverse initialization order."""

        async with self._lock:
            if self._state == "closed":
                return
            if self._state == "new":
                self._state = "closed"
                return

            stack = self._stack
            self._stack = None
            self._values = MappingProxyType({})
            self._state = "closing"
            with trace_span(
                "runtime.shutdown",
                attributes={"runtime.dependency.count": len(self._registrations)},
            ) as span:
                try:
                    if stack is not None:
                        await stack.aclose()
                except asyncio.CancelledError:
                    _record_failure(span, "cancelled", "runtime_cancelled")
                    raise
                except Exception as exc:
                    error = RuntimeLifecycleError("runtime_shutdown_failed")
                    _record_failure(span, "failed", error.code)
                    raise error from exc
                finally:
                    self._state = "closed"

                if span is not None:
                    span.set_attribute("runtime.outcome", "stopped")

    async def __aenter__(self) -> ApplicationRuntime:
        return await self.start()

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.close()

    def _require_started(self) -> None:
        if self._state != "started":
            raise RuntimeLifecycleError("runtime_not_started")


async def _enter_resource(stack: AsyncExitStack, resource: Any) -> Any:
    async_enter = getattr(resource, "__aenter__", None)
    async_exit = getattr(resource, "__aexit__", None)
    sync_enter = getattr(resource, "__enter__", None)
    sync_exit = getattr(resource, "__exit__", None)

    if callable(async_enter) or callable(async_exit):
        if not callable(async_enter) or not callable(async_exit):
            raise TypeError("asynchronous runtime resource is incomplete")
        return await stack.enter_async_context(resource)
    if callable(sync_enter) or callable(sync_exit):
        if not callable(sync_enter) or not callable(sync_exit):
            raise TypeError("synchronous runtime resource is incomplete")
        return stack.enter_context(resource)
    return resource


async def _close_after_failed_start(stack: AsyncExitStack) -> None:
    try:
        await stack.aclose()
    except BaseException:
        pass


def _record_failure(
    span: Any,
    outcome: str,
    error_code: str,
    *,
    dependency_name: str | None = None,
) -> None:
    if span is not None:
        span.set_attribute("runtime.outcome", outcome)
        span.set_attribute("error.type", error_code)
        if dependency_name is not None:
            span.set_attribute("runtime.dependency.name", dependency_name)
    set_span_error(span, error_code)
