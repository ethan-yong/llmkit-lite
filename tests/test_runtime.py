from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from opentelemetry.trace import StatusCode

from llmkit_lite.runtime import (
    ApplicationRuntime,
    RuntimeDependency,
    RuntimeLifecycleError,
)


def _dependency(
    name: str = "client",
    factory: Any | None = None,
    *,
    requires: tuple[str, ...] = (),
) -> RuntimeDependency:
    return RuntimeDependency(
        name,
        (lambda dependencies: object()) if factory is None else factory,
        requires,
    )


def test_runtime_dependency_normalizes_and_freezes_declaration() -> None:
    requirements = [" config "]
    dependency = RuntimeDependency(
        " client ",
        lambda dependencies: object(),
        requirements,
    )
    requirements.append("other")

    assert dependency.name == "client"
    assert dependency.requires == ("config",)
    with pytest.raises(FrozenInstanceError):
        dependency.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory,expected_error",
    [
        (lambda: _dependency(name=""), ValueError),
        (lambda: _dependency(name=1), TypeError),
        (lambda: _dependency(factory=object()), TypeError),
        (lambda: _dependency(requires="client"), TypeError),
        (lambda: _dependency(requires=("",)), ValueError),
        (lambda: _dependency(requires=("client",)), ValueError),
        (lambda: _dependency(requires=("config", "config")), ValueError),
    ],
)
def test_runtime_dependency_rejects_invalid_declarations(
    factory,
    expected_error,
) -> None:
    with pytest.raises(expected_error):
        factory()


def test_runtime_validates_registration_and_requirement_order() -> None:
    first = _dependency()

    with pytest.raises(TypeError, match="must be a sequence"):
        ApplicationRuntime("client")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="RuntimeDependency"):
        ApplicationRuntime((object(),))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate runtime dependency"):
        ApplicationRuntime((first, first))
    with pytest.raises(ValueError, match="not initialized"):
        ApplicationRuntime((_dependency("router", requires=("client",)),))

    runtime = ApplicationRuntime(
        (
            _dependency("client"),
            _dependency("router", requires=("client",)),
        )
    )
    assert [item.name for item in runtime.registrations] == ["client", "router"]


async def test_runtime_initializes_plain_and_async_factories_in_order() -> None:
    events: list[str] = []
    client = object()
    router = object()

    def create_client(dependencies: Mapping[str, Any]) -> object:
        assert dependencies == {}
        events.append("client")
        return client

    async def create_router(dependencies: Mapping[str, Any]) -> object:
        assert dependencies == {"client": client}
        with pytest.raises(TypeError):
            dependencies["other"] = object()  # type: ignore[index]
        events.append("router")
        return router

    runtime = ApplicationRuntime(
        (
            RuntimeDependency("client", create_client),
            RuntimeDependency("router", create_router, ("client",)),
        )
    )

    assert await runtime.start() is runtime
    assert runtime.started is True
    assert events == ["client", "router"]
    assert runtime.get(" client ") is client
    assert runtime.get("router") is router
    assert runtime.dependencies == {"client": client, "router": router}
    with pytest.raises(TypeError):
        runtime.dependencies["other"] = object()  # type: ignore[index]

    await runtime.close()


async def test_runtime_enters_and_closes_resources_in_reverse_order() -> None:
    events: list[str] = []

    @contextmanager
    def sync_resource():
        events.append("sync:start")
        try:
            yield "sync-value"
        finally:
            events.append("sync:stop")

    @asynccontextmanager
    async def async_resource():
        events.append("async:start")
        try:
            yield "async-value"
        finally:
            events.append("async:stop")

    runtime = ApplicationRuntime(
        (
            RuntimeDependency("sync", lambda dependencies: sync_resource()),
            RuntimeDependency(
                "async",
                lambda dependencies: async_resource(),
                ("sync",),
            ),
        )
    )

    await runtime.start()
    assert runtime.dependencies == {
        "sync": "sync-value",
        "async": "async-value",
    }
    await runtime.close()

    assert events == ["sync:start", "async:start", "async:stop", "sync:stop"]
    assert runtime.started is False


async def test_runtime_start_is_exactly_once_for_concurrent_callers() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def create_client(dependencies: Mapping[str, Any]) -> object:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return object()

    runtime = ApplicationRuntime((RuntimeDependency("client", create_client),))
    first = asyncio.create_task(runtime.start())
    await entered.wait()
    second = asyncio.create_task(runtime.start())
    await asyncio.sleep(0)
    release.set()

    assert await asyncio.gather(first, second) == [runtime, runtime]
    assert calls == 1
    assert await runtime.start() is runtime
    assert calls == 1
    await runtime.close()
    await runtime.close()


async def test_runtime_startup_failure_cleans_up_without_publishing_values() -> None:
    events: list[str] = []

    @asynccontextmanager
    async def managed_client():
        events.append("client:start")
        try:
            yield object()
        finally:
            events.append("client:stop")

    def fail(dependencies: Mapping[str, Any]) -> None:
        events.append("router:fail")
        raise RuntimeError("private startup failure")

    runtime = ApplicationRuntime(
        (
            RuntimeDependency("client", lambda dependencies: managed_client()),
            RuntimeDependency("router", fail, ("client",)),
        )
    )

    with pytest.raises(RuntimeLifecycleError) as exc_info:
        await runtime.start()

    assert exc_info.value.code == "runtime_startup_failed"
    assert "private" not in str(exc_info.value)
    assert events == ["client:start", "router:fail", "client:stop"]
    assert runtime.started is False
    with pytest.raises(RuntimeLifecycleError) as access_info:
        runtime.get("client")
    assert access_info.value.code == "runtime_not_started"


async def test_runtime_startup_cancellation_cleans_up_and_propagates() -> None:
    events: list[str] = []
    waiting = asyncio.Event()

    @asynccontextmanager
    async def managed_client():
        events.append("client:start")
        try:
            yield object()
        finally:
            events.append("client:stop")

    async def wait_forever(dependencies: Mapping[str, Any]) -> object:
        waiting.set()
        await asyncio.Event().wait()
        return object()

    runtime = ApplicationRuntime(
        (
            RuntimeDependency("client", lambda dependencies: managed_client()),
            RuntimeDependency("router", wait_forever, ("client",)),
        )
    )
    startup = asyncio.create_task(runtime.start())
    await waiting.wait()
    startup.cancel()

    with pytest.raises(asyncio.CancelledError):
        await startup

    assert events == ["client:start", "client:stop"]
    assert runtime.started is False


async def test_runtime_shutdown_failure_is_safe_and_continues_cleanup() -> None:
    events: list[str] = []

    @asynccontextmanager
    async def resource(name: str, *, fail: bool = False):
        events.append(f"{name}:start")
        try:
            yield name
        finally:
            events.append(f"{name}:stop")
            if fail:
                raise RuntimeError("private shutdown failure")

    runtime = ApplicationRuntime(
        (
            RuntimeDependency("first", lambda dependencies: resource("first")),
            RuntimeDependency(
                "second",
                lambda dependencies: resource("second", fail=True),
                ("first",),
            ),
        )
    )
    await runtime.start()

    with pytest.raises(RuntimeLifecycleError) as exc_info:
        await runtime.close()

    assert exc_info.value.code == "runtime_shutdown_failed"
    assert "private" not in str(exc_info.value)
    assert events == ["first:start", "second:start", "second:stop", "first:stop"]
    assert runtime.started is False
    await runtime.close()


async def test_runtime_shutdown_cancellation_propagates_after_cleanup() -> None:
    events: list[str] = []
    stopping = asyncio.Event()

    @asynccontextmanager
    async def first_resource():
        events.append("first:start")
        try:
            yield "first"
        finally:
            events.append("first:stop")

    @asynccontextmanager
    async def blocking_resource():
        events.append("second:start")
        try:
            yield "second"
        finally:
            events.append("second:stop")
            stopping.set()
            await asyncio.Event().wait()

    runtime = ApplicationRuntime(
        (
            RuntimeDependency("first", lambda dependencies: first_resource()),
            RuntimeDependency(
                "second",
                lambda dependencies: blocking_resource(),
                ("first",),
            ),
        )
    )
    await runtime.start()
    shutdown = asyncio.create_task(runtime.close())
    await stopping.wait()
    shutdown.cancel()

    with pytest.raises(asyncio.CancelledError):
        await shutdown

    assert events == ["first:start", "second:start", "second:stop", "first:stop"]
    assert runtime.started is False


async def test_runtime_access_errors_and_closed_state_are_safe() -> None:
    runtime = ApplicationRuntime((_dependency(),))

    with pytest.raises(RuntimeLifecycleError) as before_info:
        runtime.get("client")
    assert before_info.value.code == "runtime_not_started"

    await runtime.start()
    with pytest.raises(RuntimeLifecycleError) as missing_info:
        runtime.get("private-missing-name")
    assert missing_info.value.code == "runtime_dependency_not_found"
    assert "private" not in str(missing_info.value)

    await runtime.close()
    with pytest.raises(RuntimeLifecycleError) as after_info:
        runtime.get("client")
    assert after_info.value.code == "runtime_not_started"
    with pytest.raises(RuntimeLifecycleError) as restart_info:
        await runtime.start()
    assert restart_info.value.code == "runtime_already_closed"


async def test_runtime_supports_async_context_management() -> None:
    runtime = ApplicationRuntime((_dependency(),))

    async with runtime as active:
        assert active is runtime
        assert runtime.started is True

    assert runtime.started is False


async def test_runtime_telemetry_excludes_values_and_exception_messages(
    in_memory_tracing,
) -> None:
    def fail(dependencies: Mapping[str, Any]) -> None:
        raise RuntimeError("private exception message")

    runtime = ApplicationRuntime(
        (
            RuntimeDependency(
                "safe_dependency_name",
                lambda dependencies: "private credential value",
            ),
            RuntimeDependency("failing_dependency", fail),
        )
    )

    with pytest.raises(RuntimeLifecycleError):
        await runtime.start()

    span = in_memory_tracing.get_finished_spans()[0]
    assert span.name == "runtime.start"
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes == {
        "runtime.dependency.count": 2,
        "runtime.outcome": "failed",
        "error.type": "runtime_startup_failed",
        "runtime.dependency.name": "failing_dependency",
        "exception.type": "llmkit_lite.runtime.RuntimeLifecycleError",
    }
    serialized = repr(span)
    assert "private credential value" not in serialized
    assert "private exception message" not in serialized
