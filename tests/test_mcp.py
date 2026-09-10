from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry.trace import StatusCode

from llmkit_lite.authorization import AuthorizationError, Principal, principal_context
from llmkit_lite.mcp import (
    AuthorizedMcpToolExecutor,
    InMemoryMcpIdempotencyStore,
    McpExecutionPolicy,
    McpGatewayError,
    McpServer,
    McpServerRegistry,
    McpToolDescriptor,
    McpToolResult,
    StreamableHttpMcpAdapter,
)

_DEFAULT_SCHEMA = object()


class RecordingAdapter:
    def __init__(
        self,
        tools: Sequence[McpToolDescriptor] = (),
        *,
        error: BaseException | None = None,
    ) -> None:
        self.tools = tools
        self.error = error
        self.calls: list[McpServer] = []
        self.call_tool_calls: list[tuple[str, Mapping[str, Any]]] = []

    async def list_tools(
        self,
        server: McpServer,
        *,
        headers: Mapping[str, str],
    ) -> Sequence[McpToolDescriptor]:
        self.calls.append(server)
        if self.error is not None:
            raise self.error
        return self.tools

    async def call_tool(
        self,
        server: McpServer,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> McpToolResult:
        self.call_tool_calls.append((tool_name, arguments))
        if self.error is not None:
            raise self.error
        return McpToolResult(content=({"type": "text", "text": "ok"},))


def _tool(
    server_name: str = "docs",
    name: str = "search",
    *,
    schema: Any = _DEFAULT_SCHEMA,
) -> McpToolDescriptor:
    input_schema = (
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        if schema is _DEFAULT_SCHEMA
        else schema
    )
    return McpToolDescriptor(
        server_name=server_name,
        name=name,
        title=" Search documentation ",
        description=" Search internal documentation ",
        input_schema=input_schema,
    )


def _server(
    name: str = "docs",
    adapter: Any | None = None,
    endpoint: str = "https://mcp.example.test/mcp",
) -> McpServer:
    return McpServer(name, endpoint, adapter or RecordingAdapter())


def test_tool_descriptor_normalizes_and_deeply_freezes_schema() -> None:
    schema = {
        "type": "object",
        "properties": {"query": {"enum": ["one", "two"]}},
    }

    tool = _tool(schema=schema)
    schema["properties"]["query"]["enum"].append("three")

    assert tool.server_name == "docs"
    assert tool.name == "search"
    assert tool.title == "Search documentation"
    assert tool.description == "Search internal documentation"
    assert tool.qualified_name == "docs.search"
    assert tool.input_schema["properties"]["query"]["enum"] == ("one", "two")
    with pytest.raises(TypeError):
        tool.input_schema["type"] = "array"  # type: ignore[index]
    with pytest.raises(TypeError):
        tool.input_schema["properties"]["query"]["type"] = "number"
    with pytest.raises(FrozenInstanceError):
        tool.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory,expected_error",
    [
        (lambda: _tool(server_name=" "), ValueError),
        (lambda: _tool(name=""), ValueError),
        (lambda: _tool(schema=[]), TypeError),
        (lambda: _tool(schema={1: "value"}), TypeError),
        (lambda: _tool(schema={"default": object()}), TypeError),
        (lambda: _server(name=""), ValueError),
        (lambda: _server(endpoint="ftp://example.test/mcp"), ValueError),
        (lambda: _server(endpoint="https:///mcp"), ValueError),
        (lambda: _server(endpoint="https://user:secret@example.test/mcp"), ValueError),
        (lambda: _server(endpoint="https://example.test/mcp#fragment"), ValueError),
        (lambda: _server(adapter=object()), TypeError),
    ],
)
def test_models_reject_invalid_values(factory, expected_error) -> None:
    with pytest.raises(expected_error):
        factory()


def test_registry_validates_and_freezes_server_registration() -> None:
    first = _server()

    with pytest.raises(ValueError, match="duplicate MCP server"):
        McpServerRegistry((first, first))
    with pytest.raises(TypeError, match="sequence"):
        McpServerRegistry("docs")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="McpServer"):
        McpServerRegistry((object(),))  # type: ignore[arg-type]

    registry = McpServerRegistry((first,))
    assert registry.servers == (first,)
    with pytest.raises(FrozenInstanceError):
        registry._servers = {}  # type: ignore[misc]

    with pytest.raises(TypeError, match="client factory must be callable"):
        StreamableHttpMcpAdapter(client_factory=object())  # type: ignore[arg-type]


async def test_registry_discovers_all_servers_in_declaration_order() -> None:
    docs_adapter = RecordingAdapter((_tool(),))
    payments_adapter = RecordingAdapter((_tool("payments", "lookup"),))
    docs = _server(adapter=docs_adapter)
    payments = _server("payments", payments_adapter)
    registry = McpServerRegistry((docs, payments))

    tools = await registry.discover_tools()

    assert [tool.qualified_name for tool in tools] == [
        "docs.search",
        "payments.lookup",
    ]
    assert docs_adapter.calls == [docs]
    assert payments_adapter.calls == [payments]


async def test_registry_discovers_only_selected_server() -> None:
    docs_adapter = RecordingAdapter((_tool(),))
    payments_adapter = RecordingAdapter((_tool("payments", "lookup"),))
    registry = McpServerRegistry(
        (
            _server(adapter=docs_adapter),
            _server("payments", payments_adapter),
        )
    )

    tools = await registry.discover_tools(" payments ")

    assert [tool.qualified_name for tool in tools] == ["payments.lookup"]
    assert docs_adapter.calls == []
    assert len(payments_adapter.calls) == 1


@pytest.mark.parametrize("name", ["missing", " ", 1])
async def test_unknown_server_errors_are_safe(name) -> None:
    registry = McpServerRegistry((_server(),))

    with pytest.raises(McpGatewayError) as exc_info:
        await registry.discover_tools(name)  # type: ignore[arg-type]

    assert exc_info.value.code == "mcp_server_not_found"
    assert exc_info.value.detail == "MCP server is not registered"
    assert "missing" not in str(exc_info.value)


async def test_registry_rejects_duplicate_and_mismatched_tools() -> None:
    for tools in (
        (_tool(), _tool()),
        (_tool("other"),),
        (object(),),
        "invalid",
    ):
        registry = McpServerRegistry((_server(adapter=RecordingAdapter(tools)),))

        with pytest.raises(McpGatewayError) as exc_info:
            await registry.discover_tools()

        assert exc_info.value.code == "mcp_invalid_response"


class FakeClient:
    def __init__(self, pages: dict[str | None, Any]) -> None:
        self.pages = pages
        self.cursors: list[str | None] = []
        self.entered = 0
        self.exited = 0

    async def __aenter__(self) -> FakeClient:
        self.entered += 1
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.exited += 1

    async def list_tools(self, *, cursor: str | None = None) -> Any:
        self.cursors.append(cursor)
        page = self.pages[cursor]
        if isinstance(page, BaseException):
            raise page
        return page

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: float | None = None,
    ) -> Any:
        raise AssertionError("call_tool was not expected")


def _raw_tool(name: str) -> Any:
    return SimpleNamespace(
        name=name,
        title=f"{name} title",
        description=f"{name} description",
        input_schema={"type": "object"},
    )


async def test_streamable_adapter_discovers_every_page_and_closes_client() -> None:
    client = FakeClient(
        {
            None: SimpleNamespace(
                result_type="complete",
                tools=[_raw_tool("first")],
                next_cursor="page-2",
            ),
            "page-2": SimpleNamespace(
                result_type="complete",
                tools=[_raw_tool("second")],
                next_cursor=None,
            ),
        }
    )
    endpoints: list[str] = []

    def factory(endpoint: str, headers: Mapping[str, str]) -> FakeClient:
        endpoints.append(endpoint)
        assert headers == {}
        return client

    adapter = StreamableHttpMcpAdapter(client_factory=factory)
    server = _server(adapter=adapter)

    tools = await adapter.list_tools(server)

    assert [tool.qualified_name for tool in tools] == [
        "docs.first",
        "docs.second",
    ]
    assert endpoints == ["https://mcp.example.test/mcp"]
    assert client.cursors == [None, "page-2"]
    assert client.entered == 1
    assert client.exited == 1


@pytest.mark.parametrize(
    "page",
    [
        SimpleNamespace(result_type="input_required", tools=[], next_cursor=None),
        SimpleNamespace(result_type="complete", tools=None, next_cursor=None),
        SimpleNamespace(
            result_type="complete",
            tools=[SimpleNamespace(name="tool", input_schema=[])],
            next_cursor=None,
        ),
        SimpleNamespace(result_type="complete", tools=[], next_cursor=""),
    ],
)
async def test_streamable_adapter_rejects_malformed_pages(page) -> None:
    client = FakeClient({None: page})
    adapter = StreamableHttpMcpAdapter(
        client_factory=lambda endpoint, headers: client
    )

    with pytest.raises(McpGatewayError) as exc_info:
        await adapter.list_tools(_server(adapter=adapter))

    assert exc_info.value.code == "mcp_invalid_response"
    assert client.exited == 1


async def test_streamable_adapter_rejects_repeated_pagination_cursor() -> None:
    client = FakeClient(
        {
            None: SimpleNamespace(tools=[], next_cursor="repeat"),
            "repeat": SimpleNamespace(tools=[], next_cursor="repeat"),
        }
    )
    adapter = StreamableHttpMcpAdapter(
        client_factory=lambda endpoint, headers: client
    )

    with pytest.raises(McpGatewayError) as exc_info:
        await adapter.list_tools(_server(adapter=adapter))

    assert exc_info.value.code == "mcp_invalid_response"
    assert client.cursors == [None, "repeat"]
    assert client.exited == 1


async def test_streamable_adapter_maps_dependency_and_discovery_failures() -> None:
    def missing_dependency(endpoint: str, headers: Mapping[str, str]):
        raise ImportError("private dependency detail")

    missing_adapter = StreamableHttpMcpAdapter(client_factory=missing_dependency)
    with pytest.raises(McpGatewayError) as missing_info:
        await missing_adapter.list_tools(_server(adapter=missing_adapter))
    assert missing_info.value.code == "mcp_dependency_unavailable"

    client = FakeClient({None: RuntimeError("private server detail")})
    failing_adapter = StreamableHttpMcpAdapter(
        client_factory=lambda endpoint, headers: client
    )
    with pytest.raises(McpGatewayError) as failure_info:
        await failing_adapter.list_tools(_server(adapter=failing_adapter))
    assert failure_info.value.code == "mcp_discovery_failed"
    assert "private" not in str(failure_info.value)
    assert client.exited == 1


async def test_cancellation_propagates_and_closes_client() -> None:
    cancelled = asyncio.CancelledError()
    client = FakeClient({None: cancelled})
    adapter = StreamableHttpMcpAdapter(
        client_factory=lambda endpoint, headers: client
    )

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await adapter.list_tools(_server(adapter=adapter))

    assert exc_info.value is cancelled
    assert client.exited == 1


async def test_registry_wraps_unexpected_adapter_failure_safely() -> None:
    adapter = RecordingAdapter(error=RuntimeError("private remote message"))
    registry = McpServerRegistry((_server(adapter=adapter),))

    with pytest.raises(McpGatewayError) as exc_info:
        await registry.discover_tools()

    assert exc_info.value.code == "mcp_discovery_failed"
    assert "private" not in str(exc_info.value)


async def test_discovery_telemetry_excludes_sensitive_values(
    in_memory_tracing,
) -> None:
    endpoint = "https://mcp.example.test/private-endpoint?token=private-token"
    adapter = RecordingAdapter(
        error=RuntimeError("private exception message"),
    )
    registry = McpServerRegistry((_server(adapter=adapter, endpoint=endpoint),))

    with pytest.raises(McpGatewayError):
        await registry.discover_tools()

    span = in_memory_tracing.get_finished_spans()[0]
    assert span.name == "mcp.tools.discover"
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes == {
        "mcp.server.name": "docs",
        "mcp.discovery.outcome": "failed",
        "error.type": "mcp_discovery_failed",
        "exception.type": "llmkit_lite.mcp.McpGatewayError",
    }
    serialized = repr(span)
    for private_value in (
        "private-endpoint",
        "private-token",
        "private exception message",
    ):
        assert private_value not in serialized


async def test_successful_discovery_records_only_safe_counts(
    in_memory_tracing,
) -> None:
    registry = McpServerRegistry((_server(adapter=RecordingAdapter((_tool(),))),))

    await registry.discover_tools()

    span = in_memory_tracing.get_finished_spans()[0]
    assert span.status.status_code is StatusCode.UNSET
    assert span.attributes == {
        "mcp.server.name": "docs",
        "mcp.discovery.outcome": "succeeded",
        "mcp.tool.count": 1,
    }


class InvocationAdapter:
    def __init__(self, result: McpToolResult | BaseException | None = None) -> None:
        self.result = result or McpToolResult(
            content=({"type": "text", "text": "found"},),
            structured_content={"count": 1},
        )
        self.calls: list[dict[str, Any]] = []

    async def list_tools(
        self,
        server: McpServer,
        *,
        headers: Mapping[str, str],
    ) -> Sequence[McpToolDescriptor]:
        return (_tool(server.name),)

    async def call_tool(
        self,
        server: McpServer,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> McpToolResult:
        self.calls.append(
            {
                "server": server.name,
                "tool": tool_name,
                "arguments": dict(arguments),
                "headers": dict(headers),
                "timeout": timeout_seconds,
            }
        )
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class RecordingAuthenticationProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[Principal | None, str]] = []

    async def get_headers(
        self,
        principal: Principal | None,
        server_name: str,
    ) -> Mapping[str, str]:
        self.calls.append((principal, server_name))
        return {"Authorization": "Bearer private-token"}


def _remote_executor(
    adapter: InvocationAdapter,
    *,
    authentication_provider: Any | None = None,
    execution_policy: McpExecutionPolicy | None = None,
    idempotency_store: Any | None = None,
    timeout_runner: Any | None = None,
) -> AuthorizedMcpToolExecutor:
    server = McpServer(
        "docs",
        "https://mcp.example.test/mcp",
        adapter,
        authentication_provider,
    )
    kwargs: dict[str, Any] = {
        "execution_policy": execution_policy,
        "idempotency_store": idempotency_store,
    }
    if timeout_runner is not None:
        kwargs["_timeout_runner"] = timeout_runner
    return AuthorizedMcpToolExecutor(
        McpServerRegistry((server,)),
        (_tool(),),
        {"docs.search": {"mcp:docs.search:execute"}},
        **kwargs,
    )


def _remote_principal(subject: str = "user-123") -> Principal:
    return Principal(subject, scopes={"mcp:docs.search:execute"})


def test_execution_models_validate_and_freeze_values() -> None:
    content = [{"type": "text", "text": "ok", "meta": {"items": [1]}}]
    structured = {"value": [1]}
    result = McpToolResult(content=content, structured_content=structured)
    content[0]["meta"]["items"].append(2)
    structured["value"].append(2)

    assert result.content[0]["meta"]["items"] == (1,)
    assert result.structured_content["value"] == (1,)
    assert McpExecutionPolicy().timeout_seconds == 30
    assert McpExecutionPolicy().idempotency_ttl_seconds == 300
    for value in (0, -1, float("inf"), True):
        with pytest.raises((TypeError, ValueError)):
            McpExecutionPolicy(timeout_seconds=value)  # type: ignore[arg-type]


async def test_authorized_remote_call_uses_principal_credentials_and_policy() -> None:
    adapter = InvocationAdapter()
    credentials = RecordingAuthenticationProvider()
    executor = _remote_executor(
        adapter,
        authentication_provider=credentials,
        execution_policy=McpExecutionPolicy(timeout_seconds=7),
    )
    principal = _remote_principal()

    result = await executor.execute(
        " docs.search ",
        {"query": "routing"},
        principal=principal,
    )

    assert result.structured_content == {"count": 1}
    assert credentials.calls == [(principal, "docs")]
    assert adapter.calls == [
        {
            "server": "docs",
            "tool": "search",
            "arguments": {"query": "routing"},
            "headers": {"Authorization": "Bearer private-token"},
            "timeout": 7.0,
        }
    ]


async def test_active_principal_is_used_for_remote_credentials() -> None:
    adapter = InvocationAdapter()
    credentials = RecordingAuthenticationProvider()
    executor = _remote_executor(adapter, authentication_provider=credentials)
    principal = _remote_principal("context-user")

    with principal_context(principal):
        await executor.execute("docs.search", {"query": "context"})

    assert credentials.calls == [(principal, "docs")]


async def test_denial_prevents_validation_credentials_and_remote_call() -> None:
    adapter = InvocationAdapter()
    credentials = RecordingAuthenticationProvider()
    executor = _remote_executor(adapter, authentication_provider=credentials)

    with pytest.raises(AuthorizationError):
        await executor.execute(
            "docs.search",
            {"query": 123, "credential": "private"},
            principal=Principal("denied", scopes={"other"}),
        )

    assert credentials.calls == []
    assert adapter.calls == []


async def test_empty_scope_declaration_is_default_denied() -> None:
    adapter = InvocationAdapter()
    server = _server(adapter=adapter)
    executor = AuthorizedMcpToolExecutor(
        McpServerRegistry((server,)),
        (_tool(),),
        {},
    )

    with pytest.raises(AuthorizationError) as exc_info:
        await executor.execute(
            "docs.search",
            {"query": "value"},
            principal=Principal("user", scopes={"anything"}),
        )

    assert exc_info.value.reason_code == "scope_requirement_missing"
    assert adapter.calls == []


@pytest.mark.parametrize(
    "arguments",
    [{}, {"query": 42}, {"query": "ok", "x": object()}],
)
async def test_invalid_arguments_never_reach_remote_adapter(arguments) -> None:
    adapter = InvocationAdapter()
    executor = _remote_executor(adapter)

    with pytest.raises(McpGatewayError) as exc_info:
        await executor.execute(
            "docs.search",
            arguments,
            principal=_remote_principal(),
        )

    assert exc_info.value.code == "mcp_invalid_arguments"
    assert adapter.calls == []


async def test_unknown_remote_tool_error_is_safe() -> None:
    executor = _remote_executor(InvocationAdapter())

    with pytest.raises(McpGatewayError) as exc_info:
        await executor.execute("private.tool", {"secret": "value"})

    assert exc_info.value.code == "mcp_tool_not_found"
    assert "private" not in str(exc_info.value)


async def test_in_memory_idempotency_reuses_success_and_detects_conflict() -> None:
    adapter = InvocationAdapter()
    executor = _remote_executor(adapter)
    principal = _remote_principal()

    first = await executor.execute(
        "docs.search",
        {"query": "same"},
        principal=principal,
        idempotency_key="request-1",
    )
    second = await executor.execute(
        "docs.search",
        {"query": "same"},
        principal=principal,
        idempotency_key="request-1",
    )

    assert second is first
    assert len(adapter.calls) == 1
    with pytest.raises(McpGatewayError) as exc_info:
        await executor.execute(
            "docs.search",
            {"query": "changed"},
            principal=principal,
            idempotency_key="request-1",
        )
    assert exc_info.value.code == "mcp_idempotency_conflict"
    assert len(adapter.calls) == 1


async def test_idempotency_is_single_flight_for_concurrent_calls() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0
    result = McpToolResult(content=())

    async def operation() -> McpToolResult:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return result

    store = InMemoryMcpIdempotencyStore()
    first = asyncio.create_task(store.execute("key", "same", operation, ttl_seconds=5))
    await entered.wait()
    second = asyncio.create_task(store.execute("key", "same", operation, ttl_seconds=5))
    await asyncio.sleep(0)
    release.set()

    assert await asyncio.gather(first, second) == [result, result]
    assert calls == 1


async def test_idempotency_expires_and_failures_are_not_cached() -> None:
    now = 10.0
    store = InMemoryMcpIdempotencyStore(clock=lambda: now)
    calls = 0

    async def operation() -> McpToolResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise McpGatewayError("mcp_call_failed")
        return McpToolResult(content=())

    with pytest.raises(McpGatewayError):
        await store.execute("key", "same", operation, ttl_seconds=5)
    await store.execute("key", "same", operation, ttl_seconds=5)
    now = 16.0
    await store.execute("key", "same", operation, ttl_seconds=5)

    assert calls == 3


async def test_idempotency_cancellation_is_not_cached() -> None:
    store = InMemoryMcpIdempotencyStore()
    calls = 0

    async def operation() -> McpToolResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise asyncio.CancelledError()
        return McpToolResult(content=())

    with pytest.raises(asyncio.CancelledError):
        await store.execute("key", "same", operation, ttl_seconds=5)
    await store.execute("key", "same", operation, ttl_seconds=5)

    assert calls == 2


async def test_authentication_failure_is_safe_and_prevents_remote_call() -> None:
    class FailingAuthenticationProvider:
        async def get_headers(
            self,
            principal: Principal | None,
            server_name: str,
        ) -> Mapping[str, str]:
            raise RuntimeError("private credential failure")

    adapter = InvocationAdapter()
    executor = _remote_executor(
        adapter,
        authentication_provider=FailingAuthenticationProvider(),
    )

    with pytest.raises(McpGatewayError) as exc_info:
        await executor.execute(
            "docs.search",
            {"query": "value"},
            principal=_remote_principal(),
        )

    assert exc_info.value.code == "mcp_authentication_failed"
    assert "private" not in str(exc_info.value)
    assert adapter.calls == []


async def test_timeout_runner_and_remote_failures_map_safely() -> None:
    async def timeout_runner(seconds: float, operation: Any) -> McpToolResult:
        assert seconds == 2
        raise McpGatewayError("mcp_timeout")

    adapter = InvocationAdapter()
    executor = _remote_executor(
        adapter,
        execution_policy=McpExecutionPolicy(timeout_seconds=2),
        timeout_runner=timeout_runner,
    )

    with pytest.raises(McpGatewayError) as exc_info:
        await executor.execute(
            "docs.search",
            {"query": "value"},
            principal=_remote_principal(),
        )

    assert exc_info.value.code == "mcp_timeout"
    assert adapter.calls == []


async def test_remote_call_telemetry_excludes_sensitive_values(
    in_memory_tracing,
) -> None:
    adapter = InvocationAdapter(RuntimeError("private exception message"))
    credentials = RecordingAuthenticationProvider()
    executor = _remote_executor(adapter, authentication_provider=credentials)

    with pytest.raises(McpGatewayError):
        await executor.execute(
            "docs.search",
            {"query": "private prompt", "credential": "private key"},
            principal=_remote_principal("private subject"),
        )

    spans = in_memory_tracing.get_finished_spans()
    child = next(span for span in spans if span.name == "mcp.tool.call")
    assert child.status.status_code is StatusCode.ERROR
    assert child.attributes == {
        "mcp.server.name": "docs",
        "mcp.tool.name": "search",
        "mcp.call.outcome": "failed",
        "error.type": "mcp_call_failed",
        "exception.type": "llmkit_lite.mcp.McpGatewayError",
    }
    serialized = repr(spans)
    for secret in (
        "private prompt",
        "private key",
        "private-token",
        "private subject",
        "private exception message",
    ):
        assert secret not in serialized


class ToolCallClient(FakeClient):
    def __init__(self, result: Any) -> None:
        super().__init__({})
        self.result = result
        self.tool_calls: list[tuple[str, dict[str, Any] | None, float | None]] = []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: float | None = None,
    ) -> Any:
        self.tool_calls.append((name, arguments, read_timeout_seconds))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


async def test_streamable_adapter_calls_tool_once_and_converts_result() -> None:
    block = SimpleNamespace(
        model_dump=lambda **kwargs: {"type": "text", "text": "hello"}
    )
    client = ToolCallClient(
        SimpleNamespace(
            result_type="complete",
            is_error=False,
            content=[block],
            structured_content={"answer": 42},
        )
    )
    factory_calls: list[tuple[str, dict[str, str]]] = []

    def factory(
        endpoint: str,
        headers: Mapping[str, str],
    ) -> ToolCallClient:
        factory_calls.append((endpoint, dict(headers)))
        return client

    adapter = StreamableHttpMcpAdapter(client_factory=factory)
    result = await adapter.call_tool(
        _server(adapter=adapter),
        " search ",
        {"query": "value"},
        headers={"Authorization": "Bearer secret"},
        timeout_seconds=4,
    )

    assert result.content == ({"type": "text", "text": "hello"},)
    assert result.structured_content == {"answer": 42}
    assert factory_calls == [
        (
            "https://mcp.example.test/mcp",
            {"Authorization": "Bearer secret"},
        )
    ]
    assert client.tool_calls == [("search", {"query": "value"}, 4.0)]
    assert client.entered == client.exited == 1


@pytest.mark.parametrize(
    "raw_result,expected_code",
    [
        (
            SimpleNamespace(
                result_type="complete",
                is_error=True,
                content=[{"type": "text", "text": "private"}],
            ),
            "mcp_tool_failed",
        ),
        (
            SimpleNamespace(
                result_type="input_required",
                is_error=False,
                content=[],
            ),
            "mcp_invalid_response",
        ),
        (
            SimpleNamespace(
                result_type="complete",
                is_error="no",
                content=[],
            ),
            "mcp_invalid_response",
        ),
        (
            SimpleNamespace(
                result_type="complete",
                is_error=False,
                content=None,
            ),
            "mcp_invalid_response",
        ),
        (TimeoutError("private"), "mcp_timeout"),
        (RuntimeError("private"), "mcp_call_failed"),
    ],
)
async def test_streamable_adapter_maps_tool_call_failures(
    raw_result: Any,
    expected_code: str,
) -> None:
    client = ToolCallClient(raw_result)
    adapter = StreamableHttpMcpAdapter(
        client_factory=lambda endpoint, headers: client
    )

    with pytest.raises(McpGatewayError) as exc_info:
        await adapter.call_tool(
            _server(adapter=adapter),
            "search",
            {},
        )

    assert exc_info.value.code == expected_code
    assert "private" not in str(exc_info.value)
    assert client.entered == client.exited == 1


async def test_streamable_adapter_propagates_call_cancellation() -> None:
    cancelled = asyncio.CancelledError()
    client = ToolCallClient(cancelled)
    adapter = StreamableHttpMcpAdapter(
        client_factory=lambda endpoint, headers: client
    )

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await adapter.call_tool(_server(adapter=adapter), "search", {})

    assert exc_info.value is cancelled
    assert client.entered == client.exited == 1
