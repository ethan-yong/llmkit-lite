from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import Any

import pytest
from opentelemetry.trace import StatusCode

from llmkit_lite.mcp import (
    McpGatewayError,
    McpServer,
    McpServerRegistry,
    McpToolDescriptor,
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

    async def list_tools(
        self,
        server: McpServer,
    ) -> Sequence[McpToolDescriptor]:
        self.calls.append(server)
        if self.error is not None:
            raise self.error
        return self.tools


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

    def factory(endpoint: str) -> FakeClient:
        endpoints.append(endpoint)
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
    adapter = StreamableHttpMcpAdapter(client_factory=lambda endpoint: client)

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
    adapter = StreamableHttpMcpAdapter(client_factory=lambda endpoint: client)

    with pytest.raises(McpGatewayError) as exc_info:
        await adapter.list_tools(_server(adapter=adapter))

    assert exc_info.value.code == "mcp_invalid_response"
    assert client.cursors == [None, "repeat"]
    assert client.exited == 1


async def test_streamable_adapter_maps_dependency_and_discovery_failures() -> None:
    def missing_dependency(endpoint: str):
        raise ImportError("private dependency detail")

    missing_adapter = StreamableHttpMcpAdapter(client_factory=missing_dependency)
    with pytest.raises(McpGatewayError) as missing_info:
        await missing_adapter.list_tools(_server(adapter=missing_adapter))
    assert missing_info.value.code == "mcp_dependency_unavailable"

    client = FakeClient({None: RuntimeError("private server detail")})
    failing_adapter = StreamableHttpMcpAdapter(client_factory=lambda endpoint: client)
    with pytest.raises(McpGatewayError) as failure_info:
        await failing_adapter.list_tools(_server(adapter=failing_adapter))
    assert failure_info.value.code == "mcp_discovery_failed"
    assert "private" not in str(failure_info.value)
    assert client.exited == 1


async def test_cancellation_propagates_and_closes_client() -> None:
    cancelled = asyncio.CancelledError()
    client = FakeClient({None: cancelled})
    adapter = StreamableHttpMcpAdapter(client_factory=lambda endpoint: client)

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
