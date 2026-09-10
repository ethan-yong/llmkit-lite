"""MCP server registration and framework-owned tool discovery."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import urlsplit

from llmkit_lite.observability import set_span_error, trace_span

_ERROR_DETAILS = {
    "mcp_server_not_found": "MCP server is not registered",
    "mcp_discovery_failed": "MCP tool discovery failed",
    "mcp_invalid_response": "MCP server returned an invalid discovery response",
    "mcp_dependency_unavailable": "MCP support is not installed",
}


def _normalize_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _normalize_optional_text(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string or None")
    normalized = value.strip()
    return normalized or None


def _normalize_endpoint(value: str) -> str:
    endpoint = _normalize_identifier(value, "MCP endpoint")
    parsed = urlsplit(endpoint)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("MCP endpoint must be an HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("MCP endpoint must not contain credentials")
    if parsed.fragment:
        raise ValueError("MCP endpoint must not contain a fragment")
    return endpoint


def _freeze_json(value: Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} keys must be strings")
            frozen[key] = _freeze_json(item, field_name)
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item, field_name) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"{field_name} must contain only JSON values")


class McpGatewayError(Exception):
    """Safe MCP boundary failure suitable for application error mapping."""

    def __init__(self, code: str) -> None:
        if code not in _ERROR_DETAILS:
            raise ValueError("unsupported MCP gateway error code")
        detail = _ERROR_DETAILS[code]
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class McpToolDescriptor:
    """Framework-owned, immutable description of one remote MCP tool."""

    server_name: str
    name: str
    input_schema: Mapping[str, Any]
    title: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "server_name",
            _normalize_identifier(self.server_name, "MCP server name"),
        )
        object.__setattr__(
            self,
            "name",
            _normalize_identifier(self.name, "MCP tool name"),
        )
        if not isinstance(self.input_schema, Mapping):
            raise TypeError("MCP tool input schema must be a mapping")
        object.__setattr__(
            self,
            "input_schema",
            _freeze_json(self.input_schema, "MCP tool input schema"),
        )
        object.__setattr__(
            self,
            "title",
            _normalize_optional_text(self.title, "MCP tool title"),
        )
        object.__setattr__(
            self,
            "description",
            _normalize_optional_text(self.description, "MCP tool description"),
        )

    @property
    def qualified_name(self) -> str:
        """Return the server-qualified tool name used by the registry."""

        return f"{self.server_name}.{self.name}"


class McpClientAdapter(Protocol):
    """Transport-independent MCP tool discovery boundary."""

    async def list_tools(self, server: McpServer) -> Sequence[McpToolDescriptor]:
        """Return every tool exposed by one configured server."""


@dataclass(frozen=True, slots=True)
class McpServer:
    """A uniquely named MCP endpoint and its client adapter."""

    name: str
    endpoint: str
    adapter: McpClientAdapter

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalize_identifier(self.name, "MCP server name"),
        )
        object.__setattr__(self, "endpoint", _normalize_endpoint(self.endpoint))
        if not callable(getattr(self.adapter, "list_tools", None)):
            raise TypeError("MCP adapter must implement list_tools")


class _McpClient(Protocol):
    async def list_tools(self, *, cursor: str | None = None) -> Any: ...


McpClientFactory = Callable[[str], AbstractAsyncContextManager[_McpClient]]


def _default_client_factory(
    endpoint: str,
) -> AbstractAsyncContextManager[_McpClient]:
    from mcp import Client

    return Client(endpoint)


@dataclass(frozen=True, slots=True, init=False)
class StreamableHttpMcpAdapter:
    """Discover tools using the official MCP Streamable HTTP client."""

    _client_factory: McpClientFactory

    def __init__(self, *, client_factory: McpClientFactory | None = None) -> None:
        resolved_factory = (
            _default_client_factory if client_factory is None else client_factory
        )
        if not callable(resolved_factory):
            raise TypeError("MCP client factory must be callable")
        object.__setattr__(
            self,
            "_client_factory",
            resolved_factory,
        )

    async def list_tools(self, server: McpServer) -> tuple[McpToolDescriptor, ...]:
        """Open one MCP session and retrieve all pages of tools."""

        try:
            client_context = self._client_factory(server.endpoint)
        except ImportError as exc:
            raise McpGatewayError("mcp_dependency_unavailable") from exc
        except Exception as exc:
            raise McpGatewayError("mcp_discovery_failed") from exc

        discovered: list[McpToolDescriptor] = []
        seen_cursors: set[str] = set()
        cursor: str | None = None

        try:
            async with client_context as client:
                while True:
                    result = await client.list_tools(cursor=cursor)
                    discovered.extend(_convert_discovery_page(server.name, result))

                    next_cursor = getattr(result, "next_cursor", None)
                    if next_cursor is None:
                        break
                    if not isinstance(next_cursor, str) or not next_cursor.strip():
                        raise McpGatewayError("mcp_invalid_response")
                    if next_cursor in seen_cursors:
                        raise McpGatewayError("mcp_invalid_response")
                    seen_cursors.add(next_cursor)
                    cursor = next_cursor
        except McpGatewayError:
            raise
        except ImportError as exc:
            raise McpGatewayError("mcp_dependency_unavailable") from exc
        except Exception as exc:
            raise McpGatewayError("mcp_discovery_failed") from exc

        return tuple(discovered)


def _convert_discovery_page(
    server_name: str,
    result: Any,
) -> tuple[McpToolDescriptor, ...]:
    if getattr(result, "result_type", "complete") != "complete":
        raise McpGatewayError("mcp_invalid_response")
    tools = getattr(result, "tools", None)
    if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
        raise McpGatewayError("mcp_invalid_response")

    converted: list[McpToolDescriptor] = []
    for tool in tools:
        try:
            converted.append(
                McpToolDescriptor(
                    server_name=server_name,
                    name=tool.name,
                    title=getattr(tool, "title", None),
                    description=getattr(tool, "description", None),
                    input_schema=tool.input_schema,
                )
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise McpGatewayError("mcp_invalid_response") from exc
    return tuple(converted)


@dataclass(frozen=True, slots=True, init=False)
class McpServerRegistry:
    """Immutable registry for deterministic MCP tool discovery."""

    _servers: Mapping[str, McpServer]

    def __init__(self, servers: Sequence[McpServer]) -> None:
        if isinstance(servers, (str, bytes)) or not isinstance(servers, Sequence):
            raise TypeError("MCP servers must be a sequence")

        registry: dict[str, McpServer] = {}
        for server in servers:
            if not isinstance(server, McpServer):
                raise TypeError("MCP servers must contain only McpServer values")
            if server.name in registry:
                raise ValueError(f"duplicate MCP server registration: {server.name}")
            registry[server.name] = server
        object.__setattr__(self, "_servers", MappingProxyType(registry))

    @property
    def servers(self) -> tuple[McpServer, ...]:
        """Return registered servers in declaration order."""

        return tuple(self._servers.values())

    async def discover_tools(
        self,
        server_name: str | None = None,
    ) -> tuple[McpToolDescriptor, ...]:
        """Discover one server or every server in declaration order."""

        selected = self._select_servers(server_name)
        discovered: list[McpToolDescriptor] = []
        for server in selected:
            with trace_span(
                "mcp.tools.discover",
                attributes={"mcp.server.name": server.name},
            ) as span:
                try:
                    tools = await server.adapter.list_tools(server)
                    validated = _validate_discovered_tools(server, tools)
                except McpGatewayError as exc:
                    if span is not None:
                        span.set_attribute("mcp.discovery.outcome", "failed")
                        span.set_attribute("error.type", exc.code)
                    set_span_error(span, exc.code)
                    raise
                except Exception as exc:
                    error = McpGatewayError("mcp_discovery_failed")
                    if span is not None:
                        span.set_attribute("mcp.discovery.outcome", "failed")
                        span.set_attribute("error.type", error.code)
                    set_span_error(span, error.code)
                    raise error from exc

                if span is not None:
                    span.set_attribute("mcp.discovery.outcome", "succeeded")
                    span.set_attribute("mcp.tool.count", len(validated))
                discovered.extend(validated)
        return tuple(discovered)

    def _select_servers(self, server_name: str | None) -> tuple[McpServer, ...]:
        if server_name is None:
            return self.servers
        try:
            normalized_name = _normalize_identifier(server_name, "MCP server name")
        except (TypeError, ValueError) as exc:
            raise McpGatewayError("mcp_server_not_found") from exc
        server = self._servers.get(normalized_name)
        if server is None:
            raise McpGatewayError("mcp_server_not_found")
        return (server,)


def _validate_discovered_tools(
    server: McpServer,
    tools: Sequence[McpToolDescriptor],
) -> tuple[McpToolDescriptor, ...]:
    if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
        raise McpGatewayError("mcp_invalid_response")

    validated: list[McpToolDescriptor] = []
    names: set[str] = set()
    for tool in tools:
        if not isinstance(tool, McpToolDescriptor):
            raise McpGatewayError("mcp_invalid_response")
        if tool.server_name != server.name or tool.name in names:
            raise McpGatewayError("mcp_invalid_response")
        names.add(tool.name)
        validated.append(tool)
    return tuple(validated)
