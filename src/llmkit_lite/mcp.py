"""MCP discovery and authorized remote tool execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence, Set
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol
from urllib.parse import urlsplit

from llmkit_lite.authorization import AuthorizationPolicy, Principal, get_principal
from llmkit_lite.observability import set_span_error, trace_span
from llmkit_lite.tools import AuthorizedToolExecutor, ToolDefinition

_ERROR_DETAILS = {
    "mcp_server_not_found": "MCP server is not registered",
    "mcp_discovery_failed": "MCP tool discovery failed",
    "mcp_invalid_response": "MCP server returned an invalid response",
    "mcp_dependency_unavailable": "MCP support is not installed",
    "mcp_tool_not_found": "MCP tool is not registered",
    "mcp_invalid_arguments": "MCP tool arguments are invalid",
    "mcp_tool_failed": "MCP tool reported a failure",
    "mcp_call_failed": "MCP tool call failed",
    "mcp_timeout": "MCP tool call timed out",
    "mcp_idempotency_conflict": "MCP idempotency key was reused inconsistently",
    "mcp_authentication_failed": "MCP authentication could not be resolved",
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


def _normalize_positive_number(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{field_name} must be greater than zero")
    return normalized


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


def _thaw_json(value: Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        thawed: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} keys must be strings")
            thawed[key] = _thaw_json(item, field_name)
        return thawed
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_thaw_json(item, field_name) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"{field_name} must contain only JSON values")


def _normalize_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(headers, Mapping):
        raise TypeError("MCP authentication headers must be a mapping")
    copied: dict[str, str] = {}
    for name, value in headers.items():
        normalized_name = _normalize_identifier(name, "MCP authentication header name")
        if not isinstance(value, str):
            raise TypeError("MCP authentication header value must be a string")
        if not value.strip():
            raise ValueError("MCP authentication header value must not be empty")
        if normalized_name in copied:
            raise ValueError("duplicate MCP authentication header")
        copied[normalized_name] = value
    return MappingProxyType(copied)


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


@dataclass(frozen=True, slots=True)
class McpToolResult:
    """Framework-owned result returned by a successful MCP tool call."""

    content: Sequence[Mapping[str, Any]] = ()
    structured_content: Any = None

    def __post_init__(self) -> None:
        if isinstance(self.content, (str, bytes)) or not isinstance(
            self.content, Sequence
        ):
            raise TypeError("MCP result content must be a sequence of mappings")
        frozen_content: list[Mapping[str, Any]] = []
        for block in self.content:
            if not isinstance(block, Mapping):
                raise TypeError("MCP result content must contain only mappings")
            frozen_content.append(_freeze_json(block, "MCP result content"))
        object.__setattr__(self, "content", tuple(frozen_content))
        object.__setattr__(
            self,
            "structured_content",
            _freeze_json(self.structured_content, "MCP structured content"),
        )


class McpAuthenticationProvider(Protocol):
    """Resolve transport headers without receiving tool arguments."""

    async def get_headers(
        self,
        principal: Principal | None,
        server_name: str,
    ) -> Mapping[str, str]:
        """Return headers for one registered MCP server."""


class McpClientAdapter(Protocol):
    """Transport-independent MCP discovery and invocation boundary."""

    async def list_tools(
        self,
        server: McpServer,
        *,
        headers: Mapping[str, str],
    ) -> Sequence[McpToolDescriptor]:
        """Return every tool exposed by one configured server."""

    async def call_tool(
        self,
        server: McpServer,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> McpToolResult:
        """Invoke one tool on the configured server."""


@dataclass(frozen=True, slots=True)
class McpServer:
    """A uniquely named MCP endpoint and its client adapter."""

    name: str
    endpoint: str
    adapter: McpClientAdapter
    authentication_provider: McpAuthenticationProvider | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _normalize_identifier(self.name, "MCP server name"),
        )
        object.__setattr__(self, "endpoint", _normalize_endpoint(self.endpoint))
        for method_name in ("list_tools", "call_tool"):
            if not callable(getattr(self.adapter, method_name, None)):
                raise TypeError(f"MCP adapter must implement {method_name}")
        if self.authentication_provider is not None and not callable(
            getattr(self.authentication_provider, "get_headers", None)
        ):
            raise TypeError("MCP authentication provider must implement get_headers")


@dataclass(frozen=True, slots=True)
class McpExecutionPolicy:
    """Timeout and process-local idempotency defaults for MCP execution."""

    timeout_seconds: float = 30.0
    idempotency_ttl_seconds: float = 300.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "timeout_seconds",
            _normalize_positive_number(self.timeout_seconds, "MCP timeout"),
        )
        object.__setattr__(
            self,
            "idempotency_ttl_seconds",
            _normalize_positive_number(
                self.idempotency_ttl_seconds,
                "MCP idempotency TTL",
            ),
        )


class _McpClient(Protocol):
    async def list_tools(self, *, cursor: str | None = None) -> Any: ...

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        read_timeout_seconds: float | None = None,
    ) -> Any: ...


McpClientFactory = Callable[
    [str, Mapping[str, str]],
    AbstractAsyncContextManager[_McpClient],
]


@asynccontextmanager
async def _default_client_factory(
    endpoint: str,
    headers: Mapping[str, str],
) -> AsyncIterator[_McpClient]:
    from mcp import Client

    if not headers:
        async with Client(endpoint) as client:
            yield client
        return

    import httpx2
    from mcp.client.streamable_http import streamable_http_client

    async with httpx2.AsyncClient(headers=dict(headers)) as http_client:
        transport = streamable_http_client(endpoint, http_client=http_client)
        async with Client(transport) as client:
            yield client


@dataclass(frozen=True, slots=True, init=False)
class StreamableHttpMcpAdapter:
    """Discover and invoke tools using the official MCP HTTP client."""

    _client_factory: McpClientFactory

    def __init__(self, *, client_factory: McpClientFactory | None = None) -> None:
        resolved_factory = (
            _default_client_factory if client_factory is None else client_factory
        )
        if not callable(resolved_factory):
            raise TypeError("MCP client factory must be callable")
        object.__setattr__(self, "_client_factory", resolved_factory)

    async def list_tools(
        self,
        server: McpServer,
        *,
        headers: Mapping[str, str] = MappingProxyType({}),
    ) -> tuple[McpToolDescriptor, ...]:
        """Open one MCP session and retrieve all pages of tools."""

        normalized_headers = _normalize_headers(headers)
        discovered: list[McpToolDescriptor] = []
        seen_cursors: set[str] = set()
        cursor: str | None = None

        try:
            client_context = self._create_client(server.endpoint, normalized_headers)
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

    async def call_tool(
        self,
        server: McpServer,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        headers: Mapping[str, str] = MappingProxyType({}),
        timeout_seconds: float = 30.0,
    ) -> McpToolResult:
        """Open one MCP session and invoke one remote tool exactly once."""

        normalized_name = _normalize_identifier(tool_name, "MCP tool name")
        if not isinstance(arguments, Mapping):
            raise TypeError("MCP tool arguments must be a mapping")
        normalized_timeout = _normalize_positive_number(
            timeout_seconds,
            "MCP timeout",
        )
        normalized_headers = _normalize_headers(headers)
        try:
            client_context = self._create_client(server.endpoint, normalized_headers)
            async with client_context as client:
                result = await client.call_tool(
                    normalized_name,
                    dict(arguments),
                    read_timeout_seconds=normalized_timeout,
                )
        except McpGatewayError:
            raise
        except TimeoutError as exc:
            raise McpGatewayError("mcp_timeout") from exc
        except ImportError as exc:
            raise McpGatewayError("mcp_dependency_unavailable") from exc
        except Exception as exc:
            raise McpGatewayError("mcp_call_failed") from exc

        return _convert_tool_result(result)

    def _create_client(
        self,
        endpoint: str,
        headers: Mapping[str, str],
    ) -> AbstractAsyncContextManager[_McpClient]:
        try:
            return self._client_factory(endpoint, headers)
        except ImportError as exc:
            raise McpGatewayError("mcp_dependency_unavailable") from exc


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


def _convert_tool_result(result: Any) -> McpToolResult:
    if getattr(result, "result_type", "complete") != "complete":
        raise McpGatewayError("mcp_invalid_response")
    is_error = getattr(result, "is_error", None)
    if not isinstance(is_error, bool):
        raise McpGatewayError("mcp_invalid_response")
    if is_error:
        raise McpGatewayError("mcp_tool_failed")
    content = getattr(result, "content", None)
    if isinstance(content, (str, bytes)) or not isinstance(content, Sequence):
        raise McpGatewayError("mcp_invalid_response")

    blocks: list[Mapping[str, Any]] = []
    for block in content:
        if isinstance(block, Mapping):
            payload = dict(block)
        else:
            model_dump = getattr(block, "model_dump", None)
            if not callable(model_dump):
                raise McpGatewayError("mcp_invalid_response")
            try:
                payload = model_dump(mode="json", by_alias=True, exclude_none=True)
            except Exception as exc:
                raise McpGatewayError("mcp_invalid_response") from exc
        if not isinstance(payload, Mapping):
            raise McpGatewayError("mcp_invalid_response")
        blocks.append(payload)

    try:
        return McpToolResult(
            content=blocks,
            structured_content=getattr(result, "structured_content", None),
        )
    except (TypeError, ValueError) as exc:
        raise McpGatewayError("mcp_invalid_response") from exc


async def _resolve_authentication_headers(
    server: McpServer,
    principal: Principal | None,
) -> Mapping[str, str]:
    provider = server.authentication_provider
    if provider is None:
        return MappingProxyType({})
    try:
        headers = await provider.get_headers(principal, server.name)
        return _normalize_headers(headers)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise McpGatewayError("mcp_authentication_failed") from exc


@dataclass(frozen=True, slots=True, init=False)
class McpServerRegistry:
    """Immutable registry for deterministic MCP discovery and lookup."""

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
        *,
        principal: Principal | None = None,
    ) -> tuple[McpToolDescriptor, ...]:
        """Discover one server or every server in declaration order."""

        selected = self._select_servers(server_name)
        active_principal = principal if principal is not None else get_principal()
        discovered: list[McpToolDescriptor] = []
        for server in selected:
            with trace_span(
                "mcp.tools.discover",
                attributes={"mcp.server.name": server.name},
            ) as span:
                try:
                    headers = await _resolve_authentication_headers(
                        server,
                        active_principal,
                    )
                    tools = await server.adapter.list_tools(server, headers=headers)
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

    def server(self, server_name: str) -> McpServer:
        """Return one configured server or raise a safe lookup error."""

        return self._select_servers(server_name)[0]

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


McpOperation = Callable[[], Awaitable[McpToolResult]]


class McpIdempotencyStore(Protocol):
    """Coordinate repeated executions using opaque hashed keys."""

    async def execute(
        self,
        key: str,
        fingerprint: str,
        operation: McpOperation,
        *,
        ttl_seconds: float,
    ) -> McpToolResult:
        """Run or reuse one matching operation."""


@dataclass(slots=True)
class _IdempotencyEntry:
    fingerprint: str
    task: asyncio.Task[McpToolResult]
    expires_at: float


class InMemoryMcpIdempotencyStore:
    """Process-local single-flight and successful-result idempotency store."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        if not callable(clock):
            raise TypeError("MCP idempotency clock must be callable")
        self._clock = clock
        self._entries: dict[str, _IdempotencyEntry] = {}
        self._lock = asyncio.Lock()

    async def execute(
        self,
        key: str,
        fingerprint: str,
        operation: McpOperation,
        *,
        ttl_seconds: float,
    ) -> McpToolResult:
        normalized_key = _normalize_identifier(key, "MCP idempotency store key")
        normalized_fingerprint = _normalize_identifier(
            fingerprint,
            "MCP idempotency fingerprint",
        )
        normalized_ttl = _normalize_positive_number(
            ttl_seconds,
            "MCP idempotency TTL",
        )
        if not callable(operation):
            raise TypeError("MCP idempotency operation must be callable")

        async with self._lock:
            now = self._clock()
            self._remove_expired(now)
            entry = self._entries.get(normalized_key)
            if entry is not None:
                if entry.fingerprint != normalized_fingerprint:
                    raise McpGatewayError("mcp_idempotency_conflict")
                task = entry.task
            else:
                task = asyncio.create_task(operation())
                entry = _IdempotencyEntry(
                    fingerprint=normalized_fingerprint,
                    task=task,
                    expires_at=math.inf,
                )
                self._entries[normalized_key] = entry

        try:
            result = await task
        except BaseException:
            async with self._lock:
                current = self._entries.get(normalized_key)
                if current is entry:
                    self._entries.pop(normalized_key, None)
            raise

        async with self._lock:
            current = self._entries.get(normalized_key)
            if current is entry and math.isinf(entry.expires_at):
                entry.expires_at = self._clock() + normalized_ttl
        return result

    def _remove_expired(self, now: float) -> None:
        expired = [
            key
            for key, entry in self._entries.items()
            if entry.task.done() and entry.expires_at <= now
        ]
        for key in expired:
            self._entries.pop(key, None)


TimeoutRunner = Callable[[float, McpOperation], Awaitable[McpToolResult]]


async def _default_timeout_runner(
    timeout_seconds: float,
    operation: McpOperation,
) -> McpToolResult:
    try:
        async with asyncio.timeout(timeout_seconds):
            return await operation()
    except TimeoutError as exc:
        raise McpGatewayError("mcp_timeout") from exc


@dataclass(frozen=True, slots=True, init=False)
class AuthorizedMcpToolExecutor:
    """Authorize, validate, deduplicate, and invoke discovered MCP tools."""

    registry: McpServerRegistry
    execution_policy: McpExecutionPolicy
    idempotency_store: McpIdempotencyStore
    _descriptors: Mapping[str, McpToolDescriptor]
    _executor: AuthorizedToolExecutor
    _timeout_runner: TimeoutRunner

    def __init__(
        self,
        registry: McpServerRegistry,
        descriptors: Sequence[McpToolDescriptor],
        required_scopes: Mapping[str, Set[str]],
        *,
        authorization_policy: AuthorizationPolicy | None = None,
        execution_policy: McpExecutionPolicy | None = None,
        idempotency_store: McpIdempotencyStore | None = None,
        _timeout_runner: TimeoutRunner = _default_timeout_runner,
    ) -> None:
        if not isinstance(registry, McpServerRegistry):
            raise TypeError("registry must be a McpServerRegistry")
        if isinstance(descriptors, (str, bytes)) or not isinstance(
            descriptors, Sequence
        ):
            raise TypeError("MCP descriptors must be a sequence")
        if not isinstance(required_scopes, Mapping):
            raise TypeError("MCP required scopes must be a mapping")
        resolved_policy = (
            McpExecutionPolicy() if execution_policy is None else execution_policy
        )
        if not isinstance(resolved_policy, McpExecutionPolicy):
            raise TypeError("execution policy must be McpExecutionPolicy")
        resolved_store = (
            InMemoryMcpIdempotencyStore()
            if idempotency_store is None
            else idempotency_store
        )
        if not callable(getattr(resolved_store, "execute", None)):
            raise TypeError("idempotency store must implement execute")
        if not callable(_timeout_runner):
            raise TypeError("timeout runner must be callable")

        descriptor_map: dict[str, McpToolDescriptor] = {}
        for descriptor in descriptors:
            if not isinstance(descriptor, McpToolDescriptor):
                raise TypeError("MCP descriptors must contain McpToolDescriptor values")
            registry.server(descriptor.server_name)
            if descriptor.qualified_name in descriptor_map:
                raise ValueError(
                    f"duplicate MCP tool descriptor: {descriptor.qualified_name}"
                )
            descriptor_map[descriptor.qualified_name] = descriptor

        normalized_scopes: dict[str, Set[str]] = {}
        for qualified_name, scopes in required_scopes.items():
            normalized_name = _normalize_identifier(
                qualified_name,
                "MCP qualified tool name",
            )
            if normalized_name not in descriptor_map:
                raise ValueError(f"unknown MCP scope mapping: {normalized_name}")
            normalized_scopes[normalized_name] = scopes

        object.__setattr__(self, "registry", registry)
        object.__setattr__(self, "execution_policy", resolved_policy)
        object.__setattr__(self, "idempotency_store", resolved_store)
        object.__setattr__(
            self,
            "_descriptors",
            MappingProxyType(descriptor_map),
        )
        object.__setattr__(self, "_timeout_runner", _timeout_runner)

        definitions: list[ToolDefinition] = []
        for descriptor in descriptor_map.values():

            async def handler(
                envelope: Mapping[str, Any],
                bound_descriptor: McpToolDescriptor = descriptor,
            ) -> McpToolResult:
                return await self._execute_remote(bound_descriptor, envelope)

            definitions.append(
                ToolDefinition(
                    name=descriptor.qualified_name,
                    handler=handler,
                    required_scopes=normalized_scopes.get(
                        descriptor.qualified_name,
                        frozenset(),
                    ),
                )
            )
        object.__setattr__(
            self,
            "_executor",
            AuthorizedToolExecutor(definitions, policy=authorization_policy),
        )

    @classmethod
    async def from_registry(
        cls,
        registry: McpServerRegistry,
        required_scopes: Mapping[str, Set[str]],
        *,
        discovery_principal: Principal | None = None,
        authorization_policy: AuthorizationPolicy | None = None,
        execution_policy: McpExecutionPolicy | None = None,
        idempotency_store: McpIdempotencyStore | None = None,
        _timeout_runner: TimeoutRunner = _default_timeout_runner,
    ) -> AuthorizedMcpToolExecutor:
        """Discover the registry and construct an authorized executor."""

        descriptors = await registry.discover_tools(principal=discovery_principal)
        return cls(
            registry,
            descriptors,
            required_scopes,
            authorization_policy=authorization_policy,
            execution_policy=execution_policy,
            idempotency_store=idempotency_store,
            _timeout_runner=_timeout_runner,
        )

    @property
    def tools(self) -> tuple[McpToolDescriptor, ...]:
        """Return the executable MCP descriptors in discovery order."""

        return tuple(self._descriptors.values())

    async def execute(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        principal: Principal | None = None,
        idempotency_key: str | None = None,
    ) -> McpToolResult:
        """Execute one qualified MCP tool through the authorization boundary."""

        try:
            normalized_name = _normalize_identifier(
                tool_name,
                "MCP qualified tool name",
            )
        except (TypeError, ValueError) as exc:
            raise McpGatewayError("mcp_tool_not_found") from exc
        if normalized_name not in self._descriptors:
            raise McpGatewayError("mcp_tool_not_found")

        envelope = {
            "arguments": arguments,
            "idempotency_key": idempotency_key,
        }
        return await self._executor.execute(
            normalized_name,
            envelope,
            principal=principal,
        )

    async def _execute_remote(
        self,
        descriptor: McpToolDescriptor,
        envelope: Mapping[str, Any],
    ) -> McpToolResult:
        arguments = _validate_tool_arguments(
            descriptor,
            envelope.get("arguments"),
        )
        raw_idempotency_key = envelope.get("idempotency_key")
        idempotency_key = (
            _normalize_identifier(raw_idempotency_key, "MCP idempotency key")
            if raw_idempotency_key is not None
            else None
        )
        principal = get_principal()

        async def operation() -> McpToolResult:
            return await self._call_remote(descriptor, arguments, principal)

        if idempotency_key is None:
            return await operation()

        store_key = _idempotency_store_key(
            descriptor.qualified_name,
            idempotency_key,
            principal,
        )
        fingerprint = _arguments_fingerprint(arguments)
        return await self.idempotency_store.execute(
            store_key,
            fingerprint,
            operation,
            ttl_seconds=self.execution_policy.idempotency_ttl_seconds,
        )

    async def _call_remote(
        self,
        descriptor: McpToolDescriptor,
        arguments: Mapping[str, Any],
        principal: Principal | None,
    ) -> McpToolResult:
        server = self.registry.server(descriptor.server_name)

        async def operation() -> McpToolResult:
            headers = await _resolve_authentication_headers(server, principal)
            return await server.adapter.call_tool(
                server,
                descriptor.name,
                arguments,
                headers=headers,
                timeout_seconds=self.execution_policy.timeout_seconds,
            )

        with trace_span(
            "mcp.tool.call",
            attributes={
                "mcp.server.name": server.name,
                "mcp.tool.name": descriptor.name,
            },
        ) as span:
            try:
                result = await self._timeout_runner(
                    self.execution_policy.timeout_seconds,
                    operation,
                )
            except asyncio.CancelledError:
                if span is not None:
                    span.set_attribute("mcp.call.outcome", "cancelled")
                    span.set_attribute("error.type", "mcp_cancelled")
                set_span_error(span, "mcp_cancelled")
                raise
            except McpGatewayError as exc:
                if span is not None:
                    span.set_attribute("mcp.call.outcome", "failed")
                    span.set_attribute("error.type", exc.code)
                set_span_error(span, exc.code)
                raise
            except Exception as exc:
                error = McpGatewayError("mcp_call_failed")
                if span is not None:
                    span.set_attribute("mcp.call.outcome", "failed")
                    span.set_attribute("error.type", error.code)
                set_span_error(span, error.code)
                raise error from exc

            if not isinstance(result, McpToolResult):
                error = McpGatewayError("mcp_invalid_response")
                if span is not None:
                    span.set_attribute("mcp.call.outcome", "failed")
                    span.set_attribute("error.type", error.code)
                set_span_error(span, error.code)
                raise error
            if span is not None:
                span.set_attribute("mcp.call.outcome", "succeeded")
            return result


def _validate_tool_arguments(
    descriptor: McpToolDescriptor,
    arguments: Any,
) -> dict[str, Any]:
    if not isinstance(arguments, Mapping):
        raise McpGatewayError("mcp_invalid_arguments")
    try:
        copied_arguments = _thaw_json(arguments, "MCP tool arguments")
        schema = _thaw_json(descriptor.input_schema, "MCP tool input schema")
    except (TypeError, ValueError) as exc:
        raise McpGatewayError("mcp_invalid_arguments") from exc

    try:
        from jsonschema import Draft202012Validator
        from jsonschema.exceptions import SchemaError, ValidationError
    except ImportError as exc:
        raise McpGatewayError("mcp_dependency_unavailable") from exc

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise McpGatewayError("mcp_invalid_response") from exc
    try:
        Draft202012Validator(schema).validate(copied_arguments)
    except ValidationError as exc:
        raise McpGatewayError("mcp_invalid_arguments") from exc
    return copied_arguments


def _idempotency_store_key(
    qualified_name: str,
    idempotency_key: str,
    principal: Principal | None,
) -> str:
    subject = principal.subject if principal is not None else "-"
    raw_key = f"{subject}\0{qualified_name}\0{idempotency_key}".encode()
    return hashlib.sha256(raw_key).hexdigest()


def _arguments_fingerprint(arguments: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        arguments,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(serialized).hexdigest()
