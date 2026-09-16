"""Provider-agnostic helpers for OpenAI-compatible chat completion APIs."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Literal, Protocol

import httpx

from llmkit_lite.observability import (
    inject_trace_context,
    set_span_error,
    trace_span,
)

logger = logging.getLogger("llmkit_lite.llm")

MessageRole = Literal["system", "user", "assistant", "tool"]
JsonValue = (
    bool
    | int
    | float
    | str
    | None
    | tuple["JsonValue", ...]
    | Mapping[str, "JsonValue"]
)

_DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


class LlmCapability(StrEnum):
    """One optional behavior an endpoint can support."""

    TEXT = "text"
    TOOL_CALLING = "tool_calling"
    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"
    VISION = "vision"
    STREAMING = "streaming"
    REASONING_EFFORT = "reasoning_effort"


class LlmResponseFormatType(StrEnum):
    """Provider-independent structured response formats."""

    JSON_OBJECT = "json_object"
    JSON_SCHEMA = "json_schema"


class LlmToolChoiceMode(StrEnum):
    """Provider-independent tool selection modes."""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"
    NAMED = "named"


class LlmGatewayError(Exception):
    """Raised for LLM gateway failures.

    `code` is intentionally stable enough for API services to map to HTTP
    responses, metrics, and fallback behavior.
    """

    def __init__(self, code: str, detail: str, raw: str | None = None) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.raw = raw


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _normalize_capabilities(
    values: Iterable[LlmCapability | str],
    field_name: str,
) -> frozenset[LlmCapability]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise TypeError(f"{field_name} must be an iterable of capabilities")
    normalized: set[LlmCapability] = set()
    for value in values:
        if isinstance(value, LlmCapability):
            normalized.add(value)
            continue
        if not isinstance(value, str):
            raise TypeError(
                f"{field_name} must contain strings or LlmCapability values"
            )
        capability_name = _required_text(value, "capability name")
        try:
            normalized.add(LlmCapability(capability_name))
        except ValueError as exc:
            raise ValueError(
                f"unsupported LLM capability: {capability_name!r}"
            ) from exc
    return frozenset(normalized)


@dataclass(frozen=True, slots=True, init=False)
class LlmCapabilities:
    """Immutable capabilities declared by one configured model endpoint."""

    supported: frozenset[LlmCapability]

    def __init__(
        self,
        supported: Iterable[LlmCapability | str] = (LlmCapability.TEXT,),
    ) -> None:
        object.__setattr__(
            self,
            "supported",
            _normalize_capabilities(supported, "supported capabilities"),
        )

    def missing(
        self,
        required: Iterable[LlmCapability | str],
    ) -> frozenset[LlmCapability]:
        """Return requirements not supported by this endpoint."""

        normalized = _normalize_capabilities(required, "required capabilities")
        return normalized.difference(self.supported)

    def supports(self, required: Iterable[LlmCapability | str]) -> bool:
        """Return whether every requested capability is supported."""

        return not self.missing(required)


@dataclass(frozen=True, slots=True)
class LlmResponseFormat:
    """Provider-independent request for a structured model response."""

    kind: LlmResponseFormatType | str
    name: str | None = None
    schema: Mapping[str, JsonValue] | None = None
    strict: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, LlmResponseFormatType):
            try:
                object.__setattr__(self, "kind", LlmResponseFormatType(self.kind))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"unsupported response format: {self.kind!r}") from exc
        if not isinstance(self.strict, bool):
            raise TypeError("response format strict must be a boolean")
        if self.kind is LlmResponseFormatType.JSON_OBJECT:
            if self.name is not None or self.schema is not None:
                raise ValueError("JSON object response format cannot include a schema")
            if self.strict:
                raise ValueError("JSON object response format cannot be strict")
            return
        object.__setattr__(self, "name", _required_text(self.name, "schema name"))
        if not isinstance(self.schema, Mapping):
            raise TypeError("JSON schema response format requires a schema mapping")
        frozen = _freeze_json(self.schema, "response format schema")
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "schema", frozen)

    @classmethod
    def json_object(cls) -> LlmResponseFormat:
        return cls(kind=LlmResponseFormatType.JSON_OBJECT)

    @classmethod
    def json_schema(
        cls,
        *,
        name: str,
        schema: Mapping[str, JsonValue],
        strict: bool = False,
    ) -> LlmResponseFormat:
        return cls(
            kind=LlmResponseFormatType.JSON_SCHEMA,
            name=name,
            schema=schema,
            strict=strict,
        )


@dataclass(frozen=True, slots=True)
class LlmToolDefinition:
    """Provider-independent declaration of one tool available to a model."""

    name: str
    input_schema: Mapping[str, JsonValue]
    description: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, "tool name"))
        object.__setattr__(
            self,
            "description",
            _optional_text(self.description, "tool description"),
        )
        if not isinstance(self.input_schema, Mapping):
            raise TypeError("tool input schema must be a mapping")
        frozen = _freeze_json(self.input_schema, "tool input schema")
        assert isinstance(frozen, Mapping)
        if frozen.get("type") != "object":
            raise ValueError("tool input schema type must be 'object'")
        properties = frozen.get("properties")
        if properties is not None and not isinstance(properties, Mapping):
            raise ValueError("tool input schema properties must be a mapping")
        object.__setattr__(self, "input_schema", frozen)


@dataclass(frozen=True, slots=True)
class LlmToolChoice:
    """Provider-independent instruction for selecting an available tool."""

    mode: LlmToolChoiceMode | str
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, LlmToolChoiceMode):
            try:
                object.__setattr__(self, "mode", LlmToolChoiceMode(self.mode))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"unsupported tool choice mode: {self.mode!r}"
                ) from exc
        if self.mode is LlmToolChoiceMode.NAMED:
            object.__setattr__(self, "name", _required_text(self.name, "tool name"))
        elif self.name is not None:
            raise ValueError("only a named tool choice can include a tool name")

    @classmethod
    def auto(cls) -> LlmToolChoice:
        return cls(LlmToolChoiceMode.AUTO)

    @classmethod
    def none(cls) -> LlmToolChoice:
        return cls(LlmToolChoiceMode.NONE)

    @classmethod
    def required(cls) -> LlmToolChoice:
        return cls(LlmToolChoiceMode.REQUIRED)

    @classmethod
    def named(cls, name: str) -> LlmToolChoice:
        return cls(LlmToolChoiceMode.NAMED, name=name)


def _freeze_json(value: Any, field_name: str) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{field_name} keys must be strings")
            frozen[key] = _freeze_json(item, field_name)
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_json(item, field_name) for item in value)
    raise TypeError(f"{field_name} must contain only JSON-compatible values")


def _thaw_json(value: JsonValue) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Provider-independent model request to invoke one named tool."""

    id: str
    name: str
    arguments: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "tool call id"))
        object.__setattr__(self, "name", _required_text(self.name, "tool call name"))
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool call arguments must be a mapping")
        frozen = _freeze_json(self.arguments, "tool call arguments")
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "arguments", frozen)


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Provider-independent message used in requests and responses."""

    role: MessageRole
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: Sequence[ToolCall] = ()

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {self.role!r}")
        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("message content must be a string or None")
        object.__setattr__(self, "name", _optional_text(self.name, "message name"))
        object.__setattr__(
            self,
            "tool_call_id",
            _optional_text(self.tool_call_id, "message tool call id"),
        )
        if isinstance(self.tool_calls, (str, bytes)) or not isinstance(
            self.tool_calls, Sequence
        ):
            raise TypeError("message tool calls must be a sequence")
        normalized_calls = tuple(self.tool_calls)
        if not all(isinstance(call, ToolCall) for call in normalized_calls):
            raise TypeError("message tool calls must contain ToolCall values")
        if self.content is None and not normalized_calls:
            raise ValueError("message must contain content or tool calls")
        object.__setattr__(self, "tool_calls", normalized_calls)


ChatMessageInput = ChatMessage | Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Provider-independent token counts for one model response."""

    input_tokens: int
    output_tokens: int
    total_tokens: int

    def __post_init__(self) -> None:
        for field_name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ChatCompletionResponse:
    """Common response returned by every LLM provider adapter."""

    text: str | None
    provider: str
    model: str
    tool_calls: Sequence[ToolCall] = ()
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    provider_metadata: Mapping[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("response text must be a string or None")
        object.__setattr__(self, "provider", _required_text(self.provider, "provider"))
        object.__setattr__(self, "model", _required_text(self.model, "response model"))
        if isinstance(self.tool_calls, (str, bytes)) or not isinstance(
            self.tool_calls, Sequence
        ):
            raise TypeError("response tool calls must be a sequence")
        normalized_calls = tuple(self.tool_calls)
        if not all(isinstance(call, ToolCall) for call in normalized_calls):
            raise TypeError("response tool calls must contain ToolCall values")
        object.__setattr__(self, "tool_calls", normalized_calls)
        object.__setattr__(
            self,
            "finish_reason",
            _optional_text(self.finish_reason, "finish reason"),
        )
        if self.usage is not None and not isinstance(self.usage, TokenUsage):
            raise TypeError("response usage must be TokenUsage or None")
        if not isinstance(self.provider_metadata, Mapping):
            raise TypeError("provider metadata must be a mapping")
        frozen = _freeze_json(self.provider_metadata, "provider metadata")
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "provider_metadata", frozen)


def _tool_call_from_mapping(value: Any, field_name: str) -> ToolCall:
    if isinstance(value, ToolCall):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must contain ToolCall values or mappings")
    arguments = value.get("arguments", {})
    if not isinstance(arguments, Mapping):
        raise TypeError(f"{field_name} arguments must be a mapping")
    return ToolCall(
        id=value.get("id"),
        name=value.get("name"),
        arguments=arguments,
    )


def _chat_message_from_input(value: ChatMessageInput) -> ChatMessage:
    if isinstance(value, ChatMessage):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("messages must contain ChatMessage values or mappings")
    role = value.get("role")
    if not isinstance(role, str):
        raise TypeError("message role must be a string")
    raw_tool_calls = value.get("tool_calls", ())
    if isinstance(raw_tool_calls, (str, bytes)) or not isinstance(
        raw_tool_calls, Sequence
    ):
        raise TypeError("message tool calls must be a sequence")
    return ChatMessage(
        role=role,  # type: ignore[arg-type]
        content=value.get("content"),
        name=value.get("name"),
        tool_call_id=value.get("tool_call_id"),
        tool_calls=tuple(
            _tool_call_from_mapping(item, "message tool calls")
            for item in raw_tool_calls
        ),
    )


def _chat_message_body(message: ChatMessage) -> dict[str, Any]:
    body: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.name is not None:
        body["name"] = message.name
    if message.tool_call_id is not None:
        body["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        body["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(
                        _thaw_json(call.arguments),
                        separators=(",", ":"),
                    ),
                },
            }
            for call in message.tool_calls
        ]
    return body


@dataclass(frozen=True)
class LlmEndpointConfig:
    """Resolved settings for one chat-completion endpoint."""

    provider: str
    base_url: str
    model_name: str
    api_key: str | None = None
    reasoning_effort: str | None = None
    capabilities: LlmCapabilities = field(default_factory=LlmCapabilities)

    def __post_init__(self) -> None:
        if not isinstance(self.capabilities, LlmCapabilities):
            object.__setattr__(self, "capabilities", LlmCapabilities(self.capabilities))


@dataclass(frozen=True)
class ChatCompletionRequest:
    """Provider-independent inputs for one chat completion."""

    messages: Sequence[ChatMessageInput]
    max_tokens: int
    timeout_seconds: float
    temperature: float = 0
    response_format: LlmResponseFormat | None = None
    tools: Sequence[LlmToolDefinition] = ()
    tool_choice: LlmToolChoice | None = None
    extra_body: Mapping[str, Any] | None = None
    routing_metadata: Mapping[str, str] = field(default_factory=dict)
    required_capabilities: Iterable[LlmCapability | str] = field(
        default_factory=lambda: frozenset({LlmCapability.TEXT})
    )

    def __post_init__(self) -> None:
        if isinstance(self.messages, (str, bytes)) or not isinstance(
            self.messages, Sequence
        ):
            raise TypeError("messages must be a sequence")
        normalized_messages = tuple(
            _chat_message_from_input(message) for message in self.messages
        )
        if not normalized_messages:
            raise ValueError("messages must not be empty")
        object.__setattr__(
            self,
            "messages",
            normalized_messages,
        )
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens < 1
        ):
            raise ValueError("max_tokens must be an integer greater than zero")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be greater than zero")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not 0 <= self.temperature <= 2
        ):
            raise ValueError("temperature must be between zero and two")
        if self.response_format is not None and not isinstance(
            self.response_format, LlmResponseFormat
        ):
            raise TypeError("response_format must be an LlmResponseFormat or None")
        if isinstance(self.tools, (str, bytes)) or not isinstance(
            self.tools, Sequence
        ):
            raise TypeError("tools must be a sequence of LlmToolDefinition values")
        normalized_tools = tuple(self.tools)
        if not all(isinstance(tool, LlmToolDefinition) for tool in normalized_tools):
            raise TypeError("tools must contain only LlmToolDefinition values")
        tool_names: set[str] = set()
        for tool in normalized_tools:
            if tool.name in tool_names:
                raise ValueError(f"duplicate LLM tool declaration: {tool.name}")
            tool_names.add(tool.name)
        object.__setattr__(self, "tools", normalized_tools)
        if self.tool_choice is not None:
            if not isinstance(self.tool_choice, LlmToolChoice):
                raise TypeError("tool_choice must be an LlmToolChoice or None")
            if not normalized_tools:
                raise ValueError("tool_choice requires at least one tool declaration")
            if (
                self.tool_choice.mode is LlmToolChoiceMode.NAMED
                and self.tool_choice.name not in tool_names
            ):
                raise ValueError(
                    f"named tool choice is not declared: {self.tool_choice.name}"
                )
        if self.extra_body is not None:
            if not isinstance(self.extra_body, Mapping):
                raise TypeError("extra_body must be a mapping or None")
            reserved_fields = {"response_format", "tools", "tool_choice"}
            overridden_fields = reserved_fields.intersection(self.extra_body)
            if overridden_fields:
                names = ", ".join(sorted(overridden_fields))
                raise ValueError(
                    f"extra_body cannot override normalized fields: {names}"
                )
            frozen_extra = _freeze_json(self.extra_body, "extra_body")
            assert isinstance(frozen_extra, Mapping)
            object.__setattr__(
                self,
                "extra_body",
                frozen_extra,
            )
        if not isinstance(self.routing_metadata, Mapping):
            raise TypeError("routing_metadata must be a mapping")
        normalized_metadata: dict[str, str] = {}
        for key, value in self.routing_metadata.items():
            normalized_key = _required_text(key, "routing metadata key")
            if not isinstance(value, str):
                raise TypeError("routing metadata values must be strings")
            normalized_metadata[normalized_key] = value
        object.__setattr__(
            self,
            "routing_metadata",
            MappingProxyType(normalized_metadata),
        )
        required_capabilities = set(
            _normalize_capabilities(
                self.required_capabilities,
                "required capabilities",
            )
        )
        if self.response_format is not None:
            required_capabilities.add(LlmCapability(self.response_format.kind.value))
        if normalized_tools or self.tool_choice is not None:
            required_capabilities.add(LlmCapability.TOOL_CALLING)
        object.__setattr__(
            self,
            "required_capabilities",
            frozenset(required_capabilities),
        )


def missing_capabilities(
    request: ChatCompletionRequest,
    cfg: LlmEndpointConfig,
) -> frozenset[LlmCapability]:
    """Return capabilities required by the request but absent from the endpoint."""

    if not isinstance(request, ChatCompletionRequest):
        raise TypeError("request must be a ChatCompletionRequest")
    if not isinstance(cfg, LlmEndpointConfig):
        raise TypeError("cfg must be an LlmEndpointConfig")
    return cfg.capabilities.missing(request.required_capabilities)


def require_capabilities(
    request: ChatCompletionRequest,
    cfg: LlmEndpointConfig,
) -> None:
    """Reject an incompatible endpoint before provider execution."""

    missing = missing_capabilities(request, cfg)
    if not missing:
        return
    names = ", ".join(sorted(capability.value for capability in missing))
    raise LlmGatewayError(
        "llm_capability_unsupported",
        f"LLM endpoint does not support required capabilities: {names}",
    )


class LlmProviderAdapter(Protocol):
    """Transport boundary implemented by an LLM provider adapter."""

    async def complete(
        self,
        request: ChatCompletionRequest,
        *,
        cfg: LlmEndpointConfig,
        http_client: httpx.AsyncClient,
    ) -> ChatCompletionResponse:
        """Execute a request and return a normalized provider response."""


def _env_get(env: Mapping[str, str], key: str) -> str:
    return (env.get(key) or "").strip()


def resolve_llm_config(
    env: Mapping[str, str] | None = None,
    *,
    provider_var: str = "LLM_PROVIDER",
    default_provider: str = "local",
) -> LlmEndpointConfig:
    """Resolve an LLM endpoint from environment-style configuration.

    Supported provider aliases:
    - `local`, `vllm`, `litellm`: `VLLM_BASE_URL`, `VLLM_MODEL_NAME`,
      optional `VLLM_API_KEY`, optional `VLLM_REASONING_EFFORT`.
    - `deepseek`: `DEEPSEEK_BASE_URL`, `DEEPSEEK_MODEL_NAME`, optional
      `DEEPSEEK_API_KEY`.

    The generic aliases `LLM_BASE_URL`, `LLM_MODEL_NAME`, `LLM_API_KEY`, and
    `LLM_REASONING_EFFORT` are also accepted as fallbacks.
    """

    source = env if env is not None else os.environ
    provider = (_env_get(source, provider_var) or default_provider).lower()

    if provider in {"local", "vllm", "litellm"}:
        normalized_provider = "local" if provider == "local" else provider
        base_url = _env_get(source, "VLLM_BASE_URL") or _env_get(
            source, "LLM_BASE_URL"
        )
        model_name = _env_get(source, "VLLM_MODEL_NAME") or _env_get(
            source, "LLM_MODEL_NAME"
        )
        api_key = _env_get(source, "VLLM_API_KEY") or _env_get(source, "LLM_API_KEY")
        reasoning_effort = _env_get(
            source, "VLLM_REASONING_EFFORT"
        ) or _env_get(source, "LLM_REASONING_EFFORT")
        missing_hint = "VLLM_BASE_URL/LLM_BASE_URL and VLLM_MODEL_NAME/LLM_MODEL_NAME"
    elif provider == "deepseek":
        normalized_provider = "deepseek"
        base_url = (
            _env_get(source, "DEEPSEEK_BASE_URL") or _DEFAULT_DEEPSEEK_BASE_URL
        )
        model_name = _env_get(source, "DEEPSEEK_MODEL_NAME") or _env_get(
            source, "LLM_MODEL_NAME"
        )
        api_key = _env_get(source, "DEEPSEEK_API_KEY") or _env_get(
            source, "LLM_API_KEY"
        )
        reasoning_effort = ""
        missing_hint = "DEEPSEEK_MODEL_NAME/LLM_MODEL_NAME"
    else:
        raise LlmGatewayError(
            "server_misconfigured",
            f"unknown {provider_var}={provider!r} "
            "(expected local, vllm, litellm, or deepseek)",
        )

    if not base_url or not model_name:
        raise LlmGatewayError(
            "server_misconfigured",
            f"LLM provider {provider!r} missing required config: {missing_hint}",
        )

    return LlmEndpointConfig(
        provider=normalized_provider,
        base_url=base_url,
        model_name=model_name,
        api_key=api_key or None,
        reasoning_effort=reasoning_effort or None,
    )


def normalize_api_base(base_url: str) -> str:
    """Return a base URL ending in `/v1`."""

    base = base_url.rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def chat_completions_url(base_url: str) -> str:
    """Return the OpenAI-compatible chat completions URL for `base_url`."""

    return f"{normalize_api_base(base_url)}/chat/completions"


def auth_headers(cfg: LlmEndpointConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    return headers


def chat_completion_body(
    messages: Sequence[ChatMessageInput],
    *,
    cfg: LlmEndpointConfig,
    max_tokens: int,
    temperature: float = 0,
    response_format: LlmResponseFormat | None = None,
    tools: Sequence[LlmToolDefinition] = (),
    tool_choice: LlmToolChoice | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a request body for an OpenAI-compatible chat completion."""

    body: dict[str, Any] = {
        "model": cfg.model_name,
        "messages": [
            _chat_message_body(_chat_message_from_input(message))
            for message in messages
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if cfg.reasoning_effort:
        body["reasoning_effort"] = cfg.reasoning_effort
    if extra_body:
        frozen_extra = _freeze_json(extra_body, "extra_body")
        assert isinstance(frozen_extra, Mapping)
        reserved_fields = {"response_format", "tools", "tool_choice"}
        overridden_fields = reserved_fields.intersection(frozen_extra)
        if overridden_fields:
            names = ", ".join(sorted(overridden_fields))
            raise ValueError(
                f"extra_body cannot override normalized fields: {names}"
            )
        body.update(_thaw_json(frozen_extra))
    if response_format is not None:
        body["response_format"] = _openai_response_format(response_format)
    if tools:
        body["tools"] = [_openai_tool_definition(tool) for tool in tools]
    if tool_choice is not None:
        body["tool_choice"] = _openai_tool_choice(tool_choice)
    return body


def _openai_response_format(response_format: LlmResponseFormat) -> dict[str, Any]:
    if response_format.kind is LlmResponseFormatType.JSON_OBJECT:
        return {"type": "json_object"}
    assert response_format.name is not None
    assert response_format.schema is not None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_format.name,
            "strict": response_format.strict,
            "schema": _thaw_json(response_format.schema),
        },
    }


def _openai_tool_definition(tool: LlmToolDefinition) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": tool.name,
        "parameters": _thaw_json(tool.input_schema),
    }
    if tool.description is not None:
        function["description"] = tool.description
    return {"type": "function", "function": function}


def _openai_tool_choice(tool_choice: LlmToolChoice) -> str | dict[str, Any]:
    if tool_choice.mode is not LlmToolChoiceMode.NAMED:
        return tool_choice.mode.value
    assert tool_choice.name is not None
    return {
        "type": "function",
        "function": {"name": tool_choice.name},
    }


def _parse_openai_tool_calls(value: Any) -> tuple[ToolCall, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("message tool_calls must be a sequence")

    calls: list[ToolCall] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("message tool_calls entries must be mappings")
        function = item.get("function")
        if not isinstance(function, Mapping):
            raise TypeError("tool call function must be a mapping")
        encoded_arguments = function.get("arguments", "{}")
        if not isinstance(encoded_arguments, str):
            raise TypeError("tool call arguments must be encoded as JSON text")
        arguments = json.loads(encoded_arguments)
        if not isinstance(arguments, Mapping):
            raise TypeError("tool call arguments must decode to an object")
        calls.append(
            ToolCall(
                id=item.get("id"),
                name=function.get("name"),
                arguments=arguments,
            )
        )
    return tuple(calls)


def _parse_token_usage(value: Any) -> TokenUsage | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("usage must be a mapping")

    def _count(key: str) -> int:
        count = value.get(key, 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise TypeError(f"usage {key} must be a non-negative integer")
        return count

    input_tokens = _count("prompt_tokens")
    output_tokens = _count("completion_tokens")
    total_tokens = value.get("total_tokens", input_tokens + output_tokens)
    if (
        isinstance(total_tokens, bool)
        or not isinstance(total_tokens, int)
        or total_tokens < 0
    ):
        raise TypeError("usage total_tokens must be a non-negative integer")
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _safe_provider_metadata(payload: Mapping[str, Any]) -> Mapping[str, JsonValue]:
    metadata: dict[str, JsonValue] = {}
    response_id = payload.get("id")
    if isinstance(response_id, str) and response_id.strip():
        metadata["response_id"] = response_id
    created = payload.get("created")
    if isinstance(created, int) and not isinstance(created, bool) and created >= 0:
        metadata["created"] = created
    fingerprint = payload.get("system_fingerprint")
    if isinstance(fingerprint, str) and fingerprint.strip():
        metadata["system_fingerprint"] = fingerprint
    return MappingProxyType(metadata)


class OpenAICompatibleAdapter:
    """Adapter for OpenAI-compatible chat completion APIs."""

    async def complete(
        self,
        request: ChatCompletionRequest,
        *,
        cfg: LlmEndpointConfig,
        http_client: httpx.AsyncClient,
    ) -> ChatCompletionResponse:
        """Call an endpoint and normalize its response."""

        require_capabilities(request, cfg)
        url = chat_completions_url(cfg.base_url)

        body = chat_completion_body(
            request.messages,
            cfg=cfg,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            response_format=request.response_format,
            tools=request.tools,
            tool_choice=request.tool_choice,
            extra_body=request.extra_body,
        )

        attributes: dict[str, str | int | float] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": cfg.provider,
            "gen_ai.request.model": cfg.model_name,
            "llmkit.timeout_seconds": request.timeout_seconds,
        }
        with trace_span(
            "llm.chat_completion",
            kind="client",
            attributes=attributes,
        ) as span:
            headers = auth_headers(cfg)
            inject_trace_context(headers)
            start = time.perf_counter()
            try:
                resp = await http_client.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=request.timeout_seconds,
                )
            except httpx.TimeoutException as exc:
                elapsed = time.perf_counter() - start
                logger.error("LLM gateway timed out after %.2fs", elapsed)
                if span is not None:
                    span.set_attribute("error.type", "llm_timeout")
                    set_span_error(span, "llm_timeout")
                raise LlmGatewayError("llm_timeout", str(exc)) from exc
            except httpx.HTTPError as exc:
                elapsed = time.perf_counter() - start
                logger.error(
                    "LLM gateway request failed after %.2fs: %s", elapsed, exc
                )
                if span is not None:
                    span.set_attribute("error.type", "llm_fetch_failed")
                    set_span_error(span, "llm_fetch_failed")
                raise LlmGatewayError("llm_fetch_failed", str(exc)) from exc

            elapsed = time.perf_counter() - start
            if span is not None:
                span.set_attribute("http.response.status_code", resp.status_code)
            if resp.status_code < 200 or resp.status_code >= 300:
                body_text = resp.text[:500]
                error_code = f"llm_http_{resp.status_code}"
                logger.error(
                    "LLM gateway returned HTTP %d after %.2fs: %s",
                    resp.status_code,
                    elapsed,
                    body_text,
                )
                if span is not None:
                    span.set_attribute("error.type", error_code)
                    set_span_error(span, error_code)
                raise LlmGatewayError(
                    error_code, "non-2xx from LLM gateway", body_text
                )

            try:
                payload = resp.json()
                if not isinstance(payload, Mapping):
                    raise TypeError("response payload must be a mapping")
                choices = payload["choices"]
                if isinstance(choices, (str, bytes)) or not isinstance(
                    choices, Sequence
                ):
                    raise TypeError("response choices must be a sequence")
                choice = choices[0]
                if not isinstance(choice, Mapping):
                    raise TypeError("response choice must be a mapping")
                message = choice["message"]
                if not isinstance(message, Mapping):
                    raise TypeError("response message must be a mapping")
                content = message.get("content")
                if content is not None and not isinstance(content, str):
                    raise TypeError("message content must be a string or None")
                tool_calls = _parse_openai_tool_calls(message.get("tool_calls"))
                finish_reason = choice.get("finish_reason")
                if finish_reason is not None and not isinstance(finish_reason, str):
                    raise TypeError("finish_reason must be a string or None")
                usage = _parse_token_usage(payload.get("usage"))
                returned_model = payload.get("model", cfg.model_name)
                if not isinstance(returned_model, str) or not returned_model.strip():
                    raise TypeError("response model must be a non-empty string")
                result = ChatCompletionResponse(
                    text=content,
                    tool_calls=tool_calls,
                    finish_reason=finish_reason,
                    usage=usage,
                    provider=cfg.provider,
                    model=returned_model,
                    provider_metadata=_safe_provider_metadata(payload),
                )
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                logger.error(
                    "LLM gateway returned an unexpected response shape: %s", exc
                )
                if span is not None:
                    span.set_attribute("error.type", "llm_invalid_response_json")
                    set_span_error(span, "llm_invalid_response_json")
                raise LlmGatewayError("llm_invalid_response_json", str(exc)) from exc

            if span is not None:
                span.set_attribute("gen_ai.response.model", result.model)
                if result.usage is not None:
                    _record_usage_attributes(span, result.usage)

            logger.info(
                "LLM gateway call ok in %.2fs (provider=%s model=%s)",
                elapsed,
                cfg.provider,
                result.model,
            )
            return result


_OPENAI_COMPATIBLE_ADAPTER = OpenAICompatibleAdapter()


async def call_chat_completion_response(
    messages: Sequence[ChatMessageInput],
    *,
    cfg: LlmEndpointConfig,
    http_client: httpx.AsyncClient,
    max_tokens: int,
    timeout_seconds: float,
    temperature: float = 0,
    response_format: LlmResponseFormat | None = None,
    tools: Sequence[LlmToolDefinition] = (),
    tool_choice: LlmToolChoice | None = None,
    extra_body: Mapping[str, Any] | None = None,
    required_capabilities: Iterable[LlmCapability | str] = (LlmCapability.TEXT,),
) -> ChatCompletionResponse:
    """Call an endpoint once and return its normalized provider response."""

    request = ChatCompletionRequest(
        messages=messages,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        extra_body=extra_body,
        required_capabilities=required_capabilities,
    )
    return await _OPENAI_COMPATIBLE_ADAPTER.complete(
        request,
        cfg=cfg,
        http_client=http_client,
    )


async def call_chat_completion(
    messages: Sequence[ChatMessageInput],
    *,
    cfg: LlmEndpointConfig,
    http_client: httpx.AsyncClient,
    max_tokens: int,
    timeout_seconds: float,
    temperature: float = 0,
    response_format: LlmResponseFormat | None = None,
    tools: Sequence[LlmToolDefinition] = (),
    tool_choice: LlmToolChoice | None = None,
    extra_body: Mapping[str, Any] | None = None,
    required_capabilities: Iterable[LlmCapability | str] = (LlmCapability.TEXT,),
) -> str:
    """Call an endpoint once and return text for compatibility-oriented callers."""

    response = await call_chat_completion_response(
        messages,
        cfg=cfg,
        http_client=http_client,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        response_format=response_format,
        tools=tools,
        tool_choice=tool_choice,
        extra_body=extra_body,
        required_capabilities=required_capabilities,
    )
    if response.text is None:
        raise LlmGatewayError(
            "llm_text_response_required",
            "LLM response did not contain text content",
        )
    return response.text


def _record_usage_attributes(span: Any, usage: TokenUsage) -> None:
    span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
    span.set_attribute("llmkit.usage.total_tokens", usage.total_tokens)
