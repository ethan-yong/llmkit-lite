"""Provider-agnostic helpers for OpenAI-compatible chat completion APIs."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
ChatMessage = Mapping[str, str]

_DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


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


@dataclass(frozen=True)
class LlmEndpointConfig:
    """Resolved settings for one chat-completion endpoint."""

    provider: str
    base_url: str
    model_name: str
    api_key: str | None = None
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class ChatCompletionRequest:
    """Provider-independent inputs for one chat completion."""

    messages: Sequence[ChatMessage]
    max_tokens: int
    timeout_seconds: float
    temperature: float = 0
    use_response_format: bool = True
    extra_body: Mapping[str, Any] | None = None
    routing_metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "messages",
            tuple(MappingProxyType(dict(message)) for message in self.messages),
        )
        if self.extra_body is not None:
            object.__setattr__(
                self,
                "extra_body",
                MappingProxyType(dict(self.extra_body)),
            )
        object.__setattr__(
            self,
            "routing_metadata",
            MappingProxyType(dict(self.routing_metadata)),
        )


class LlmProviderAdapter(Protocol):
    """Transport boundary implemented by an LLM provider adapter."""

    async def complete(
        self,
        request: ChatCompletionRequest,
        *,
        cfg: LlmEndpointConfig,
        http_client: httpx.AsyncClient,
    ) -> str:
        """Execute one chat-completion request and return its text content."""


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
    messages: Sequence[ChatMessage],
    *,
    cfg: LlmEndpointConfig,
    max_tokens: int,
    temperature: float = 0,
    response_format: dict[str, str] | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a request body for an OpenAI-compatible chat completion."""

    body: dict[str, Any] = {
        "model": cfg.model_name,
        "messages": [dict(message) for message in messages],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        body["response_format"] = dict(response_format)
    if cfg.reasoning_effort:
        body["reasoning_effort"] = cfg.reasoning_effort
    if extra_body:
        body.update(extra_body)
    return body


class OpenAICompatibleAdapter:
    """Adapter for OpenAI-compatible chat completion APIs."""

    async def complete(
        self,
        request: ChatCompletionRequest,
        *,
        cfg: LlmEndpointConfig,
        http_client: httpx.AsyncClient,
    ) -> str:
        """Call an endpoint and return message content.

        A single retry without `response_format` is attempted after HTTP 400
        because several compatible gateways reject that parameter even when
        they can still produce JSON.
        """

        url = chat_completions_url(cfg.base_url)

        def _body(with_response_format: bool) -> dict[str, Any]:
            return chat_completion_body(
                request.messages,
                cfg=cfg,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                response_format=(
                    {"type": "json_object"}
                    if with_response_format and request.use_response_format
                    else None
                ),
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
                    json=_body(True),
                    timeout=request.timeout_seconds,
                )
                if resp.status_code == 400 and request.use_response_format:
                    if span is not None:
                        span.add_event(
                            "llm.response_format_retry",
                            {"http.response.status_code": 400},
                        )
                    resp = await http_client.post(
                        url,
                        headers=headers,
                        json=_body(False),
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
                content = payload["choices"][0]["message"]["content"]
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                logger.error(
                    "LLM gateway returned an unexpected response shape: %s", exc
                )
                if span is not None:
                    span.set_attribute("error.type", "llm_invalid_response_json")
                    set_span_error(span, "llm_invalid_response_json")
                raise LlmGatewayError("llm_invalid_response_json", str(exc)) from exc

            if not isinstance(content, str):
                if span is not None:
                    span.set_attribute("error.type", "llm_invalid_response_json")
                    set_span_error(span, "llm_invalid_response_json")
                raise LlmGatewayError(
                    "llm_invalid_response_json", "message content was not a string"
                )

            returned_model = payload.get("model", cfg.model_name)
            if span is not None:
                if isinstance(returned_model, str):
                    span.set_attribute("gen_ai.response.model", returned_model)
                usage = payload.get("usage")
                if isinstance(usage, dict):
                    _record_usage_attributes(span, usage)

            logger.info(
                "LLM gateway call ok in %.2fs (provider=%s model=%s)",
                elapsed,
                cfg.provider,
                returned_model,
            )
            return content


_OPENAI_COMPATIBLE_ADAPTER = OpenAICompatibleAdapter()


async def call_chat_completion(
    messages: Sequence[ChatMessage],
    *,
    cfg: LlmEndpointConfig,
    http_client: httpx.AsyncClient,
    max_tokens: int,
    timeout_seconds: float,
    temperature: float = 0,
    use_response_format: bool = True,
    extra_body: Mapping[str, Any] | None = None,
) -> str:
    """Call a configured chat-completion endpoint and return message content."""

    request = ChatCompletionRequest(
        messages=messages,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        use_response_format=use_response_format,
        extra_body=extra_body,
    )
    return await _OPENAI_COMPATIBLE_ADAPTER.complete(
        request,
        cfg=cfg,
        http_client=http_client,
    )


def _record_usage_attributes(span: Any, usage: Mapping[str, Any]) -> None:
    attribute_by_usage_key = {
        "prompt_tokens": "gen_ai.usage.input_tokens",
        "completion_tokens": "gen_ai.usage.output_tokens",
        "total_tokens": "llmkit.usage.total_tokens",
    }
    for usage_key, attribute_name in attribute_by_usage_key.items():
        value = usage.get(usage_key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            span.set_attribute(attribute_name, value)
