import json

import httpx
import pytest
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from llmkit_lite.llm import (
    LlmEndpointConfig,
    LlmGatewayError,
    auth_headers,
    call_chat_completion,
    chat_completion_body,
    chat_completions_url,
    normalize_api_base,
    resolve_llm_config,
)


def _chat_content(text: str) -> str:
    return json.dumps(
        {"model": "test-model", "choices": [{"message": {"content": text}}]}
    )


def _cfg(**overrides: object) -> LlmEndpointConfig:
    defaults: dict[str, object] = {
        "provider": "local",
        "base_url": "https://gateway.example.com",
        "model_name": "test-model",
        "api_key": None,
        "reasoning_effort": None,
    }
    defaults.update(overrides)
    return LlmEndpointConfig(**defaults)  # type: ignore[arg-type]


def test_resolve_llm_config_local_default() -> None:
    cfg = resolve_llm_config(
        {
            "LLM_BASE_URL": "https://gateway.example.com",
            "LLM_MODEL_NAME": "qwen2.5-vl",
        }
    )
    assert cfg.provider == "local"
    assert cfg.base_url == "https://gateway.example.com"
    assert cfg.model_name == "qwen2.5-vl"


def test_resolve_llm_config_vllm_specific_vars_win() -> None:
    cfg = resolve_llm_config(
        {
            "LLM_PROVIDER": "vllm",
            "LLM_BASE_URL": "https://generic.example.com",
            "LLM_MODEL_NAME": "generic",
            "VLLM_BASE_URL": "https://vllm.example.com",
            "VLLM_MODEL_NAME": "vllm-model",
            "VLLM_REASONING_EFFORT": "low",
        }
    )
    assert cfg.provider == "vllm"
    assert cfg.base_url == "https://vllm.example.com"
    assert cfg.model_name == "vllm-model"
    assert cfg.reasoning_effort == "low"


def test_resolve_llm_config_deepseek() -> None:
    cfg = resolve_llm_config(
        {
            "LLM_PROVIDER": "deepseek",
            "DEEPSEEK_MODEL_NAME": "deepseek-chat",
            "DEEPSEEK_API_KEY": "sk-test",
            "LLM_REASONING_EFFORT": "ignored",
        }
    )
    assert cfg.provider == "deepseek"
    assert cfg.base_url == "https://api.deepseek.com/v1"
    assert cfg.api_key == "sk-test"
    assert cfg.reasoning_effort is None


def test_resolve_llm_config_missing_vars_raises() -> None:
    with pytest.raises(LlmGatewayError) as exc_info:
        resolve_llm_config({})
    assert exc_info.value.code == "server_misconfigured"


def test_resolve_llm_config_unknown_provider() -> None:
    with pytest.raises(LlmGatewayError) as exc_info:
        resolve_llm_config({"LLM_PROVIDER": "unknown"})
    assert exc_info.value.code == "server_misconfigured"
    assert "unknown LLM_PROVIDER" in exc_info.value.detail


def test_normalize_api_base() -> None:
    assert normalize_api_base("https://gateway.example.com") == (
        "https://gateway.example.com/v1"
    )
    assert normalize_api_base("https://gateway.example.com/v1/") == (
        "https://gateway.example.com/v1"
    )


def test_chat_completions_url() -> None:
    assert chat_completions_url("https://gateway.example.com") == (
        "https://gateway.example.com/v1/chat/completions"
    )


def test_auth_headers_without_key() -> None:
    assert auth_headers(_cfg()) == {"Content-Type": "application/json"}


def test_auth_headers_with_key() -> None:
    assert auth_headers(_cfg(api_key="sk-test")) == {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-test",
    }


def test_chat_completion_body() -> None:
    body = chat_completion_body(
        [{"role": "user", "content": "hi"}],
        cfg=_cfg(reasoning_effort="medium"),
        max_tokens=50,
        temperature=0.2,
        response_format={"type": "json_object"},
        extra_body={"seed": 123},
    )
    assert body == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
        "max_tokens": 50,
        "response_format": {"type": "json_object"},
        "reasoning_effort": "medium",
        "seed": 123,
    }


async def test_call_chat_completion_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, content=_chat_content('{"ok": true}'))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        content = await call_chat_completion(
            [{"role": "user", "content": "hi"}],
            cfg=_cfg(),
            http_client=client,
            max_tokens=100,
            timeout_seconds=5,
        )
    assert content == '{"ok": true}'


async def test_call_chat_completion_creates_safe_client_span(
    in_memory_tracing,
) -> None:
    captured_headers: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_headers.update(request.headers)
        return httpx.Response(
            200,
            json={
                "model": "returned-model",
                "choices": [{"message": {"content": "private response"}}],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tracer = trace.get_tracer("test")
        with tracer.start_as_current_span("parent"):
            content = await call_chat_completion(
                [{"role": "user", "content": "private prompt"}],
                cfg=_cfg(api_key="private-api-key"),
                http_client=client,
                max_tokens=100,
                timeout_seconds=5,
            )

    spans = in_memory_tracing.get_finished_spans()
    parent_span = next(span for span in spans if span.name == "parent")
    llm_span = next(span for span in spans if span.name == "llm.chat_completion")
    assert content == "private response"
    assert captured_headers["traceparent"].startswith("00-")
    assert llm_span.kind.name == "CLIENT"
    assert llm_span.parent.span_id == parent_span.context.span_id
    assert llm_span.attributes["gen_ai.provider.name"] == "local"
    assert llm_span.attributes["gen_ai.request.model"] == "test-model"
    assert llm_span.attributes["gen_ai.response.model"] == "returned-model"
    assert llm_span.attributes["http.response.status_code"] == 200
    assert llm_span.attributes["gen_ai.usage.input_tokens"] == 7
    assert llm_span.attributes["gen_ai.usage.output_tokens"] == 3
    assert llm_span.attributes["llmkit.usage.total_tokens"] == 10
    exported = str(llm_span.attributes) + str(llm_span.events)
    assert "private prompt" not in exported
    assert "private response" not in exported
    assert "private-api-key" not in exported


async def test_call_chat_completion_retries_without_response_format_on_400(
    in_memory_tracing,
) -> None:
    calls: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": "unsupported"})
        return httpx.Response(200, content=_chat_content('{"ok": true}'))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        content = await call_chat_completion(
            [{"role": "user", "content": "hi"}],
            cfg=_cfg(),
            http_client=client,
            max_tokens=100,
            timeout_seconds=5,
        )
    assert content == '{"ok": true}'
    assert len(calls) == 2
    assert "response_format" not in calls[1]
    llm_span = next(
        span
        for span in in_memory_tracing.get_finished_spans()
        if span.name == "llm.chat_completion"
    )
    assert [event.name for event in llm_span.events] == [
        "llm.response_format_retry"
    ]


async def test_call_chat_completion_timeout_raises(in_memory_tracing) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(LlmGatewayError) as exc_info:
            await call_chat_completion(
                [{"role": "user", "content": "hi"}],
                cfg=_cfg(),
                http_client=client,
                max_tokens=100,
                timeout_seconds=5,
            )
    assert exc_info.value.code == "llm_timeout"
    span = next(
        span
        for span in in_memory_tracing.get_finished_spans()
        if span.name == "llm.chat_completion"
    )
    assert span.attributes["error.type"] == "llm_timeout"
    assert span.status.status_code is StatusCode.ERROR


async def test_call_chat_completion_transport_failure_is_traced(
    in_memory_tracing,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(LlmGatewayError) as exc_info:
            await call_chat_completion(
                [{"role": "user", "content": "hi"}],
                cfg=_cfg(),
                http_client=client,
                max_tokens=100,
                timeout_seconds=5,
            )

    span = next(
        span
        for span in in_memory_tracing.get_finished_spans()
        if span.name == "llm.chat_completion"
    )
    assert exc_info.value.code == "llm_fetch_failed"
    assert span.attributes["error.type"] == "llm_fetch_failed"
    assert span.status.status_code is StatusCode.ERROR


async def test_call_chat_completion_non_2xx_raises(in_memory_tracing) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream broke")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(LlmGatewayError) as exc_info:
            await call_chat_completion(
                [{"role": "user", "content": "hi"}],
                cfg=_cfg(),
                http_client=client,
                max_tokens=100,
                timeout_seconds=5,
            )
    assert exc_info.value.code == "llm_http_500"
    assert exc_info.value.raw == "upstream broke"
    span = next(
        span
        for span in in_memory_tracing.get_finished_spans()
        if span.name == "llm.chat_completion"
    )
    assert span.attributes["error.type"] == "llm_http_500"
    assert span.status.status_code is StatusCode.ERROR


async def test_call_chat_completion_invalid_response_shape_raises(
    in_memory_tracing,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(LlmGatewayError) as exc_info:
            await call_chat_completion(
                [{"role": "user", "content": "hi"}],
                cfg=_cfg(),
                http_client=client,
                max_tokens=100,
                timeout_seconds=5,
            )
    assert exc_info.value.code == "llm_invalid_response_json"
    span = next(
        span
        for span in in_memory_tracing.get_finished_spans()
        if span.name == "llm.chat_completion"
    )
    assert span.attributes["error.type"] == "llm_invalid_response_json"
    assert span.status.status_code is StatusCode.ERROR
