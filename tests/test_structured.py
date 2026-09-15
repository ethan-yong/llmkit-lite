import json

import httpx
import pytest
from pydantic import BaseModel, Field

from llmkit_lite.llm import LlmCapabilities, LlmEndpointConfig, LlmGatewayError
from llmkit_lite.structured import (
    StructuredOutputPolicy,
    StructuredOutputStrategy,
    clamp_float,
    extract_json_object,
    parse_structured_json,
    positive_float_or_none,
    select_structured_output_strategy,
    str_or_none,
    structured_json_call,
)


class DemoOutput(BaseModel):
    name: str
    score: float = Field(ge=0, le=1)


def _cfg(
    capabilities: set[str] | None = None,
) -> LlmEndpointConfig:
    return LlmEndpointConfig(
        provider="local",
        base_url="https://gateway.example.com",
        model_name="test-model",
        capabilities=LlmCapabilities(
            capabilities
            if capabilities is not None
            else {"text", "json_object"}
        ),
    )


def _chat_content(text: str) -> str:
    return json.dumps(
        {"model": "test-model", "choices": [{"message": {"content": text}}]}
    )


def test_extract_json_object_plain() -> None:
    assert extract_json_object('{"a": 1}') == '{"a": 1}'


def test_extract_json_object_strips_think_block() -> None:
    raw = '<think>reasoning here</think>\n{"a": 1}'
    assert extract_json_object(raw) == '{"a": 1}'


def test_extract_json_object_strips_markdown_fence() -> None:
    raw = '```json\n{"a": 1}\n```'
    assert extract_json_object(raw) == '{"a": 1}'


def test_extract_json_object_no_braces_returns_none() -> None:
    assert extract_json_object("no json here") is None


def test_parse_structured_json_accepts_valid_model() -> None:
    result = parse_structured_json('{"name":"alpha","score":0.8}', DemoOutput)
    assert result.ok
    assert result.value == DemoOutput(name="alpha", score=0.8)


def test_parse_structured_json_rejects_missing_json() -> None:
    result = parse_structured_json("plain text", DemoOutput)
    assert not result.ok
    assert result.error_code == "llm_unparseable_content"


def test_parse_structured_json_rejects_invalid_json() -> None:
    result = parse_structured_json('{"name": }', DemoOutput)
    assert not result.ok
    assert result.error_code == "llm_invalid_json"


def test_parse_structured_json_rejects_schema_mismatch() -> None:
    result = parse_structured_json('{"name":"alpha","score":5}', DemoOutput)
    assert not result.ok
    assert result.error_code == "schema_validation_failed"


async def test_structured_json_call_validates_llm_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}
        assert "JSON Schema" in body["messages"][0]["content"]
        return httpx.Response(
            200, content=_chat_content('{"name":"alpha","score":0.9}')
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await structured_json_call(
            [{"role": "user", "content": "score alpha"}],
            response_model=DemoOutput,
            cfg=_cfg(),
            http_client=client,
            max_tokens=100,
            timeout_seconds=5,
        )
    assert result.ok
    assert result.value == DemoOutput(name="alpha", score=0.9)


@pytest.mark.parametrize(
    ("capabilities", "policy", "expected"),
    [
        (
            {"text", "json_schema", "json_object"},
            StructuredOutputPolicy.REQUIRE_NATIVE,
            StructuredOutputStrategy.JSON_SCHEMA,
        ),
        (
            {"text", "json_object"},
            StructuredOutputPolicy.REQUIRE_NATIVE,
            StructuredOutputStrategy.JSON_OBJECT,
        ),
        (
            {"text"},
            StructuredOutputPolicy.ALLOW_PROMPT_FALLBACK,
            StructuredOutputStrategy.PROMPT_ONLY,
        ),
    ],
)
def test_given_capabilities_when_selecting_strategy_then_strongest_mode_is_used(
    capabilities: set[str],
    policy: StructuredOutputPolicy,
    expected: StructuredOutputStrategy,
) -> None:
    assert (
        select_structured_output_strategy(LlmCapabilities(capabilities), policy)
        is expected
    )


async def test_given_schema_capability_when_calling_then_pydantic_schema_is_sent(
) -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200, content=_chat_content('{"name":"alpha","score":0.9}')
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await structured_json_call(
            [{"role": "user", "content": "score alpha"}],
            response_model=DemoOutput,
            cfg=_cfg({"text", "json_object", "json_schema"}),
            http_client=client,
            max_tokens=100,
            timeout_seconds=5,
        )

    assert result.ok
    response_format = bodies[0]["response_format"]
    assert isinstance(response_format, dict)
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"] == {
        "name": "DemoOutput",
        "strict": False,
        "schema": DemoOutput.model_json_schema(),
    }


async def test_given_text_only_endpoint_when_native_required_then_call_is_rejected(
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, content=_chat_content('{"name":"alpha","score":0.9}')
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(LlmGatewayError) as exc_info:
            await structured_json_call(
                [{"role": "user", "content": "score alpha"}],
                response_model=DemoOutput,
                cfg=_cfg({"text"}),
                http_client=client,
                max_tokens=100,
                timeout_seconds=5,
            )

    assert exc_info.value.code == "llm_capability_unsupported"
    assert calls == 0


async def test_given_prompt_fallback_when_calling_then_native_format_is_omitted(
) -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200, content=_chat_content('{"name":"alpha","score":0.9}')
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await structured_json_call(
            [{"role": "user", "content": "score alpha"}],
            response_model=DemoOutput,
            cfg=_cfg({"text"}),
            http_client=client,
            max_tokens=100,
            timeout_seconds=5,
            policy=StructuredOutputPolicy.ALLOW_PROMPT_FALLBACK,
        )

    assert result.ok
    assert "response_format" not in bodies[0]
    messages = bodies[0]["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert "JSON Schema" in messages[0]["content"]


def test_given_schema_policy_when_only_json_object_exists_then_selection_is_rejected(
) -> None:
    with pytest.raises(LlmGatewayError) as exc_info:
        select_structured_output_strategy(
            LlmCapabilities({"text", "json_object"}),
            StructuredOutputPolicy.REQUIRE_JSON_SCHEMA,
        )

    assert exc_info.value.code == "llm_capability_unsupported"


def test_clamp_float_bounds() -> None:
    assert clamp_float(-5) == 0.0
    assert clamp_float(5) == 1.0
    assert clamp_float(0.42) == pytest.approx(0.42)
    assert clamp_float("not a number") == 0.0


def test_str_or_none() -> None:
    assert str_or_none("  hi  ") == "hi"
    assert str_or_none("   ") is None
    assert str_or_none(123) is None


def test_positive_float_or_none() -> None:
    assert positive_float_or_none(18.5) == 18.5
    assert positive_float_or_none("18.5") == 18.5
    assert positive_float_or_none(-1) is None
    assert positive_float_or_none(True) is None
    assert positive_float_or_none(0, allow_zero=False) is None
