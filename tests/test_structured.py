import json

import httpx
import pytest
from pydantic import BaseModel, Field

from llmkit_lite.llm import LlmEndpointConfig
from llmkit_lite.structured import (
    clamp_float,
    extract_json_object,
    parse_structured_json,
    positive_float_or_none,
    str_or_none,
    structured_json_call,
)


class DemoOutput(BaseModel):
    name: str
    score: float = Field(ge=0, le=1)


def _cfg() -> LlmEndpointConfig:
    return LlmEndpointConfig(
        provider="local",
        base_url="https://gateway.example.com",
        model_name="test-model",
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
