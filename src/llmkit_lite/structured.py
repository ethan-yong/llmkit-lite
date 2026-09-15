"""Structured-output helpers for LLM calls."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from llmkit_lite.llm import (
    ChatMessage,
    ChatMessageInput,
    LlmCapabilities,
    LlmCapability,
    LlmEndpointConfig,
    LlmGatewayError,
    LlmResponseFormat,
    call_chat_completion,
)

TModel = TypeVar("TModel", bound=BaseModel)


class StructuredOutputPolicy(StrEnum):
    """How strongly a structured call must be enforced by the provider."""

    REQUIRE_JSON_SCHEMA = "require_json_schema"
    REQUIRE_NATIVE = "require_native"
    ALLOW_PROMPT_FALLBACK = "allow_prompt_fallback"


class StructuredOutputStrategy(StrEnum):
    """Concrete strategy selected from an endpoint's capabilities."""

    JSON_SCHEMA = "json_schema"
    JSON_OBJECT = "json_object"
    PROMPT_ONLY = "prompt_only"


def select_structured_output_strategy(
    capabilities: LlmCapabilities,
    policy: StructuredOutputPolicy | str = StructuredOutputPolicy.REQUIRE_NATIVE,
) -> StructuredOutputStrategy:
    """Select the strongest allowed strategy without contacting the provider."""

    if not isinstance(capabilities, LlmCapabilities):
        raise TypeError("capabilities must be an LlmCapabilities value")
    try:
        normalized_policy = StructuredOutputPolicy(policy)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported structured output policy: {policy!r}") from exc
    if not capabilities.supports({LlmCapability.TEXT}):
        raise LlmGatewayError(
            "llm_capability_unsupported",
            "structured output requires the text capability",
        )
    if capabilities.supports({LlmCapability.JSON_SCHEMA}):
        return StructuredOutputStrategy.JSON_SCHEMA
    if normalized_policy is StructuredOutputPolicy.REQUIRE_JSON_SCHEMA:
        raise LlmGatewayError(
            "llm_capability_unsupported",
            "structured output requires the json_schema capability",
        )
    if capabilities.supports({LlmCapability.JSON_OBJECT}):
        return StructuredOutputStrategy.JSON_OBJECT
    if normalized_policy is StructuredOutputPolicy.ALLOW_PROMPT_FALLBACK:
        return StructuredOutputStrategy.PROMPT_ONLY
    raise LlmGatewayError(
        "llm_capability_unsupported",
        "native structured output requires json_schema or json_object capability",
    )


@dataclass(frozen=True)
class StructuredCallResult(Generic[TModel]):
    """Typed result for a structured LLM call or parse attempt."""

    value: TModel | None
    raw_content: str
    json_text: str | None = None
    error_code: str | None = None
    error_detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.value is not None and self.error_code is None

    @classmethod
    def success(
        cls,
        *,
        value: TModel,
        raw_content: str,
        json_text: str,
    ) -> StructuredCallResult[TModel]:
        return cls(value=value, raw_content=raw_content, json_text=json_text)

    @classmethod
    def failure(
        cls,
        *,
        raw_content: str,
        error_code: str,
        error_detail: str,
        json_text: str | None = None,
    ) -> StructuredCallResult[TModel]:
        return cls(
            value=None,
            raw_content=raw_content,
            json_text=json_text,
            error_code=error_code,
            error_detail=error_detail,
        )


def extract_json_object(raw: str) -> str | None:
    """Extract the first plausible JSON object from model output.

    This strips common reasoning blocks and markdown fences before slicing from
    the first `{` to the last `}`. It is deliberately defensive for local and
    OpenAI-compatible backends that can be chatty even when asked for JSON.
    """

    s = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE)
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, flags=re.IGNORECASE)
    if fence:
        s = fence.group(1)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return s[start : end + 1]


def parse_structured_json(
    raw: str,
    response_model: type[TModel],
) -> StructuredCallResult[TModel]:
    """Parse model output as a Pydantic model without making an LLM call."""

    json_text = extract_json_object(raw)
    if json_text is None:
        return StructuredCallResult.failure(
            raw_content=raw,
            error_code="llm_unparseable_content",
            error_detail="no JSON object found in LLM response",
        )
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return StructuredCallResult.failure(
            raw_content=raw,
            json_text=json_text,
            error_code="llm_invalid_json",
            error_detail=str(exc),
        )
    try:
        value = response_model.model_validate(payload)
    except ValidationError as exc:
        return StructuredCallResult.failure(
            raw_content=raw,
            json_text=json_text,
            error_code="schema_validation_failed",
            error_detail=exc.json(),
        )
    return StructuredCallResult.success(
        value=value,
        raw_content=raw,
        json_text=json_text,
    )


async def structured_json_call(
    messages: Sequence[ChatMessageInput],
    *,
    response_model: type[TModel],
    cfg: LlmEndpointConfig,
    http_client: httpx.AsyncClient,
    max_tokens: int,
    timeout_seconds: float,
    temperature: float = 0,
    policy: StructuredOutputPolicy | str = StructuredOutputPolicy.REQUIRE_NATIVE,
    extra_body: Mapping[str, Any] | None = None,
) -> StructuredCallResult[TModel]:
    """Call an LLM and validate the response against a Pydantic model."""

    strategy = select_structured_output_strategy(cfg.capabilities, policy)
    schema = response_model.model_json_schema()
    schema_text = json.dumps(schema, separators=(",", ":"), sort_keys=True)
    structured_messages: tuple[ChatMessageInput, ...] = (
        ChatMessage(
            role="system",
            content=(
                "Return only a JSON object matching the following JSON Schema. "
                "Do not include Markdown or commentary.\nJSON Schema:\n"
                f"{schema_text}"
            ),
        ),
        *messages,
    )

    response_format: LlmResponseFormat | None = None
    if strategy is StructuredOutputStrategy.JSON_SCHEMA:
        response_format = LlmResponseFormat.json_schema(
            name=_response_schema_name(response_model),
            schema=schema,
        )
    elif strategy is StructuredOutputStrategy.JSON_OBJECT:
        response_format = LlmResponseFormat.json_object()

    content = await call_chat_completion(
        structured_messages,
        cfg=cfg,
        http_client=http_client,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        response_format=response_format,
        extra_body=extra_body,
    )
    return parse_structured_json(content, response_model)


def _response_schema_name(response_model: type[BaseModel]) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]", "_", response_model.__name__)
    return (normalized.strip("_") or "structured_response")[:64]


def clamp_float(value: object, *, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        n = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return minimum
    if n != n:
        return minimum
    return max(minimum, min(maximum, n))


def str_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def positive_float_or_none(value: object, *, allow_zero: bool = True) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        n = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if n != n or n < 0:
        return None
    if n == 0 and not allow_zero:
        return None
    return n
