"""Structured-output helpers for LLM calls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from llmkit_lite.llm import ChatMessage, LlmEndpointConfig, call_chat_completion

TModel = TypeVar("TModel", bound=BaseModel)


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
    ) -> "StructuredCallResult[TModel]":
        return cls(value=value, raw_content=raw_content, json_text=json_text)

    @classmethod
    def failure(
        cls,
        *,
        raw_content: str,
        error_code: str,
        error_detail: str,
        json_text: str | None = None,
    ) -> "StructuredCallResult[TModel]":
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
    messages: list[ChatMessage],
    *,
    response_model: type[TModel],
    cfg: LlmEndpointConfig,
    http_client: httpx.AsyncClient,
    max_tokens: int,
    timeout_seconds: float,
    temperature: float = 0,
    use_response_format: bool = True,
    extra_body: dict[str, Any] | None = None,
) -> StructuredCallResult[TModel]:
    """Call an LLM and validate the response against a Pydantic model."""

    content = await call_chat_completion(
        messages,
        cfg=cfg,
        http_client=http_client,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        use_response_format=use_response_format,
        extra_body=extra_body,
    )
    return parse_structured_json(content, response_model)


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
