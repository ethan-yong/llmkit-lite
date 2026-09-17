"""Bounded orchestration for normalized model tool calls."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from llmkit_lite.authorization import Principal
from llmkit_lite.llm import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    LlmToolDefinition,
    ToolCall,
)
from llmkit_lite.observability import set_span_error, trace_span

CompletionCallback = Callable[
    [ChatCompletionRequest],
    Awaitable[ChatCompletionResponse],
]

_ERROR_DETAILS = {
    "tool_loop_declaration_mismatch": (
        "request tool declarations do not match the authorized executor"
    ),
    "tool_loop_invalid_response": "tool loop received an invalid normalized response",
    "tool_loop_invalid_tool_message": (
        "tool executor returned an invalid tool message"
    ),
    "tool_loop_limit_exceeded": "tool loop exceeded its configured round limit",
    "tool_loop_no_tools": "tool loop executor has no model-visible tools",
}


class ToolLoopError(Exception):
    """Safe orchestration failure suitable for application error mapping."""

    def __init__(self, code: str) -> None:
        if code not in _ERROR_DETAILS:
            raise ValueError("unsupported tool loop error code")
        detail = _ERROR_DETAILS[code]
        super().__init__(detail)
        self.code = code
        self.detail = detail


class ModelToolExecutor(Protocol):
    """Execution boundary required by the tool-loop runner."""

    @property
    def llm_tools(self) -> tuple[LlmToolDefinition, ...]:
        """Return tools that may be exposed to the model."""

    async def execute_call(
        self,
        call: ToolCall,
        *,
        principal: Principal | None = None,
    ) -> ChatMessage:
        """Authorize and execute one normalized model tool call."""


@dataclass(frozen=True, slots=True)
class ToolLoopResult:
    """Final response and transcript produced by a completed tool loop."""

    response: ChatCompletionResponse
    messages: tuple[ChatMessage, ...]
    tool_rounds: int
    tool_calls: int


def _executor_tools(executor: ModelToolExecutor) -> tuple[LlmToolDefinition, ...]:
    tools = getattr(executor, "llm_tools", None)
    if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
        raise TypeError("executor llm_tools must be a sequence")
    normalized = tuple(tools)
    if not all(isinstance(tool, LlmToolDefinition) for tool in normalized):
        raise TypeError("executor llm_tools must contain LlmToolDefinition values")
    if not normalized:
        raise ToolLoopError("tool_loop_no_tools")
    return normalized


def _assistant_message(response: ChatCompletionResponse) -> ChatMessage:
    if response.text is None and not response.tool_calls:
        raise ToolLoopError("tool_loop_invalid_response")
    return ChatMessage(
        role="assistant",
        content=response.text,
        tool_calls=response.tool_calls,
    )


def _validate_tool_message(message: ChatMessage, call: ToolCall) -> None:
    if (
        message.role != "tool"
        or message.tool_call_id != call.id
        or message.name != call.name
    ):
        raise ToolLoopError("tool_loop_invalid_tool_message")


async def run_tool_loop(
    request: ChatCompletionRequest,
    *,
    complete: CompletionCallback,
    executor: ModelToolExecutor,
    principal: Principal | None = None,
    max_tool_rounds: int = 4,
) -> ToolLoopResult:
    """Run sequential model/tool rounds until the model returns no tool calls."""

    if not isinstance(request, ChatCompletionRequest):
        raise TypeError("request must be a ChatCompletionRequest")
    if not callable(complete):
        raise TypeError("complete must be callable")
    if (
        isinstance(max_tool_rounds, bool)
        or not isinstance(max_tool_rounds, int)
        or max_tool_rounds < 0
    ):
        raise ValueError("max_tool_rounds must be a non-negative integer")

    tools = _executor_tools(executor)
    if request.tools and tuple(request.tools) != tools:
        raise ToolLoopError("tool_loop_declaration_mismatch")

    current_request = replace(request, tools=tools)
    messages = list(current_request.messages)
    tool_rounds = 0
    tool_calls = 0

    with trace_span(
        "llm.tool_loop",
        attributes={"llmkit.tool_loop.max_rounds": max_tool_rounds},
    ) as span:
        try:
            while True:
                response = await complete(current_request)
                if not isinstance(response, ChatCompletionResponse):
                    raise ToolLoopError("tool_loop_invalid_response")
                messages.append(_assistant_message(response))

                if not response.tool_calls:
                    if span is not None:
                        span.set_attribute("llmkit.tool_loop.outcome", "completed")
                        span.set_attribute("llmkit.tool_loop.rounds", tool_rounds)
                        span.set_attribute("llmkit.tool_loop.calls", tool_calls)
                    return ToolLoopResult(
                        response=response,
                        messages=tuple(messages),
                        tool_rounds=tool_rounds,
                        tool_calls=tool_calls,
                    )

                if tool_rounds >= max_tool_rounds:
                    raise ToolLoopError("tool_loop_limit_exceeded")

                for call in response.tool_calls:
                    tool_message = await executor.execute_call(
                        call,
                        principal=principal,
                    )
                    if not isinstance(tool_message, ChatMessage):
                        raise ToolLoopError("tool_loop_invalid_tool_message")
                    _validate_tool_message(tool_message, call)
                    messages.append(tool_message)
                    tool_calls += 1

                tool_rounds += 1
                current_request = replace(
                    current_request,
                    messages=tuple(messages),
                )
        except BaseException as exc:
            if span is not None:
                error_code = getattr(exc, "code", type(exc).__name__)
                span.set_attribute("llmkit.tool_loop.outcome", "failed")
                span.set_attribute("llmkit.tool_loop.rounds", tool_rounds)
                span.set_attribute("llmkit.tool_loop.calls", tool_calls)
                span.set_attribute("error.type", str(error_code))
                set_span_error(span, str(error_code))
            raise
