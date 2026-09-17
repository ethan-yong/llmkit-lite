import asyncio
from collections.abc import Sequence

import pytest
from opentelemetry.trace import StatusCode

from llmkit_lite.authorization import Principal
from llmkit_lite.llm import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    LlmCapability,
    LlmToolDefinition,
    ToolCall,
)
from llmkit_lite.tool_loop import ToolLoopError, run_tool_loop
from llmkit_lite.tools import AuthorizedToolExecutor, ToolDefinition

WEATHER_TOOL = LlmToolDefinition(
    "weather",
    {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
    "Get the current weather for a city.",
)


def _request(
    *,
    tools: Sequence[LlmToolDefinition] = (),
) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "Weather in Kuala Lumpur?"}],
        max_tokens=100,
        timeout_seconds=5,
        tools=tools,
    )


def _response(
    *,
    text: str | None = None,
    tool_calls: Sequence[ToolCall] = (),
) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        text=text,
        provider="test",
        model="test-model",
        tool_calls=tool_calls,
    )


class ScriptedCompletion:
    def __init__(
        self,
        outcomes: Sequence[ChatCompletionResponse | BaseException],
    ) -> None:
        self._outcomes = list(outcomes)
        self.requests: list[ChatCompletionRequest] = []

    async def __call__(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        self.requests.append(request)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class RecordingExecutor:
    def __init__(
        self,
        *,
        tools: Sequence[LlmToolDefinition] = (WEATHER_TOOL,),
        failure: BaseException | None = None,
        invalid_message: bool = False,
    ) -> None:
        self.llm_tools = tuple(tools)
        self.failure = failure
        self.invalid_message = invalid_message
        self.calls: list[tuple[ToolCall, Principal | None]] = []

    async def execute_call(
        self,
        call: ToolCall,
        *,
        principal: Principal | None = None,
    ) -> ChatMessage:
        self.calls.append((call, principal))
        if self.failure is not None:
            raise self.failure
        if self.invalid_message:
            return ChatMessage(
                role="tool",
                content="invalid",
                name=call.name,
                tool_call_id="different-call",
            )
        return ChatMessage(
            role="tool",
            content=f"result for {call.id}",
            name=call.name,
            tool_call_id=call.id,
        )


async def test_given_text_response_then_returns_without_tool_execution() -> None:
    completion = ScriptedCompletion((_response(text="Sunny."),))
    executor = RecordingExecutor()

    result = await run_tool_loop(
        _request(),
        complete=completion,
        executor=executor,
    )

    assert result.response.text == "Sunny."
    assert result.messages == (
        ChatMessage(role="user", content="Weather in Kuala Lumpur?"),
        ChatMessage(role="assistant", content="Sunny."),
    )
    assert result.tool_rounds == 0
    assert result.tool_calls == 0
    assert executor.calls == []
    assert completion.requests[0].tools == (WEATHER_TOOL,)
    assert LlmCapability.TOOL_CALLING in completion.requests[0].required_capabilities


async def test_given_tool_call_then_replays_transcript_in_order() -> None:
    call = ToolCall("call-1", "weather", {"city": "Kuala Lumpur"})
    completion = ScriptedCompletion(
        (
            _response(tool_calls=(call,)),
            _response(text="It is sunny."),
        )
    )
    executor = RecordingExecutor()

    result = await run_tool_loop(
        _request(),
        complete=completion,
        executor=executor,
    )

    assistant_call = ChatMessage(role="assistant", tool_calls=(call,))
    tool_result = ChatMessage(
        role="tool",
        content="result for call-1",
        name="weather",
        tool_call_id="call-1",
    )
    assert completion.requests[1].messages == (
        ChatMessage(role="user", content="Weather in Kuala Lumpur?"),
        assistant_call,
        tool_result,
    )
    assert result.messages == (
        *completion.requests[1].messages,
        ChatMessage(role="assistant", content="It is sunny."),
    )
    assert result.tool_rounds == 1
    assert result.tool_calls == 1


async def test_given_multiple_calls_then_executes_sequentially() -> None:
    calls = (
        ToolCall("call-1", "weather", {"city": "Kuala Lumpur"}),
        ToolCall("call-2", "weather", {"city": "Penang"}),
    )
    completion = ScriptedCompletion(
        (
            _response(tool_calls=calls),
            _response(text="Both are sunny."),
        )
    )
    executor = RecordingExecutor()

    result = await run_tool_loop(
        _request(),
        complete=completion,
        executor=executor,
    )

    assert [call.id for call, _principal in executor.calls] == ["call-1", "call-2"]
    assert result.tool_rounds == 1
    assert result.tool_calls == 2


async def test_given_principal_then_forwards_same_identity() -> None:
    principal = Principal("user-123", scopes={"tools:weather:execute"})
    call = ToolCall("call-1", "weather", {"city": "Kuala Lumpur"})
    completion = ScriptedCompletion(
        (_response(tool_calls=(call,)), _response(text="Sunny."))
    )
    executor = RecordingExecutor()

    await run_tool_loop(
        _request(),
        complete=completion,
        executor=executor,
        principal=principal,
    )

    assert executor.calls == [(call, principal)]


async def test_given_authorized_executor_then_runs_real_tool_boundary() -> None:
    principal = Principal("user-123", scopes={"tools:weather:execute"})
    observed: list[str] = []

    async def weather(arguments) -> dict[str, str]:
        city = str(arguments["city"])
        observed.append(city)
        return {"city": city, "forecast": "sunny"}

    executor = AuthorizedToolExecutor(
        (
            ToolDefinition(
                "weather",
                weather,
                required_scopes={"tools:weather:execute"},
                input_schema=WEATHER_TOOL.input_schema,
                description=WEATHER_TOOL.description,
            ),
        )
    )
    call = ToolCall("call-1", "weather", {"city": "Kuala Lumpur"})
    completion = ScriptedCompletion(
        (_response(tool_calls=(call,)), _response(text="It is sunny."))
    )

    result = await run_tool_loop(
        _request(),
        complete=completion,
        executor=executor,
        principal=principal,
    )

    assert observed == ["Kuala Lumpur"]
    assert result.messages[-2].content == ('{"city":"Kuala Lumpur","forecast":"sunny"}')


async def test_given_mismatched_tools_then_rejects_before_calls() -> None:
    other_tool = LlmToolDefinition("clock", {"type": "object"})
    completion = ScriptedCompletion((_response(text="unused"),))
    executor = RecordingExecutor()

    with pytest.raises(ToolLoopError) as exc_info:
        await run_tool_loop(
            _request(tools=(other_tool,)),
            complete=completion,
            executor=executor,
        )

    assert exc_info.value.code == "tool_loop_declaration_mismatch"
    assert completion.requests == []
    assert executor.calls == []


async def test_given_no_visible_tools_then_rejects_before_completion() -> None:
    completion = ScriptedCompletion((_response(text="unused"),))
    executor = RecordingExecutor(tools=())

    with pytest.raises(ToolLoopError) as exc_info:
        await run_tool_loop(
            _request(),
            complete=completion,
            executor=executor,
        )

    assert exc_info.value.code == "tool_loop_no_tools"
    assert completion.requests == []


async def test_given_round_limit_then_no_extra_tool_runs() -> None:
    first_call = ToolCall("call-1", "weather", {"city": "Kuala Lumpur"})
    second_call = ToolCall("call-2", "weather", {"city": "Penang"})
    completion = ScriptedCompletion(
        (
            _response(tool_calls=(first_call,)),
            _response(tool_calls=(second_call,)),
        )
    )
    executor = RecordingExecutor()

    with pytest.raises(ToolLoopError) as exc_info:
        await run_tool_loop(
            _request(),
            complete=completion,
            executor=executor,
            max_tool_rounds=1,
        )

    assert exc_info.value.code == "tool_loop_limit_exceeded"
    assert [call.id for call, _principal in executor.calls] == ["call-1"]
    assert len(completion.requests) == 2


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("private failure"), asyncio.CancelledError()],
)
async def test_given_executor_failure_when_running_loop_then_propagates_unchanged(
    failure: BaseException,
) -> None:
    call = ToolCall("call-1", "weather", {"city": "Kuala Lumpur"})
    completion = ScriptedCompletion((_response(tool_calls=(call,)),))
    executor = RecordingExecutor(failure=failure)

    with pytest.raises(type(failure)) as exc_info:
        await run_tool_loop(
            _request(),
            complete=completion,
            executor=executor,
        )

    assert exc_info.value is failure


async def test_given_provider_failure_then_propagates_and_traces_safely(
    in_memory_tracing,
) -> None:
    failure = RuntimeError("private provider failure")
    completion = ScriptedCompletion((failure,))
    executor = RecordingExecutor()

    with pytest.raises(RuntimeError) as exc_info:
        await run_tool_loop(
            _request(),
            complete=completion,
            executor=executor,
        )

    assert exc_info.value is failure
    loop_span = next(
        span
        for span in in_memory_tracing.get_finished_spans()
        if span.name == "llm.tool_loop"
    )
    assert loop_span.status.status_code is StatusCode.ERROR
    assert loop_span.attributes["llmkit.tool_loop.outcome"] == "failed"
    assert loop_span.attributes["llmkit.tool_loop.rounds"] == 0
    assert loop_span.attributes["llmkit.tool_loop.calls"] == 0
    assert loop_span.attributes["error.type"] == "RuntimeError"
    assert "private provider failure" not in repr(
        (loop_span.attributes, loop_span.events)
    )


async def test_given_invalid_tool_message_then_rejects_it() -> None:
    call = ToolCall("call-1", "weather", {"city": "Kuala Lumpur"})
    completion = ScriptedCompletion((_response(tool_calls=(call,)),))
    executor = RecordingExecutor(invalid_message=True)

    with pytest.raises(ToolLoopError) as exc_info:
        await run_tool_loop(
            _request(),
            complete=completion,
            executor=executor,
        )

    assert exc_info.value.code == "tool_loop_invalid_tool_message"


async def test_given_tracing_when_loop_completes_then_records_only_safe_counts(
    in_memory_tracing,
) -> None:
    secret = "private Kuala Lumpur request"
    call = ToolCall("call-1", "weather", {"city": secret})
    completion = ScriptedCompletion(
        (_response(tool_calls=(call,)), _response(text="private result"))
    )
    executor = RecordingExecutor()

    await run_tool_loop(
        ChatCompletionRequest(
            messages=[{"role": "user", "content": secret}],
            max_tokens=100,
            timeout_seconds=5,
        ),
        complete=completion,
        executor=executor,
        principal=Principal("private-user", scopes={"private-scope"}),
    )

    spans = in_memory_tracing.get_finished_spans()
    loop_span = next(span for span in spans if span.name == "llm.tool_loop")
    assert loop_span.status.status_code is StatusCode.UNSET
    assert loop_span.attributes == {
        "llmkit.tool_loop.max_rounds": 4,
        "llmkit.tool_loop.outcome": "completed",
        "llmkit.tool_loop.rounds": 1,
        "llmkit.tool_loop.calls": 1,
    }
    telemetry = repr((loop_span.attributes, loop_span.events))
    assert secret not in telemetry
    assert "private result" not in telemetry
    assert "private-user" not in telemetry
    assert "private-scope" not in telemetry


@pytest.mark.parametrize("max_tool_rounds", [-1, 1.5, True])
async def test_given_invalid_limit_when_starting_then_rejects_it(
    max_tool_rounds: object,
) -> None:
    completion = ScriptedCompletion((_response(text="unused"),))
    executor = RecordingExecutor()

    with pytest.raises(ValueError, match="non-negative integer"):
        await run_tool_loop(
            _request(),
            complete=completion,
            executor=executor,
            max_tool_rounds=max_tool_rounds,  # type: ignore[arg-type]
        )

    assert completion.requests == []
