import json

import httpx
from fastapi import FastAPI
from pydantic import BaseModel, Field

from llmkit_lite.api import add_request_id_middleware, get_request_id
from llmkit_lite.llm import LlmEndpointConfig
from llmkit_lite.structured import structured_json_call


class LabelOut(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


def _chat_content(text: str) -> str:
    return json.dumps(
        {"model": "test-model", "choices": [{"message": {"content": text}}]}
    )


async def test_fastapi_tracing_gateway_and_structured_output_compose(
    in_memory_tracing,
) -> None:
    def llm_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["messages"][0]["content"] == "Classify: hello"
        return httpx.Response(
            200,
            content=_chat_content('{"label":"greeting","confidence":0.97}'),
        )

    app = FastAPI()
    add_request_id_middleware(app)
    app.state.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(llm_handler)
    )
    cfg = LlmEndpointConfig(
        provider="local",
        base_url="https://gateway.example.com",
        model_name="test-model",
    )

    @app.get("/classify")
    async def classify():
        result = await structured_json_call(
            [{"role": "user", "content": "Classify: hello"}],
            response_model=LabelOut,
            cfg=cfg,
            http_client=app.state.http_client,
            max_tokens=100,
            timeout_seconds=5,
        )
        assert result.value is not None
        return {
            "request_id": get_request_id(),
            "label": result.value.label,
            "confidence": result.value.confidence,
        }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/classify", headers={"X-Request-ID": "trace-1"})
    await app.state.http_client.aclose()

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "trace-1"
    assert response.json() == {
        "request_id": "trace-1",
        "label": "greeting",
        "confidence": 0.97,
    }
    spans = in_memory_tracing.get_finished_spans()
    request_span = next(span for span in spans if span.name == "GET /classify")
    llm_span = next(span for span in spans if span.name == "llm.chat_completion")
    assert llm_span.context.trace_id == request_span.context.trace_id
    assert llm_span.parent.span_id == request_span.context.span_id
