import asyncio
import logging
from contextlib import asynccontextmanager

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from opentelemetry.trace import StatusCode

from llmkit_lite.api import (
    RequestIdFilter,
    add_request_id_middleware,
    configure_logging,
    get_request_id,
    http_client_lifespan,
    llm_exception_handler,
    runtime_lifespan,
)
from llmkit_lite.llm import LlmGatewayError
from llmkit_lite.observability import (
    execution_context,
    get_thread_id,
    get_trace_id,
    trace_span,
)
from llmkit_lite.observability import (
    get_request_id as get_observability_request_id,
)
from llmkit_lite.runtime import ApplicationRuntime, RuntimeDependency


async def test_request_id_middleware_uses_header_and_isolates_concurrent_requests():
    app = FastAPI()
    add_request_id_middleware(app)

    @app.get("/id")
    async def read_id():
        await asyncio.sleep(0.01)
        return {
            "request_id": get_request_id(),
            "observability_request_id": get_observability_request_id(),
        }

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first, second = await asyncio.gather(
            client.get("/id", headers={"X-Request-ID": "req-a"}),
            client.get("/id", headers={"X-Request-ID": "req-b"}),
        )

    assert first.json() == {
        "request_id": "req-a",
        "observability_request_id": "req-a",
    }
    assert second.json() == {
        "request_id": "req-b",
        "observability_request_id": "req-b",
    }
    assert first.headers["X-Request-ID"] == "req-a"
    assert second.headers["X-Request-ID"] == "req-b"
    assert get_request_id() == "-"
    assert get_observability_request_id() == "-"


def test_configure_logging_adds_request_id_filter() -> None:
    root = logging.getLogger()
    handler = logging.StreamHandler()
    root.addHandler(handler)
    try:
        configure_logging(level="INFO")
        assert any(isinstance(f, RequestIdFilter) for f in handler.filters)
    finally:
        root.removeHandler(handler)


def test_request_id_filter_adds_execution_and_trace_context(
    in_memory_tracing,
) -> None:
    record = logging.LogRecord("test", logging.INFO, "", 0, "message", (), None)

    with execution_context("request-1", "thread-1"):
        with trace_span("operation"):
            RequestIdFilter().filter(record)

    assert record.request_id == "request-1"
    assert record.thread_id == "thread-1"
    assert len(record.trace_id) == 32


async def test_request_middleware_creates_server_span_from_inbound_context(
    in_memory_tracing,
) -> None:
    app = FastAPI()
    add_request_id_middleware(app)

    @app.get("/items/{item_id}")
    async def read_item(item_id: str):
        return {
            "item_id": item_id,
            "thread_id": get_thread_id(),
            "trace_id": get_trace_id(),
        }

    parent_trace_id = "00000000000000000000000000000011"
    parent_span_id = "0000000000000022"
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/items/secret-item-id",
            headers={
                "X-Request-ID": "request-1",
                "X-Thread-ID": "thread-1",
                "traceparent": f"00-{parent_trace_id}-{parent_span_id}-01",
            },
        )

    spans = in_memory_tracing.get_finished_spans()
    server_span = next(span for span in spans if span.name == "GET /items/{item_id}")
    assert response.json() == {
        "item_id": "secret-item-id",
        "thread_id": "thread-1",
        "trace_id": parent_trace_id,
    }
    assert server_span.kind.name == "SERVER"
    assert server_span.context.trace_id == int(parent_trace_id, 16)
    assert server_span.parent.span_id == int(parent_span_id, 16)
    assert server_span.attributes["http.request.method"] == "GET"
    assert server_span.attributes["http.route"] == "/items/{item_id}"
    assert server_span.attributes["http.response.status_code"] == 200
    assert "secret-item-id" not in str(server_span.attributes)


async def test_request_middleware_marks_server_errors(in_memory_tracing) -> None:
    app = FastAPI()
    add_request_id_middleware(app)

    @app.get("/unavailable")
    async def unavailable():
        return PlainTextResponse("unavailable", status_code=503)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/unavailable")

    span = next(
        span
        for span in in_memory_tracing.get_finished_spans()
        if span.name == "GET /unavailable"
    )
    assert response.status_code == 503
    assert span.status.status_code is StatusCode.ERROR


async def test_http_client_lifespan_sets_and_closes_client() -> None:
    class DummyClient:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    app = FastAPI(lifespan=http_client_lifespan(client_factory=DummyClient))

    @app.get("/client")
    async def client_status():
        return {"closed": app.state.http_client.closed}

    # ASGITransport does not drive lifespan itself; call the lifespan directly
    # for a focused unit test of the helper.
    lifespan = http_client_lifespan(client_factory=DummyClient)
    async with lifespan(app):
        assert app.state.http_client.closed is False
    assert app.state.http_client.closed is True


async def test_runtime_lifespan_exposes_runtime_and_closes_resources() -> None:
    events: list[str] = []

    @asynccontextmanager
    async def client_resource():
        events.append("start")
        try:
            yield "client"
        finally:
            events.append("stop")

    runtime = ApplicationRuntime(
        (
            RuntimeDependency(
                "client",
                lambda dependencies: client_resource(),
            ),
        )
    )
    app = FastAPI(lifespan=runtime_lifespan(runtime))
    lifespan = runtime_lifespan(runtime)

    async with lifespan(app):
        assert app.state.runtime is runtime
        assert app.state.runtime.get("client") == "client"

    assert events == ["start", "stop"]
    assert runtime.started is False
    with pytest.raises(AttributeError):
        _ = app.state.runtime


async def test_runtime_lifespan_restores_existing_state_value() -> None:
    runtime = ApplicationRuntime(())
    app = FastAPI()
    app.state.services = "previous"

    async with runtime_lifespan(runtime, state_attr=" services ")(app):
        assert app.state.services is runtime

    assert app.state.services == "previous"


def test_runtime_lifespan_validates_inputs() -> None:
    with pytest.raises(TypeError, match="ApplicationRuntime"):
        runtime_lifespan(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be a string"):
        runtime_lifespan(ApplicationRuntime(()), state_attr=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must not be empty"):
        runtime_lifespan(ApplicationRuntime(()), state_attr=" ")


async def test_llm_exception_handler_maps_errors() -> None:
    app = FastAPI()
    app.add_exception_handler(
        LlmGatewayError,
        llm_exception_handler({"server_misconfigured": 500}, expose_detail=True),
    )

    @app.get("/boom")
    async def boom():
        raise LlmGatewayError("server_misconfigured", "missing model")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/boom")

    assert response.status_code == 500
    assert response.json() == {
        "error": "server_misconfigured",
        "detail": "missing model",
    }


async def test_llm_exception_handler_defaults_to_502() -> None:
    handler = llm_exception_handler()
    response = await handler(None, LlmGatewayError("llm_timeout", "slow"))  # type: ignore[arg-type]
    assert response.status_code == 502
    assert response.body == b'{"error":"llm_timeout"}'


def test_pytest_is_available() -> None:
    assert pytest
