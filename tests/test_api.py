import asyncio
import logging

import httpx
import pytest
from fastapi import FastAPI

from llmkit_lite.api import (
    RequestIdFilter,
    add_request_id_middleware,
    configure_logging,
    get_request_id,
    http_client_lifespan,
    llm_exception_handler,
)
from llmkit_lite.llm import LlmGatewayError


async def test_request_id_middleware_uses_header_and_isolates_concurrent_requests():
    app = FastAPI()
    add_request_id_middleware(app)

    @app.get("/id")
    async def read_id():
        await asyncio.sleep(0.01)
        return {"request_id": get_request_id()}

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first, second = await asyncio.gather(
            client.get("/id", headers={"X-Request-ID": "req-a"}),
            client.get("/id", headers={"X-Request-ID": "req-b"}),
        )

    assert first.json() == {"request_id": "req-a"}
    assert second.json() == {"request_id": "req-b"}
    assert first.headers["X-Request-ID"] == "req-a"
    assert second.headers["X-Request-ID"] == "req-b"
    assert get_request_id() == "-"


def test_configure_logging_adds_request_id_filter() -> None:
    root = logging.getLogger()
    handler = logging.StreamHandler()
    root.addHandler(handler)
    try:
        configure_logging(level="INFO")
        assert any(isinstance(f, RequestIdFilter) for f in handler.filters)
    finally:
        root.removeHandler(handler)


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

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # ASGITransport does not drive lifespan itself; call the lifespan directly
        # for a focused unit test of the helper.
        lifespan = http_client_lifespan(client_factory=DummyClient)
        async with lifespan(app):
            assert app.state.http_client.closed is False
        assert app.state.http_client.closed is True


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
