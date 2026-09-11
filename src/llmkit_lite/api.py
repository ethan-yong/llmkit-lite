"""FastAPI-friendly helpers for LLM services."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx

from llmkit_lite.llm import LlmGatewayError
from llmkit_lite.observability import (
    execution_context,
    get_thread_id,
    get_trace_id,
    set_span_error,
    trace_span,
)
from llmkit_lite.observability import get_request_id as _get_request_id
from llmkit_lite.runtime import ApplicationRuntime

DEFAULT_LOG_FORMAT = (
    "%(asctime)s %(levelname)s "
    "[request_id=%(request_id)s thread_id=%(thread_id)s trace_id=%(trace_id)s] "
    "%(name)s: %(message)s"
)


class RequestIdFilter(logging.Filter):
    """Attach active execution identifiers to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        record.thread_id = get_thread_id() or "-"
        record.trace_id = get_trace_id() or "-"
        return True


def get_request_id() -> str:
    """Return the request identifier bound to the current execution."""

    return _get_request_id()


def configure_logging(
    *,
    level: str | int | None = None,
    log_format: str = DEFAULT_LOG_FORMAT,
) -> None:
    """Configure root logging with request IDs.

    Calling this more than once is harmless: the filter is attached only to
    handlers that do not already have a `RequestIdFilter`.
    """

    resolved_level = level or os.environ.get("LLMKIT_LOG_LEVEL", "INFO")
    logging.basicConfig(level=resolved_level, format=log_format)
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, RequestIdFilter) for f in handler.filters):
            handler.addFilter(RequestIdFilter())


def add_request_id_middleware(
    app: Any,
    *,
    header_name: str = "X-Request-ID",
    thread_header_name: str = "X-Thread-ID",
    response_header: bool = True,
    id_factory: Callable[[], str] | None = None,
) -> None:
    """Install middleware that scopes request context and creates a server span."""

    try:
        from fastapi import Request
    except ImportError as exc:  # pragma: no cover - exercised by packaging users
        raise ImportError("Install llmkit-lite[api] to use FastAPI helpers") from exc

    make_id = id_factory or (lambda: uuid.uuid4().hex[:12])

    @app.middleware("http")
    async def _assign_request_id(request: Request, call_next: Callable[..., Awaitable]):
        incoming = request.headers.get(header_name)
        request_id = incoming.strip() if incoming else ""
        if not request_id:
            request_id = make_id()
        incoming_thread = request.headers.get(thread_header_name)
        thread_id = incoming_thread.strip() if incoming_thread else None
        if not thread_id:
            thread_id = None

        attributes: dict[str, str | int] = {
            "http.request.method": request.method,
            "url.scheme": request.url.scheme,
            "llmkit.request_id": request_id,
        }
        if request.url.hostname:
            attributes["server.address"] = request.url.hostname
        if request.url.port:
            attributes["server.port"] = request.url.port
        if thread_id is not None:
            attributes["llmkit.thread_id"] = thread_id

        with execution_context(request_id=request_id, thread_id=thread_id):
            with trace_span(
                f"HTTP {request.method}",
                kind="server",
                attributes=attributes,
                carrier=request.headers,
            ) as span:
                response = None
                try:
                    response = await call_next(request)
                finally:
                    route = request.scope.get("route")
                    route_path = getattr(route, "path", None)
                    if span is not None:
                        if isinstance(route_path, str):
                            span.update_name(f"{request.method} {route_path}")
                            span.set_attribute("http.route", route_path)
                        if response is not None:
                            span.set_attribute(
                                "http.response.status_code", response.status_code
                            )
                            if response.status_code >= 500:
                                set_span_error(span, f"HTTP {response.status_code}")
                if response_header:
                    response.headers[header_name] = get_request_id()
                return response


def http_client_lifespan(
    *,
    state_attr: str = "http_client",
    client_factory: Callable[[], httpx.AsyncClient] | None = None,
):
    """Return a FastAPI lifespan context manager with one shared HTTP client."""

    factory = client_factory or httpx.AsyncClient

    @asynccontextmanager
    async def _lifespan(app: Any):
        client = factory()
        setattr(app.state, state_attr, client)
        try:
            yield
        finally:
            await client.aclose()

    return _lifespan


def runtime_lifespan(
    runtime: ApplicationRuntime,
    *,
    state_attr: str = "runtime",
):
    """Expose an application runtime for one FastAPI lifespan."""

    if not isinstance(runtime, ApplicationRuntime):
        raise TypeError("runtime must be an ApplicationRuntime")
    if not isinstance(state_attr, str):
        raise TypeError("runtime state attribute must be a string")
    normalized_state_attr = state_attr.strip()
    if not normalized_state_attr:
        raise ValueError("runtime state attribute must not be empty")

    @asynccontextmanager
    async def _lifespan(app: Any):
        missing = object()
        previous = getattr(app.state, normalized_state_attr, missing)
        await runtime.start()
        attached = False
        try:
            setattr(app.state, normalized_state_attr, runtime)
            attached = True
            yield
        finally:
            try:
                await runtime.close()
            finally:
                if attached:
                    if previous is missing:
                        try:
                            delattr(app.state, normalized_state_attr)
                        except AttributeError:
                            pass
                    else:
                        setattr(app.state, normalized_state_attr, previous)

    return _lifespan


def llm_exception_handler(
    status_by_code: Mapping[str, int] | None = None,
    *,
    default_status: int = 502,
    expose_detail: bool = False,
):
    """Create a FastAPI exception handler for `LlmGatewayError`."""

    try:
        from fastapi import Request
        from fastapi.responses import JSONResponse
    except ImportError as exc:  # pragma: no cover - exercised by packaging users
        raise ImportError("Install llmkit-lite[api] to use FastAPI helpers") from exc

    statuses = dict(status_by_code or {})

    async def _handler(_: Request, exc: LlmGatewayError) -> JSONResponse:
        status_code = statuses.get(exc.code, default_status)
        body: dict[str, Any] = {"error": exc.code}
        if expose_detail:
            body["detail"] = exc.detail
        return JSONResponse(status_code=status_code, content=body)

    return _handler
