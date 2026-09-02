"""FastAPI-friendly helpers for LLM services."""

from __future__ import annotations

import contextvars
import logging
import os
import uuid
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any

import httpx

from llmkit_lite.llm import LlmGatewayError

_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "llmkit_request_id", default="-"
)

DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"


class RequestIdFilter(logging.Filter):
    """Attach the active request ID to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


def get_request_id() -> str:
    return _request_id.get()


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
    response_header: bool = True,
    id_factory: Callable[[], str] | None = None,
) -> None:
    """Install middleware that scopes a request ID via `contextvars`."""

    try:
        from fastapi import Request
    except ImportError as exc:  # pragma: no cover - exercised by packaging users
        raise ImportError("Install llmkit-lite[api] to use FastAPI helpers") from exc

    make_id = id_factory or (lambda: uuid.uuid4().hex[:12])

    @app.middleware("http")
    async def _assign_request_id(request: Request, call_next: Callable[..., Awaitable]):
        incoming = request.headers.get(header_name)
        token = _request_id.set(incoming.strip() if incoming else make_id())
        try:
            response = await call_next(request)
            if response_header:
                response.headers[header_name] = get_request_id()
            return response
        finally:
            _request_id.reset(token)


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
