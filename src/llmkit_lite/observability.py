"""Execution context and OpenTelemetry setup for LLM application services."""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider


_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "llmkit_request_id", default="-"
)
_thread_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "llmkit_thread_id", default=None
)
_configuration_lock = threading.Lock()
_configured_settings: TracingSettings | None = None
_configured_provider: TracerProvider | None = None


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Correlation identifiers associated with the current execution."""

    request_id: str
    thread_id: str | None = None
    trace_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "request_id",
            _normalize_required_identifier(self.request_id, "request_id"),
        )
        for field_name in ("thread_id", "trace_id"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _normalize_required_identifier(value, field_name),
                )


@dataclass(frozen=True, slots=True)
class TracingSettings:
    """Settings used to configure OpenTelemetry trace export."""

    enabled: bool = False
    service_name: str = "llmkit-lite"
    service_version: str | None = None
    environment: str | None = None
    otlp_endpoint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "service_name",
            _normalize_required_identifier(self.service_name, "service_name"),
        )
        for field_name in ("service_version", "environment", "otlp_endpoint"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(
                    self,
                    field_name,
                    _normalize_required_identifier(value, field_name),
                )


def _normalize_required_identifier(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


def get_request_id() -> str:
    """Return the request identifier bound to the current execution."""

    return _request_id.get()


def get_thread_id() -> str | None:
    """Return the persistent workflow identifier bound to this execution."""

    return _thread_id.get()


def get_trace_id() -> str | None:
    """Return the active OpenTelemetry trace ID, if one is available."""

    try:
        from opentelemetry import trace
    except ImportError:
        return None

    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return None
    return f"{span_context.trace_id:032x}"


def get_execution_context() -> ExecutionContext:
    """Return a snapshot of the current execution identifiers."""

    return ExecutionContext(
        request_id=get_request_id(),
        thread_id=get_thread_id(),
        trace_id=get_trace_id(),
    )


@contextmanager
def execution_context(
    request_id: str,
    thread_id: str | None = None,
) -> Iterator[ExecutionContext]:
    """Bind request and workflow identifiers for the duration of a context."""

    normalized_request_id = _normalize_required_identifier(request_id, "request_id")
    normalized_thread_id = (
        _normalize_required_identifier(thread_id, "thread_id")
        if thread_id is not None
        else None
    )
    request_token = _request_id.set(normalized_request_id)
    thread_token = _thread_id.set(normalized_thread_id)
    try:
        yield get_execution_context()
    finally:
        _thread_id.reset(thread_token)
        _request_id.reset(request_token)


def configure_tracing(settings: TracingSettings) -> TracerProvider | None:
    """Configure global OTLP trace export when observability is enabled.

    Repeating configuration with the same settings returns the existing provider.
    A conflicting second configuration is rejected because OpenTelemetry supports
    only one global tracer provider per process.

    Raises:
        ImportError: If the observability extra is not installed.
        RuntimeError: If tracing was already configured with different settings,
            or another library has already installed a global tracer provider.
    """

    if not settings.enabled:
        return None

    global _configured_provider, _configured_settings

    with _configuration_lock:
        if _configured_settings is not None:
            if settings != _configured_settings:
                raise RuntimeError(
                    "OpenTelemetry tracing is already configured with "
                    "different settings"
                )
            return _configured_provider

        try:
            from opentelemetry import trace
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError as exc:
            raise ImportError(
                "Install llmkit-lite[observability] to enable OpenTelemetry tracing"
            ) from exc

        resource_attributes = {"service.name": settings.service_name}
        if settings.service_version is not None:
            resource_attributes["service.version"] = settings.service_version
        if settings.environment is not None:
            resource_attributes["deployment.environment.name"] = settings.environment

        exporter_options = {}
        if settings.otlp_endpoint is not None:
            exporter_options["endpoint"] = settings.otlp_endpoint
        exporter = OTLPSpanExporter(**exporter_options)
        provider = TracerProvider(resource=Resource.create(resource_attributes))
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        if trace.get_tracer_provider() is not provider:
            provider.shutdown()
            raise RuntimeError(
                "OpenTelemetry already has a global tracer provider; configure "
                "llmkit-lite before other tracing integrations"
            )

        _configured_settings = settings
        _configured_provider = provider
        return provider
