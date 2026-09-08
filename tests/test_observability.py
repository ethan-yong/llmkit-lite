import asyncio
import builtins

import pytest
from opentelemetry import trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, use_span

from llmkit_lite.observability import (
    ExecutionContext,
    TracingSettings,
    configure_tracing,
    execution_context,
    get_execution_context,
    get_request_id,
    get_thread_id,
    get_trace_id,
)


def test_default_execution_context_has_compatibility_request_id() -> None:
    assert get_execution_context() == ExecutionContext(
        request_id="-",
        thread_id=None,
        trace_id=None,
    )


def test_execution_context_strips_identifiers_and_restores_nested_values() -> None:
    with execution_context(request_id=" outer ", thread_id=" workflow "):
        assert get_request_id() == "outer"
        assert get_thread_id() == "workflow"

        with execution_context(request_id="inner"):
            assert get_execution_context() == ExecutionContext(
                request_id="inner",
                thread_id=None,
                trace_id=None,
            )

        assert get_request_id() == "outer"
        assert get_thread_id() == "workflow"

    assert get_request_id() == "-"
    assert get_thread_id() is None


@pytest.mark.parametrize(
    "request_id,thread_id",
    [("", None), ("   ", None), ("request", ""), ("request", "   ")],
)
def test_execution_context_rejects_empty_identifiers(
    request_id: str,
    thread_id: str | None,
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        with execution_context(request_id=request_id, thread_id=thread_id):
            pass


def test_execution_context_model_normalizes_identifiers() -> None:
    assert ExecutionContext(" request ", " thread ", " trace ") == ExecutionContext(
        "request",
        "thread",
        "trace",
    )
    with pytest.raises(ValueError, match="request_id must not be empty"):
        ExecutionContext(" ")


async def test_execution_context_isolates_concurrent_tasks() -> None:
    ready = asyncio.Event()
    observed: list[ExecutionContext] = []

    async def capture(request_id: str, thread_id: str) -> None:
        with execution_context(request_id=request_id, thread_id=thread_id):
            ready.set()
            await asyncio.sleep(0.01)
            observed.append(get_execution_context())

    await asyncio.gather(capture("req-a", "thread-a"), capture("req-b", "thread-b"))

    assert set((item.request_id, item.thread_id) for item in observed) == {
        ("req-a", "thread-a"),
        ("req-b", "thread-b"),
    }
    assert get_execution_context() == ExecutionContext("-", None, None)


def test_get_trace_id_formats_active_valid_span_context() -> None:
    span_context = SpanContext(
        trace_id=0x1234,
        span_id=0x5678,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    span = NonRecordingSpan(span_context)

    assert get_trace_id() is None
    with use_span(span, end_on_exit=False):
        assert get_trace_id() == "00000000000000000000000000001234"
        assert get_execution_context().trace_id == get_trace_id()
    assert get_trace_id() is None


def test_disabled_tracing_does_not_import_opentelemetry_sdk(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("opentelemetry.sdk"):
            raise AssertionError("disabled tracing imported the OpenTelemetry SDK")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert configure_tracing(TracingSettings()) is None


def test_enabled_tracing_is_idempotent_and_rejects_conflicts(monkeypatch) -> None:
    exported: list[object] = []

    def record_export(self, spans):
        exported.extend(spans)
        from opentelemetry.sdk.trace.export import SpanExportResult

        return SpanExportResult.SUCCESS

    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter.export",
        record_export,
    )
    settings = TracingSettings(
        enabled=True,
        service_name=" support-agent ",
        service_version=" 1.0.0 ",
        environment=" test ",
        otlp_endpoint=" http://collector.test/v1/traces ",
    )

    provider = configure_tracing(settings)

    assert provider is not None
    assert configure_tracing(settings) is provider
    assert settings.service_name == "support-agent"
    assert settings.service_version == "1.0.0"
    assert settings.environment == "test"
    assert settings.otlp_endpoint == "http://collector.test/v1/traces"
    with pytest.raises(RuntimeError, match="different settings"):
        configure_tracing(TracingSettings(enabled=True, service_name="other-service"))

    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("operation"):
        assert get_trace_id() is not None
    assert provider.force_flush()
    assert len(exported) == 1


def test_enabled_tracing_requires_observability_extra(monkeypatch) -> None:
    import llmkit_lite.observability as observability

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("opentelemetry.exporter"):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(observability, "_configured_settings", None)
    monkeypatch.setattr(observability, "_configured_provider", None)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(
        ImportError,
        match=r"Install llmkit-lite\[observability\]",
    ):
        configure_tracing(TracingSettings(enabled=True))
