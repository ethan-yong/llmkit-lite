import asyncio
import builtins

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
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
    inject_trace_context,
    set_span_error,
    trace_span,
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


def test_trace_span_extracts_and_injects_w3c_context(in_memory_tracing) -> None:
    parent_trace_id = "00000000000000000000000000000011"
    parent_span_id = "0000000000000022"
    outbound: dict[str, str] = {}

    with trace_span(
        "server operation",
        kind="server",
        carrier={
            "traceparent": f"00-{parent_trace_id}-{parent_span_id}-01",
        },
    ) as span:
        assert span is not None
        inject_trace_context(outbound)

    exported = in_memory_tracing.get_finished_spans()[0]
    assert exported.context.trace_id == int(parent_trace_id, 16)
    assert exported.parent.span_id == int(parent_span_id, 16)
    assert outbound["traceparent"].startswith(f"00-{parent_trace_id}-")


def test_set_span_error_marks_recording_span(in_memory_tracing) -> None:
    with trace_span("failed operation") as span:
        set_span_error(span, "failed")

    exported = in_memory_tracing.get_finished_spans()[0]
    assert exported.status.status_code.name == "ERROR"
    assert exported.status.description == "failed"


def test_trace_span_records_only_safe_exception_type(in_memory_tracing) -> None:
    with pytest.raises(RuntimeError, match="private exception detail"):
        with trace_span("failed operation"):
            raise RuntimeError("private exception detail")

    exported = in_memory_tracing.get_finished_spans()[0]
    assert exported.attributes["exception.type"] == "builtins.RuntimeError"
    assert exported.status.status_code.name == "ERROR"
    assert exported.status.description == "builtins.RuntimeError"
    assert exported.events == ()
    assert "private exception detail" not in str(exported.attributes)


def test_disabled_tracing_does_not_import_opentelemetry_sdk(monkeypatch) -> None:
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("opentelemetry.sdk"):
            raise AssertionError("disabled tracing imported the OpenTelemetry SDK")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    assert configure_tracing(TracingSettings()) is None


def test_enabled_tracing_is_idempotent_and_rejects_conflicts(monkeypatch) -> None:
    exporters: list[InMemorySpanExporter] = []

    class InMemoryOTLPSpanExporter(InMemorySpanExporter):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.options = kwargs
            exporters.append(self)

    import opentelemetry.exporter.otlp.proto.http.trace_exporter as trace_exporter

    monkeypatch.setattr(
        trace_exporter,
        "OTLPSpanExporter",
        InMemoryOTLPSpanExporter,
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
    assert exporters[0].options == {"endpoint": "http://collector.test/v1/traces"}
    assert len(exporters[0].get_finished_spans()) == 1


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
