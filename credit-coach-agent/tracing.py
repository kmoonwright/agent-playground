"""Arize AX OpenInference / OpenTelemetry setup.

Must run (call ensure_tracing()) before the OpenAI SDK is imported anywhere,
so auto-instrumentation can patch it.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

from config import arize_api_key, arize_project_name, arize_space_id, openai_api_key

CHAIN = "CHAIN"
TOOL = "TOOL"
EVALUATOR = "EVALUATOR"
GUARDRAIL = "GUARDRAIL"

_provider = None
_instrumented = False


def ensure_tracing(*, log_to_console: bool = False):
    """Register the Arize exporter and OpenAI instrumentor. Idempotent.

    Warns (does not crash) if Arize credentials are missing so the agent still
    runs locally without them.
    """
    global _provider, _instrumented

    if _provider is not None or _instrumented:
        return _provider

    space_id = arize_space_id()
    api_key = arize_api_key()
    project_name = arize_project_name()

    if not space_id or not api_key:
        print(
            "Warning: ARIZE_SPACE_ID / ARIZE_API_KEY not set. "
            "Traces will not export to Arize AX. Agent will still run locally."
        )
        _provider = _local_tracer_provider()
        _instrumented = True
        if openai_api_key():
            _instrument_openai(_provider)
        return _provider

    from arize.otel import register

    # batch=False (SimpleSpanProcessor) so traces land in AX live during a demo
    # instead of sitting in a batch until process exit.
    _provider = register(
        space_id=space_id,
        api_key=api_key,
        project_name=project_name,
        batch=False,
        log_to_console=log_to_console,
    )
    _instrument_openai(_provider)
    _instrumented = True
    print(f"Arize AX tracing initialized -> project '{project_name}'.")
    return _provider


def _local_tracer_provider():
    """In-process provider so nested spans still get real ids without Arize creds."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    existing = trace.get_tracer_provider()
    if isinstance(existing, TracerProvider):
        return existing
    provider = TracerProvider()
    trace.set_tracer_provider(provider)
    return provider


def _instrument_openai(tracer_provider) -> None:
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
    except ImportError:
        print("Warning: openinference-instrumentation-openai not installed.")
        return
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


def flush_tracing() -> None:
    """Flush + shut down the tracer provider so short-lived CLI runs export."""
    global _provider, _instrumented
    if _provider is None:
        return
    try:
        _provider.force_flush()
    finally:
        try:
            _provider.shutdown()
        except Exception:
            pass
        _provider = None
        _instrumented = False


def get_tracer():
    from opentelemetry import trace

    return trace.get_tracer("credit-coach-agent")


def _context_attributes() -> dict[str, Any]:
    """Attributes set by an enclosing using_session()/using_attributes() block."""
    try:
        from openinference.instrumentation import get_attributes_from_context

        return dict(get_attributes_from_context())
    except Exception:
        return {}


def using_session(session_id: str, extra: dict[str, Any] | None = None):
    """Group every span opened inside this block -- manual and auto-instrumented,
    across as many separate traces as are opened -- under one Arize AX session.
    """
    from openinference.instrumentation import using_attributes

    return using_attributes(
        session_id=session_id,
        metadata=extra or {},
        tags=["credit-coach-agent"],
    )


@contextmanager
def traced_span(
    name: str,
    kind: str,
    *,
    input_value: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any]:
    from opentelemetry.trace import Status, StatusCode

    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        for key, value in _context_attributes().items():
            span.set_attribute(key, value)
        span.set_attribute("openinference.span.kind", kind)
        if input_value is not None:
            span.set_attribute("input.value", input_value)
        if attributes:
            for key, value in attributes.items():
                if value is None:
                    continue
                if isinstance(value, (dict, list)):
                    span.set_attribute(key, json.dumps(value, default=str))
                else:
                    span.set_attribute(key, value)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def set_output(span, value: Any) -> None:
    if isinstance(value, (dict, list)):
        span.set_attribute("output.value", json.dumps(value, default=str))
        span.set_attribute("output.mime_type", "application/json")
    else:
        span.set_attribute("output.value", str(value))


def current_ids() -> tuple[str, str]:
    """Return (trace_id, span_id) hex strings for the current span."""
    from opentelemetry import trace
    from opentelemetry.trace import format_span_id, format_trace_id

    ctx = trace.get_current_span().get_span_context()
    return format_trace_id(ctx.trace_id), format_span_id(ctx.span_id)
