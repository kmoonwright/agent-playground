"""Arize AX / OpenInference tracing setup. Must be imported, and
ensure_tracing() called, before the openai SDK is imported anywhere in the
process, so OpenAIInstrumentor can patch it.

All diagnostic prints go to stderr, not stdout: when this module is loaded by
mcp_server.py, stdout is the MCP stdio wire protocol itself, and anything
else written there corrupts it.
"""

import json
import sys
from contextlib import contextmanager

import config

TRACER_NAME = "anydocs-agent"
MAX_ATTR_CHARS = 4000

# OpenInference span kinds used in this project.
CHAIN = "CHAIN"
AGENT = "AGENT"
TOOL = "TOOL"
RETRIEVER = "RETRIEVER"
EMBEDDING = "EMBEDDING"
RERANKER = "RERANKER"
GUARDRAIL = "GUARDRAIL"
PROMPT = "PROMPT"
LLM = "LLM"
EVALUATOR = "EVALUATOR"

_provider = None


def _warn(message: str) -> None:
    print(message, file=sys.stderr)


def ensure_tracing():
    """Register the Arize exporter and the OpenAI + MCP instrumentors.

    Idempotent. Warns (to stderr) rather than crashing if Arize credentials
    are missing, so a local run still produces a real span tree, just
    without export.
    """
    global _provider

    if _provider is not None:
        return _provider

    space_id = config.arize_space_id()
    api_key = config.arize_api_key()
    project_name = config.arize_project_name()

    if not space_id or not api_key:
        _warn(
            "Warning: ARIZE_SPACE_ID / ARIZE_API_KEY not set. "
            "Traces will not export to Arize AX. The agent will still run."
        )
        _provider = _local_tracer_provider()
    else:
        from arize.otel import register

        _provider = register(
            space_id=space_id,
            api_key=api_key,
            project_name=project_name,
            log_to_console=False,
            # register() defaults verbose=True and prints a banner straight
            # to stdout regardless of log_to_console — fatal in mcp_server.py,
            # where stdout is the JSON-RPC wire. Always off, host process too.
            verbose=False,
        )
        _warn(f"Arize AX tracing initialized -> project '{project_name}'.")

    _instrument(_provider)
    return _provider


def _local_tracer_provider():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    existing = trace.get_tracer_provider()
    if isinstance(existing, TracerProvider):
        return existing
    provider = TracerProvider()
    trace.set_tracer_provider(provider)
    return provider


def _instrument(tracer_provider) -> None:
    from openinference.instrumentation.mcp import MCPInstrumentor

    MCPInstrumentor().instrument(tracer_provider=tracer_provider)

    if config.openai_api_key():
        from openinference.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)


def flush_tracing() -> None:
    """Flush + shut down the tracer provider so a short-lived process exports."""
    global _provider
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


def get_tracer():
    from opentelemetry import trace

    return trace.get_tracer(TRACER_NAME)


def truncate(text: str, limit: int = MAX_ATTR_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


def set_attributes(span, attributes: dict) -> None:
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            span.set_attribute(key, json.dumps(value))
        elif isinstance(value, str):
            span.set_attribute(key, truncate(value))
        else:
            span.set_attribute(key, value)


def set_output(span, value) -> None:
    if isinstance(value, (dict, list)):
        span.set_attribute("output.value", truncate(json.dumps(value)))
        span.set_attribute("output.mime_type", "application/json")
    else:
        span.set_attribute("output.value", truncate(str(value)))


def using_session_context(session_id: str, extra: dict | None = None):
    """Group every span under one AX session, including auto-instrumented LLM
    spans. Pass the same session_id across every turn of one conversation —
    a fresh id per call would put every turn in its own ungrouped session.

    This only affects spans created in *this* process. openinference's
    using_attributes() stores session_id/metadata/tags as a plain in-process
    OTel Context value, not OTel Baggage — it is not something
    MCPInstrumentor (or any propagator) carries across the MCP wire. Spans
    created inside mcp_server.py need session_id passed to them explicitly
    (see mcp_server.py's search_docs)."""
    from openinference.instrumentation import using_attributes

    metadata = {"session": session_id}
    if extra:
        metadata.update(extra)
    return using_attributes(session_id=session_id, metadata=metadata, tags=["anydocs"])


def _context_attributes() -> dict:
    """Pull whatever using_session_context() set (session id, metadata,
    tags) off the current OTel Context, so manual spans carry it too — the
    auto-instrumented LLM spans get this for free from the instrumentor;
    manual spans need to ask for it explicitly."""
    try:
        from openinference.instrumentation import get_attributes_from_context

        return dict(get_attributes_from_context())
    except Exception:
        return {}


@contextmanager
def traced_span(name: str, kind: str, *, input_value=None, attributes: dict | None = None):
    """Manual OpenInference span. Records exceptions and re-raises."""
    from opentelemetry.trace import Status, StatusCode

    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        for key, value in _context_attributes().items():
            span.set_attribute(key, value)
        span.set_attribute("openinference.span.kind", kind)
        if input_value is not None:
            span.set_attribute("input.value", truncate(input_value))
        if attributes:
            set_attributes(span, attributes)
        try:
            yield span
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
