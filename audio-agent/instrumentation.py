"""Arize AX / OpenInference tracing setup. Must be imported, and
ensure_tracing() called, before the openai/langchain SDKs are imported
anywhere in the process, so their instrumentors can patch them.
"""

import json
import sys
from contextlib import contextmanager

import config

TRACER_NAME = "audio-agent"
MAX_ATTR_CHARS = 4000

# OpenInference span kinds used in this project.
CHAIN = "CHAIN"
AGENT = "AGENT"
TOOL = "TOOL"
LLM = "LLM"

_provider = None


def _warn(message: str) -> None:
    print(message, file=sys.stderr)


def ensure_tracing():
    """Register the Arize exporter and the LangChain + OpenAI instrumentors.

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
    # LangGraph node execution and every langchain chat-model call route
    # through LangChain's callback system, so this instrumentor is
    # unconditional -- it's what turns the graph's shape itself into spans,
    # regardless of which provider key (if any) is set.
    from openinference.instrumentation.langchain import LangChainInstrumentor

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

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


def set_audio_attributes(span, *, data: bytes, mime_type: str, transcript: str | None = None) -> None:
    """Attach raw audio to a span using OpenInference's AudioAttributes --
    documented but not auto-populated by any instrumentor, so this is the
    only way the audio itself (not just a path/duration) ends up on a span.
    Skips the actual attachment above config.max_audio_attach_bytes(), since
    an unbounded base64 data URI risks span bloat and OTLP payload limits."""
    import base64

    from openinference.semconv.trace import AudioAttributes

    span.set_attribute(AudioAttributes.AUDIO_MIME_TYPE, mime_type)
    if transcript:
        span.set_attribute(AudioAttributes.AUDIO_TRANSCRIPT, truncate(transcript))

    limit = config.max_audio_attach_bytes()
    if len(data) > limit:
        span.set_attribute("audio.attach_skipped_reason", f"{len(data)} bytes exceeds {limit} byte cap")
        return
    encoded = base64.b64encode(data).decode("ascii")
    span.set_attribute(AudioAttributes.AUDIO_URL, f"data:{mime_type};base64,{encoded}")


def using_run_context(run_id: str, extra: dict | None = None):
    """Group every span from one run -- including auto-instrumented
    LangChain spans -- under one AX session id."""
    from openinference.instrumentation import using_attributes

    metadata = {"run_id": run_id}
    if extra:
        metadata.update(extra)
    return using_attributes(session_id=run_id, metadata=metadata, tags=["audio-agent"])


def _context_attributes() -> dict:
    """Pull whatever using_run_context() set (session id, metadata, tags)
    off the current OTel Context, so manual spans carry it too -- the
    auto-instrumented LangChain spans get this for free; manual spans
    need to ask for it explicitly."""
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
