"""Arize AX OpenInference / OpenTelemetry setup for a multi-threaded agent.

Must run before the Anthropic or OpenAI SDK is imported so auto-instrumentation
can patch them — every LLM import in this project is therefore local to a
function, after `ensure_tracing()`.

Two things here that the sibling demos don't need:

  * BOTH SDKs are instrumented, so `--provider claude` and `--provider openai`
    produce the same LLM span shape.
  * `propagate_to_thread()` — OpenTelemetry context does NOT cross thread
    boundaries by itself. The DJ runs the brain, the loader, the conductor, and
    the sub-agent pool on separate threads, so every hand-off must carry the
    captured context or its spans become trace roots instead of children.

The audio callback thread is deliberately absent from all of this: a synchronous
span export inside the PortAudio callback would be an audible dropout. `mixer.py`
imports nothing from this module.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterator

from config import (
    MAX_ATTR_CHARS,
    anthropic_api_key,
    arize_api_key,
    arize_project_name,
    arize_space_id,
    openai_api_key,
)

TRACER_NAME = "dj-agent"

# OpenInference span kinds used in this project.
CHAIN = "CHAIN"
AGENT = "AGENT"
TOOL = "TOOL"
GUARDRAIL = "GUARDRAIL"

_provider = None


def ensure_tracing(*, log_to_console: bool = False, batch: bool = False):
    """Register the Arize exporter and both LLM instrumentors. Idempotent.

    Warns (does not crash) if Arize credentials are missing, so a local run
    still produces a real span tree with real ids — just no export.

    `batch=False` uses a SimpleSpanProcessor so spans land in AX live during a
    demo. That export is synchronous, which is fine on the brain, loader, and
    conductor threads -- and is why the audio thread never traces.
    """
    global _provider

    if _provider is not None:
        return _provider

    space_id = arize_space_id()
    api_key = arize_api_key()
    project_name = arize_project_name()

    if not space_id or not api_key:
        print(
            "Warning: ARIZE_SPACE_ID / ARIZE_API_KEY not set. "
            "Traces will not export to Arize AX. The DJ will still run."
        )
        _provider = _local_tracer_provider()
        _instrument_llms(_provider)
        return _provider

    from arize.otel import register

    _provider = register(
        space_id=space_id,
        api_key=api_key,
        project_name=project_name,
        batch=batch,
        log_to_console=log_to_console,
    )
    _instrument_llms(_provider)
    mode = "batched" if batch else "live (SimpleSpanProcessor)"
    print(f"Arize AX tracing initialized -> project '{project_name}', {mode}.")
    return _provider


def use_provider(provider) -> None:
    """Install a provider directly. Only for tests / the tracing selftest."""
    global _provider
    _provider = provider


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


def _instrument_llms(tracer_provider) -> None:
    """Patch whichever SDKs we have credentials for. Never fatal."""
    kwargs = {"tracer_provider": tracer_provider} if tracer_provider is not None else {}

    if anthropic_api_key():
        try:
            from openinference.instrumentation.anthropic import AnthropicInstrumentor

            AnthropicInstrumentor().instrument(**kwargs)
        except ImportError:
            print("Warning: openinference-instrumentation-anthropic not installed.")
        except Exception as exc:  # already instrumented, version skew, etc.
            print(f"Warning: could not instrument Anthropic SDK: {exc}")

    if openai_api_key():
        try:
            from openinference.instrumentation.openai import OpenAIInstrumentor

            OpenAIInstrumentor().instrument(**kwargs)
        except ImportError:
            print("Warning: openinference-instrumentation-openai not installed.")
        except Exception as exc:
            print(f"Warning: could not instrument OpenAI SDK: {exc}")


def flush_tracing() -> None:
    """Flush + shut down the provider so a short-lived process exports."""
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
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [truncated {len(text) - limit} chars]"


# ---------------------------------------------------------------- threading


def capture_context():
    """Snapshot the current OTel context on the *submitting* thread."""
    from opentelemetry import context as otel_context

    return otel_context.get_current()


@contextmanager
def propagate_to_thread(ctx) -> Iterator[None]:
    """Re-attach a captured context inside a worker thread.

    Without this, a span started on the loader thread, the conductor, or a
    sub-agent worker becomes its own trace root, and the Arize waterfall loses
    the parent/child structure that makes a multi-agent run legible.
    """
    from opentelemetry import context as otel_context

    if ctx is None:
        yield
        return
    token = otel_context.attach(ctx)
    try:
        yield
    finally:
        otel_context.detach(token)


def in_thread(ctx, fn, *args, **kwargs):
    """Convenience wrapper: run `fn` on this thread under a captured context."""
    with propagate_to_thread(ctx):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------- spans


def _context_attributes() -> dict[str, Any]:
    try:
        from openinference.instrumentation import get_attributes_from_context

        return dict(get_attributes_from_context())
    except Exception:
        return {}


@contextmanager
def traced_span(
    name: str,
    kind: str,
    *,
    input_value: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> Iterator[Any]:
    """Manual OpenInference span that also inherits using_attributes() context."""
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


def set_attributes(span, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            span.set_attribute(key, truncate(json.dumps(value, default=str)))
        elif isinstance(value, (str, bool, int, float)):
            span.set_attribute(key, value)
        else:
            span.set_attribute(key, truncate(str(value)))


def set_output(span, value: Any) -> None:
    if isinstance(value, (dict, list)):
        span.set_attribute("output.value", truncate(json.dumps(value, default=str)))
        span.set_attribute("output.mime_type", "application/json")
    else:
        span.set_attribute("output.value", truncate(str(value)))


def using_set_context(set_id: str, extra: dict[str, Any] | None = None):
    """Group every span in one DJ set under a single AX session.

    Also attaches to auto-instrumented LLM spans, which is the only way the
    Anthropic/OpenAI spans pick up the set id.
    """
    from openinference.instrumentation import using_attributes

    metadata = {"set": set_id}
    if extra:
        metadata.update(extra)
    return using_attributes(session_id=set_id, metadata=metadata, tags=["dj-set"])
