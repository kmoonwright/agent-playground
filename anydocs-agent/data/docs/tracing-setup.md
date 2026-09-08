# Setting up OpenInference tracing with arize-otel

Arize AX ingests traces over standard OpenTelemetry. The `arize-otel` package's `register()` function is a convenience wrapper that configures an OpenTelemetry `TracerProvider` with Arize-aware defaults — it's optional; you could wire the OpenTelemetry SDK up by hand instead.

## Minimal setup

```python
from arize.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

tracer_provider = register(
    space_id="your-arize-space-id",
    api_key="your-arize-api-key",
    project_name="your-project-name",
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

`register()` returns the `TracerProvider`. You then hand that provider to one or more OpenInference **auto-instrumentors** — one per SDK you use (OpenAI, Anthropic, and so on). An instrumentor patches its target library so every call through that library automatically produces a well-formed `LLM`-kind span, with no manual span code at each call site.

## Manual spans for everything else

Auto-instrumentation only covers the LLM call itself. Any other step you want visible in a trace — retrieval, guardrails, tool execution, prompt construction — needs a manual span, created the ordinary OpenTelemetry way and given an explicit `openinference.span.kind` attribute so Arize AX renders it consistently with the auto-instrumented ones.

## Ordering matters

Tracing setup has to run before the instrumented SDK is imported anywhere in the process — instrumentors work by patching the SDK's client classes at instrument-time, so a client constructed before `instrument()` runs won't be patched.

## Timing of export

`register()` can be configured to export spans synchronously (useful for a short-lived script or a live demo, since spans show up immediately) or batched (lower overhead, but nothing exports until a flush). Either way, a short-lived process should flush and shut down the provider before exiting, or its last spans may never be sent.
