"""Offline tests -- run the full LangGraph pipeline via --selftest, no API
keys or real audio file needed."""

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import instrumentation
from agent import run


def test_selftest_run_produces_all_fields():
    result = run(None, selftest=True)

    assert result["duration_s"] > 0
    assert result["sample_rate"] > 0
    assert result["stubbed"] is True
    assert result["transcript"]
    assert result["analysis"] == {"sentiment": "neutral", "topic": "unknown"}
    assert result["response"].startswith("Stubbed response:")


def test_transcribe_span_attaches_audio(monkeypatch):
    # A local TracerProvider handed to instrumentation.get_tracer(), not
    # opentelemetry.trace.set_tracer_provider() -- the global provider can
    # only ever be set once per process, so a second test setting it would
    # silently no-op instead of failing loudly.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(instrumentation, "get_tracer", lambda: provider.get_tracer("test"))

    run(None, selftest=True)

    transcribe_span = next(s for s in exporter.get_finished_spans() if s.name == "transcribe")
    assert transcribe_span.attributes["audio.mime_type"] == "audio/wav"
    assert transcribe_span.attributes["audio.url"].startswith("data:audio/wav;base64,")
    assert transcribe_span.attributes["audio.transcript"]
