from __future__ import annotations

import os
from contextlib import nullcontext
from typing import Any

_CONFIGURED = False
_STATUS: dict[str, Any] = {"enabled": False, "message": "Phoenix tracing is disabled."}
_TRACER = None


def configure_observability() -> dict[str, Any]:
    """Opt-in Phoenix setup. Call before importing Gemini/LangChain clients."""
    global _CONFIGURED, _STATUS, _TRACER
    if _CONFIGURED:
        return dict(_STATUS)
    _CONFIGURED = True
    enabled = os.getenv("PHOENIX_TRACING_ENABLED", "false").strip().casefold() in {
        "1", "true", "yes", "on"
    }
    if not enabled:
        _STATUS = {
            "enabled": False,
            "endpoint": os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"),
            "message": "Phoenix tracing is disabled (set PHOENIX_TRACING_ENABLED=true to enable).",
        }
        return dict(_STATUS)

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces").strip()
    project_name = os.getenv("PHOENIX_PROJECT_NAME", "rag-demo-intern").strip() or "rag-demo-intern"
    try:
        from phoenix.otel import register
        from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
        from openinference.instrumentation.langchain import LangChainInstrumentor

        provider = register(endpoint=endpoint, project_name=project_name)
        # Both provider and orchestration layers are instrumented: LangChain
        # spans alone do not capture direct google.genai client requests.
        GoogleGenAIInstrumentor().instrument(tracer_provider=provider)
        LangChainInstrumentor().instrument(tracer_provider=provider)
        from opentelemetry import trace

        _TRACER = trace.get_tracer("rag_demo")
        _STATUS = {
            "enabled": True,
            "endpoint": endpoint,
            "project_name": project_name,
            "message": "Tracing enabled; spans export to the configured Phoenix collector.",
        }
    except Exception as error:
        _STATUS = {
            "enabled": False,
            "endpoint": endpoint,
            "project_name": project_name,
            "error": str(error)[:500],
            "message": "Phoenix tracing could not start; the chatbot continues without tracing.",
        }
    return dict(_STATUS)


def tracing_status() -> dict[str, Any]:
    return dict(_STATUS)


def trace_span(name: str, attributes: dict[str, Any] | None = None):
    """Create a short OpenTelemetry span when opt-in tracing is available."""
    if not _STATUS.get("enabled") or _TRACER is None:
        return nullcontext()
    span_context = _TRACER.start_as_current_span(name)

    class _SpanAttributes:
        def __enter__(self):
            self.span = span_context.__enter__()
            for key, value in (attributes or {}).items():
                if value is not None and isinstance(value, (str, int, float, bool)):
                    self.span.set_attribute(key, value)
            return self.span

        def __exit__(self, exc_type, exc, traceback):
            if exc is not None:
                self.span.record_exception(exc)
                from opentelemetry.trace import Status, StatusCode

                self.span.set_status(Status(StatusCode.ERROR, str(exc)[:500]))
            return span_context.__exit__(exc_type, exc, traceback)

    return _SpanAttributes()
