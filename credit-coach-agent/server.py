"""Optional local HTTP wrapper for demoing the agent live.

    uvicorn server:app --reload --port 8787
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, Request
from opentelemetry import context as otel_context
from opentelemetry.propagate import extract
from pydantic import BaseModel

import tracing

tracing.ensure_tracing()

import agent  # noqa: E402  (must import after ensure_tracing() patches openai)

app = FastAPI(title="credit-coach-agent")


class ChatRequest(BaseModel):
    user_id: str
    message: str
    history: list[dict] = []
    session_id: str | None = None  # pass the prior response's session_id back to keep one session


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    session_id = req.session_id or f"chat-{req.user_id}-{uuid.uuid4().hex[:6]}"
    with tracing.using_session(session_id, extra={"user_id": req.user_id}):
        result = agent.run_agent(req.user_id, req.message, req.history)
    return {
        "response_text": result.response_text,
        "tool_calls": result.tool_calls,
        "trace_id": result.trace_id,
        "session_id": session_id,
    }


class RemoteAgentRequest(BaseModel):
    """Matches Arize AX's Remote Agent input schema: templated fields at the
    top level, plus the `arize_metadata` block AX always injects. Register
    this endpoint under More > Remote Agents with an Input Schema covering
    `user_id` / `message` / `history` (not `arize_metadata` -- AX adds that).
    """

    user_id: str
    message: str
    history: list[dict] = []
    arize_metadata: dict | None = None


@app.post("/remote-agent")
def remote_agent(req: RemoteAgentRequest, request: Request) -> dict:
    meta = req.arize_metadata or {}
    session_id = meta.get("run_id") or meta.get("example_id") or (
        f"remote-agent-{uuid.uuid4().hex[:6]}"
    )

    # Link this run's spans under AX's experiment-run trace via the
    # traceparent/baggage headers it sends alongside the request.
    parent_ctx = extract(dict(request.headers))
    token = otel_context.attach(parent_ctx)
    try:
        with tracing.using_session(session_id, extra={"user_id": req.user_id, **meta}):
            result = agent.run_agent(req.user_id, req.message, req.history)
    finally:
        otel_context.detach(token)

    return {
        "response_text": result.response_text,
        "tool_calls": result.tool_calls,
        "trace_id": result.trace_id,
        "session_id": session_id,
        "guardrails_triggered": result.guardrails_triggered,
    }


@app.on_event("shutdown")
def _shutdown() -> None:
    tracing.flush_tracing()
