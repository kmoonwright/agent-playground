"""Optional local HTTP wrapper for demoing the agent live.

    uvicorn server:app --reload --port 8787
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI
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


@app.on_event("shutdown")
def _shutdown() -> None:
    tracing.flush_tracing()
