"""OpenAI chat calls shared by host and librarian, with a same-shape mock
fallback when there's no OPENAI_API_KEY — the LLM-kind span always looks
the same, real or mocked.
"""

import json
from typing import Callable

import config
from instrumentation import LLM, set_output, traced_span


class ToolCall:
    def __init__(self, name: str, arguments: dict, call_id: str):
        self.name = name
        self.arguments = arguments
        self.id = call_id


def _client():
    from openai import OpenAI

    return OpenAI(api_key=config.openai_api_key())


def decide_tool_call(messages: list[dict], tool_spec: dict, mock_arguments: dict) -> ToolCall | None:
    """Ask the model whether to call the one tool offered — possibly again,
    if `messages` already has an earlier round of it in this same loop.

    When mocked, deterministically calls it exactly once: recognizing "this
    question has a second, separate part worth another search" is exactly
    the judgment the dumb fallback can't fake, so it doesn't pretend to —
    it stops as soon as `messages` shows one hop already happened here.
    """
    if not config.openai_api_key():
        already_hopped = any(m.get("role") == "tool" for m in messages)
        if already_hopped:
            return None
        with traced_span(
            "mock_chat_completion", LLM, input_value=json.dumps(messages), attributes={"llm.model_name": "mock"}
        ) as span:
            set_output(span, {"tool_call": tool_spec["function"]["name"], "arguments": mock_arguments})
        return ToolCall(tool_spec["function"]["name"], mock_arguments, call_id="mock_call_1")

    response = _client().chat.completions.create(
        model=config.openai_model(), messages=messages, tools=[tool_spec]
    )
    message = response.choices[0].message
    if not message.tool_calls:
        return None
    call = message.tool_calls[0]
    return ToolCall(call.function.name, json.loads(call.function.arguments), call_id=call.id)


def draft_text(messages: list[dict], mock_text: Callable[[], str]) -> str:
    """Ask the model for a free-text completion, or fall back to a
    deterministic mock string when there's no API key."""
    if not config.openai_api_key():
        with traced_span(
            "mock_chat_completion", LLM, input_value=json.dumps(messages), attributes={"llm.model_name": "mock"}
        ) as span:
            text = mock_text()
            set_output(span, text)
        return text

    response = _client().chat.completions.create(model=config.openai_model(), messages=messages)
    return response.choices[0].message.content or ""
