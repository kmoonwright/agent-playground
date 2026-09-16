"""Tool schemas + dispatch for the agent's tool-calling loop.

Each tool call gets its own OTel TOOL span (name, input args, output) -- this
is what the code-based evaluator checks against ground truth later.
"""

from __future__ import annotations

import json
import time
from typing import Any

import config
import db
import tracing

# DB-backed tools get a nested "mcp call -> db query" span pair simulating the
# real Finn/Kikoff-equivalent agent's actual architecture (MCP server fronting
# a database) -- still no real MCP server or database process, just spans that
# read the same way in Arize AX. general_faq has no lookup, so it stays flat.
_DB_TABLES = {
    "get_credit_report": "credit_reports",
    "list_disputable_items": "disputable_items",
    "check_dispute_eligibility": "disputable_items",
}

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_credit_report",
            "description": "Get a user's credit report summary and score.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_disputable_items",
            "description": "List items on a user's credit report that may be eligible for dispute.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_dispute_eligibility",
            "description": "Check whether a specific report item is eligible to dispute, and why.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "item_id": {"type": "string"},
                },
                "required": ["user_id", "item_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "general_faq",
            "description": (
                f"Answer a general question about how {config.company_name()} works. "
                "Use this for questions that are not about a specific user's report or disputes."
            ),
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
    },
]

_FAQ_ANSWERS = {
    "how_it_works": (
        f"{config.company_name()} helps you build credit with a small credit-builder "
        "account: you make on-time payments toward a small line of credit, and that "
        "positive payment history gets reported to the credit bureaus every month."
    ),
    "pricing": f"{config.company_name()} plans are a few dollars a month -- check the app for current pricing.",
    "credit_building": (
        "Building credit is mostly about two things: paying on time, every time, "
        "and keeping balances low relative to your limits."
    ),
}
_DEFAULT_FAQ_ANSWER = (
    f"{config.company_name()} is a credit-building app. For account-specific "
    "questions, ask about your credit report or a specific item you'd like to dispute."
)


def _faq(topic: str) -> str:
    key = topic.strip().lower().replace(" ", "_")
    return _FAQ_ANSWERS.get(key, _DEFAULT_FAQ_ANSWER)


_HANDLERS = {
    "get_credit_report": lambda args: db.get_report(args["user_id"]),
    "list_disputable_items": lambda args: db.list_disputable_items(args["user_id"]),
    "check_dispute_eligibility": lambda args: db.check_dispute_eligibility(
        args["user_id"], args["item_id"]
    ),
    "general_faq": lambda args: _faq(args.get("topic", "")),
}


def _call_handler(name: str, args: dict[str, Any], handler) -> Any:
    """Call handler(args), nesting fake mcp/db spans for DB-backed tools."""
    table = _DB_TABLES.get(name)
    if table is None:
        return handler(args)

    with tracing.traced_span(
        f"mcp.{name}",
        tracing.TOOL,
        attributes={
            "mcp.method": name,
            "mcp.server": "credit-coach-mcp (simulated)",
            "mcp.simulated": True,
        },
    ) as mcp_span:
        time.sleep(0.02)
        with tracing.traced_span(
            f"db.query:{table}",
            tracing.TOOL,
            attributes={
                "db.system": "fake-json-store",
                "db.table": table,
                "db.statement": f"SELECT * FROM {table} WHERE user_id = {args.get('user_id')!r}",
                "db.simulated": True,
            },
        ) as db_span:
            time.sleep(0.01)
            result = handler(args)
            tracing.set_output(db_span, result)
        tracing.set_output(mcp_span, result)
    return result


def dispatch(name: str, args: dict[str, Any]) -> str:
    """Run a tool call inside its own TOOL span. Returns a string observation."""
    handler = _HANDLERS.get(name)
    with tracing.traced_span(
        name,
        tracing.TOOL,
        input_value=json.dumps(args),
        attributes={"tool.name": name, **args},
    ) as span:
        if handler is None:
            result = f"error: unknown tool '{name}'"
            tracing.set_output(span, result)
            return result
        try:
            result = _call_handler(name, args, handler)
        except (KeyError, TypeError) as exc:
            result = f"error: invalid arguments for '{name}': {exc}"
            tracing.set_output(span, result)
            return result
        tracing.set_output(span, result)
        return json.dumps(result) if isinstance(result, (dict, list)) else str(result)
