"""Librarian sub-agent: decides a search query, calls the real MCP tool,
returns the raw excerpts. A nested AGENT span under the host, same thread.

The host may call this more than once per turn — once per sub-topic on a
compound question (see agents/host.py's hop loop) — so this always
describes one delegation, not "the" delegation for a turn.
"""

import mcp_client
from agents import llm, tools
from agents.prompts import LIBRARIAN_SYSTEM
from instrumentation import AGENT, set_output, traced_span


async def run(session, session_id: str, query: str) -> list[dict]:
    with traced_span("librarian", AGENT, input_value=query) as span:
        messages = [
            {"role": "system", "content": LIBRARIAN_SYSTEM},
            {"role": "user", "content": query},
        ]
        call = llm.decide_tool_call(messages, tools.SEARCH_DOCS, mock_arguments={"query": query})
        search_query = call.arguments.get("query", query) if call else query
        excerpts = await mcp_client.call_search_docs(session, session_id, search_query)
        set_output(span, excerpts)
    return excerpts
