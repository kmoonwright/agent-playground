"""Spawns mcp_server.py once (stdio transport) and holds an MCP client
session against it for the caller's lifetime.
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from instrumentation import TOOL, set_output, traced_span

_SERVER_SCRIPT = Path(__file__).resolve().parent / "mcp_server.py"


@asynccontextmanager
async def open_session(docs_dir: str | None = None):
    args = [str(_SERVER_SCRIPT)] + ([docs_dir] if docs_dir else [])
    params = StdioServerParameters(command=sys.executable, args=args, cwd=str(_SERVER_SCRIPT.parent))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call_search_docs(session: ClientSession, session_id: str, query: str) -> list[dict]:
    """session_id is plumbing between our client and server code, passed as
    a plain tool argument — it's not exposed in agents/tools.py's
    SEARCH_DOCS schema, so the librarian's LLM never sees or decides it."""
    with traced_span(
        "mcp_client.search_docs",
        TOOL,
        input_value=query,
        attributes={
            "mcp.role": "client",
            "mcp.transport": "stdio",
            "mcp.server.name": "anydocs-search-server",
        },
    ) as span:
        result = await session.call_tool(
            "search_docs", arguments={"query": query, "session_id": session_id}
        )
        if result.is_error:
            message = result.content[0].text if result.content else "unknown MCP tool error"
            raise RuntimeError(f"mcp_server.py's search_docs failed: {message}")
        results = result.structured_content
        if isinstance(results, dict) and "result" in results:
            results = results["result"]
        set_output(span, results)
        return results
