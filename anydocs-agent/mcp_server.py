"""Standalone MCP server (stdio transport). Exposes one tool, search_docs.

Runs as its own OS process, spawned by mcp_client.py. stdout is the MCP
JSON-RPC wire here — see instrumentation.py's stderr-only logging rule.
"""

import sys
from pathlib import Path

import config
import instrumentation
import retrieval
import sources
import store
from instrumentation import TOOL, set_output, traced_span
from mcp.server.mcpserver import MCPServer

instrumentation.ensure_tracing()

_docs_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else config.DOCS_DIR
_chunks = sources.load_folder(_docs_dir)
# Normalized once here, not per query — see retrieval.search_docs's docstring.
_vectors = retrieval.normalize(store.embed([chunk.text for chunk in _chunks]))

mcp = MCPServer("anydocs-search-server")


@mcp.tool()
def search_docs(query: str, session_id: str) -> list[dict]:
    """Search the loaded documentation corpus for chunks relevant to query.

    session_id is passed by mcp_client.py, not decided by any LLM — it's
    what lets these server-side spans join the same AX session as the
    client-side ones (see instrumentation.using_session_context)."""
    with instrumentation.using_session_context(session_id):
        with traced_span(
            "mcp_server.search_docs",
            TOOL,
            input_value=query,
            attributes={"mcp.role": "server", "mcp.transport": "stdio"},
        ) as span:
            results = retrieval.search_docs(query, _chunks, _vectors)
            payload = [
                {
                    "doc_id": r.chunk.doc_id,
                    "heading": r.chunk.heading,
                    "text": r.chunk.text,
                    "score": r.score,
                }
                for r in results
            ]
            set_output(span, payload)
            return payload


if __name__ == "__main__":
    mcp.run(transport="stdio")
