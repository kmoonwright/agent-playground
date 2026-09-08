"""Calls the real MCP server (stdio subprocess) through mcp_client — a real
protocol round trip, deterministic since embeddings fall back to the hashed
pseudo-embedding without an OPENAI_API_KEY.

Each test opens its own session inline rather than via a fixture: an anyio
task group's enter/exit must happen in the same asyncio Task, which an
async-generator pytest fixture's separate teardown call can't guarantee.
"""

import mcp_client


async def test_search_docs_returns_relevant_chunk():
    async with mcp_client.open_session() as session:
        results = await mcp_client.call_search_docs(session, "test-session", "eleven OpenInference span kinds")
    assert results
    assert results[0]["doc_id"] == "span-kinds"


async def test_search_docs_ranks_an_on_topic_query_above_an_off_topic_one():
    # An absolute score cutoff isn't portable across embedding backends: real
    # OpenAI embeddings score anything Arize-domain-adjacent (e.g. "pricing")
    # much higher than the hashed mock does, even though neither doc actually
    # answers it. What should hold regardless of backend is the ranking.
    async with mcp_client.open_session() as session:
        on_topic = await mcp_client.call_search_docs(session, "test-session", "eleven OpenInference span kinds")
        off_topic = await mcp_client.call_search_docs(session, "test-session", "Arize Enterprise pricing")
    assert max(r["score"] for r in on_topic) > max(r["score"] for r in off_topic)
