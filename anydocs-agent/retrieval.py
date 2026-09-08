"""search_docs: EMBEDDING -> RETRIEVER -> RERANKER. Runs inside the MCP
server process; called by its "search_docs" tool handler.
"""

import re

import numpy as np

import store
from instrumentation import EMBEDDING, RERANKER, RETRIEVER, set_output, traced_span
from schema import DocChunk, SearchResult

CANDIDATE_K = 8
RESULT_K = 3


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def _overlap(query: str, text: str) -> float:
    query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not query_words:
        return 0.0
    text_words = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(query_words & text_words) / len(query_words)


def search_docs(
    query: str, chunks: list[DocChunk], normalized_corpus_vectors: np.ndarray
) -> list[SearchResult]:
    """normalized_corpus_vectors must already be L2-normalized (see
    mcp_server.py, which normalizes once at startup) — normalizing the
    whole corpus again on every query is wasted, repeated work."""
    with traced_span("embed_query", EMBEDDING, input_value=query) as span:
        query_vector = store.embed([query])[0]
        set_output(span, {"dim": len(query_vector)})

    with traced_span("vector_search", RETRIEVER, input_value=query) as span:
        similarities = normalized_corpus_vectors @ normalize(query_vector[None, :])[0]
        top = np.argsort(similarities)[::-1][:CANDIDATE_K]
        candidates = [SearchResult(chunk=chunks[i], score=float(similarities[i])) for i in top]
        set_output(span, [c.chunk.id for c in candidates])

    with traced_span("rerank_results", RERANKER, input_value=query) as span:
        reranked = sorted(
            candidates,
            key=lambda c: 0.7 * c.score + 0.3 * _overlap(query, c.chunk.text),
            reverse=True,
        )[:RESULT_K]
        set_output(span, [r.chunk.id for r in reranked])

    return reranked
