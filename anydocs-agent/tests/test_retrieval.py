"""Cosine ranking math, no MCP layer — direct calls into retrieval.search_docs
with hand-built fixture chunks."""

import retrieval
import store
from schema import DocChunk


def test_search_docs_ranks_the_relevant_chunk_first():
    chunks = [
        DocChunk(id="a#1", doc_id="a", heading="Cats", text="Cats are small feline mammals that purr."),
        DocChunk(id="b#1", doc_id="b", heading="Rockets", text="Rockets use combustion to reach orbit."),
    ]
    vectors = retrieval.normalize(store.embed([c.text for c in chunks]))
    results = retrieval.search_docs("tell me about feline mammals", chunks, vectors)
    assert results[0].chunk.doc_id == "a"
