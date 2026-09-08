"""Embeddings. Real OpenAI embeddings if a key is set, otherwise a
deterministic hashed pseudo-embedding — same interface, same span shape,
so the retrieval pipeline runs and traces identically either way.

Real embeddings are cached to disk by text hash: mcp_server.py is a fresh
OS process every time it's spawned, and without this it would re-embed the
whole corpus — a real, billed OpenAI call — on every single run.
"""

import hashlib
import json
import re

import numpy as np

import config

_HASH_DIM = 256
_EMBEDDING_MODEL = "text-embedding-3-small"

# Dropped so two unrelated chunks don't look similar just for sharing common
# words (including "arize", present in nearly every chunk here).
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to", "of",
    "in", "on", "for", "and", "or", "it", "its", "this", "that", "as", "by",
    "with", "from", "at", "you", "your", "so", "if", "not", "no", "can",
    "arize", "ax",
}


def _hashed_embedding(text: str) -> np.ndarray:
    vec = np.zeros(_HASH_DIM)
    words = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS]
    for word in words:
        index = int(hashlib.sha1(word.encode()).hexdigest(), 16) % _HASH_DIM
        vec[index] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def _cache_key(text: str) -> str:
    return hashlib.sha1(f"{_EMBEDDING_MODEL}:{text}".encode()).hexdigest()


def _load_cache() -> dict[str, list[float]]:
    try:
        return json.loads(config.EMBEDDING_CACHE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def embed(texts: list[str]) -> np.ndarray:
    """Embed a batch of texts. Returns an (n, dim) array."""
    if config.openai_api_key():
        cache = _load_cache()
        keys = [_cache_key(text) for text in texts]
        missing = [text for text, key in zip(texts, keys) if key not in cache]

        if missing:
            from openai import OpenAI

            client = OpenAI(api_key=config.openai_api_key())
            response = client.embeddings.create(model=_EMBEDDING_MODEL, input=missing)
            for text, item in zip(missing, response.data):
                cache[_cache_key(text)] = item.embedding
            config.EMBEDDING_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            config.EMBEDDING_CACHE_PATH.write_text(json.dumps(cache))

        return np.array([cache[key] for key in keys])
    return np.array([_hashed_embedding(text) for text in texts])
