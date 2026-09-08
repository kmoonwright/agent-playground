"""Tool specs, OpenAI function-calling shape."""

ASK_LIBRARIAN = {
    "type": "function",
    "function": {
        "name": "ask_librarian",
        "description": "Delegate a documentation question to the librarian, who searches the docs corpus and returns relevant excerpts.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
        },
    },
}

SEARCH_DOCS = {
    "type": "function",
    "function": {
        "name": "search_docs",
        "description": "Search the documentation corpus for chunks relevant to a query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query"}},
            "required": ["query"],
        },
    },
}
