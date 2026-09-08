# The eleven OpenInference span kinds

OpenInference spans carry an `openinference.span.kind` attribute that says what a span represents. There are eleven of them, and they're best understood as one taxonomy rather than eleven separate topics.

**LLM** — a call to a large language model. Example: querying GPT-4 or Claude directly.

**CHAIN** — a starting point or a link between different steps of an LLM application. Example: orchestrating pre- or post-processing logic between other operations.

**AGENT** — encompasses calls to LLMs and tools, typically the top-level wrapper for an agent loop. Example: the main coordinator in a research assistant.

**TOOL** — a call to an external tool, API, or function on behalf of an LLM. Example: executing a knowledge-base search the model requested.

**RETRIEVER** — a data retrieval query for context from a datastore, such as a vector database similarity search. Example: a vector query run inside a tool span to find relevant chunks.

**EMBEDDING** — an encoding of unstructured data (usually text) into a vector. Example: converting a document or a query into an embedding before storing or searching it. EMBEDDING is the encoding step; RETRIEVER is the querying step — easy to conflate, but they're different work.

**RERANKER** — a relevance-based re-ordering of documents. Example: reordering a first-pass set of search results by a finer relevance score.

**GUARDRAIL** — a validation of LLM input or output for safety, policy, or compliance. Example: filtering or blocking an unsafe model response before it reaches a user.

**EVALUATOR** — an evaluation process: the type, configuration, and result. Example: a span recording the scoring of an answer's quality after generation.

**PROMPT** — prompt construction or templating. Example: substituting variables into a prompt template before sending it to a model.

**UNKNOWN** — the default when no kind is set explicitly on a span. Not something you deliberately create — it just means nobody classified that span.
