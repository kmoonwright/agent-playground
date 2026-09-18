# AnyDocsAgent

A small Q&A agent for a documentation corpus, instrumented with OpenInference and
traced to [Arize AX](https://app.arize.com). The name is literal — the pipeline is
generic (point `--docs` at any folder of markdown you have) — but the bundled
default corpus is six short pages on Arize AX / OpenInference concepts, so asking
this agent questions about span kinds, evaluators, and MCP tracing is itself traced
by the very platform it's answering questions about.

Architecture: a **host** agent decides whether a question needs the docs, and if so
delegates to a **librarian** sub-agent — once per sub-topic, so a genuinely
compound question gets two delegations, not one — which searches over a real
**MCP server** for each: a separate OS process, talking stdio JSON-RPC, not an
in-process function call.

## Quick start

```bash
uv sync
cp .env.example .env          # fill in keys, or leave blank to run fully mocked
uv run python cli.py --selftest
uv run python cli.py                       # interactive REPL
uv run python -m pytest -q                 # offline test suite
uv run python run_experiment.py            # offline eval (needs Arize creds)
uv run python generate_sessions.py         # 6 multi-turn AX sessions (needs Arize creds)
```

Every value in `.env` is optional. With no `OPENAI_API_KEY`, embeddings fall back to
a deterministic hashed pseudo-embedding and chat calls fall back to a scripted stub —
same span shape either way, so the whole pipeline runs and traces with zero
credentials. With no `ARIZE_API_KEY`/`ARIZE_SPACE_ID`, tracing falls back to a local
in-process provider (a warning is printed once) — everything still runs, it just
doesn't export.

See [`EXAMPLE_PROMPTS.md`](EXAMPLE_PROMPTS.md) for prompts known to produce each
interesting trace shape — multi-hop delegation, the guardrail refusal, the MCP
cross-process branch — instead of guessing. For a live customer demo, skip the
REPL and seed AX directly:

```bash
uv run python generate_sessions.py                 # one of each journey (6 sessions × 4 turns)
uv run python generate_sessions.py --sessions 3    # faster smoke
```

That groups every turn of one persona under a single `session.id` so AX's
Sessions view is a conversation, not 24 unrelated traces. The six journeys
cover the workflows you'd actually walk a customer through: span taxonomy,
online vs offline evals, the reliability loop, MCP client/server tracing,
compound multi-hop retrieval, and a coverage-boundary refusal. Multi-hop
traces need `OPENAI_API_KEY`; without it the sessions still export, but every
turn takes exactly one hop.

## The span tree

One turn produces this tree. `LLM` spans are auto-instrumented by
`OpenAIInstrumentor`; everything else is a manual span with an explicit
`openinference.span.kind`. The `mcp_server.*` branch runs in a **separate OS
process** — `mcp_server.py`, spawned once and held open — joined into this same
trace purely by `MCPInstrumentor`'s OpenTelemetry-context propagation over the
stdio pipe.

The `AGENT librarian` branch below can repeat: `agents/host.py::run()` is a bounded
loop (`MAX_HOPS = 3`), and the host's own system prompt tells it to delegate again,
with a different focused query, when a question has a genuinely separate second
part — not to rephrase or double-check. A simple question takes one hop; a compound
one ("what's the difference between EMBEDDING and RETRIEVER, and how does online
vs. offline evaluation relate to that?") takes two, each its own full nested branch.
The `host` `AGENT` span carries `hops` and `hop_queries` attributes, so it's visible
directly on the span without having to count children.

```
CHAIN   turn
└─ AGENT   host                                  hops=2, hop_queries=[...]
     ├─ LLM         hop 1: delegate, or answer directly?
     ├─ AGENT   librarian (hop 1)                nested sub-agent, same thread as host
     │    ├─ LLM     librarian picks a search query
     │    └─ TOOL   mcp_client.search_docs       client side: mcp.role=client, mcp.transport=stdio
     │         ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄ ┄    (stdio JSON-RPC to a separate process)
     │           TOOL   mcp_server.search_docs    server side: mcp.role=server, mcp.transport=stdio
     │                ├─ EMBEDDING   embed_query
     │                ├─ RETRIEVER   vector_search       cosine similarity over the chunk store
     │                └─ RERANKER    rerank_results       lexical-overlap re-score of the top hits
     ├─ LLM         hop 2: a second, distinct sub-topic — delegate again?
     ├─ AGENT   librarian (hop 2)                same shape as hop 1, different query
     │    └─ ...
     ├─ PROMPT   build_answer_prompt              inject the deduped excerpts from every hop
     ├─ LLM       host drafts the answer, with [doc-id] citations
     └─ GUARDRAIL   check_groundedness            strips fabricated citations; forces a refusal on a weak match
```

That's 9 of the 11 OpenInference span kinds in one request. `EVALUATOR` shows up
separately in `run_experiment.py`. `UNKNOWN` is just the default when nobody sets a
kind — nothing here is meant to hit it deliberately.

### Why MCP gets its own client/server span names

`openinference-instrumentation-mcp`'s `MCPInstrumentor` is unusual: it emits **no
spans of its own**. Its only job is propagating OpenTelemetry context across the
wire protocol, so spans created independently on the client and server sides join
into one trace. That means nothing in the trace says "this was MCP" unless you name
it yourself — so the client- and server-side spans here are deliberately named
`mcp_client.search_docs` / `mcp_server.search_docs`, each carrying `mcp.role` and
`mcp.transport` attributes, rather than both just being called `search_docs`. See
`data/docs/mcp-tracing.md` for more.

One operational consequence: the server is a stdio-transport process, so its stdout
*is* the JSON-RPC wire — it never prints anything (see `instrumentation.py`'s
stderr-only logging).

### Sessions: grouping every turn for session-level evals

Every turn of one conversation shares a single `session_id`, generated once per
run (`cli.py`) — not regenerated per turn, which would put each turn in its own
ungrouped session and make session-level evaluation impossible. Getting this right
took two things, not one:

- `agents/host.py` wraps each turn in `instrumentation.using_session_context(session_id)`, and `instrumentation.traced_span()` explicitly pulls that context onto every manual span it creates (`get_attributes_from_context()`). Auto-instrumented `LLM` spans get this for free from the instrumentor; manual spans don't unless you ask.
- `session_id` is threaded across the MCP process boundary as a plain tool argument (`mcp_client.call_search_docs` → `mcp_server.py`'s `search_docs(query, session_id)`), because — see `data/docs/mcp-tracing.md` — `MCPInstrumentor` only propagates standard trace context, not OpenInference's session/metadata context. Without this, the server-side spans (`mcp_server.search_docs`, `EMBEDDING`, `RETRIEVER`, `RERANKER`) would silently have no `session.id` even after fixing the first half.

## Evaluators: online vs. offline

- **Online** — configured entirely in the Arize AX UI, no code here. Point an
  LLM-as-a-judge "groundedness" evaluator at the host's `AGENT` span
  (`input.value` = question + retrieved excerpts, `output.value` = the final
  answer), running continuously on new traces.
- **Offline** — `run_experiment.py`. Uploads `data/eval_dataset.json` as an Arize
  dataset, runs the real pipeline as the task, and scores it with two evaluators:
  - `citation_valid` (code eval): does the answer cite the doc it was supposed to
    (or refuse, for the out-of-scope row)?
  - `answer_correctness_judge` (LLM-as-judge, traced as an `EVALUATOR` span): is the
    answer consistent with the expected summary?

  Two rows are deliberately the hard ones — a nuance question (does `MCPInstrumentor`
  emit its own spans?) and an out-of-scope one (Arize's Enterprise pricing, which
  isn't in any of the bundled docs). With `gpt-4o-mini` and the current prompts, the
  guardrail and `ANSWER_SYSTEM`'s own "say plainly it isn't covered" instruction
  both handle these correctly — a clean 10/10 pass is the actual current result, not
  a hand-waved claim. The value of keeping them in the dataset isn't a guaranteed
  visible failure right now; it's that a future prompt change, model swap, or corpus
  edit that breaks either behavior would show up here as a regression.

## Layout

| File | Purpose |
|---|---|
| `config.py` | env accessors; the only module that reads the environment |
| `instrumentation.py` | span-kind constants, tracing setup, manual span helper |
| `sources.py` | loads a folder of markdown into chunks |
| `store.py` | embeddings, real or deterministic-mock |
| `retrieval.py` | EMBEDDING → RETRIEVER → RERANKER |
| `mcp_server.py` | the search tool, as a real MCP server (stdio) |
| `mcp_client.py` | spawns and talks to `mcp_server.py` |
| `agents/host.py` | the host agent: bounded delegate loop (`MAX_HOPS`), draft, guardrail |
| `agents/librarian.py` | the sub-agent: pick a query, call the MCP tool |
| `agents/llm.py` | shared OpenAI-or-mock chat call |
| `evaluators/` | the offline eval's code eval + LLM judge |
| `run_experiment.py` | uploads the offline eval as an Arize experiment |
| `cli.py` | REPL / `--selftest` entry point |
| `generate_sessions.py` | multi-turn demo conversations, grouped by `session.id` |
| `data/docs/` | the bundled corpus |
| `data/eval_dataset.json` | the offline eval's dataset |

## Known limitations

The hashed pseudo-embedding fallback is a bag-of-words proxy, not a real semantic
embedding — it's there so the whole pipeline runs and traces without an API key, not
to be a good retriever. Real answer quality needs `OPENAI_API_KEY`.

`agents/host.py`'s `SCORE_THRESHOLD` is a backstop for retrieval finding nothing
plausibly relevant, not the primary defense against an uncovered-but-domain-adjacent
question. Real embeddings score anything Arize-adjacent moderately high (e.g. a
pricing question against docs about Arize concepts, ~0.45), much higher than the
hashed mock's score for the same query (~0.12) — an absolute cutoff tuned to one
embedding backend doesn't transfer to the other. What actually catches the
domain-adjacent-but-uncovered case is `ANSWER_SYSTEM`'s own instruction to the model
to say plainly when the excerpts don't answer the question.

Markdown corpus files chunk by `## ` heading — fine for docs with a few genuinely
separable sections, wrong for a flat taxonomy (`span-kinds.md` deliberately uses
inline `**bold**` labels instead of headings, so the whole file stays one chunk;
splitting it per-kind meant no top-k ever retrieved the complete list of eleven).

The mock (no-key) fallback always takes exactly one hop, even for a compound
question — recognizing "this question has two distinct parts worth separate
searches" is a real capability, not something the deterministic fallback can fake,
so it doesn't pretend to. Multi-hop only happens with a real `OPENAI_API_KEY`.
