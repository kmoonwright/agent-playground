# Example prompts, and what they produce

Every prompt below has been run against the real agent (real OpenAI + Arize
credentials) in the course of building this demo, not guessed. The `hops` value
is read straight off the `host` `AGENT` span's `hops` attribute — check that
directly in Arize AX rather than eyeballing the tree; it's the fastest way to
confirm what actually happened.

## Baseline: one hop

```
What are the eleven OpenInference span kinds?
```

`hops: 1`. This is the same question `cli.py --selftest` runs. Produces the full
single-hop span tree: `CHAIN turn` → `AGENT host` → `AGENT librarian` → `TOOL
mcp_client.search_docs` → (cross-process) → `TOOL mcp_server.search_docs` →
`EMBEDDING`/`RETRIEVER`/`RERANKER` → back up through `PROMPT
build_answer_prompt` → `GUARDRAIL check_groundedness`.

## A comparison alone does NOT trigger a second hop

```
What's the difference between EMBEDDING and RETRIEVER spans?
```

`hops: 1`. Worth having as a contrast case: what decides a second hop is whether
the *first search covers the whole question*, not whether the question is
phrased as a comparison. Both terms are defined in the same `span-kinds.md`
chunk, so one search is enough.

## Multi-hop: two different docs

```
What's the difference between EMBEDDING and RETRIEVER spans, and how does online vs offline evaluation relate to that?
```

`hops: 2`, `hop_queries` shows two distinct queries. Produces two full nested
`AGENT librarian` branches (each with its own `TOOL`/`EMBEDDING`/`RETRIEVER`/
`RERANKER` under it) as siblings under one `AGENT host`.

## Multi-hop: same doc, two sections

```
What are the five silent failure modes, and what does the eight-step improvement loop look like?
```

`hops: 2`, even though both halves come from `agent-reliability.md` — different
sections of one doc still count as separate sub-topics. Good for showing the
loop keys on *topic* separation, not *document* separation.

## Guardrail: out-of-scope refusal

```
What is Arize's Enterprise pricing?
```

The answer refuses (its own wording, or the exact `NOT_COVERED` string if the
guardrail's score threshold forces it — both paths are real and both were
observed in this session). `citations` comes back empty either way. Good
paired with `run_experiment.py`'s row 9, which scores this exact case offline.

## The MCP cross-process branch

Any prompt above that searches at all already produces this — call it out
explicitly when reading the trace: `mcp_client.search_docs` (host process) and
`mcp_server.search_docs` (a separate OS process, spawned by `mcp_client.py`)
share one trace purely through `MCPInstrumentor`'s context propagation. Look
for `mcp.role: client` vs `mcp.role: server` on those two spans, and the
`EMBEDDING`/`RETRIEVER`/`RERANKER` spans nested under the *server* one.

## What's not reliably promptable

The guardrail's fabricated-citation catch and the `MAX_HOPS` bound are both
real, but a capable real model rarely fabricates a citation or insists on more
than 2-3 hops on demand — there's no prompt that "reliably" forces either one
live. They're covered instead by scripted-fake tests:
`tests/test_agents.py::test_guardrail_strips_a_fabricated_citation` and
`::test_host_stops_hopping_at_max_hops`.

## The one span kind that's not from the REPL

`EVALUATOR` only shows up from `run_experiment.py` (the offline eval), not from
any REPL prompt — see the README's "Evaluators" section.
