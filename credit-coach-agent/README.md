# credit-coach-agent

A synthetic fintech credit-coach agent for a guided Arize AX POC. The agent
runs a tool-calling loop against a fake local JSON "DB" of credit profiles,
fully traced to Arize AX. A dataset generator produces synthetic rows across
4 question categories (with message history), uploads them as an Arize
Dataset, and an experiment runner scores the agent against all of them with
4 evaluator types: code-based, LLM-based, agent-based, and remote-agent-based.

Nothing here calls a real company system or touches real user data — every
persona, credit score, and disputable item is invented. The agent's identity
(company name, agent name) is set via `.env` — see `COMPANY_NAME` /
`AGENT_NAME` below — so this demo can be re-skinned for any customer without
touching code.

## Setup

```bash
uv sync
cp .env.example .env   # fill in keys below
```

| Var | Required for | Notes |
|---|---|---|
| `OPENAI_API_KEY` | `agent.py`, `server.py`, LLM/agent/remote evals | without it, evaluators fall back to a mock judge |
| `ARIZE_SPACE_ID` | tracing export, `dataset_gen.py`, `run_experiment.py` | Space Settings in app.arize.com |
| `ARIZE_API_KEY` | same as above | Space Settings in app.arize.com |
| `ARIZE_PROJECT_NAME` | tracing export | defaults to `credit-coach-agent` |
| `COMPANY_NAME` | prompts, FAQ copy, dataset questions | defaults to `Acme` |
| `AGENT_NAME` | prompts, CLI output | defaults to `Sage` |

Everything degrades gracefully without Arize creds — `tracing.py` falls back
to a local, unexported tracer so `agent.py` and the offline tests still run.

## How to run

**Offline, no keys needed:**
```bash
uv run pytest                          # tool-loop + span-nesting selftest
uv run python dataset_gen.py --dry-run # builds data/dataset.csv locally, no Arize push
```

**Live, once `.env` is filled in:**
```bash
uv run python agent.py --user-id P01 --message "what's my credit score?"
uv run python dataset_gen.py           # generates ~60 rows, pushes to Arize AX as a Dataset
uv run python run_experiment.py        # runs the agent over every row, logs an Experiment with 4 evals
uv run uvicorn server:app --reload --port 8787   # optional: talk to the agent live during a demo

# Populate the tenant with a batch of traces before a demo (default pool already
# mixes plain / compound / guardrail rows, so this alone covers the full range):
uv run python generate_traces.py --count 30
uv run python generate_traces.py --compound-only --count 10   # just want deep tool-chain traces, fast
uv run python generate_traces.py --guardrail-only --count 6   # just want the guardrail-catch traces
uv run python generate_sessions.py --sessions 8                # multi-turn conversations, grouped by session.id
```

## Demo prompts for richer traces

Every turn now always opens an `input_guardrail` span before the LLM loop
and an `output_guardrail` span after it (see "Guardrails" below), and every
DB-backed tool call nests a fake `mcp.*` → `db.*` span pair underneath it
(see "Traces" below). So for `k` tool calls in one turn:

`spans = 1 (CHAIN) + 2 (guardrails) + k x (LLM + TOOL + MCP + DB) + 1 (final LLM) = 4k + 4`

- No tool needed (a plain FAQ) → 4 spans (`k=0`; +2 more if the model still
  calls `general_faq`, which has no MCP/DB nesting).
- One tool needed (a report/dispute lookup) → 8 spans (`k=1`).
- A compound question needing 3 tools → **16 spans** (`k=3`) — confirmed by
  running one offline.

`dataset_gen.COMPOUND_PROMPTS` are hand-authored to push toward `k=2-3` —
copy/paste these live, or run `generate_traces.py --compound-only` (above)
to fire them automatically:

- "Can you explain my credit report, and let me know if I have anything worth disputing?" (`user_id=P07`)
- "What's my credit score, and can I dispute DI-07 on my file?" (`user_id=P10`)
- "Give me my full report, tell me what's disputable, and confirm whether DI-10 specifically is eligible." (`user_id=P16`)
- "Walk me through my report and flag anything I should dispute." (`user_id=P12`)
- "What's on my report, and is that inquiry from Apex Lending something I can dispute?" (`user_id=P08`)

## Guardrails

Two guardrails demonstrate Arize AX's risk-reduction/compliance story --
both actually change the agent's behavior when triggered, not just log it.
Each opens its own `GUARDRAIL`-kind span (`guardrails.py`):

- **`input_scope_guardrail`** (`guardrails.check_input`) — runs before the
  LLM is ever called. Blocks out-of-scope legal/financial-advice requests
  (suing, bankruptcy, "legal advice") and replaces the reply with a canned
  redirect. When it triggers, the LLM is never invoked at all for that turn.
- **`output_compliance_guardrail`** (`guardrails.check_output`) — runs after
  the agent's reply is drafted. Catches guarantee/outcome-promise language
  ("guarantee", "100%") and credit-score hallucinations (a stated score that
  doesn't match what `get_credit_report` actually returned), and rewrites
  the reply using the **real facts from the tool calls already made** -- not
  a generic refusal.

`AgentResult.guardrails_triggered` records which (if any) fired.
`dataset_gen.GUARDRAIL_PROMPTS` reliably trip these — copy/paste live, or run
`generate_traces.py --guardrail-only`:

- "Should I sue over this collections account, or would filing bankruptcy be smarter?" (`user_id=P07`, trips `input_scope_guardrail`)
- "Can you guarantee that disputing DI-07 will 100% remove it and raise my score by 50 points?" (`user_id=P10`, trips `output_compliance_guardrail`)
- "I want legal advice on whether to press charges over DI-10 -- what should I do?" (`user_id=P16`, trips `input_scope_guardrail`)

## Sessions

Every trace carries a `session.id` attribute -- the OpenInference convention
Arize AX's Sessions view groups traces by. Multiple separate turns (each its
own trace) that share a `session.id` show up as one longer-style conversation
instead of scattered, unrelated traces.

- `tracing.using_session(session_id, extra=None)` is the context manager that
  sets it -- wrap any block of `run_agent(...)` calls in it and every span
  inside, manual (`CHAIN`/`TOOL`) and auto-instrumented alike, picks up
  `session.id` (and `agent.py`'s CLI and `server.py`'s `/chat` already do
  this for you; see below).
- `agent.py --user-id ... --message ...` generates a fresh one-off
  `cli-<user_id>-<hex>` session per run and prints it.
- `server.py`'s `POST /chat` accepts an optional `session_id` in the request
  body and always returns one in the response -- pass the same `session_id`
  back on your next `/chat` call to keep building one session live during a
  demo.
- `generate_sessions.py` (below) is the fast way to build several full
  multi-turn sessions at once.

## What to look at in Arize AX

- **Traces**: each turn is one `credit-coach-agent-turn` span (kind `CHAIN`),
  with an `input_guardrail` `GUARDRAIL` span, a `TOOL` span per tool call
  (`get_credit_report`, `list_disputable_items`, `check_dispute_eligibility`,
  `general_faq`), an `output_guardrail` `GUARDRAIL` span, and the
  auto-instrumented OpenAI call(s) nested underneath. The 3 DB-backed tool
  calls each nest a further `mcp.<tool>` → `db.query:<table>` span pair,
  simulating the MCP-server-in-front-of-a-database shape the original ask
  described -- still no real MCP process or database, just spans that read
  the same way.
- **Sessions**: traces sharing a `session.id` (see "Sessions" above) group
  into one conversation in the Sessions view -- `generate_sessions.py`
  produces several ready-made ones.
- **Datasets**: the `credit-coach-agent` dataset, ~60 rows across
  `how_it_works`, `explain_report`, `can_i_dispute`, `disputable_items`, each
  with `message_history`, `expected_tool`, `expected_tool_args`, and
  `expected_facts`.
- **Experiments**: one row per dataset example, with 4 evaluator columns:
  - `code_eval` — deterministic: right tool, right args
  - `llm_eval` — LLM-as-judge, structured `correctness`/`completeness`/`tone`
  - `agent_eval` — holistic: does the full trace's *behavior* match
    `expected_facts` (right score, right eligibility, right CTA)?
  - `remote_eval` — same holistic judge, wrapped as a simulated external
    round-trip (its own `EVALUATOR` span, small delay) — stands in for
    calling out to the customer's own external judge.

## Layout

| File | Purpose |
|---|---|
| `tracing.py` | Arize AX OTel setup, manual span helpers, `using_session` |
| `config.py` | env vars + paths |
| `db.py` | fake persona/credit-report JSON store |
| `tools.py` | 4 tool schemas + dispatch; nests fake `mcp.*`/`db.*` spans for DB-backed tools |
| `guardrails.py` | the 2 input/output guardrails, each its own `GUARDRAIL` span |
| `agent.py` | the agent's tool-calling loop, wired through both guardrails |
| `server.py` | `POST /chat` FastAPI wrapper for live demos |
| `dataset_gen.py` | synthetic dataset builder + Arize upload; also `COMPOUND_PROMPTS`/`GUARDRAIL_PROMPTS` |
| `evaluators.py` | the 4 evaluator functions |
| `run_experiment.py` | runs the agent over the dataset, logs the experiment |
| `generate_traces.py` | fires a batch of single-turn agent runs to populate Arize AX with demo traces |
| `generate_sessions.py` | builds multi-turn conversations grouped into Arize AX sessions |
| `data/personas.json` | 16 hand-authored fake credit profiles |
| `tests/test_agent.py` | offline selftest, no network calls |
| `tests/test_guardrails.py` | offline checks on the 2 guardrails' trigger logic |
| `tests/test_dataset_gen.py` | offline checks on `COMPOUND_PROMPTS`/`GUARDRAIL_PROMPTS` + pool-building logic |
| `tests/test_sessions.py` | offline checks on session-plan logic |

## Safety

All data is synthetic and invented for this repo. No real company system,
API, or customer record is called or referenced anywhere in this code.
