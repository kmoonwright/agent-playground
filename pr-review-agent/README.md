# Lite PR-review agent

A small, readable code-review **agent** — a model in a tool-calling loop, not a linear pipeline.

It is a lite port of the [OpenHands + Arize workshop](https://github.com/nikanand04/openhands-arize-workshop) reviewer. The workshop hides the loop inside the OpenHands Agent SDK. This directory makes the loop the thing you read.

Sibling to [`pdf-agent/`](../pdf-agent/README.md), which is a traced document pipeline (not an agent). Same Arize wiring; different shape of work.

## What you should learn

Open `agent.py`, `tools.py`, and `prompt.py`. That is the whole agent.

1. **The review only counts if a tool is called.** `submit_review` writes JSON + markdown to `out/`. Chat prose is discarded. This is the workshop's real failure mode: models write an excellent review as a message and never submit it.
2. **One nudge, then stop.** If the model ends a turn with no tool call, we resend a short "call the tool now" message once. A second miss is a genuine failure (red CHAIN span), not something to paper over.
3. **Line numbers come from `read_file`, not from the diff.** Diff hunk headers lie. The tool description and the prompt both say so.
4. **The file manifest is complete even when a patch is truncated.** `[patch abbreviated]` means "read the file", not "the file is missing".
5. **A cheap, error-free run can still be wrong.** `--max-iterations 2` on `buggy-auth` often approves a PR with a backdoor in it. Cost and latency look better. The review is worse. That is why the CHAIN span carries the review itself, not just status=OK.

## Setup

Python 3.11+. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

| Variable | Required | Where from |
|---|---|---|
| `OPENAI_API_KEY` | yes | OpenAI dashboard |
| `OPENAI_MODEL` | no | defaults to `gpt-4o-mini` |
| `ARIZE_API_KEY`, `ARIZE_SPACE_ID` | no | [app.arize.com](https://app.arize.com) → Settings → API Keys. Without them the agent still runs; traces stay local. |
| `ARIZE_PROJECT_NAME` | no | defaults to `pr-review-agent` |
| `GITHUB_TOKEN` | no | only for `--repo/--pr` (read-only; `gh auth token` is enough) |

## How to run

All commands from `pr-review-agent/`.

### 1. Fixture with a real bug (healthy budget)

```bash
python review.py --fixture buggy-auth
```

The PR claims to add login rate limiting. It increments a counter it never checks, compares a password hash to plaintext, and ships a `"letmein"` admin backdoor. A healthy run should call `read_file` on `auth.py`, then `submit_review` with `needs_rework` and a cited line.

Artifacts land in `out/demo__shop__42.json` and `.md`.

### 2. Starve the budget

```bash
python review.py --fixture buggy-auth --max-iterations 2
```

Same PR, two model turns. Often the agent never reads the file, never calls `submit_review`, or approves the PR. The trace can be clean, cheap, and short — and still wrong.

### 3. A clean rename (calibration)

```bash
python review.py --fixture clean-rename
```

`get_user` → `fetch_user`, call sites updated, no behavior change. The honest review is `worth_merging` with an empty (or near-empty) findings list. A model that invents nits to "look thorough" fails this one.

### 4. A live public PR (optional)

```bash
python review.py --repo pydantic/pydantic --pr 123
```

Same loop. `read_file` hits the GitHub Contents API at the PR head SHA, allowlisted to files in the PR. **No clone, no shell, nothing posted to GitHub.** Prefer small open PRs; large ones blow the default 8-iteration budget — raise `--max-iterations`.

## What to look at in Arize AX

Project `pr-review-agent` → **Tracing**. One trace per review:

1. `review_pull_request` (CHAIN) — PR prompt in, review markdown out, plus `review.verdict` / `review.finding_count`
2. `read_file` / `submit_review` (TOOL) — what the agent actually did
3. Auto-instrumented LLM spans — messages, tokens, latency

Session id is the PR slug (`demo/shop#42`), so every run of the same PR groups together.

If the agent never submitted, the CHAIN span is **ERROR** (`agent finished without submitting a review`). That is intentional.

## Layout

```
review.py            CLI
agent.py             the while-loop (read this)
tools.py             read_file + submit_review
prompt.py            system + user + one nudge
schema.py            Finding / Review
source.py            ReviewTarget: fixtures or GitHub
github.py            GET-only PR + file fetch
instrumentation.py   arize-otel + OpenAI instrumentor
data/prs/            buggy-auth, clean-rename
out/                 review artifacts (gitignored)
```

## Safety

- Tools cannot run a shell or write to GitHub.
- `read_file` only opens paths listed in the PR's changed-files manifest.
- The GitHub path is GET-only (PR metadata, file list, file contents).
