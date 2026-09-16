# Agent playground

Six small, self-contained Python demos of LLM work instrumented with OpenInference and evaluated in [Arize AX](https://app.arize.com). Each directory has its own venv (or `uv` project), `.env`, and README.

They are meant to be read side by side. Same tracing shape (CHAIN → nested spans → auto-instrumented LLM calls), six different control flows: a **pipeline**, a **single agent**, a **multi-agent system running against a realtime deadline**, a **multi-agent system with a real cross-process MCP call**, a **LangGraph pipeline with audio input**, and a **tool-calling agent paired with its own synthetic dataset + 4-evaluator experiment run**.

| Directory | What it is | Start here |
|---|---|---|
| [`pdf-agent/`](pdf-agent/README.md) | Linear PDF extraction workflow. Not an agent. Traces, mixed evals, dataset/experiment, SDK annotation. | `python extract.py --file data/pdfs/DN-001_walmart.pdf` |
| [`pr-review-agent/`](pr-review-agent/README.md) | Visible tool-calling loop that reviews a PR. Lite port of the [OpenHands + Arize workshop](https://github.com/nikanand04/openhands-arize-workshop) reviewer, without the SDK. | `python review.py --fixture buggy-auth` |
| [`dj-agent/`](dj-agent/README.md) | Live DJ. A host agent delegates crate-digging and transition-planning to sub-agents while beatmatched audio plays through the speakers. Shows cross-thread span propagation and an agent that is deliberately kept *out* of the deadline path. | `python dj.py --make-test-crate && python dj.py --offline --source local --provider rule` |
| [`anydocs-agent/`](anydocs-agent/README.md) | Q&A over a docs corpus. A host agent delegates search to a librarian sub-agent — once per sub-topic on a compound question, not just once — which calls a real MCP server (a separate OS process) over stdio. Bundled corpus is Arize AX/OpenInference concept docs, so the agent explaining Arize AX is itself traced by Arize AX. `uv`-managed. | `uv sync && uv run python cli.py --selftest` |
| [`audio-agent/`](audio-agent/README.md) | Transcribes an audio clip, analyzes it, and replies. Built as a **LangGraph** graph instead of raw SDK calls, so `openinference-instrumentation-langchain` turns the graph's own node structure into spans, alongside a hand-instrumented raw Whisper call. `uv`-managed. | `uv sync && uv run python cli.py --selftest` |
| [`credit-coach-agent/`](credit-coach-agent/README.md) | Synthetic fintech credit-coach agent — tool-calling loop, fake JSON DB, traced to Arize AX, with a dataset generator + 4-evaluator (code/LLM/agent/remote-agent) experiment run. Identity is `.env`-configurable, re-skinnable for any customer. `uv`-managed. | `uv sync && uv run pytest` |

`pdf-agent` and `pr-review-agent` want Python 3.11+ and a plain venv; `dj-agent` wants 3.12+ (librosa 1.0) with a plain venv; `anydocs-agent`, `audio-agent`, and `credit-coach-agent` want 3.11+ and use `uv` instead. Copy each demo's `.env.example` to `.env` and fill in the keys that README describes — all six degrade gracefully without them, and `dj-agent` runs its whole audio path with no credentials at all.
