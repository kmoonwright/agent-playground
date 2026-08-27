# Agent playground

Three small, self-contained Python demos of LLM work instrumented with OpenInference and evaluated in [Arize AX](https://app.arize.com). Each directory has its own venv, `.env`, and README.

They are meant to be read side by side. Same tracing shape (CHAIN → nested spans → auto-instrumented LLM calls), three different control flows: a **pipeline**, a **single agent**, and a **multi-agent system running against a realtime deadline**.

| Directory | What it is | Start here |
|---|---|---|
| [`pdf-agent/`](pdf-agent/README.md) | Linear PDF extraction workflow. Not an agent. Traces, mixed evals, dataset/experiment, SDK annotation. | `python extract.py --file data/pdfs/DN-001_walmart.pdf` |
| [`pr-review-agent/`](pr-review-agent/README.md) | Visible tool-calling loop that reviews a PR. Lite port of the [OpenHands + Arize workshop](https://github.com/nikanand04/openhands-arize-workshop) reviewer, without the SDK. | `python review.py --fixture buggy-auth` |
| [`dj-agent/`](dj-agent/README.md) | Live DJ. A host agent delegates crate-digging and transition-planning to sub-agents while beatmatched audio plays through the speakers. Shows cross-thread span propagation and an agent that is deliberately kept *out* of the deadline path. | `python dj.py --make-test-crate && python dj.py --offline --source local --provider rule` |

`pdf-agent` and `pr-review-agent` want Python 3.11+; `dj-agent` wants 3.12+ (librosa 1.0). Copy each demo's `.env.example` to `.env` and fill in the keys that README describes — all three degrade gracefully without them, and `dj-agent` runs its whole audio path with no credentials at all.
