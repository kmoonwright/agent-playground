# Agent playground

Two small, self-contained Python demos of LLM work instrumented with OpenInference and evaluated in [Arize AX](https://app.arize.com). Each directory has its own venv, `.env`, and README.

They are meant to be read side by side: one is a **pipeline**, the other is an **agent**. Same tracing shape (CHAIN → nested spans → auto-instrumented LLM calls). Different control flow.

| Directory | What it is | Start here |
|---|---|---|
| [`pdf-agent/`](pdf-agent/README.md) | Linear PDF extraction workflow. Not an agent. Traces, mixed evals, dataset/experiment, SDK annotation. | `python extract.py --file data/pdfs/DN-001_walmart.pdf` |
| [`pr-review-agent/`](pr-review-agent/README.md) | Visible tool-calling loop that reviews a PR. Lite port of the [OpenHands + Arize workshop](https://github.com/nikanand04/openhands-arize-workshop) reviewer, without the SDK. | `python review.py --fixture buggy-auth` |

Python 3.11+. Copy each demo's `.env.example` to `.env` and fill in Arize / OpenAI keys as that README describes.
