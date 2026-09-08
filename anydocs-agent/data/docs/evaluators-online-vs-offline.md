# Online vs. offline evaluators on Arize AX

The split is about who orchestrates the evaluation: Arize runs online evaluators on its own platform; you orchestrate offline evaluators yourself, outside the platform.

## Online evaluators

Online evaluators run entirely inside Arize AX. The platform triggers them, executes them, and joins the results back onto the originating spans as `eval.<name>.*` attributes — no separate download/upload step.

Two flavors, same output shape:
- **LLM-as-a-judge**: an LLM scores each span, trace, or session against a prompt, model, and output schema you configure entirely in the UI. No code required.
- **Code evaluators**: small Python functions (including built-in templates like "contains any keyword") that score spans with minimal code.

They can run continuously on new traces as they arrive, or as a historical batch on demand. Rate-limiting and retries are handled by the platform.

## Offline evaluators

Offline evaluation is a three-step, user-orchestrated workflow: download spans (or a dataset) from Arize, run your own evaluation pipeline against them, upload the results back.

You control the whole pipeline — multi-stage logic, evaluators running in parallel, custom data transforms, non-standard models, CI integration. This is the right fit for complex evaluation logic the platform's UI doesn't expose, but it's the wrong fit if all you want is "score every new trace as it comes in" — that's what online evaluators are for.

One common offline pattern is an Arize **experiment**: upload a labeled dataset, define a task function that reproduces your pipeline, attach one or more evaluators, and run them together — the experiment view then shows pass/fail per row per evaluator.

## They compose

Both can evaluate the same spans at the same time. A typical setup layers a cheap, continuous online evaluator (say, a fast groundedness check on every trace) with a periodic, more expensive offline evaluation (an experiment run against a curated dataset before a release) — you don't have to pick just one.
