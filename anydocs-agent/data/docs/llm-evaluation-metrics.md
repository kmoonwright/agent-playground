# LLM evaluation metrics

Evaluating an LLM application spans several dimensions, and no single evaluator reliably measures every one of them — most real setups layer several.

## Quality metrics
Correctness, groundedness (is the answer actually supported by the source material?), relevance, hallucination detection, safety, and fluency.

## RAG-specific metrics
Split into retrieval-side and generation-side:
- Retrieval: document relevance, whether the retrieved context is sufficient to answer the question, ranking quality.
- Generation: answer correctness, faithfulness to the retrieved sources, citation accuracy.

## Operational metrics
Latency, cost, token usage, and task completion rate — not about answer quality, but still part of a production scorecard.

## Agent-specific checks
Tool selection accuracy (did it call the right tool?), parameter extraction validity (did it pass the tool correct arguments?), path quality (was the sequence of steps reasonable?), and recovery behavior (does it handle a failed tool call sensibly?).

## Code evals vs. LLM-as-a-judge
- **Code-based evaluation** uses deterministic logic — schema validity, exact-value checks, whether a tool argument or latency falls in range. Fast, reproducible, cheap, but limited to things you can actually verify with code.
- **LLM-as-a-judge** deploys a model against a written rubric to score semantic qualities that don't reduce to a deterministic check — answer correctness, relevance, groundedness, helpfulness. Results vary with the judge model, the prompt, and the context given, so a judge's scores should be validated against human labels before being trusted.

## Where evals fit in the lifecycle
- **Pre-production**: curated datasets test expected behavior before anything ships, establishing a quality baseline.
- **CI/CD**: the same evaluators run against a proposed change to catch regressions before release.
- **Production**: evaluation runs on real traffic — guardrails block unsafe outputs synchronously, while other evaluators run asynchronously on sampled traces.
