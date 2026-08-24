# Live demo runbook (5–10 min)

Goal: show tracing, mixed types, mixed evals, and API annotation on a PDF workflow — not an agent.

Have these ready before the call: `.env` filled, `pip install -r requirements.txt` done, annotation config `extraction_quality` created, Arize AX project `pdf-extraction-demo` open in a browser.

---

### 0:00 — Frame it (30s)

Typical starting point: a PDF pipeline with ad-hoc output checks, no traces or formal evals. This repo is a linear extraction workflow on synthetic deduction notices, instrumented on Arize AX.

### 0:30 — Run one document (1 min)

```bash
python extract.py --file data/pdfs/DN-001_walmart.pdf
```

Point at the printed JSON: `retailer_name` (string), `deduction_amount` (int cents), `is_valid_claim` (boolean). Mention the fourth field `claim_reason` exists so the judge has something to read.

### 1:30 — Tracing view (2 min)

Arize AX → project **pdf-extraction-demo** → **Tracing** → latest trace (filter `document_id = DN-001` or session id `DN-001`).

Walk the tree top-down:

1. **`process_document`** — custom attributes `document.id`, `document.filename`. This is how they’d filter a specific invoice later.
2. **`pdf_ingestion`** — raw page text. No LLM yet; proves non-LLM steps are visible.
3. **LLM span** (OpenAI auto-instrumentation) — prompt, structured output, tokens, latency, cost. This is “visibility into every LLM call.”
4. **`parse_validate`** — Pydantic typed parse. Type mismatches show up here as span errors, not silent JSON.

If they ask about another provider (Gemini, Anthropic, etc.): same OpenInference shape; swap the instrumentor. The span tree does not depend on an agent framework.

### 3:30 — Experiment / mixed evals (3 min)

```bash
python run_experiment.py
```

While it runs: 12 docs, 8 clean + 4 traps. Dataset is labeled (retailer, amount cents, `is_valid_claim`).

Arize AX → **Datasets** → `deduction-notices-*` → **Experiments**.

Call out the three eval columns:

| Column | Kind | Field | Failure to click |
|---|---|---|---|
| `retailer_name_match` | code | string enum | **DN-009** Kroger |
| `deduction_amount_valid` | code | int range | **DN-010** negative, **DN-012** \$50M |
| `claim_validity_judge` | LLM-as-judge | boolean | **DN-011** draft footer vs extracted `true` |

Open DN-011: body looks like damaged goods; footer is `INTERNAL DRAFT — DO NOT PROCESS`. Ground truth `is_valid_claim=false`. That’s the subjective judgment you cannot regex.

If they ask about comparing prompt/model versions: this experiment is the baseline; a second `run_experiment.py --experiment-name …` on the same dataset is the comparison view.

### 6:30 — Annotate via API (2 min)

Can a specific run be annotated from an external system?

```bash
python annotate_demo.py
```

This is a standalone script any other service could call. It uses `client.spans.update_annotations` against the span id in `.last_run.json` (defaults to DN-011). No UI click.

Refresh the DN-011 trace. Annotation `extraction_quality = incorrect` should be on the span.

Optional, if there’s time:

```bash
python annotate_demo.py --also-experiment-run --label incorrect
```

Same idea on the experiment row (`client.experiments.annotate_runs`). Mention the one-time annotation-config prerequisite — configs live in the space; the API only writes values.

### 8:30 — Close (30s)

They get traces on the real pipeline, typed evals per field, dataset/experiment for regression, and annotations from their own tools. Next step if they want it: point this same instrumentation at a handful of the customer’s PDFs.

---

## Backup if something is slow

- Traces not in UI yet: `python extract.py --file data/pdfs/DN-009_kroger_trap.pdf` again; SimpleSpanProcessor is on (`batch=False`). Wait ~10s and refresh.
- No OpenAI key: pipeline still traces (mock LLM span). Say so up front; the tree is the same.
- Experiment already exists from rehearsal: that’s fine — new timestamped names are created each run.
- Annotation 400: config `extraction_quality` missing. Create it (README) and retry; do not try to invent labels in the UI on the fly unless the config is there.
