# PDF Document-Processing Demo on Arize AX

Self-contained Python demo of a **linear PDF extraction workflow** (not an agent) instrumented with OpenInference / OpenTelemetry and evaluated in Arize AX.

Shows tracing every LLM call in a document pipeline, mixed output types (`int` / `boolean` / `string`), mixed deterministic + LLM-as-judge evals, and SDK annotation from an external script.

## What this shows

| Capability | Where it lands |
|---|---|
| Tracing PDF extraction | One trace per document: `process_document` → `pdf_ingestion` → auto-instrumented LLM span → `parse_validate` |
| Mixed output types | `deduction_amount` (int, US cents), `is_valid_claim` (boolean), `retailer_name` (string) |
| Mixed evals | Code evals on retailer enum + amount range; LLM-as-judge on claim validity, scored against ground truth |
| Dataset + experiment | `run_experiment.py` uploads 12 labeled docs and runs all three evaluators |
| Annotate via API | `annotate_demo.py` calls `client.spans.update_annotations` (optional `experiments.annotate_runs`) |

Trap documents **DN-009 … DN-012** are designed to fail in the experiment view. A clean run with zero failures is not a useful demo.

## Setup

Python 3.11+. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

| Variable | Required | Where to get it |
|---|---|---|
| `ARIZE_API_KEY` | yes (for AX) | [app.arize.com](https://app.arize.com) → Settings → API Keys |
| `ARIZE_SPACE_ID` | yes (for AX) | Space Settings (or `ax spaces list`) |
| `ARIZE_PROJECT_NAME` | no | defaults to `pdf-extraction-demo` |
| `OPENAI_API_KEY` | no | If unset, extraction and the judge are stubbed **with the same span shape** |
| `OPENAI_MODEL` | no | defaults to `gpt-4o-mini` |

Generate the synthetic PDFs (idempotent):

```bash
python ingest.py --generate
```

### One-time: annotation config

SDK annotation **requires a config that already exists in the space**. Create it once in the Arize UI (**Annotation Configs** → new categorical config) or via CLI:

```bash
ax annotation-configs create \
  --name "extraction_quality" \
  --space "$ARIZE_SPACE_ID" \
  --type categorical \
  --value correct \
  --value incorrect \
  --optimization-direction maximize
```

The name must be exactly `extraction_quality`. Optional freeform config if you also pass `--also-experiment-run` with notes:

```bash
ax annotation-configs create \
  --name "reviewer_notes" \
  --space "$ARIZE_SPACE_ID" \
  --type freeform
```

## How to run

All commands from `pdf-agent/`.

### 1. Single document (one trace)

```bash
python extract.py --file data/pdfs/DN-001_walmart.pdf
```

Then open Arize AX → project `pdf-extraction-demo` → **Tracing**. Filter metadata `document_id = DN-001`. Expand the trace:

1. `process_document` (CHAIN) — document id + filename attributes
2. `pdf_ingestion` (CHAIN) — raw page text
3. LLM span — prompt, structured JSON, tokens / latency / cost (auto-instrumented when using OpenAI)
4. `parse_validate` (GUARDRAIL) — typed `int` / `boolean` / `string` fields

### 2. Dataset + experiment (mixed evals)

```bash
python run_experiment.py
```

Use `--dry-run` to exercise the pipeline locally without creating AX records. `--concurrency 1` (default) keeps traces easy to narrate live.

**Expected experiment columns**

| Column | Type | What fails |
|---|---|---|
| `retailer_name_match` | code | **DN-009** (Kroger — not in enum) |
| `deduction_amount_valid` | code | **DN-010** (negative), **DN-012** (\$50M) |
| `claim_validity_judge` | LLM-as-judge | **DN-011** (looks valid; footer says DRAFT / DO NOT PROCESS). **DN-010** / **DN-012** also fail GT (`is_valid_claim=false`) if extraction marks them valid |

### 3. Annotate from an external system

After a pipeline or experiment run, `.last_run.json` stores span ids. Then:

```bash
python annotate_demo.py
```

This calls `client.spans.update_annotations` for the DN-011 span by default (override with `--span-id` / `--doc-id` / `--label correct`).

To also annotate an experiment run via API:

```bash
python annotate_demo.py --also-experiment-run
```

Refresh the trace (or experiment row) in Arize AX — the annotation appears **without using the UI to apply it**.

## Synthetic data

12 single-page deduction notices under `data/pdfs/`, labels in `data/ground_truth.json`.

Known retailers: **Walmart, Target, CVS, Amazon**. Amounts are **integer US cents**.

| ID | Retailer | Amount | Valid? | Role |
|---|---|---|---|---|
| DN-001 … DN-008 | known | sane | yes | pass path |
| DN-009 | Kroger | \$150 | yes | unknown retailer |
| DN-010 | Target | -\$1,250 | no | negative amount |
| DN-011 | Walmart | \$95 | **no** | draft footer (LLM-judge) |
| DN-012 | Amazon | \$50,000,000 | no | absurd amount + goodwill |

No real customer documents are used.

## Live walkthrough

See [DEMO_RUNBOOK.md](DEMO_RUNBOOK.md) (5–10 minutes).

## Layout

```
ingest.py            PDF generation + ingestion span
extract.py           LLM extract (auto-instrumented) + parse/validate + process_document
evaluators/          code evals + LLM-as-judge
run_experiment.py    dataset upload + experiment
annotate_demo.py     SDK annotation (external-system stand-in)
instrumentation.py   arize-otel + OpenAI instrumentor
data/ground_truth.json
data/pdfs/
```
