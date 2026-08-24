"""LLM extraction + parse/validate, wrapped in a per-document trace."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from config import KNOWN_RETAILERS, openai_api_key, openai_model
from instrumentation import (
    current_ids,
    ensure_tracing,
    flush_tracing,
    set_output,
    traced_span,
    using_document_context,
)
from schema import ExtractionResult, ParsedDocument

SYSTEM_PROMPT = """You extract structured fields from retailer deduction notices.

Return JSON matching the schema. Rules:
- retailer_name: the retailer as written on the notice.
- deduction_amount: integer US cents. $125.00 → 12500. Negative amounts stay negative.
  If the amount is missing or not numeric (e.g. "TBD"), you MUST still produce an integer;
  use 0 and put the problem in claim_reason.
- is_valid_claim: true ONLY if this is a real, processable deduction claim.
  False if the notice is a draft, placeholder, void, goodwill credit, or says do not process.
- claim_reason: a short reason copied from the document.

Read the entire document, including footers and disclaimers.
"""


def process_document(
    pdf_path: Path,
    *,
    doc_id: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full pipeline: ingest → LLM extract → parse/validate. One trace per document."""
    from ingest import ingest_pdf

    pdf_path = Path(pdf_path)
    doc_id = doc_id or pdf_path.stem.split("_")[0]
    filename = pdf_path.name

    with using_document_context(doc_id, filename, extra_metadata):
        with traced_span(
            "process_document",
            "CHAIN",
            input_value=filename,
            attributes={
                "document.id": doc_id,
                "document.filename": filename,
            },
        ) as root:
            ingested = ingest_pdf(pdf_path, doc_id=doc_id)
            raw = extract_fields(ingested["text"], doc_id=doc_id, filename=filename)
            parsed = parse_and_validate(raw)
            trace_id, span_id = current_ids()
            output = {
                **parsed.as_output(),
                "doc_id": doc_id,
                "filename": filename,
                "source_text": ingested["text"],
                "trace_id": trace_id,
                "span_id": span_id,
                "model": raw.get("model"),
                "mocked": raw.get("mocked", False),
            }
            set_output(
                root,
                {
                    "retailer_name": parsed.retailer_name,
                    "deduction_amount": parsed.deduction_amount,
                    "is_valid_claim": parsed.is_valid_claim,
                    "claim_reason": parsed.claim_reason,
                    "parse_ok": parsed.parse_ok,
                    "types": {
                        "retailer_name": "string",
                        "deduction_amount": "int",
                        "is_valid_claim": "boolean",
                    },
                },
            )
            return output


def extract_fields(source_text: str, *, doc_id: str, filename: str) -> dict[str, Any]:
    """LLM call (auto-instrumented) or mock with the same span shape if no API key."""
    user_prompt = _user_prompt(source_text, doc_id, filename)
    if openai_api_key():
        return _extract_openai(user_prompt)
    return _extract_mock(source_text, user_prompt)


def _user_prompt(source_text: str, doc_id: str, filename: str) -> str:
    return (
        f"Document ID: {doc_id}\nFilename: {filename}\n\n"
        f"--- deduction notice text ---\n{source_text}\n--- end ---"
    )


def _extract_openai(user_prompt: str) -> dict[str, Any]:
    # Imported here so OpenAIInstrumentor can patch the SDK first.
    from openai import OpenAI

    client = OpenAI()
    model = openai_model()
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=ExtractionResult,
        temperature=0,
    )
    parse = getattr(client.chat.completions, "parse", None)
    if parse is None:
        completion = client.beta.chat.completions.parse(**kwargs)
    else:
        completion = parse(**kwargs)

    message = completion.choices[0].message
    if getattr(message, "refusal", None):
        return {
            "raw_json": None,
            "parsed": None,
            "refusal": message.refusal,
            "model": model,
            "mocked": False,
        }
    parsed: ExtractionResult | None = message.parsed
    raw_json = parsed.model_dump() if parsed is not None else None
    if raw_json is None and message.content:
        raw_json = json.loads(message.content)
    return {
        "raw_json": raw_json,
        "parsed": parsed,
        "refusal": None,
        "model": model,
        "mocked": False,
    }


def _extract_mock(source_text: str, user_prompt: str) -> dict[str, Any]:
    """Heuristic extractor used when OPENAI_API_KEY is unset.

    Emits a manual LLM span with prompt / structured output / token attributes
    so traces look the same as the auto-instrumented path.
    """
    with traced_span(
        "openai.chat.completions",
        "LLM",
        input_value=user_prompt,
        attributes={
            "llm.model_name": "mock-extractor",
            "llm.provider": "openai",
            "llm.system": "openai",
            "llm.input_messages.0.message.role": "system",
            "llm.input_messages.0.message.content": SYSTEM_PROMPT,
            "llm.input_messages.1.message.role": "user",
            "llm.input_messages.1.message.content": user_prompt,
        },
    ) as span:
        extracted = _heuristic_extract(source_text)
        payload = extracted.model_dump()
        span.set_attribute("llm.output_messages.0.message.role", "assistant")
        span.set_attribute(
            "llm.output_messages.0.message.content", json.dumps(payload)
        )
        prompt_tokens = max(len(user_prompt) // 4, 1)
        completion_tokens = max(len(json.dumps(payload)) // 4, 1)
        span.set_attribute("llm.token_count.prompt", prompt_tokens)
        span.set_attribute("llm.token_count.completion", completion_tokens)
        span.set_attribute("llm.token_count.total", prompt_tokens + completion_tokens)
        set_output(span, payload)
        return {
            "raw_json": payload,
            "parsed": extracted,
            "refusal": None,
            "model": "mock-extractor",
            "mocked": True,
        }


def _heuristic_extract(source_text: str) -> ExtractionResult:
    retailer = _find_retailer(source_text)
    amount = _find_amount_cents(source_text)
    lowered = source_text.lower()
    # Deliberately ignore a late footer so the draft trap still produces a
    # false-positive extraction (the interesting LLM-judge failure).
    body = source_text[:800].lower()
    invalid_markers = (
        "goodwill",
        "not a contractual",
        "credit issued in error",
        "negative deduction",
    )
    is_valid = amount is not None and amount >= 0 and not any(m in body for m in invalid_markers)
    # Keep the draft footer from flipping the mock extraction to false.
    if "internal draft" in lowered and "internal draft" not in body:
        is_valid = True
    reason = _find_reason(source_text)
    return ExtractionResult(
        retailer_name=retailer or "UNKNOWN",
        deduction_amount=amount if amount is not None else 0,
        is_valid_claim=is_valid,
        claim_reason=reason,
    )


def _find_retailer(text: str) -> str | None:
    match = re.search(r"Retailer:\s*(.+)", text)
    if match:
        return match.group(1).strip()
    for name in (*KNOWN_RETAILERS, "Kroger"):
        if re.search(rf"\b{re.escape(name)}\b", text, re.I):
            return name
    return None


def _find_amount_cents(text: str) -> int | None:
    match = re.search(
        r"Claimed deduction amount:\s*(-?\$[0-9,]+(?:\.[0-9]{2})?)", text
    )
    if not match:
        match = re.search(r"(-?\$[0-9,]+(?:\.[0-9]{2})?)", text)
    if not match:
        return None
    raw = match.group(1).replace("$", "").replace(",", "")
    try:
        return int(round(float(raw) * 100))
    except ValueError:
        return None


def _find_reason(text: str) -> str:
    match = re.search(
        r"Reason for deduction\s*(.+?)(?:Please remit|INTERNAL DRAFT|$)",
        text,
        re.S | re.I,
    )
    if match:
        return " ".join(match.group(1).split())
    return text[:240].strip()


def parse_and_validate(raw: dict[str, Any]) -> ParsedDocument:
    """Parse LLM JSON into typed fields. Type mismatches are recorded on the span."""
    with traced_span(
        "parse_validate",
        "GUARDRAIL",
        input_value=json.dumps(raw.get("raw_json"), default=str),
        attributes={"document.id": raw.get("doc_id")},
    ) as span:
        if raw.get("refusal"):
            parsed = ParsedDocument(parse_ok=False, parse_error=f"model refusal: {raw['refusal']}")
            set_output(span, parsed.as_output())
            return parsed

        payload = raw.get("raw_json")
        already = raw.get("parsed")
        try:
            if isinstance(already, ExtractionResult):
                result = already
            elif payload is None:
                raise ValueError("LLM returned no structured payload")
            else:
                result = ExtractionResult.model_validate(payload)
        except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
            parsed = ParsedDocument(parse_ok=False, parse_error=str(exc))
            if isinstance(payload, dict):
                parsed.retailer_name = payload.get("retailer_name")
                parsed.claim_reason = payload.get("claim_reason")
                parsed.is_valid_claim = payload.get("is_valid_claim")
                amount = payload.get("deduction_amount")
                parsed.deduction_amount = amount if isinstance(amount, int) else None
            span.set_attribute("validation.ok", False)
            span.set_attribute("validation.error", str(exc))
            set_output(span, parsed.as_output())
            return parsed

        parsed = ParsedDocument(
            retailer_name=result.retailer_name,
            deduction_amount=result.deduction_amount,
            is_valid_claim=result.is_valid_claim,
            claim_reason=result.claim_reason,
            parse_ok=True,
        )
        span.set_attribute("validation.ok", True)
        span.set_attribute("output.retailer_name", result.retailer_name)
        span.set_attribute("output.deduction_amount", result.deduction_amount)
        span.set_attribute("output.is_valid_claim", result.is_valid_claim)
        span.set_attribute("output.types.retailer_name", "string")
        span.set_attribute("output.types.deduction_amount", "int")
        span.set_attribute("output.types.is_valid_claim", "boolean")
        set_output(span, parsed.as_output())
        return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the extraction pipeline on one PDF.")
    parser.add_argument("--file", type=Path, required=True, help="Path to a deduction-notice PDF.")
    parser.add_argument("--doc-id", type=str, default=None)
    args = parser.parse_args()

    ensure_tracing()
    try:
        result = process_document(args.file, doc_id=args.doc_id)
        printable = {k: v for k, v in result.items() if k != "source_text"}
        print(json.dumps(printable, indent=2, default=str))
        _write_last_run_single(result)
    finally:
        flush_tracing()


def _write_last_run_single(result: dict[str, Any]) -> None:
    from datetime import datetime, timezone

    from config import LAST_RUN_PATH, arize_project_name

    LAST_RUN_PATH.write_text(
        json.dumps(
            {
                "project_name": arize_project_name(),
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "spans": [
                    {
                        "doc_id": result.get("doc_id"),
                        "span_id": result.get("span_id"),
                        "trace_id": result.get("trace_id"),
                        "filename": result.get("filename"),
                    }
                ],
            },
            indent=2,
        )
    )
    print(f"\nSaved span id {result.get('span_id')} to {LAST_RUN_PATH.name}")


if __name__ == "__main__":
    main()
