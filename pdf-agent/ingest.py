"""Synthetic deduction-notice PDF generation and traced ingestion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import GROUND_TRUTH_PATH, PDF_DIR
from instrumentation import set_output, traced_span

DEFAULT_FOOTER = (
    "This deduction notice is issued under the applicable vendor trading agreement. "
    "Questions: deductions@example-retailer.com"
)


def load_ground_truth() -> dict[str, Any]:
    return json.loads(GROUND_TRUTH_PATH.read_text())


def pdf_path_for(doc: dict[str, Any]) -> Path:
    return PDF_DIR / doc["filename"]


def generate_pdfs(*, overwrite: bool = True) -> list[Path]:
    """Render one-page deduction notices from the ground-truth fixture."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    fixture = load_ground_truth()
    written: list[Path] = []

    for doc in fixture["documents"]:
        path = pdf_path_for(doc)
        if path.exists() and not overwrite:
            written.append(path)
            continue

        c = canvas.Canvas(str(path), pagesize=letter)
        width, height = letter
        y = height - 72

        c.setFont("Times-Bold", 18)
        c.drawString(72, y, "DEDUCTION NOTICE")
        y -= 18
        c.setFont("Times-Roman", 10)
        c.drawString(72, y, "Vendor chargeback / bill-back — do not discard")
        y -= 28

        c.setFont("Times-Bold", 11)
        c.drawString(72, y, f"Notice ID:  {doc['doc_id']}")
        y -= 16
        c.drawString(72, y, f"Notice date:  {doc['notice_date']}")
        y -= 16
        c.drawString(72, y, f"Retailer:  {doc['retailer_name']}")
        y -= 16
        c.drawString(72, y, f"Claimed deduction amount:  {doc['deduction_amount_display']}")
        y -= 28

        c.setFont("Times-Bold", 12)
        c.drawString(72, y, "Reason for deduction")
        y -= 18
        c.setFont("Times-Roman", 11)
        for line in _wrap(doc["claim_reason"], 90):
            c.drawString(72, y, line)
            y -= 14

        y -= 24
        c.setFont("Times-Roman", 10)
        c.drawString(72, y, "Please remit or dispute within 15 days of the notice date.")
        y -= 36

        footer = doc.get("footer") or DEFAULT_FOOTER
        c.setFont("Times-Italic", 9 if doc.get("trap_type") == "draft_not_a_claim" else 8)
        for line in _wrap(footer, 100):
            c.drawString(72, y, line)
            y -= 12

        if doc.get("is_trap"):
            c.setFont("Times-Roman", 7)
            c.setFillGray(0.5)
            c.drawString(72, 48, f"[synthetic trap document — {doc['trap_type']}]")

        c.showPage()
        c.save()
        written.append(path)

    return written


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word])
        if len(trial) > width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def ingest_pdf(path: Path, *, doc_id: str | None = None) -> dict[str, Any]:
    """Extract raw page text. Emits a nested OpenInference span."""
    from pypdf import PdfReader

    path = Path(path)
    doc_id = doc_id or path.stem.split("_")[0]
    attributes = {
        "document.id": doc_id,
        "document.filename": path.name,
        "input.mime_type": "application/pdf",
    }
    with traced_span(
        "pdf_ingestion",
        "CHAIN",
        input_value=str(path),
        attributes=attributes,
    ) as span:
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            pages.append({"page": i + 1, "text": page.extract_text() or ""})
        combined = "\n\n".join(p["text"] for p in pages)
        result = {
            "doc_id": doc_id,
            "filename": path.name,
            "path": str(path),
            "page_count": len(pages),
            "pages": pages,
            "text": combined,
        }
        span.set_attribute("document.page_count", len(pages))
        set_output(span, {"page_count": len(pages), "chars": len(combined), "text": combined})
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and/or ingest synthetic deduction PDFs.")
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Write PDFs under data/pdfs/ from data/ground_truth.json.",
    )
    parser.add_argument(
        "--ingest",
        type=Path,
        help="Ingest a single PDF and print extracted text.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip existing PDFs when generating.",
    )
    args = parser.parse_args()

    if args.generate or not args.ingest:
        paths = generate_pdfs(overwrite=not args.no_overwrite)
        print(f"Wrote {len(paths)} PDFs to {PDF_DIR}")
        for path in paths:
            print(f"  {path.name}")

    if args.ingest:
        from instrumentation import ensure_tracing, flush_tracing

        ensure_tracing()
        try:
            result = ingest_pdf(args.ingest)
            print(json.dumps({k: v for k, v in result.items() if k != "pages"}, indent=2))
            print("\n--- page text ---\n")
            print(result["text"])
        finally:
            flush_tracing()


if __name__ == "__main__":
    main()
