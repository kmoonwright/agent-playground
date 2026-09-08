"""Pluggable doc sources. Point this at any folder of markdown/text files."""

import re
from pathlib import Path

from schema import DocChunk


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"


def _chunk_markdown(doc_id: str, text: str) -> list[DocChunk]:
    """Split on '## ' headings. A file with no such heading is one chunk."""
    sections = re.split(r"^## ", text, flags=re.MULTILINE)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        heading, _, body = section.partition("\n")
        heading = heading.lstrip("#").strip()
        body = body.strip() or heading
        chunks.append(
            DocChunk(id=f"{doc_id}#{_slug(heading)}", doc_id=doc_id, heading=heading, text=body)
        )
    return chunks


def load_folder(path: Path) -> list[DocChunk]:
    """Load every *.md file in `path` into DocChunks, chunked by heading."""
    chunks = []
    for file in sorted(Path(path).glob("*.md")):
        doc_id = file.stem
        chunks.extend(_chunk_markdown(doc_id, file.read_text()))
    return chunks
