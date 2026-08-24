"""One ReviewTarget type, two backends: local fixtures and live GitHub PRs.

The agent never cares which backend it got. It sees a file list, patches, and
read_file(path). GitHub is GET-only; fixtures never leave the repo.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from config import MAX_FILE_PATCH_CHARS, MAX_TOTAL_PATCH_CHARS, PRS_DIR


@dataclass
class ChangedFile:
    filename: str
    status: str
    additions: int
    deletions: int
    patch: str | None
    truncated: bool = False

    @property
    def patch_block(self) -> str:
        if self.patch is None:
            return f"[patch omitted: no textual diff available for {self.filename}]"
        if self.truncated:
            return (
                f"{self.patch}\n"
                f"[patch abbreviated: {self.filename} exceeds the prompt budget — "
                f"read the file with read_file for the full contents]"
            )
        return self.patch


@dataclass
class ReviewTarget:
    slug: str
    repo: str
    number: int
    title: str
    body: str
    author: str
    base_ref: str
    head_ref: str
    head_sha: str
    html_url: str
    additions: int
    deletions: int
    changed_files_count: int
    files: list[ChangedFile] = field(default_factory=list)
    reader: Callable[[str], str] = field(repr=False, default=lambda path: f"error: no reader for {path}")

    def read_file(self, path: str) -> str:
        allowed = {f.filename for f in self.files}
        if path not in allowed:
            listed = ", ".join(sorted(allowed)) or "(none)"
            return (
                f"error: {path!r} is not in this PR's changed files. "
                f"Readable paths: {listed}"
            )
        return self.reader(path)


def apply_patch_budget(files: list[ChangedFile]) -> list[ChangedFile]:
    """Cap per-file and total patch size. Truncated files stay in the manifest."""
    budget = MAX_TOTAL_PATCH_CHARS
    out: list[ChangedFile] = []
    for item in files:
        patch = item.patch
        truncated = item.truncated
        if patch is not None:
            cap = min(MAX_FILE_PATCH_CHARS, budget)
            if cap <= 0:
                patch, truncated = None, False
            elif len(patch) > cap:
                patch, truncated = patch[:cap], True
            budget -= len(patch) if patch else 0
        out.append(replace(item, patch=patch, truncated=truncated))
    return out


def list_fixtures() -> list[str]:
    if not PRS_DIR.is_dir():
        return []
    return sorted(p.name for p in PRS_DIR.iterdir() if (p / "meta.json").is_file())


def load_fixture(name: str) -> ReviewTarget:
    root = PRS_DIR / name
    meta_path = root / "meta.json"
    if not meta_path.is_file():
        known = ", ".join(list_fixtures()) or "(none)"
        raise SystemExit(f"Unknown fixture {name!r}. Available: {known}")

    meta = json.loads(meta_path.read_text())
    workspace = (root / "workspace").resolve()
    files = [
        ChangedFile(
            filename=raw["filename"],
            status=raw.get("status", "modified"),
            additions=raw.get("additions", 0),
            deletions=raw.get("deletions", 0),
            patch=raw.get("patch"),
            truncated=bool(raw.get("truncated", False)),
        )
        for raw in meta.get("files", [])
    ]
    return target_from_meta(meta, apply_patch_budget(files), _fixture_reader(workspace))


def _fixture_reader(workspace: Path) -> Callable[[str], str]:
    def read(path: str) -> str:
        target = (workspace / path).resolve()
        if workspace not in target.parents and target != workspace:
            return f"error: {path!r} escapes the workspace"
        if not target.is_file():
            return f"error: {path} not found in workspace (deleted or missing at PR head)"
        return target.read_text()

    return read


def target_from_meta(
    meta: dict,
    files: list[ChangedFile],
    reader: Callable[[str], str],
) -> ReviewTarget:
    repo = meta["repo"]
    number = int(meta["number"])
    return ReviewTarget(
        slug=f"{repo}#{number}",
        repo=repo,
        number=number,
        title=meta.get("title", ""),
        body=meta.get("body") or "",
        author=meta.get("author", "unknown"),
        base_ref=meta.get("base_ref", "main"),
        head_ref=meta.get("head_ref", "head"),
        head_sha=meta.get("head_sha", "fixture"),
        html_url=meta.get("html_url", ""),
        additions=int(meta.get("additions", sum(f.additions for f in files))),
        deletions=int(meta.get("deletions", sum(f.deletions for f in files))),
        changed_files_count=int(meta.get("changed_files_count", len(files))),
        files=files,
        reader=reader,
    )
