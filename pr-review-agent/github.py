"""Read-only GitHub access. Every request is a GET. Nothing is posted."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request

from source import ChangedFile, ReviewTarget, apply_patch_budget, target_from_meta

API = "https://api.github.com"


def fetch_pull_request(repo: str, number: int, token: str | None = None, max_files: int = 40) -> ReviewTarget:
    """Fetch PR metadata + per-file patches, then wire read_file to the Contents API."""
    pr = _get(f"/repos/{repo}/pulls/{number}", token)
    files_raw: list[dict] = []
    page = 1
    while len(files_raw) < max_files:
        batch = _get(
            f"/repos/{repo}/pulls/{number}/files",
            token,
            params={"per_page": 100, "page": page},
        )
        if not isinstance(batch, list) or not batch:
            break
        files_raw.extend(batch)
        if len(batch) < 100:
            break
        page += 1

    files = [
        ChangedFile(
            filename=raw["filename"],
            status=raw.get("status", "modified"),
            additions=raw.get("additions", 0),
            deletions=raw.get("deletions", 0),
            patch=raw.get("patch"),
        )
        for raw in files_raw[:max_files]
    ]
    files = apply_patch_budget(files)
    head = pr.get("head") or {}
    base = pr.get("base") or {}
    head_sha = head.get("sha", "")
    meta = {
        "repo": repo,
        "number": number,
        "title": pr.get("title", ""),
        "body": pr.get("body") or "",
        "author": (pr.get("user") or {}).get("login", "unknown"),
        "base_ref": (base.get("ref") or "unknown"),
        "head_ref": (head.get("ref") or "unknown"),
        "head_sha": head_sha,
        "html_url": pr.get("html_url", ""),
        "additions": pr.get("additions", 0),
        "deletions": pr.get("deletions", 0),
        "changed_files_count": pr.get("changed_files", len(files_raw)),
    }
    return target_from_meta(meta, files, _github_reader(repo, files, head_sha, token))


def _github_reader(
    repo: str,
    files: list[ChangedFile],
    head_sha: str,
    token: str | None,
):
    by_name = {f.filename: f for f in files}

    def read(path: str) -> str:
        info = by_name.get(path)
        if info is not None and info.status == "removed":
            return f"error: {path} was deleted in this PR"
        return _get_file_text(repo, path, head_sha, token)

    return read


def _get_file_text(repo: str, path: str, ref: str, token: str | None) -> str:
    quoted = urllib.parse.quote(path, safe="/")
    try:
        data = _get(f"/repos/{repo}/contents/{quoted}", token, params={"ref": ref})
    except SystemExit as exc:
        return f"error: could not read {path}: {exc}"
    if isinstance(data, list):
        return f"error: {path} is a directory"
    encoding = data.get("encoding")
    content = data.get("content")
    if encoding == "base64" and isinstance(content, str):
        raw = base64.b64decode(content)
        return raw.decode("utf-8", errors="replace")
    if data.get("download_url"):
        return f"error: {path} is too large for the contents API; skipped"
    return f"error: unexpected contents payload for {path}"


def _get(path: str, token: str | None, params: dict | None = None) -> dict | list:
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "pr-review-agent",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:300]
        if exc.code == 404:
            raise SystemExit(
                f"{path} not found (HTTP 404). Check the repo/PR, and note that "
                f"private repos need GITHUB_TOKEN. {body}"
            ) from exc
        raise SystemExit(f"GitHub GET {path} failed: HTTP {exc.code}. {body}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"GitHub GET {path} failed: {exc}") from exc
