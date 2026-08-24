"""Review prompt. Short on purpose — a long skill will outvote a one-line instruction.

The workshop's code-review skill is ~15k characters and tells the model to write a
GitHub review body. That is why their agent often "submitted" in chat and never
called the tool. We keep delivery rules here, at the top, and make submit_review
the only output that counts.
"""

from __future__ import annotations

from source import ReviewTarget

SYSTEM_PROMPT = """You are a read-only code review agent. You cannot post to GitHub or modify code.

The review is only delivered by calling the submit_review tool. Writing it as a chat
message does nothing — it is discarded.

Standards:
- Flag real bugs, security issues, and breaking changes. Do not manufacture nits for a clean PR.
- Only set `line` on a finding after you verified it by calling read_file. Do not count lines from a diff hunk header.
- An empty findings list is correct for a genuinely clean PR.
- A file in the Files Changed manifest is in the PR even if its patch is abbreviated or omitted. Read the file; do not call it missing.

Call submit_review exactly once when you are done.
"""

USER_PROMPT = """Review the pull request below and identify issues that need to be addressed.

## Pull Request

- **Title**: {title}
- **Description**: {body}
- **Repository**: {repo}
- **Author**: {author}
- **Base**: {base_ref}  **Head**: {head_ref}
- **PR**: {number}  **SHA**: {head_sha}
- **Size**: +{additions} / -{deletions} across {changed_files_count} files

{files_manifest}
## Patches

Individual patches may be **abbreviated** (`[patch abbreviated: ...]`) or **omitted**
(`[patch omitted: ...]`). Files in the manifest whose patch is missing or short are
still in the PR — call read_file to inspect them.

```diff
{diff}
```

Analyze the changes (read files to verify line numbers), then call submit_review.
That tool call is the only output that counts.
"""

NUDGE = """You ended your turn without calling submit_review, so no review has been recorded.
Anything you wrote as a chat message was discarded.

Call submit_review now with the review you already worked out: taste_rating, verdict,
risk_level, key_insight, findings, and review_markdown. Only set line if you verified
it with read_file. Do not investigate further.
"""


def format_files_manifest(pr: ReviewTarget) -> str:
    """Every changed file, never abbreviated. Lets the agent tell 'not in the PR'
    apart from 'patch got cut'.
    """
    if not pr.files:
        return "## Files Changed\n\n_(no files reported)_\n"

    additions = sum(f.additions for f in pr.files)
    deletions = sum(f.deletions for f in pr.files)
    lines = [
        f"## Files Changed ({len(pr.files)} files, +{additions} / -{deletions})",
        "",
        (
            "All files in the PR are listed here. If a file's patch is missing or "
            "abbreviated below, call read_file rather than treating it as absent."
        ),
        "",
    ]
    for item in pr.files:
        lines.append(
            f"- `{item.filename}` [{item.status}] +{item.additions} / -{item.deletions}"
        )
    return "\n".join(lines) + "\n"


def format_patches(pr: ReviewTarget) -> str:
    blocks = [f"--- {item.filename} ---\n{item.patch_block}" for item in pr.files]
    return "\n\n".join(blocks) if blocks else "(no patches available)"


def build_prompt(pr: ReviewTarget) -> str:
    return USER_PROMPT.format(
        title=pr.title,
        body=pr.body.strip() or "No description provided",
        repo=pr.repo,
        author=pr.author,
        base_ref=pr.base_ref,
        head_ref=pr.head_ref,
        number=pr.number,
        head_sha=pr.head_sha,
        additions=pr.additions,
        deletions=pr.deletions,
        changed_files_count=pr.changed_files_count,
        files_manifest=format_files_manifest(pr),
        diff=format_patches(pr),
    )
