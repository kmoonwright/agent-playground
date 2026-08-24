"""Read-only PR review agent. Reviews a fixture or a public GitHub PR.

    python review.py --fixture buggy-auth
    python review.py --fixture buggy-auth --max-iterations 2
    python review.py --repo owner/name --pr 123
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import DEFAULT_MAX_ITERATIONS, OUT_DIR, github_token, openai_model
from source import ReviewTarget, list_fixtures, load_fixture


def parse_args() -> argparse.Namespace:
    fixtures = list_fixtures()
    parser = argparse.ArgumentParser(
        description="Read-only PR review agent. Reviews a local fixture or a public GitHub PR."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--fixture",
        choices=fixtures or None,
        metavar="NAME",
        help="local PR under data/prs/ (%s)" % (", ".join(fixtures) or "none found"),
    )
    source.add_argument("--repo", help="owner/name of a public GitHub repository")
    parser.add_argument("--pr", type=int, help="pull request number (with --repo)")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"agent step budget (default {DEFAULT_MAX_ITERATIONS}); try 2 to starve it",
    )
    parser.add_argument("--model", default=None, help="OpenAI model; defaults to $OPENAI_MODEL")
    parser.add_argument("--out", default=None, help="artifact path; defaults to out/<slug>.json")
    args = parser.parse_args()
    if args.repo and not args.pr:
        parser.error("--pr is required with --repo")
    if args.pr and not args.repo:
        parser.error("--repo is required with --pr")
    return args


def load_target(args: argparse.Namespace) -> ReviewTarget:
    if args.fixture:
        return load_fixture(args.fixture)
    from github import fetch_pull_request

    return fetch_pull_request(args.repo, args.pr, token=github_token())


def artifact_path(args: argparse.Namespace, target: ReviewTarget) -> Path:
    if args.out:
        return Path(args.out)
    slug = target.slug.replace("/", "__").replace("#", "__")
    return OUT_DIR / f"{slug}.json"


def report(submitted, artifact: Path) -> int:
    print("\n" + "=" * 72)
    if submitted is None:
        print("The agent finished without calling submit_review.")
        print("That is itself a failure mode — look at the CHAIN span in Arize.")
        print("=" * 72)
        return 1

    unverified = sum(1 for f in submitted.findings if f.line is None)
    print(f"Taste rating : {submitted.taste_rating}")
    print(f"Verdict      : {submitted.verdict}")
    print(f"Risk level   : {submitted.risk_level}")
    print(f"Findings     : {len(submitted.findings)} ({unverified} without a line number)")
    print(f"Key insight  : {submitted.key_insight}")
    print(f"\nArtifacts    : {artifact}")
    print(f"               {artifact.with_suffix('.md')}")
    print("=" * 72)
    return 0


def main() -> int:
    args = parse_args()
    target = load_target(args)
    out = artifact_path(args, target)

    print(f"Reviewing {target.slug}: {target.title}")
    if target.html_url:
        print(f"  {target.html_url}")
    print(
        f"  +{target.additions} / -{target.deletions} "
        f"across {len(target.files)} files"
    )
    print(f"  model={args.model or openai_model()}  max_iterations={args.max_iterations}\n")

    from agent import run_review

    submitted = run_review(
        target,
        artifact_path=out,
        max_iterations=args.max_iterations,
        model=args.model,
    )
    return report(submitted, out)


if __name__ == "__main__":
    raise SystemExit(main())
