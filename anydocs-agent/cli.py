"""CLI + REPL entry point."""

import argparse
import asyncio
import uuid

import instrumentation
import mcp_client
from agents import host


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anydocs", description="Ask questions over a docs corpus, traced to Arize AX."
    )
    parser.add_argument(
        "--docs", help="path to a folder of markdown docs (default: the bundled Arize/OpenInference docs)"
    )
    parser.add_argument("--selftest", action="store_true", help="run one turn and exit")
    return parser


async def _selftest(docs_dir: str | None) -> None:
    session_id = str(uuid.uuid4())
    async with mcp_client.open_session(docs_dir) as session:
        answer = await host.run(session, session_id, "What are the eleven OpenInference span kinds?")
        print(f"selftest answer:\n{answer}")


async def _repl(docs_dir: str | None) -> None:
    # One session_id for the whole conversation, not one per turn — that's
    # what lets Arize AX group every turn together for session-level evals.
    session_id = str(uuid.uuid4())
    async with mcp_client.open_session(docs_dir) as session:
        print("anydocs-agent — ask a question about the bundled docs. /quit to exit.\n")
        while True:
            try:
                question = input("ask> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                continue
            if question in {"/quit", "/q", "quit", "exit"}:
                break
            answer = await host.run(session, session_id, question)
            print(f"\n{answer}\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    instrumentation.ensure_tracing()
    try:
        if args.selftest:
            asyncio.run(_selftest(args.docs))
        else:
            asyncio.run(_repl(args.docs))
    finally:
        instrumentation.flush_tracing()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
