"""CLI entry point."""

import argparse

import config
import instrumentation
from agent import make_sample_file, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-agent", description="Transcribe and respond to an audio clip, traced to Arize AX."
    )
    parser.add_argument("--file", help="path to a .wav/.mp3 file to transcribe")
    parser.add_argument(
        "--selftest", action="store_true", help="run against a synthetic in-memory tone, no API keys needed"
    )
    parser.add_argument("--model", choices=sorted(config.CHAT_MODELS), help="chat model for analyze/respond")
    parser.add_argument(
        "--make-sample",
        nargs="?",
        const="samples/some.wav",
        metavar="PATH",
        help="write a synthetic tone WAV to PATH (default samples/some.wav) and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.make_sample is not None:
        path = make_sample_file(args.make_sample)
        print(f"wrote sample audio to {path}")
        return 0

    if not args.file and not args.selftest:
        raise SystemExit("Pass --file <path> or --selftest.")

    instrumentation.ensure_tracing()
    try:
        result = run(args.file, selftest=args.selftest, model_name=args.model)
    finally:
        instrumentation.flush_tracing()

    print(f"transcript: {result['transcript']}")
    print(f"analysis:   {result['analysis']}")
    print(f"response:   {result['response']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
