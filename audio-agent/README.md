# audio-agent

A small audio-in demo, instrumented with OpenInference and traced to
[Arize AX](https://app.arize.com). It transcribes an audio clip, analyzes the
transcript, and replies to it -- built as a **LangGraph** graph rather than a
plain function pipeline, so the graph's own node structure shows up as spans
via `openinference-instrumentation-langchain`, with almost no manual span
code.

## Quick start

```bash
uv sync
cp .env.example .env                        # fill in keys, or leave blank to run stubbed
uv run python cli.py --selftest             # synthetic tone, no API keys needed
uv run python cli.py --make-sample          # write samples/some.wav (gitignored)
uv run python cli.py --file samples/some.wav
uv run python -m pytest -q                  # offline test suite
```

`--make-sample` writes a synthetic 440Hz tone so there's a real file to point
`--file` at. The `samples/` directory is gitignored.

## The span tree

```
CHAIN   audio-agent run
├── TOOL   load_audio        soundfile: duration_s, sample_rate, channels
├── LLM    transcribe        raw OpenAI whisper-1 call -- manual span (LangChain
│                            has no wrapper for the transcription endpoint).
│                            Carries audio.url (base64 data URI, capped by
│                            AUDIO_AGENT_MAX_ATTACH_BYTES), audio.mime_type,
│                            audio.transcript -- OpenInference's AudioAttributes,
│                            set by hand since no instrumentor populates them.
├── LLM    analyze           langchain chat model -- auto-instrumented -- JSON
│                            {sentiment, topic}
└── LLM    respond           langchain chat model -- auto-instrumented -- a
                             short reply to the transcript
```

`load_audio`/`transcribe`/`analyze`/`respond` are each their own LangGraph
node, so `LangChainInstrumentor` gives each one a span automatically. The
`analyze` and `respond` spans additionally nest a real `ChatOpenAI` /
`ChatAnthropic` call span underneath, since those are genuine LangChain
runnables -- `transcribe`'s Whisper call isn't, so it gets a hand-written
`traced_span` instead. That contrast (auto vs. manual instrumentation, and
a non-LangChain SDK call living inside a LangGraph node) is the interesting
part of this trace.

Every run shares one `session_id` via `instrumentation.using_run_context()`,
so all four spans -- manual and auto-instrumented alike -- group together in
Arize AX under one run.

## Graceful degradation

- No `ARIZE_SPACE_ID`/`ARIZE_API_KEY` -> tracing falls back to a local
  in-process provider (a warning prints once); everything still runs, it
  just doesn't export.
- `--selftest` (or a missing `OPENAI_API_KEY`) -> skips the real Whisper call
  and the two chat-model calls, using canned text instead -- same span shape
  either way, so the whole pipeline runs and traces with zero credentials.
  A synthetic 440Hz tone stands in for a real audio file.
- `--model` picks the chat model used by `analyze`/`respond` (see
  `config.CHAT_MODELS`); transcription always uses OpenAI's `whisper-1`.

### Audio attachment: unverified in the Arize AX UI

`audio.url`/`audio.mime_type`/`audio.transcript` are real, documented
`openinference-semantic-conventions` constants (`AudioAttributes`, mirroring
`ImageAttributes.IMAGE_URL`), but as of this writing no instrumentor
populates them and no doc confirms how Arize AX's UI renders them -- this
repo sets them by hand on the `transcribe` span on the theory that AX treats
`audio.url` like it treats `image.url`. Worth checking in the UI after a real
run; if it doesn't render, the attributes are still there to inspect as raw
span data.

## Layout

| File | Purpose |
|---|---|
| `config.py` | env accessors + the chat-model table; the only module that reads the environment |
| `instrumentation.py` | span-kind constants, tracing setup, manual span + audio-attachment helpers |
| `agent.py` | the LangGraph pipeline: `load_audio -> transcribe -> analyze -> respond` |
| `cli.py` | `--file` / `--selftest` / `--make-sample` entry point |
| `tests/test_agent.py` | offline tests via `--selftest`, incl. the audio-attachment span attributes |

## Deliberately is not

Not a multi-agent system, not a mic-recording tool (file input only), not a
production transcription service -- it's a short, readable trace to look at
in Arize AX.
