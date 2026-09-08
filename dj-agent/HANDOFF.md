# dj-agent — handoff

Read this first in a new session, then `README.md` for the design rationale and
`audio/mixer.py` for the realtime core.

**State as of the last session:** 11 commits on `main`, working tree clean,
nothing pushed. 125 tests pass, pyflakes clean. A security and correctness sweep
just landed. Python 3.14.7 in `.venv`. ~7,900 lines across 23 modules and 10 test
files.

---

## What it is

An interactive live DJ: you type requests at a REPL while beatmatched audio keeps
playing through the speakers. A host agent (`claude-opus-5`) delegates crate
digging and transition planning to two cheap sub-agents (`claude-haiku-4-5`) on a
thread pool. Tracks come from ccMixter or a local crate behind one `Source`
protocol. Everything is traced to Arize AX with OpenInference span kinds.

The point of the demo is the thing most agent demos avoid: **an agent that is
deliberately kept out of the deadline path.** Deck feeding is deterministic and
lives on a conductor thread; the LLM is asked 45s early and re-asked every 20s,
and if no answer arrives the set fades out and says so.

## Run it

```bash
.venv/bin/python -m pytest -q                    # 125 tests, ~9s, no network/device/key
.venv/bin/python dj.py --selftest-mixer          # DSP asserts + writes out/selftest.wav
.venv/bin/python dj.py --trace-selftest          # OTel context across threads + negative control
.venv/bin/python dj.py --render out/set.wav --duration 200 --offline --source local --provider rule
.venv/bin/python dj.py --offline --source local --provider rule   # AUDIBLE live set
```

Expected numbers, unchanged since the sweep — treat any drift as a regression:

| Gate | Must print |
|---|---|
| `--selftest-mixer` | worst beat offset **0.068 ms**, RMS spread **1.09 dB**, peak **0.7374** |
| `--render ... --duration 200` | 3 transitions at **66.87 / 119.98 / 171.13s**, 0 underruns, 0 dropouts, 2.45 dB spread |
| live `/status` | `Audio dropouts: 0` |

**Ask the user before running anything audible.** The live REPL and any command
that opens a PortAudio stream play sound out of the speakers; that surprised them
once already. The `--render` and selftest paths are silent.

The crate (`data/crate/`, gitignored) already holds 5 synthetic CC0 tracks and
`cache/tracks/` holds 9 analysed tracks, so the offline paths work with no setup.
`dj.py --make-test-crate` regenerates the crate; `dj.py --seed-cache N` fills the
cache from the network.

## Layout

```
dj.py     503   argv, REPL, console printer, offline renderer. Only module that
                touches argv, the terminal, or PortAudio.
config.py 189   paths, audio constants, per-role model table, sanitize()
schema.py 302   Pydantic records, tool-arg validation, validate_plan guardrail
instrumentation.py 268   Arize/OTel, span kinds, propagate_to_thread
session.py 1092  DJSession: state, loader thread, conductor, every tool

audio/    mixer.py 523 (read first)   analysis.py 246
agents/   tools.py 416  llm.py 322  brain.py 231  subagents.py 214  prompt.py 98
music/    ccmixter.py 343  library.py 342  local.py 191  source.py 100  registry.py 54
devtools/ selftest.py 430  testcrate.py 137
tests/    10 files, 125 tests
```

Threads: T1 REPL, T2 brain + sub-agent pool, T3 loader, T4 audio callback,
T5 conductor.

## Rules that must not be broken

1. **The audio callback never blocks.** No lock, no I/O, no librosa, no span, no
   unbounded allocation in `Mixer.process` or anything it calls. Commands in via
   a bounded deque, state out via one immutable `Snapshot` rebind, finished ~80 MB
   buffers out via a retire queue so the conductor pays the `free()`.
   `audio/mixer.py` imports nothing from `instrumentation.py`.
2. **Deck state belongs to the callback thread.** Read `session.snapshot()`, never
   `mixer.a` / `mixer.b`, from any other thread.
3. **±8% varispeed** (`RATE_MIN=0.92`, `RATE_MAX=1.08`) is the project's hard
   constraint. It is pushed up into the search tools, the guardrail, the stage-time
   check, and the system prompt. Changing it changes all of those.
4. **Playback rate comes from metadata BPM, not librosa.** librosa supplies the
   *phase*, and the tempo only when the source has none. A 0.3% tempo error drifts
   ~48 ms over a 16s crossfade, which flams audibly.
5. **Bar alignment happens inside the callback** (`StartXfade(align_to_bar=True)`).
   The snapshot is published every 8 blocks, so a delay computed by the conductor
   is up to ~186 ms stale — that was measured as transitions landing up to 370 ms
   late before it was moved.

## Gotchas that already cost debugging time

- **ccMixter needs both a browser `User-Agent` and `Referer: https://ccmixter.org/`.**
  No headers → 403, UA alone → 403, both → 200. The 403 explains nothing.
- **ccMixter's TLS handshake takes 20–30s** while TCP connect finishes in 0.16s,
  and `requests` charges it to the *connect* timeout. A normal 10s timeout fails
  every call. Hence one reused `Session`, a long connect timeout, one retry, disk cache.
- **librosa beat-tracks a real 120 BPM track at 161.5.** `grid_confidence` catches
  the disagreement and `synthetic_bar_grid` rebuilds the grid arithmetically.
- **`Mixer.process()` returns a view into a reused scratch buffer.** Any offline
  caller keeping the result across calls must `.copy()` it. There is a test pinning
  this contract.
- **Equal-power assertions need band-limited random-phase sinusoids as probes.**
  Beatmatched clicks are perfectly correlated (+3 dB by amplitude addition) and
  white noise loses 1.76 dB to the resampler's lowpass. Both look like mixer bugs
  and are not.
- **`ANALYSIS_SCHEMA_VERSION` is 3.** Bumping it silently invalidates every cached
  `analysis.json`; the next run re-analyses (slow, librosa + numba JIT).
- **Anthropic: leave adaptive thinking on.** Disabling it can make Opus 5 emit a
  tool call as plain visible text. All `tool_result` blocks go in **one** user
  message; OpenAI deliberately does the opposite. Never send sampling parameters.
  Check `stop_reason == "refusal"` before reading content.
- **`--render` waits up to 60s per tick** while a load is in flight. With a crate
  full of broken files a 90s render can take 5+ minutes. Pre-existing, untouched.

## Verified vs not

**Verified this session:** the DSP numbers above; cross-thread OTel propagation,
with a negative control proving `propagate_to_thread` is load-bearing; a live
80s set through real PortAudio at 0 dropouts; the crossfade-overrun crash
(`operands could not be broadcast together with shapes (1024,2) (428,1)`)
reproduced and then guarded so that reverting *either* half of the fix fails the
test; path traversal, SSRF, terminal escapes, size caps; a corrupt crate file now
reporting and fading out instead of going silently dead.

**Not verified at all — say so rather than implying otherwise:**

- **The real LLM path.** No `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` exists on this
  machine and there is no `.env`. Both agent loops are tested only against scripted
  fake clients.
- **Arize export.** Only the graceful-degradation path (no credentials → local
  `TracerProvider` + warning) has ever run.

To close both: `cp .env.example .env`, fill in the keys, then
`dj.py --offline --source local --provider claude` and check the waterfall for one
`dj.turn` root with `host` under it and the sub-agent spans as its children. If
sub-agent spans appear as separate trace roots, context propagation broke.

## Open items, deliberately not done

Each was considered and declined with a reason — re-litigate only with new information.

- **`ValidatedPlan.rate_a` / `rate_b` are computed and unused.** Only deck A is
  rate-ramped. Wiring them changes what you hear and needs a listening test, not
  just the gates. The docstring says so.
- **No `TOOL` spans in production.** Tool calls are a `tool_calls` attribute on the
  parent AGENT span (that is what `Dispatch.calls` is for), which keeps a
  6-iteration turn to one span per agent. The README documents this rather than the
  program growing to match an earlier, wrong diagram.
- **The autopilot rule-selector fallback does not exist.** There is no timer and no
  fallback brain; the conductor re-asks and then fades out. Docs corrected to match.
- **`session.py` is 1092 lines and not split.** Only `validate_plan` was moved out
  (to `schema.py`, which made an existing docstring true).
- **libsndfile and mutagen parse untrusted bytes unsandboxed.** Inherent to decoding
  audio. Size and duration caps bound memory, not parser bugs.
- **The crate `LICENSE.json` is unsigned**, so `license_verified` on a local file
  means only "this filename appears in a sidecar". Documented as intentional.

## Conventions

- **Commits:** conventional style with a scope, matching the repo
  (`feat(dj-agent): ...`), subject plus a short body. **No authorship attributed
  to Claude** — no `Co-Authored-By`, no "Generated with", no mention of AI. Author
  is Kyle Moonwright <kmoonwright@outlook.com>. Commits land on `main`, as the
  other two demos did. Do not push unless asked.
- **Standing preferences for this repo** (`~/Dev/agent-playground`): Arize tracing
  is always wanted; models must be configurable via a table plus CLI/env overrides,
  never hardcoded at a call site; sub-agent orchestration is welcome, not
  over-engineering; sources should be pluggable with an offline local option.
- **Style:** dense, honest comments that explain *why* — especially where a
  constraint is non-obvious or a number was measured. State limitations plainly
  instead of hiding them. Prefer correcting a document over growing the program to
  match it.

## The 11 commits

```
10ecf00 chore(dj-agent): add project scaffolding and dependencies
c1f8921 feat(dj-agent): add config table and Arize AX tracing
bb0db34 feat(dj-agent): add validated schema and the transition guardrail
940e119 feat(dj-agent): add pluggable sources and the on-disk cache
8d09846 feat(dj-agent): add the varispeed mixer and beat analysis
d2a1032 feat(dj-agent): add the session state machine, loader, and conductor
fb6ab0a feat(dj-agent): add the host agent and two sub-agents
2fe9ed8 feat(dj-agent): add the CLI, REPL, and offline selftests
3390dd5 test(dj-agent): add 125 offline tests
83bec5a docs(dj-agent): document the design, tracing, and limitations
afe345c docs: add dj-agent to the demo index
```

Ordered by the import graph, so each commit references only files already present
(`music/` before `audio/`, because `audio/analysis.py` imports `music.library` at
module level while `music/library.py` defers its `import audio.analysis` into a
function). `pytest` only passes from `3390dd5` onward.
