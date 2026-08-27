# dj-agent

A live, traced, multi-agent DJ. You talk to it mid-set; it digs through
royalty-free music, beatmatches, and crossfades on the beat while you type.

```
dj> something with a vocal, a bit more upbeat
[dj] Taking it up to 126 and bringing in Solar Drift — dubby but it has the hook you wanted.
[loaded] Solar Drift — Nadia K — 124.0 BPM (grid confidence 0.91)
[transition] 8 bars (15.2s), starting in 0.42s on the next bar
[now playing] Solar Drift — Nadia K — 124.0 BPM
```

This is the third demo in `agent-playground`. The other two share one tracing
shape with different control flow — `pdf-agent` is a pipeline, `pr-review-agent`
is a single tool-calling agent. This one is a **live, interactive, multi-agent
system**: a host agent that delegates to sub-agents while realtime audio plays.

**Is:** a live REPL DJ with real beatmatched crossfades, pluggable music sources,
and an Arize AX trace of the whole agent hierarchy.
**Deliberately is not:** an EQ/filter/scratch mixer, a harmonic-key mixer, a
recommender, a re-host of anybody's audio, or a streaming service.

## Quick start

```bash
python3.14 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # optional: Arize + an LLM key

# 1. Prove the DSP works. No network, no audio device, no API key.
.venv/bin/python dj.py --selftest-mixer

# 2. Prove the span tree nests across threads. Also fully offline.
.venv/bin/python dj.py --trace-selftest

# 3. Play a set with no API key and no network at all.
.venv/bin/python dj.py --make-test-crate
.venv/bin/python dj.py --offline --source local --provider rule

# 4. Play a real set from ccMixter, driven by an LLM.
.venv/bin/python dj.py --seed-cache 6     # once; their TLS handshake is slow
.venv/bin/python dj.py
```

Nothing above requires credentials. Without Arize keys you get a local tracer and
a warning; without an LLM key you get the rule selector. The audio path is
identical in every case.

## How the mix works

Two decks. Deck A is playing; the conductor stages the next track on deck B and
crossfades on a bar boundary before deck A runs out.

**Beatmatching is varispeed** — the incoming track is resampled to the target
tempo, so its pitch moves with it, exactly like a turntable pitch fader. That is
cheap, it is adjustable live (so "drop to 100 BPM" can ramp the track that is
*already playing*), and it is what makes the whole thing feel like DJing rather
than rendering. The cost is a hard constraint: playback rate is clamped to
**±8%**, so a track more than 8% off the target tempo cannot be played, however
well it fits musically. That constraint is pushed all the way up into the search
tools, the transition guardrail, and the host's system prompt.

**Playback rate comes from the source's metadata BPM, not the beat tracker.** A
0.3% tempo error accumulates ~48 ms of drift over a 16-second crossfade, which
flams audibly. Producer-authored BPM is usually exact; librosa's float is not.
librosa supplies the *phase* — and the tempo only when the source has none.

**There is no dependable downbeat tracker**, so bars are assumed 4/4 and taken
every fourth detected beat. When metadata BPM and the beat tracker disagree
(`grid_confidence` near 0 — a real 120 BPM house track beat-tracks at 161.5), the
bar grid is rebuilt arithmetically from the trusted tempo, anchored to the first
detected onset. `/nudge ±20` fixes the rest by ear.

## The agents

```
CHAIN  dj.turn                     one user utterance
└─ AGENT  host                     claude-opus-5 — conversation and decisions
   ├─ AGENT crate_digger           claude-haiku-4-5 — searches, inspects, ranks
   ├─ AGENT transition_planner     claude-haiku-4-5 — crossfade length, cue, tempo
   │  └─ GUARDRAIL validate_plan   clamps the plan; the validator wins
   └─ (tool calls appear as the `tool_calls` attribute on this span)
CHAIN  track.load                  loader thread — one queued track
└─ CHAIN fetch_and_analyze         download or cache hit, then librosa
CHAIN  track.prime                 the opening track, before the stream opens
CHAIN  transition                  conductor thread — the actual blend
```

With `--provider rule` the host span is `rule_selector` instead, and there are
no sub-agents. Those ten names are the complete set the program emits.

Tool calls are **not** separate spans. Each agent span carries the ordered list
of tools it called as one attribute (`Dispatch.calls`), which keeps a 6-iteration
turn to one span per agent instead of a dozen. A sub-agent's own span is a child
of the host's, so the delegation boundary is still visible in the waterfall.

The expensive model hosts the conversation and makes decisions. The sub-agents
that do the reading and the arithmetic run on something cheap. Sub-agents cannot
touch the mixer, the queue, or the loader — they read the library and return
data; the host does the queueing. "Who can mutate playback" is exactly one agent.

The **planner is advisory**: its numbers pass through `schema.validate_plan`,
which owns the crossfade length, the cue point, and the one tempo both decks are
asked to play. A bad plan becomes a corrected plan with the corrections reported,
not a broken mix — and when the two tracks are further apart than ±8% allows, it
says so instead of reporting a match it cannot deliver. (`rate_a`/`rate_b` are
reported for the trace; only deck A is actually rate-ramped.)

**The LLM is never in the deadline path.** Deck feeding is deterministic and
lives on the conductor thread. If the queue runs empty the conductor *asks* the
brain for a track 45 seconds early and re-asks every 20 seconds. If nothing is
staged by the time deck A runs out, deck A fades out over 4 bars and the set
ends with a message — the audio callback is never stalled and never silent by
accident.

## Threads

| Thread | Blocks on | Owns |
|---|---|---|
| T1 REPL | `input()` | terminal |
| T2 brain | the utterance queue | LLM history, sub-agent pool |
| T2a–c pool | pool queue | one sub-agent invocation each |
| T3 loader | the load queue | network, cache, librosa |
| T4 audio callback | PortAudio | all deck state |
| T5 conductor | a 0.25s tick | scheduling, credits, transition spans |

Two rules make this safe, and both are stated at the top of the files that
enforce them:

1. **The audio callback never blocks.** No lock, no filesystem, no network, no
   librosa, no span, no unbounded allocation. Commands reach it through a bounded
   deque; state leaves through a single immutable-snapshot rebind; finished
   80 MB buffers leave through a retire queue so the *conductor* pays for the
   `free()`. `audio/mixer.py` imports nothing from `instrumentation.py`.
2. **OpenTelemetry context does not cross threads by itself.** Every hand-off
   carries a captured context through `propagate_to_thread`, or its spans become
   trace roots and the multi-agent waterfall stops being legible.
   `--trace-selftest` asserts this, including a negative control that confirms
   the helper is load-bearing.

## Sources

`ccmixter` and `local`, behind one `Source` protocol in `music/source.py`. Adding a
third means one new file in `music/` plus a name and a constructor branch in
`music/registry.py`.

**ccMixter** has producer-authored BPM in the payload and indexes it as 5-BPM
`bpm_XXX_YYY` tags, so a tempo range becomes a set of band queries. Two things
to know:

- Downloading audio requires **both** a browser-like `User-Agent` **and**
  `Referer: https://ccmixter.org/`. With only one you get a bare 403. Verified:
  no headers → 403, UA alone → 403, UA + Referer → 200.
- Their **TLS handshake takes 20–30 seconds** (TCP connect finishes in 0.16s).
  `requests` counts that against the *connect* timeout, so a normal 10s timeout
  fails every call. The session is reused so the cost is paid once per process,
  and every response is cached to disk for 6 hours. `robots.txt` disallows
  `/api/`, so requests are rate-limited to one per 1.5s.

**local** is any folder of audio. `--crate DIR`, BPM/title/artist from tags where
they exist, `bpm=None` where they don't so analysis computes the tempo. A local
file has **no license unless the crate ships a `LICENSE.json`** mapping filenames
to one; without it the track is `local (unverified)` and every credits line says
so.

## Licensing

Every track that actually starts sounding — not every track queued — is appended
to `cache/credits.jsonl` with its license, its page URL, and whether that license
could be verified. `/credits` shows the running log; ending the set writes
`out/SETLIST.md`, which flags unverified tracks explicitly. `cache/` is
gitignored; nothing is re-hosted.

## Configuration

Models are a table with overrides, resolved most-specific-first:
`--model-host` → `--model` → `DJ_MODEL_HOST` → `DJ_MODEL` → the built-in default.
No model string appears at a call site, and every LLM span carries the role and
the resolved model, so a swap is visible in the trace.

```bash
dj.py --provider openai                      # same tools, same spans
dj.py --model-crate-digger claude-sonnet-5   # spend more on digging
dj.py --model claude-opus-5                  # everything on one model
```

`--provider auto` (the default) picks Claude if `ANTHROPIC_API_KEY` is set, then
OpenAI, then the keyless rule selector.

## Commands

| | |
|---|---|
| *anything else* | talk to the DJ |
| `/status` | deck state, queue, target BPM, dropouts |
| `/credits` | what has played, with licenses |
| `/nudge ±ms` | shift deck B to fix a mis-phased bar grid by ear |
| `/quit` | end the set, write the setlist |

## Modes

| | |
|---|---|
| `--selftest-mixer` | offline DSP asserts: equal power, no clipping, beat lock, exact length. Writes `out/selftest.wav` to listen to. |
| `--trace-selftest` | asserts that OTel context survives a thread hand-off, with a negative control that confirms the helper is load-bearing. |
| `--render WAV --duration N` | render a set to a file with no audio device, for inspecting a transition in a waveform editor. |
| `--seed-cache N` | fetch and analyse N tracks, then exit. Do this once and every later set can run `--offline` — worth it given ccMixter's 20–30s handshake. |
| `--make-test-crate` | write synthetic CC0 tracks into the crate directory. |
| `--offline` | never touch the network. |

## Layout

Five files at the top level; everything else groups by subject. The 21-column
gutter and the fenced block match `pr-review-agent/README.md`.

```
dj.py                CLI, REPL, offline renderer
config.py            paths, audio constants, the per-role model table
schema.py            Pydantic records, tool-argument models, the plan guardrail
instrumentation.py   arize-otel + Anthropic/OpenAI instrumentors, thread propagation
session.py           DJSession: state, loader thread, conductor, tool implementations

audio/               the DSP. Imports nothing from instrumentation.
  mixer.py           decks, the audio callback, crossfade (read this first)
  analysis.py        decode, resample, normalize, beat-track

agents/              everything that talks to a model
  llm.py             the two agent loops + the host brain (read this)
  tools.py           provider-neutral tool specs, adapters, dispatch
  subagents.py       crate_digger + transition_planner
  brain.py           RuleBrain (keyless) + the brain thread
  prompt.py          three system prompts + the live state block

music/               where tracks come from and where they are kept
  source.py          the Source protocol, the Track record, tempo ranking
  ccmixter.py        the networked source
  local.py           the static local crate
  registry.py        which sources are enabled this run
  library.py         the on-disk cache, credits, the setlist

devtools/            reachable only as dj.py flags, never from a running set
  selftest.py        the DSP asserts + the cross-thread tracing assert
  testcrate.py       synthetic CC0 crate generation

tests/               125 offline tests
data/crate/          the local crate
cache/  out/         gitignored
```

Two of the project's function-local imports exist to break an import cycle, and
both say so where they sit: `music/library.py`'s `import audio.analysis` and
`agents/llm.py`'s `import agents.subagents`. The rest are deferred for cost —
librosa, numba, soundfile, and the two LLM SDKs (which `instrumentation` has to
patch before they are imported).

## Tests

```bash
.venv/bin/python -m pytest
```

All offline: no network, no audio device, no API key. Both agent loops are tested
against scripted fake clients, because the mechanics that are easy to get wrong
fail *silently* against the real API — batching every `tool_result` into one user
message (and, on the OpenAI side, deliberately *not* doing that), echoing
assistant content verbatim, never sending sampling parameters, and checking
`stop_reason == "refusal"` before reading `content`.

## Known limitations

- **Beat phase on real music.** Slow intros, non-4/4, and no downbeat tracker
  mean some transitions land off-phase. `grid_confidence` flags it, the
  arithmetic grid fallback contains it, and `/nudge` fixes it. Honest limitation,
  not a bug being chased.
- **Varispeed shifts pitch.** ±8% is roughly a semitone at the extreme.
- **ccMixter's catalog** skews remix and stem material rather than polished full
  tracks, and much of it is `-NC`. `lic=open` is the default.
- **Memory.** A 4-minute stereo float32 track is ~84 MB; up to three are in
  flight.
- **Track titles, artists, and tags are untrusted input.** Anyone can upload to
  ccMixter, and a crate is whatever files are in a folder; all three reach the
  model's context. With track ids constrained to `^[0-9a-f]{12}$` at the schema
  boundary, the whole blast radius of a prompt injection is: the model can be
  talked into playing the attacker's track, moving the tempo within ±8%, and
  running more searches. There is no shell, no `eval`, no `subprocess`, and no
  tool that takes a path or a URL. Two mitigations are deliberate: the search
  results handed to the model carry no URLs at all, and sub-agents cannot touch
  the mixer, the queue, or the loader. Text on its way to the terminal has
  control characters stripped by `config.sanitize`, so a crafted title cannot
  rewrite the scrollback to forge an attribution.
- **libsndfile and mutagen parse untrusted bytes.** That is inherent to decoding
  audio and is not sandboxed here. Size and duration caps bound the damage a
  malformed file can do to memory, not what a parser bug could do.
- **`license_verified` on a local crate means "this filename appears in
  `LICENSE.json`".** The sidecar is unsigned, so it is a statement of intent, not
  proof. Anything without one is labelled unverified everywhere it appears.
