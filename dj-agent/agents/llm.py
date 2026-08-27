"""Provider shims and the agent loop, for both Anthropic and OpenAI.

The loop shape is the same for both and for all three roles: call the model, run
whatever tools it asked for, feed the results back, repeat until it stops asking
or the iteration budget runs out. The message plumbing differs enough between
the two SDKs that they get one clean implementation each rather than a leaky
abstraction over both.

Both SDKs are imported lazily, inside functions, because `instrumentation`
must patch them before first import or the LLM spans never appear.

Anthropic specifics worth stating outright, since each is easy to get wrong:

  * Adaptive thinking is left ON (the `thinking` parameter is omitted, which is
    the default on claude-opus-5). Disabling it can make the model emit a tool
    call as plain visible text -- the turn succeeds, the call silently never
    runs, and the bogus text poisons later turns. For a tool-calling agent that
    is the worst available failure mode, so latency is controlled with
    `output_config.effort` instead.
  * All tool_result blocks go back in ONE user message. Splitting them teaches
    the model to stop making parallel calls, which would defeat the concurrent
    find_tracks fan-out.
  * `response.content` is appended verbatim, so thinking and tool_use blocks
    survive the round trip.
  * No temperature / top_p / top_k -- they are rejected on claude-opus-5.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import config
import instrumentation as ins
import agents.prompt as prompts
import agents.tools as tools
from session import DJSession

MAX_TOKENS = 2048
EFFORT = "low"  # this is a live REPL; the set does not wait
HISTORY_LIMIT = 24


@dataclass
class AgentResult:
    text: str
    iterations: int
    terminal: Any = None
    refused: bool = False
    tool_calls: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- Anthropic


def run_anthropic_agent(
    *,
    role: str,
    system: str,
    messages: list[dict[str, Any]],
    dispatch: tools.Dispatch,
    max_iterations: int,
) -> AgentResult:
    import anthropic

    client = anthropic.Anthropic()
    model = config.resolve_model("claude", role)
    specs = tools.to_anthropic(tools.TOOLS_BY_ROLE[role])
    terminal_name = tools.TERMINAL_TOOL.get(role)

    text_out: list[str] = []
    nudged = False

    for step in range(1, max_iterations + 1):
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=specs,
            output_config={"effort": EFFORT},
            messages=messages,
        )

        # HTTP 200 with an empty or partial body. Check before reading content.
        if response.stop_reason == "refusal":
            return AgentResult(
                text="(the model declined that request)",
                iterations=step,
                refused=True,
                tool_calls=list(dispatch.calls),
            )

        # Verbatim: thinking and tool_use blocks must survive the round trip.
        messages.append({"role": "assistant", "content": response.content})

        said = "".join(b.text for b in response.content if b.type == "text").strip()
        if said:
            text_out.append(said)

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            if not nudged and terminal_name and dispatch.terminal is None:
                nudged = True
                messages.append({"role": "user", "content": prompts.NUDGE})
                continue
            break

        results = []
        for block in tool_uses:
            observation = dispatch.run(block.name, dict(block.input or {}))
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": observation,
                }
            )
        messages.append({"role": "user", "content": results})

        if dispatch.terminal is not None:
            break

    return AgentResult(
        text="\n".join(text_out).strip(),
        iterations=step,
        terminal=dispatch.terminal,
        tool_calls=list(dispatch.calls),
    )


# ---------------------------------------------------------------- OpenAI


def run_openai_agent(
    *,
    role: str,
    system: str,
    messages: list[dict[str, Any]],
    dispatch: tools.Dispatch,
    max_iterations: int,
) -> AgentResult:
    from openai import OpenAI

    client = OpenAI()
    model = config.resolve_model("openai", role)
    specs = tools.to_openai(tools.TOOLS_BY_ROLE[role])
    terminal_name = tools.TERMINAL_TOOL.get(role)

    if not messages or messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": system})

    text_out: list[str] = []
    nudged = False

    for step in range(1, max_iterations + 1):
        response = client.chat.completions.create(model=model, messages=messages, tools=specs)
        choice = response.choices[0].message
        messages.append(_openai_assistant(choice))

        if choice.content:
            text_out.append(choice.content.strip())

        if not choice.tool_calls:
            if not nudged and terminal_name and dispatch.terminal is None:
                nudged = True
                messages.append({"role": "user", "content": prompts.NUDGE})
                continue
            break

        for call in choice.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                observation = f"Could not parse your arguments as JSON: {exc}"
            else:
                observation = dispatch.run(call.function.name, args)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": observation}
            )

        if dispatch.terminal is not None:
            break

    return AgentResult(
        text="\n".join(text_out).strip(),
        iterations=step,
        terminal=dispatch.terminal,
        tool_calls=list(dispatch.calls),
    )


def _openai_assistant(choice) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": choice.content}
    if choice.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in choice.tool_calls
        ]
    return message


def run_agent(provider: str, **kwargs) -> AgentResult:
    if provider == "claude":
        return run_anthropic_agent(**kwargs)
    if provider == "openai":
        return run_openai_agent(**kwargs)
    raise ValueError(f"no agent runner for provider {provider!r}")


# ---------------------------------------------------------------- host brain


class LlmBrain:
    """The host agent. Delegates crate digging and transition planning."""

    def __init__(
        self,
        provider: str,
        session: DJSession,
        say: Callable[[str], None],
        *,
        max_iterations: int | None = None,
    ) -> None:
        self.name = provider
        self.provider = provider
        self.session = session
        self.say = say
        self.max_iterations = max_iterations or config.MAX_ITERATIONS["host"]
        self.history: list[dict[str, Any]] = []
        self._pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="subagent")

    def close(self) -> None:
        """Shut the sub-agent pool down.

        The workers are non-daemon, so without this a clean exit blocks in
        `threading._shutdown` until every in-flight sub-agent finishes.
        """
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _delegate(self, name: str, args: dict[str, Any]) -> str:
        # Imported here to break the llm <-> subagents cycle; subagents imports
        # this module at the top, so one of the two directions has to be lazy.
        import agents.subagents as subagents

        ctx = ins.capture_context()
        if name == "find_tracks":
            future = self._pool.submit(
                ins.in_thread, ctx, subagents.run_crate_digger, self.provider, self.session, args
            )
        else:
            future = self._pool.submit(
                ins.in_thread,
                ctx,
                subagents.run_transition_planner,
                self.provider,
                self.session,
                args,
            )
        try:
            return future.result(timeout=120)
        except Exception as exc:
            return f"{name} could not complete: {type(exc).__name__}: {exc}"

    def handle(self, utterance: str) -> None:
        with ins.using_set_context(self.session.set_id, {"brain": self.provider}):
            with ins.traced_span("dj.turn", ins.CHAIN, input_value=utterance) as turn:
                dispatch = tools.Dispatch(self.session, "host", delegate=self._delegate)

                # The state block is regenerated every turn, so it must not be
                # carried forward -- a stale deck snapshot in history is worse
                # than none. It is dropped by identity below rather than by a
                # marker key, so nothing extra is ever sent to the API.
                state_block = {"role": "user", "content": prompts.state_block(self.session)}
                messages = [
                    *self.history[-HISTORY_LIMIT:],
                    state_block,
                    {"role": "user", "content": utterance},
                ]

                with ins.traced_span(
                    "host",
                    ins.AGENT,
                    attributes={
                        "role": "host",
                        "model": config.resolve_model(self.provider, "host"),
                        "provider": self.provider,
                        "effort": EFFORT,
                    },
                ) as agent_span:
                    result = run_agent(
                        self.provider,
                        role="host",
                        system=prompts.HOST_SYSTEM,
                        messages=messages,
                        dispatch=dispatch,
                        max_iterations=self.max_iterations,
                    )
                    ins.set_attributes(
                        agent_span,
                        {
                            "iterations": result.iterations,
                            "tool_calls": result.tool_calls,
                            "refused": result.refused,
                        },
                    )
                    ins.set_output(agent_span, result.text or "(no text)")

                self.history = [
                    m for m in messages if m is not state_block and m.get("role") != "system"
                ]
                reply = result.text or "(queued)"
                self.say(f"[dj] {reply}")
                ins.set_output(turn, reply)
