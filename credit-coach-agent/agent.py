"""A synthetic credit-coach agent.

A tool-calling loop against OpenAI, wrapped in one manual CHAIN span so tool
spans and the auto-instrumented OpenAI spans nest underneath it and read as
one coherent agent run in Arize AX.

The agent's identity is configurable via config.AGENT_NAME / config.COMPANY_NAME
(set AGENT_NAME / COMPANY_NAME in .env) so this demo can be re-skinned for any
customer without touching code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import config
import guardrails
import tools
import tracing

SYSTEM_PROMPT = f"""You are {config.agent_name()}, {config.company_name()}'s AI credit coach.

Always use a tool to answer questions about a specific user's credit report,
score, or disputable items -- never guess or fabricate a score, balance, or
dispute status. Use general_faq only for general "how does {config.company_name()} work"
questions that are not about a specific user's report. Keep replies short,
plain-English, and actionable.
"""


@dataclass
class AgentResult:
    response_text: str
    tool_calls: list[dict] = field(default_factory=list)
    trace_id: str = ""
    guardrails_triggered: list[str] = field(default_factory=list)


def run_agent(user_id: str, message: str, history: list[dict] | None = None) -> AgentResult:
    from openai import OpenAI

    client = OpenAI()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": message})

    tool_calls: list[dict] = []
    guardrails_triggered: list[str] = []

    with tracing.traced_span(
        "credit-coach-agent-turn",
        tracing.CHAIN,
        input_value=message,
        attributes={"user_id": user_id, "history_len": len(history or [])},
    ) as span:
        trace_id, _ = tracing.current_ids()
        response_text = ""

        input_check = guardrails.check_input(user_id, message)
        if input_check.triggered:
            guardrails_triggered.append("input_scope_guardrail")
            response_text = input_check.safe_response or ""
        else:
            for _ in range(config.MAX_ITERATIONS):
                completion = client.chat.completions.create(
                    model=config.MODEL,
                    messages=messages,
                    tools=tools.TOOLS,
                )
                choice = completion.choices[0].message
                messages.append(choice.model_dump(exclude_none=True))

                if not choice.tool_calls:
                    response_text = choice.content or ""
                    break

                for call in choice.tool_calls:
                    name = call.function.name
                    args = json.loads(call.function.arguments or "{}")
                    result = tools.dispatch(name, args)
                    tool_calls.append({"name": name, "args": args, "result": result})
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": result,
                        }
                    )

            output_check = guardrails.check_output(response_text, tool_calls)
            if output_check.triggered:
                guardrails_triggered.append("output_compliance_guardrail")
                response_text = output_check.safe_response or response_text

        tracing.set_output(span, response_text)
        span.set_attribute(
            "agent.tool_calls", json.dumps([tc["name"] for tc in tool_calls])
        )
        span.set_attribute("agent.guardrails_triggered", json.dumps(guardrails_triggered))

    return AgentResult(
        response_text=response_text,
        tool_calls=tool_calls,
        trace_id=trace_id,
        guardrails_triggered=guardrails_triggered,
    )


def main() -> None:
    import argparse
    import uuid

    parser = argparse.ArgumentParser(description="Ask the credit-coach agent a question, ad hoc.")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--message", required=True)
    args = parser.parse_args()

    tracing.ensure_tracing()
    session_id = f"cli-{args.user_id}-{uuid.uuid4().hex[:6]}"
    try:
        with tracing.using_session(session_id, extra={"user_id": args.user_id}):
            result = run_agent(args.user_id, args.message)
        print(f"{config.agent_name()}: {result.response_text}")
        if result.tool_calls:
            print("\nTool calls:")
            for tc in result.tool_calls:
                print(f"  - {tc['name']}({tc['args']}) -> {tc['result']}")
        if result.guardrails_triggered:
            print(f"\nguardrails triggered: {result.guardrails_triggered}")
        print(f"\ntrace_id: {result.trace_id}")
        print(f"session_id: {session_id}")
    finally:
        tracing.flush_tracing()


if __name__ == "__main__":
    main()
