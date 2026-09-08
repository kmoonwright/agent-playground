"""Host agent: a bounded tool-calling loop that can delegate to the
librarian more than once per turn — for a genuinely compound question, one
search per sub-topic, not one search total — then builds a grounded answer
prompt, drafts an answer, and guardrail-checks it.
"""

import json
import re

from agents import librarian, llm, tools
from agents.prompts import ANSWER_SYSTEM, HOST_SYSTEM
from instrumentation import (
    AGENT,
    CHAIN,
    GUARDRAIL,
    PROMPT,
    set_attributes,
    set_output,
    traced_span,
    using_session_context,
)

# A backstop for clearly-irrelevant retrieval, not the primary defense against
# an uncovered-but-domain-adjacent question ("Arize pricing" against docs about
# Arize concepts) — real embeddings score those moderately high too. The
# ANSWER_SYSTEM prompt's own "say plainly it isn't covered" instruction is
# what actually catches that case; this only fires when retrieval found
# nothing plausibly relevant at all.
SCORE_THRESHOLD = 0.15
NOT_COVERED = "That isn't covered in these docs."

# Bounds how many times one turn can delegate to the librarian. HOST_SYSTEM
# is what decides whether a second hop is actually warranted; this is just
# the backstop against a runaway loop.
MAX_HOPS = 3


def _dedupe_excerpts(excerpts: list[dict]) -> list[dict]:
    """Keep the best-scoring occurrence of each (doc, section) across hops.

    Keying on doc_id alone would collapse two different sections of the
    same doc retrieved by two different hops down to just one of them —
    exactly the case a compound question spanning one doc's sections hits.
    """
    best: dict[tuple[str, str], dict] = {}
    for excerpt in excerpts:
        key = (excerpt["doc_id"], excerpt["heading"])
        if key not in best or excerpt["score"] > best[key]["score"]:
            best[key] = excerpt
    return list(best.values())


CITATION_PATTERN = re.compile(r"\[([a-z0-9-]+)\]")


def extract_citations(text: str) -> set[str]:
    return set(CITATION_PATTERN.findall(text))


def _build_prompt(question: str, excerpts: list[dict]) -> list[dict]:
    context = (
        "\n\n".join(f"[{e['doc_id']}] {e['heading']}\n{e['text']}" for e in excerpts)
        or "(no excerpts found)"
    )
    user = f"Question: {question}\n\nExcerpts:\n{context}"
    return [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": user}]


def _mock_answer(excerpts: list[dict]) -> str:
    if not excerpts:
        return NOT_COVERED
    top = excerpts[0]
    return f"{top['text'][:200]} [{top['doc_id']}]"


def _check_groundedness(draft: str, excerpts: list[dict], searched: bool) -> tuple[str, dict]:
    """Strip any citation that doesn't match a retrieved doc, and force the
    "not covered" answer if the best retrieval score was too low to trust."""
    valid_ids = {e["doc_id"] for e in excerpts}
    cited_ids = extract_citations(draft)
    fabricated = cited_ids - valid_ids

    answer = draft
    for bad_id in fabricated:
        answer = answer.replace(f"[{bad_id}]", "")

    best_score = max((e["score"] for e in excerpts), default=0.0)
    forced_refusal = searched and best_score < SCORE_THRESHOLD
    if forced_refusal:
        answer = NOT_COVERED

    attributes = {
        "guardrail.triggered": bool(fabricated) or forced_refusal,
        "guardrail.fabricated_citations": sorted(fabricated),
        "guardrail.best_score": best_score,
    }
    return answer, attributes


async def run(session, session_id: str, question: str) -> str:
    """session_id groups every turn of one conversation under one AX
    session — pass the same id across turns; a fresh one per call would put
    each turn in its own ungrouped session."""
    with using_session_context(session_id):
        with traced_span("turn", CHAIN, input_value=question):
            with traced_span("host", AGENT, input_value=question) as agent_span:
                messages = [
                    {"role": "system", "content": HOST_SYSTEM},
                    {"role": "user", "content": question},
                ]
                excerpts: list[dict] = []
                hop_queries: list[str] = []

                while len(hop_queries) < MAX_HOPS:
                    call = llm.decide_tool_call(
                        messages, tools.ASK_LIBRARIAN, mock_arguments={"query": question}
                    )
                    if call is None:
                        break
                    query = call.arguments.get("query", question)
                    hop_queries.append(query)
                    hop_excerpts = await librarian.run(session, session_id, query)
                    excerpts.extend(hop_excerpts)
                    # Real message threading, not just bookkeeping: the next
                    # decide_tool_call call sees this hop's results and
                    # decides for itself whether a further hop is warranted.
                    messages.append(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": call.id,
                                    "type": "function",
                                    "function": {
                                        "name": call.name,
                                        "arguments": json.dumps(call.arguments),
                                    },
                                }
                            ],
                        }
                    )
                    messages.append(
                        {"role": "tool", "tool_call_id": call.id, "content": json.dumps(hop_excerpts)}
                    )

                excerpts = _dedupe_excerpts(excerpts)

                with traced_span("build_answer_prompt", PROMPT, input_value=question) as prompt_span:
                    prompt_messages = _build_prompt(question, excerpts)
                    set_output(prompt_span, prompt_messages)

                draft = llm.draft_text(prompt_messages, mock_text=lambda: _mock_answer(excerpts))

                with traced_span("check_groundedness", GUARDRAIL, input_value=draft) as guardrail_span:
                    answer, guardrail_attrs = _check_groundedness(draft, excerpts, searched=bool(hop_queries))
                    set_attributes(guardrail_span, guardrail_attrs)
                    set_output(guardrail_span, answer)

                set_attributes(agent_span, {"hops": len(hop_queries), "hop_queries": hop_queries})
                set_output(agent_span, answer)
    return answer
