"""LLM-as-judge — is the answer consistent with the expected summary? The
judge call itself is traced as an EVALUATOR-kind span, distinct from the
pipeline's own LLM/AGENT/GUARDRAIL spans.
"""

import json

import config
from evaluators.util import evaluation_result
from instrumentation import EVALUATOR, set_output, traced_span
from schema import AnswerJudgeResult

JUDGE_SYSTEM = """You are grading a documentation Q&A answer. Given the question, the
expected-answer summary, and the actual answer, decide whether the actual answer is
consistent with the expected summary. Respond with JSON: {"correct": bool, "explanation": str}."""


def _judge_openai(question: str, expected_summary: str, answer: str) -> AnswerJudgeResult:
    from openai import OpenAI

    client = OpenAI(api_key=config.openai_api_key())
    response = client.chat.completions.create(
        model=config.openai_model(),
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": f"Question: {question}\nExpected summary: {expected_summary}\nActual answer: {answer}",
            },
        ],
        response_format={"type": "json_object"},
    )
    return AnswerJudgeResult.model_validate(json.loads(response.choices[0].message.content))


def _judge_mock(expected_summary: str, answer: str) -> AnswerJudgeResult:
    overlap = set(expected_summary.lower().split()) & set(answer.lower().split())
    correct = len(overlap) >= 3
    return AnswerJudgeResult(correct=correct, explanation=f"(mock) {len(overlap)} overlapping words")


def answer_correctness_judge(output, dataset_row=None, **_):
    dataset_row = dataset_row or {}
    question = dataset_row.get("question", "")
    expected_summary = dataset_row.get("expected_answer_summary", "")
    answer = output.get("answer", "") if isinstance(output, dict) else str(output)

    with traced_span(
        "answer_correctness_judge",
        EVALUATOR,
        input_value=json.dumps({"question": question, "expected_summary": expected_summary, "answer": answer}),
    ) as span:
        judge = (
            _judge_openai(question, expected_summary, answer)
            if config.openai_api_key()
            else _judge_mock(expected_summary, answer)
        )
        set_output(span, {"correct": judge.correct, "explanation": judge.explanation})

    return evaluation_result(
        score=1.0 if judge.correct else 0.0,
        label="pass" if judge.correct else "fail",
        explanation=judge.explanation,
    )
