"""Code eval + judge behavior on rows shaped like eval_dataset.json's."""

from evaluators.code_evals import citation_valid
from evaluators.llm_judge import answer_correctness_judge


def test_citation_valid_passes_when_expected_doc_is_cited():
    output = {"answer": "... [span-kinds]", "citations": ["span-kinds"], "refused": False}
    assert citation_valid(output, {"expected_doc_id": "span-kinds"}).score == 1.0


def test_citation_valid_fails_when_expected_doc_is_missing():
    output = {"answer": "... [wrong-doc]", "citations": ["wrong-doc"], "refused": False}
    assert citation_valid(output, {"expected_doc_id": "span-kinds"}).score == 0.0


def test_citation_valid_passes_a_refusal_on_the_out_of_scope_row():
    output = {"answer": "not covered", "citations": [], "refused": True}
    assert citation_valid(output, {"expected_doc_id": None}).score == 1.0


def test_answer_correctness_judge_mock_passes_on_overlap():
    summary = "false completion no-progress partial completion constraint loss harmful success"
    output = {"answer": summary}
    assert answer_correctness_judge(output, {"expected_answer_summary": summary}).score == 1.0


def test_answer_correctness_judge_mock_fails_with_no_overlap():
    output = {"answer": "completely unrelated text about rockets"}
    result = answer_correctness_judge(output, {"expected_answer_summary": "false completion no-progress"})
    assert result.score == 0.0
