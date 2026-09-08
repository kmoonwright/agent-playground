"""Code eval — does the answer cite the doc it was supposed to (or refuse,
for the out-of-scope trap rows)?
"""

from evaluators.util import evaluation_result


def citation_valid(output, dataset_row=None, **_):
    dataset_row = dataset_row or {}
    citations = output.get("citations", []) if isinstance(output, dict) else []
    expected = dataset_row.get("expected_doc_id")

    if expected is None:
        passed = bool(output.get("refused")) if isinstance(output, dict) else False
        explanation = f"expected a refusal (out-of-scope question); refused={passed}"
    else:
        passed = expected in citations
        explanation = f"expected a citation of '{expected}'; got {citations}"

    return evaluation_result(score=1.0 if passed else 0.0, label="pass" if passed else "fail", explanation=explanation)
