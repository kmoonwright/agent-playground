"""Bridge to the Arize experiments SDK's result type."""


def evaluation_result(*, score: float, label: str, explanation: str):
    from arize.experiments import EvaluationResult

    return EvaluationResult(score=score, label=label, explanation=explanation)
