"""Mixed evaluators: two deterministic code evals + one LLM-as-judge."""

from .code_evals import deduction_amount_valid, retailer_name_match
from .llm_judge import claim_validity_judge

EVALUATORS = [
    retailer_name_match,
    deduction_amount_valid,
    claim_validity_judge,
]

__all__ = [
    "EVALUATORS",
    "retailer_name_match",
    "deduction_amount_valid",
    "claim_validity_judge",
]
