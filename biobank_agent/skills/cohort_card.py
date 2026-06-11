"""Agentic cohort card generation skill."""

from biobank_agent.data.cohort_cards import (
    build_cohort_card,
    record_cohort_card_to_action_graph,
)
from biobank_agent.registry import skill


@skill(
    name="cohort_card",
    description=(
        "Generate an auditable cohort design card for UKB/HPP/CKB workflows. "
        "Chooses case-control, incident, survival, target-trial, or trajectory design from the query."
    ),
    parameters={
        "query": {
            "type": "string",
            "description": "Research question or cohort construction intent",
        },
        "endpoint": {
            "type": "string",
            "description": "Endpoint or phenotype concept, e.g. I21 or acute_myocardial_infarction",
        },
        "banks": {
            "type": "string",
            "description": "Comma-separated bank IDs, default ukb,hpp,ckb",
            "default": "ukb,hpp,ckb",
        },
        "cohort_key": {
            "type": "string",
            "description": "Optional active cohort key from session state to attach counts/missingness",
            "default": "",
        },
    },
    required=["query", "endpoint"],
)
def cohort_card(
    query: str,
    endpoint: str,
    banks: str = "ukb,hpp,ckb",
    cohort_key: str = "",
    *,
    ctx=None,
) -> dict:
    bank_ids = [b.strip().lower() for b in banks.split(",") if b.strip()]
    cohort_df = None
    if ctx is not None and cohort_key:
        cohort_df = getattr(getattr(ctx, "state", None), "cohorts", {}).get(cohort_key)
    card = build_cohort_card(
        query=query,
        endpoint=endpoint,
        banks=bank_ids,
        cohort_df=cohort_df,
    )
    if ctx is not None:
        record_cohort_card_to_action_graph(getattr(ctx, "memory", None), card)
    return card.to_dict()
