"""Cross-cohort phenotype harmonisation skill."""

from biobank_agent.data.phenotype import (
    harmonize_phenotype,
    record_phenotype_to_action_graph,
)
from biobank_agent.registry import skill


@skill(
    name="phenotype_harmonize",
    description=(
        "Create an auditable UKB/HPP/CKB phenotype mapping for a disease or biomarker. "
        "Records code sources, field IDs, drift risks, and recommended validation checks."
    ),
    parameters={
        "concept": {
            "type": "string",
            "description": "Phenotype concept, ICD code, or biomarker alias (e.g. I21, E11, LDL, HbA1c)",
        },
        "phenotype_type": {
            "type": "string",
            "description": "'disease' or 'biomarker'. If omitted, the concept is inferred.",
            "default": "disease",
        },
        "banks": {
            "type": "string",
            "description": "Comma-separated bank IDs, default ukb,hpp,ckb",
            "default": "ukb,hpp,ckb",
        },
    },
    required=["concept"],
)
def phenotype_harmonize(
    concept: str,
    phenotype_type: str = "disease",
    banks: str = "ukb,hpp,ckb",
    *,
    ctx=None,
) -> dict:
    bank_ids = [b.strip().lower() for b in banks.split(",") if b.strip()]
    phenotype = harmonize_phenotype(concept, bank_ids, phenotype_type=phenotype_type)
    if ctx is not None:
        record_phenotype_to_action_graph(getattr(ctx, "memory", None), phenotype)
    return phenotype.to_dict()
