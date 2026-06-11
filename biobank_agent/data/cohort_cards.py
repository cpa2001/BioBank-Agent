"""Agentic cohort construction cards for cross-biobank studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CohortCard:
    """Auditable cohort design summary."""

    cohort_id: str
    cohort_type: str
    endpoint: str
    banks: list[str]
    inclusion: list[str]
    exclusion: list[str]
    index_date: str
    lookback_window: str
    follow_up_window: str
    n_cases: int | None = None
    n_controls: int | None = None
    missingness: dict[str, float] = field(default_factory=dict)
    bias_flags: list[str] = field(default_factory=list)
    status: str = "PARTIAL"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_cohort_type(query: str) -> str:
    """Choose a cohort design from natural-language intent."""
    q = (query or "").lower()
    if "target-trial" in q or "target trial" in q or "intervention" in q:
        return "target_trial_emulation"
    if "trajectory" in q or "forecast" in q or "healthformer" in q:
        return "trajectory_prediction"
    if "survival" in q or "incident" in q or "time-to-event" in q:
        return "survival_cohort"
    if "prevalent" in q or "case-control" in q:
        return "prevalent_case_control"
    return "incident_cohort"


def build_cohort_card(
    *,
    query: str,
    endpoint: str,
    banks: list[str] | None = None,
    cohort_df: pd.DataFrame | None = None,
    cohort_type: str | None = None,
) -> CohortCard:
    """Create an auditable cohort design card."""
    banks = banks or ["ukb", "hpp", "ckb"]
    ctype = cohort_type or infer_cohort_type(query)
    inclusion, exclusion, index_date, lookback, followup = _defaults_for_type(ctype, endpoint)

    n_cases = None
    n_controls = None
    missingness: dict[str, float] = {}
    if cohort_df is not None and not cohort_df.empty:
        if "label" in cohort_df.columns:
            n_cases = int(cohort_df["label"].sum())
            n_controls = int((cohort_df["label"] == 0).sum())
        numeric = cohort_df.select_dtypes(include="number")
        if not numeric.empty:
            missingness = {
                str(col): round(float(cohort_df[col].isna().mean()), 6)
                for col in numeric.columns
                if col != "label"
            }

    flags = _bias_flags(ctype, banks=banks, n_cases=n_cases, missingness=missingness)
    status = "PASS" if n_cases is not None and n_cases >= 100 and not any("critical" in f for f in flags) else "PARTIAL"
    return CohortCard(
        cohort_id=f"{endpoint}:{ctype}:{'-'.join(banks)}",
        cohort_type=ctype,
        endpoint=endpoint,
        banks=banks,
        inclusion=inclusion,
        exclusion=exclusion,
        index_date=index_date,
        lookback_window=lookback,
        follow_up_window=followup,
        n_cases=n_cases,
        n_controls=n_controls,
        missingness=missingness,
        bias_flags=flags,
        status=status,
    )


def record_cohort_card_to_action_graph(memory: Any, card: CohortCard) -> None:
    """Persist cohort card into Action Graph when available."""
    if memory is None or not hasattr(memory, "upsert_node"):
        return
    memory.upsert_node("cohort_card", card.cohort_id, payload=card.to_dict(), score=1.0)
    memory.upsert_node("phenotype", card.endpoint, payload={"endpoint": card.endpoint}, score=0.7)
    memory.link_nodes(
        "cohort_card",
        card.cohort_id,
        "phenotype",
        card.endpoint,
        relation="defines_endpoint",
        weight=0.8,
        evidence={"source": "agentic_cohort_card"},
    )
    for bank in card.banks:
        memory.upsert_node("bank", bank, payload={"bank_id": bank}, score=0.6)
        memory.link_nodes("cohort_card", card.cohort_id, "bank", bank, relation="applies_to", weight=0.6)


def _defaults_for_type(cohort_type: str, endpoint: str) -> tuple[list[str], list[str], str, str, str]:
    common_inclusion = [f"valid participant identifier", f"available phenotype definition for {endpoint}"]
    common_exclusion = ["withdrawn participants", "invalid or conflicting identifiers"]
    if cohort_type == "prevalent_case_control":
        return (
            common_inclusion + ["case has endpoint before or at baseline; controls have no endpoint record"],
            common_exclusion + ["controls with endpoint in diagnoses/death registry"],
            "baseline assessment",
            "all records before baseline when available",
            "not required for prevalent design",
        )
    if cohort_type == "survival_cohort":
        return (
            common_inclusion + ["participants at risk at index date", "linked follow-up and death/disease outcome"],
            common_exclusion + ["outcome before index date for incident analyses"],
            "baseline or first eligible diagnosis date",
            ">= 1 year preferred for covariate ascertainment",
            "from index date to outcome, death, loss to follow-up, or administrative censoring",
        )
    if cohort_type == "target_trial_emulation":
        return (
            common_inclusion + ["eligible for intervention at time zero", "baseline covariates before treatment assignment"],
            common_exclusion + ["contraindication or prior exposure when defining new-user design"],
            "time zero defined by eligibility and treatment decision",
            "pre-specified baseline covariate window",
            "trial-aligned follow-up window with censoring rule",
        )
    if cohort_type == "trajectory_prediction":
        return (
            common_inclusion + ["at least one baseline measurement", "future target measurement or outcome window"],
            common_exclusion + ["target value measured before context window"],
            "first complete context window",
            "observed context sequence before prediction time",
            "future query horizon defined by target timestamp",
        )
    return (
        common_inclusion + ["free of endpoint at baseline", "linked longitudinal outcome records"],
        common_exclusion + ["pre-baseline endpoint"],
        "baseline assessment",
        "baseline covariate window",
        "incident endpoint follow-up window",
    )


def _bias_flags(
    cohort_type: str,
    *,
    banks: list[str],
    n_cases: int | None,
    missingness: dict[str, float],
) -> list[str]:
    flags = []
    if n_cases is None:
        flags.append("case count unavailable; execution required before confirmatory claims")
    elif n_cases < 100:
        flags.append("critical: n_cases < 100 guardrail")
    if any(v > 0.3 for v in missingness.values()):
        flags.append("high missingness in one or more covariates")
    if len(set(banks)) > 1:
        flags.append("cross-cohort phenotype drift must be audited")
    if cohort_type == "target_trial_emulation":
        flags.append("requires explicit confounding, adherence, censoring and positivity checks")
    if cohort_type == "survival_cohort":
        flags.append("immortal-time and informative-censoring checks required")
    return flags
