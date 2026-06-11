"""里程碑4: bio-research safety guardrails — new methodology families + a reachable gate.

Pure + deterministic. Each new family is checked positive and clean; a conservatism test pins
that a correct GWAS narrative trips none of them; gate tests prove the (previously dead)
methodology_gate_enabled flag now reaches CompletionGate and enforces.
"""

from __future__ import annotations

from types import SimpleNamespace

from biobank_agent.runtime.completion import CompletionGate
from biobank_agent.runtime.engine import ProviderRouter
from biobank_agent.runtime.methodology import review_methodology
from biobank_agent.runtime.types import PlanState, PlanStep, ProviderResponse, RuntimeConfig


def _issues(text: str = "", payload=None) -> set[str]:
    return {f["issue"] for f in review_methodology(payload or {}, text=text)}


# ── new check families: positive + clean ─────────────────────────────────────

def test_causal_overreach_flags_unhedged_causal_claim_on_observational():
    assert "causal_overreach" in _issues(
        "In this cross-sectional GWAS, the variant causes disease and leads to higher risk.")


def test_causal_overreach_silent_on_hedged_association():
    assert "causal_overreach" not in _issues(
        "In this cross-sectional GWAS, the variant is associated with disease risk.")


def test_causal_overreach_silent_when_causal_design_present():
    assert "causal_overreach" not in _issues(
        "Two-sample Mendelian randomization shows the exposure causes the outcome.")


def test_phenotype_encoding_flags_binary_with_linear():
    assert "phenotype_encoding_mismatch" in _issues(
        "We analyzed the case-control outcome with linear regression.")


def test_phenotype_encoding_silent_on_logistic_for_binary():
    assert "phenotype_encoding_mismatch" not in _issues(
        "We analyzed the case-control outcome with logistic regression and report odds ratios.")


def test_missingness_flags_complete_case_without_mitigation():
    assert "missingness_or_selection_bias" in _issues(
        "We used a complete-case analysis and excluded participants with missing covariates.")


def test_missingness_silent_with_imputation():
    assert "missingness_or_selection_bias" not in _issues(
        "Missing covariates were handled with multiple imputation before modelling.")


def test_uncontrolled_confounding_flags_crude_association():
    assert "uncontrolled_confounding" in _issues(
        "We report the crude, unadjusted association between exposure and outcome.")


def test_uncontrolled_confounding_silent_when_adjusted():
    assert "uncontrolled_confounding" not in _issues(
        "We report the association adjusted for age, sex, and principal components.")


def test_covariate_leakage_blocks_outcome_in_covariates():
    flags = review_methodology({"outcome": "T2D", "covariates": ["age", "sex", "T2D"]})
    leak = [f for f in flags if f["issue"] == "covariate_leakage"]
    assert leak and leak[0]["severity"] == "block"


def test_covariate_leakage_silent_when_clean():
    assert "covariate_leakage" not in {
        f["issue"] for f in review_methodology({"outcome": "T2D", "covariates": ["age", "sex", "bmi"]})}


def test_clean_corrected_gwas_narrative_trips_no_new_checks():
    narrative = (
        "We ran a GWAS, adjusted for age, sex and 10 ancestry principal components, applied "
        "Benjamini-Hochberg FDR correction, and report odds ratios with 95% confidence intervals. "
        "Variants were associated with the case-control phenotype via logistic regression."
    )
    new = {"causal_overreach", "phenotype_encoding_mismatch", "missingness_or_selection_bias",
           "uncontrolled_confounding", "covariate_leakage"}
    assert _issues(narrative).isdisjoint(new)


# ── gate reachability / rollout ──────────────────────────────────────────────

class _FakeProvider:
    def complete(self, request):
        return ProviderResponse(
            text='{"accepted": true, "missing": [], "reasons": "done", "checked": true}',
            provider="fake", model=request.model or "m")


def _router() -> ProviderRouter:
    cfg = RuntimeConfig(primary_model="m", planner_model="m", critic_model="m",
                        summarizer_model="m", safety_reviewer_model="m")
    return ProviderRouter({"m": _FakeProvider()}, cfg)


def _plan() -> PlanState:
    return PlanState(objective="sc analysis",
                     steps=[PlanStep(id="a", title="Analyze", status="done", tool_scope=["x"])])


# Omics velocity claim with no spliced/unspliced counts -> a block-severity methodology sin.
_SIN = "Single-cell RNA velocity was computed from the neighbor graph to infer the trajectory."


def test_methodology_gate_blocks_when_enabled():
    gate = CompletionGate(_router(), RuntimeConfig(methodology_gate_enabled=True))
    assessment = gate.assess("sc analysis", _plan(), evidence_summary=_SIN, report_present=False)
    assert assessment.accepted is False
    assert any("velocity" in m for m in assessment.missing)


def test_methodology_gate_silent_when_disabled():
    gate = CompletionGate(_router(), RuntimeConfig(methodology_gate_enabled=False))
    assessment = gate.assess("sc analysis", _plan(), evidence_summary=_SIN, report_present=False)
    assert assessment.accepted is True


def test_runtime_config_carries_and_maps_methodology_flag():
    assert RuntimeConfig().methodology_gate_enabled is False
    mapped = RuntimeConfig.from_settings(SimpleNamespace(methodology_gate_enabled=True))
    assert mapped.methodology_gate_enabled is True
