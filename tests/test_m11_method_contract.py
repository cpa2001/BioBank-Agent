"""Tests for M11: MethodContract dataclass (omics reviewer + completion wiring tests added once Codex signs off)."""

from __future__ import annotations

from biobank_agent.runtime.method_contract import MethodContract


def test_roundtrips_through_dict():
    c = MethodContract(
        name="sc_qc_preprocess",
        summary="scRNA QC and normalization",
        inputs=["raw h5ad"],
        outputs=["filtered, normalized h5ad"],
        postconditions=["obs has n_genes_by_counts", "X is log1p-normalized"],
        statistical_assumptions=["doublets removed before clustering"],
        citation="doi:10.1000/example",
    )
    d = c.to_dict()
    assert d["name"] == "sc_qc_preprocess" and len(d["postconditions"]) == 2
    assert MethodContract.from_dict(d) == c


def test_from_dict_ignores_unknown_keys():
    c = MethodContract.from_dict({"name": "x", "bogus": 1, "outputs": ["y"]})
    assert c.name == "x" and c.outputs == ["y"]


def test_is_empty():
    assert MethodContract(name="x", citation="doi:..").is_empty()
    assert not MethodContract(name="x", outputs=["y"]).is_empty()


def test_as_text_includes_declared_sections():
    txt = MethodContract(
        name="rna_velocity",
        summary="estimate RNA velocity",
        postconditions=["requires spliced/unspliced layers"],
    ).as_text().lower()
    assert "velocity" in txt and "spliced" in txt and "postconditions" in txt


# ── Omics methodology checks + check_artifact ───────────────────────────────
from biobank_agent.runtime.methodology import review_methodology, methodology_blocks, check_artifact  # noqa: E402


def _issues(text: str) -> set:
    return {f["issue"] for f in review_methodology({}, text=text)}


def test_omics_pseudoreplicated_de_blocks():
    flags = review_methodology({}, text="scanpy single-cell differential expression treating each cell as an independent replicate across clusters")
    assert any(f["issue"] == "pseudoreplicated_de" and f["severity"] == "block" for f in flags)


def test_omics_velocity_requires_splicing():
    assert "velocity_without_splicing" in _issues("scanpy RNA velocity on the single-cell UMAP embedding")
    assert "velocity_without_splicing" not in _issues("scVelo RNA velocity using spliced/unspliced loom layers")


def test_omics_checks_do_not_fire_on_tabular_gwas():
    txt = "GWAS association: 200 SNP tests, BH-FDR corrected, odds ratio with 95% CI, PCA covariates."
    assert _issues(txt) == set()


def test_coloc_unharmonized_is_advisory_not_block():
    flags = review_methodology({}, text="colocalization analysis with coloc.abf across two GWAS summary-statistic datasets")
    coloc = [f for f in flags if f["issue"] == "coloc_unharmonized_alleles"]
    assert coloc and coloc[0]["severity"] == "advisory"
    assert methodology_blocks(coloc) == []


def test_check_artifact_flags_missing_postconditions():
    summary = {"n_obs": 100, "n_vars": 2000, "obs_keys": ["leiden"], "layers": []}
    issues = {f["issue"] for f in check_artifact(summary, min_obs=50, require_obs_keys=("leiden", "cell_type"), require_layers=("spliced",))}
    assert "artifact_missing_obs_key" in issues and "artifact_missing_layer" in issues


def test_check_artifact_passes_when_satisfied():
    # Pure dict checks — no scanpy/anndata import needed to validate the artifact shape.
    summary = {"n_obs": 100, "n_vars": 2000, "obs_keys": ["leiden", "cell_type"], "layers": ["spliced", "unspliced"]}
    assert check_artifact(summary, min_obs=50, require_obs_keys=("leiden", "cell_type"), require_layers=("spliced",)) == []


# ── Completion-gate wiring (default OFF; on only with the flag) ──────────────
import types as _t  # noqa: E402

from biobank_agent.runtime.completion import CompletionGate  # noqa: E402
from biobank_agent.runtime.engine import ProviderRouter  # noqa: E402
from biobank_agent.runtime.types import PlanState, PlanStep, ProviderResponse, RuntimeConfig  # noqa: E402


class _FakeProvider:
    def __init__(self, handler):
        self.handler = handler

    def complete(self, request):
        return ProviderResponse(text=self.handler(request), provider="fake", model=request.model or "m")


def _router(handler):
    cfg = RuntimeConfig(primary_model="m", planner_model="m", critic_model="m", summarizer_model="m", safety_reviewer_model="m")
    return ProviderRouter({"m": _FakeProvider(handler)}, cfg)


def _plan():
    return PlanState(objective="sc analysis", steps=[PlanStep(id="a", title="analyze", status="done", tool_scope=["python_exec"])])


_ACCEPT = lambda req: '{"accepted": true, "missing": [], "reasons": "done"}'  # noqa: E731
_OMICS_SIN_EV = "scanpy single-cell differential expression treating each cell as an independent replicate across clusters"


def test_completion_gate_off_by_default_accepts_omics_sin():
    gate = CompletionGate(_router(_ACCEPT))  # config None -> flag defaults off, no side effects
    a = gate.assess("sc", _plan(), evidence_summary=_OMICS_SIN_EV, report_present=False)
    assert a.accepted is True


def test_completion_gate_on_blocks_omics_sin():
    gate = CompletionGate(_router(_ACCEPT), _t.SimpleNamespace(methodology_gate_enabled=True))
    a = gate.assess("sc", _plan(), evidence_summary=_OMICS_SIN_EV, report_present=False)
    assert a.accepted is False
    assert "pseudoreplicated_de" in a.missing


def test_completion_gate_on_still_accepts_clean_tabular_goal():
    gate = CompletionGate(_router(_ACCEPT), _t.SimpleNamespace(methodology_gate_enabled=True))
    ev = "Tools run: vcf_qc:ok, vcf_association:ok. Reported odds ratio with 95% CI; BH-FDR applied across tests."
    a = gate.assess("gwas", _plan(), evidence_summary=ev, report_present=False)
    assert a.accepted is True
