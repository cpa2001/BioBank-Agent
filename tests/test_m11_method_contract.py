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
