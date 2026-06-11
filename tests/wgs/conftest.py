"""Shared fixtures for WGS pipeline tests — all mocked, no real VCFs."""

from __future__ import annotations

import os
import importlib.machinery
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


try:  # pragma: no cover - exercised only when the optional dependency exists
    import cyvcf2 as _cyvcf2  # noqa: F401
except ModuleNotFoundError:
    cyvcf2_stub = types.ModuleType("cyvcf2")
    cyvcf2_stub.__spec__ = importlib.machinery.ModuleSpec("cyvcf2", loader=None)

    def _missing_vcf(*args, **kwargs):
        raise ImportError("cyvcf2 is required for VCF parsing: pip install cyvcf2")

    cyvcf2_stub.VCF = _missing_vcf
    sys.modules.setdefault("cyvcf2", cyvcf2_stub)

# ---------------------------------------------------------------------------
# Sample manifest (28 samples: J=10, S=12, V=6)
# ---------------------------------------------------------------------------
_J_IDS = [f"J{i:03d}" for i in range(1, 11)]
_S_IDS = [f"S{i:03d}" for i in range(1, 13)]
_V_IDS = [f"V{i:03d}" for i in range(1, 7)]
ALL_SAMPLE_IDS = _J_IDS + _S_IDS + _V_IDS
ALL_GROUPS = ["J"] * 10 + ["S"] * 12 + ["V"] * 6


def _build_pheno_df() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": ALL_SAMPLE_IDS,
        "phenotype_group": ALL_GROUPS,
        "vcf_path": [f"/fake/vcf/{s}.vcf.gz" for s in ALL_SAMPLE_IDS],
    })


# ---------------------------------------------------------------------------
# ctx fixture
# ---------------------------------------------------------------------------
@pytest.fixture()
def mock_ctx(tmp_path):
    """Minimal ctx matching biobank_agent agent._build_ctx() shape."""
    pheno_df = _build_pheno_df()

    dm = SimpleNamespace(
        query=lambda sql, *a, **kw: pheno_df.copy(),
        subject_id_col="sample_id",
    )
    state = SimpleNamespace(custom_data={}, figures=[], cohorts={})
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    return SimpleNamespace(dm=dm, state=state, report_dir=report_dir)


# ---------------------------------------------------------------------------
# Fake cyvcf2 objects
# ---------------------------------------------------------------------------
class FakeVariant:
    """Drop-in replacement for a cyvcf2.Variant."""

    def __init__(
        self,
        chrom="chr22",
        pos=1000,
        ref="A",
        alt=None,
        qual=50.0,
        filt=None,
        gt_types=None,
        rsid=None,
        dp=30,
        gq=40,
        n_samples=5,
    ):
        self.CHROM = chrom
        self.POS = pos
        self.REF = ref
        self.ALT = alt if alt is not None else ["T"]
        self.QUAL = qual
        self.FILTER = filt
        self.ID = rsid
        self._dp = dp
        self._gq = gq
        if gt_types is not None:
            self.gt_types = np.asarray(gt_types, dtype=np.int8)
        else:
            self.gt_types = np.zeros(n_samples, dtype=np.int8)

    def format(self, tag):
        if tag == "DP":
            return np.array([[self._dp]] * len(self.gt_types))
        if tag == "GQ":
            return np.array([[self._gq]] * len(self.gt_types))
        raise KeyError(tag)


class FakeVCF:
    """Drop-in replacement for cyvcf2.VCF."""

    def __init__(self, samples=None, variants=None):
        self.samples = samples or ["S1", "S2", "S3", "S4", "S5"]
        self._variants = variants or []
        self._closed = False

    def __iter__(self):
        return iter(self._variants)

    def __call__(self, region=None):
        return iter(self._variants)

    def close(self):
        self._closed = True


def make_variants(n=100, n_samples=5, chrom="chr22", start_pos=16000000, seed=42):
    """Generate *n* fake SNP variants with mixed genotypes."""
    rng = np.random.default_rng(seed)
    transitions = [("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")]
    transversions = [("A", "T"), ("A", "C"), ("G", "T"), ("G", "C")]
    all_pairs = transitions + transversions
    variants = []
    for i in range(n):
        pair = all_pairs[i % len(all_pairs)]
        gt = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])
        variants.append(FakeVariant(
            chrom=chrom,
            pos=start_pos + i * 100,
            ref=pair[0],
            alt=[pair[1]],
            qual=float(rng.integers(20, 100)),
            filt=None,
            gt_types=gt,
            dp=int(rng.integers(10, 60)),
            gq=int(rng.integers(15, 60)),
            n_samples=n_samples,
        ))
    return variants


# ---------------------------------------------------------------------------
# Monkeypatch helpers
# ---------------------------------------------------------------------------
@pytest.fixture()
def patch_cyvcf2(monkeypatch):
    """Return a helper that patches cyvcf2.VCF to return a FakeVCF."""
    def _patch(target_module, fake_vcf):
        monkeypatch.setattr(target_module, "VCF", lambda path: fake_vcf)
    return _patch


@pytest.fixture()
def patch_merge_vcfs(monkeypatch, tmp_path):
    """Patch merge_vcfs to create an empty file and record calls."""
    calls = []

    def _fake_merge(vcf_paths, output_path, region=None, **kw):
        calls.append({"vcf_paths": vcf_paths, "output_path": str(output_path), "region": region})
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        return p

    monkeypatch.setattr("biobank_agent.utils.bcftools.merge_vcfs", _fake_merge)
    return calls


@pytest.fixture()
def patch_get_sample_vcf_paths(monkeypatch):
    """Patch get_sample_vcf_paths to return fake mapping."""
    mapping = {s: f"/fake/vcf/{s}.vcf.gz" for s in ALL_SAMPLE_IDS}

    def _fake(ctx):
        return dict(mapping)

    monkeypatch.setattr(
        "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths", _fake,
    )
    return mapping


@pytest.fixture()
def patch_build_genotype_matrix(monkeypatch):
    """Patch build_genotype_matrix with configurable output."""
    state = {"n_samples": 28, "n_variants": 500, "samples": list(ALL_SAMPLE_IDS)}

    def _fake(merged_vcf_path, maf_min=0.05, maf_max=0.95,
              max_variants=50000, snv_only=True, region=None):
        ns = state["n_samples"]
        nv = state["n_variants"]
        rng = np.random.default_rng(42)
        G = rng.choice([0, 1, 2], size=(ns, nv)).astype(np.float64)
        vids = [f"chr22:{16000000 + i * 100}:A:T" for i in range(nv)]
        return G, vids, state["samples"][:ns]

    monkeypatch.setattr(
        "biobank_agent.utils.vcf_genotypes.build_genotype_matrix", _fake,
    )
    return state


@pytest.fixture()
def patch_build_allele_counts(monkeypatch):
    """Patch build_allele_counts with configurable output."""
    records = []

    def _factory(n=50, chrom="chr22"):
        rng = np.random.default_rng(42)
        out = []
        for i in range(n):
            out.append({
                "chrom": chrom,
                "pos": 16000000 + i * 100,
                "ref": "A",
                "alt": "T",
                "rsid": f"rs{100000 + i}",
                "case_alt": int(rng.integers(0, 10)),
                "case_ref": int(rng.integers(10, 30)),
                "ctrl_alt": int(rng.integers(0, 10)),
                "ctrl_ref": int(rng.integers(10, 30)),
                "maf": round(float(rng.uniform(0.01, 0.4)), 4),
            })
        records.clear()
        records.extend(out)
        return out

    def _fake(merged_vcf_path, case_ids, ctrl_ids, maf_min=0.0,
              region=None, max_variants=0):
        if not records:
            _factory()
        return list(records)

    monkeypatch.setattr(
        "biobank_agent.utils.vcf_genotypes.build_allele_counts", _fake,
    )
    return _factory


@pytest.fixture()
def patch_filter_vcf(monkeypatch):
    """Patch filter_vcf to create an empty file."""
    calls = []

    def _fake(input_path, output_path, include_expr=None, exclude_expr=None, **kw):
        calls.append({"input": str(input_path), "output": str(output_path),
                       "include": include_expr, "exclude": exclude_expr})
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        return p

    monkeypatch.setattr("biobank_agent.utils.bcftools.filter_vcf", _fake)
    return calls
