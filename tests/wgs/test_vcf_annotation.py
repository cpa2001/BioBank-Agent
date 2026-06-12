"""Comprehensive tests for biobank_agent/skills/vcf_annotation.py — all I/O mocked."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.wgs.conftest import ALL_SAMPLE_IDS, FakeVCF, FakeVariant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(tmp_path, custom_data=None):
    """Build a minimal ctx object mirroring agent._build_ctx() shape."""
    import pandas as pd

    pheno_df = pd.DataFrame({
        "sample_id": ALL_SAMPLE_IDS,
        "vcf_path": [f"/fake/vcf/{s}.vcf.gz" for s in ALL_SAMPLE_IDS],
    })
    dm = SimpleNamespace(
        query=lambda sql, *a, **kw: pheno_df.copy(),
        subject_id_col="sample_id",
    )
    state = SimpleNamespace(
        custom_data=custom_data if custom_data is not None else {},
        figures=[],
        cohorts={},
    )
    report_dir = tmp_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(dm=dm, state=state, report_dir=report_dir)


def _patch_infra(monkeypatch, tmp_path, raise_merge=False):
    """Patch merge_vcfs, get_tmp_dir, and get_sample_vcf_paths."""
    sample_map = {s: f"/fake/vcf/{s}.vcf.gz" for s in ALL_SAMPLE_IDS}

    monkeypatch.setattr(
        "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
        lambda ctx: dict(sample_map),
    )

    tmp_dir = tmp_path / "vcf_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def _fake_get_tmp_dir(ctx):
        return tmp_dir

    monkeypatch.setattr(
        "biobank_agent.utils.bcftools.get_tmp_dir",
        _fake_get_tmp_dir,
    )

    def _fake_merge(vcf_paths, output_path, region=None, **kw):
        if raise_merge:
            raise RuntimeError("bcftools merge failed")
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        return p

    monkeypatch.setattr(
        "biobank_agent.utils.bcftools.merge_vcfs",
        _fake_merge,
    )

    return sample_map


def _noop_query(vcf_path, gene, chrom, start, end, min_qual=0.0, max_variants=10000):
    """_query_gene_variants stub that returns no hits."""
    return []


def _single_hit_query(vcf_path, gene, chrom, start, end, min_qual=0.0, max_variants=10000):
    """_query_gene_variants stub that returns one SNV hit per gene."""
    return [{
        "chrom": chrom,
        "pos": start + 1000,
        "ref": "A",
        "alt": "T",
        "rsid": ".",
        "qual": 55.0,
        "gene": gene,
        "n_carriers": 3,
        "carrier_samples": ["J001", "J002", "J003"],
        "gt_types": [0, 1, 0, 2, 0],
    }]


def _indel_hit_query(vcf_path, gene, chrom, start, end, min_qual=0.0, max_variants=10000):
    """Returns one indel hit (REF length > 1)."""
    return [{
        "chrom": chrom,
        "pos": start + 500,
        "ref": "ACGT",
        "alt": "A",
        "rsid": ".",
        "qual": 40.0,
        "gene": gene,
        "n_carriers": 1,
        "carrier_samples": ["S001"],
        "gt_types": [0, 1, 0, 0, 0],
    }]


# ---------------------------------------------------------------------------
# Main skill tests
# ---------------------------------------------------------------------------

class TestDefaultGenesAllVitiligo:
    """test_default_genes_all_vitiligo — no genes param → all 20 queried."""

    def test_default_genes_all_vitiligo(self, monkeypatch, tmp_path):
        ctx = _make_ctx(tmp_path)
        _patch_infra(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "biobank_agent.skills.vcf_annotation._query_gene_variants", _noop_query
        )

        from biobank_agent.skills.vcf_annotation import vcf_annotation, VITILIGO_GENES_HG38

        result = vcf_annotation(genes="", ctx=ctx)

        assert "error" not in result
        assert result["n_genes_queried"] == len(VITILIGO_GENES_HG38)
        assert result["n_genes_queried"] == 20
        assert result["unknown_genes"] == []


class TestCustomGeneList:
    """test_custom_gene_list — genes='TYR,OCA2' → only 2 genes queried."""

    def test_custom_gene_list(self, monkeypatch, tmp_path):
        ctx = _make_ctx(tmp_path)
        _patch_infra(monkeypatch, tmp_path)

        queried = []

        def _tracking_query(vcf_path, gene, chrom, start, end, **kw):
            queried.append(gene)
            return []

        monkeypatch.setattr(
            "biobank_agent.skills.vcf_annotation._query_gene_variants", _tracking_query
        )

        from biobank_agent.skills.vcf_annotation import vcf_annotation

        result = vcf_annotation(genes="TYR,OCA2", ctx=ctx)

        assert "error" not in result
        assert result["n_genes_queried"] == 2
        assert set(queried) == {"TYR", "OCA2"}


class TestUnknownGenesReported:
    """test_unknown_genes_reported — genes='FAKE,TYR' → unknown_genes=['FAKE']."""

    def test_unknown_genes_reported(self, monkeypatch, tmp_path):
        ctx = _make_ctx(tmp_path)
        _patch_infra(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "biobank_agent.skills.vcf_annotation._query_gene_variants", _noop_query
        )

        from biobank_agent.skills.vcf_annotation import vcf_annotation

        result = vcf_annotation(genes="FAKE,TYR", ctx=ctx)

        assert "error" not in result
        assert result["n_genes_queried"] == 1
        assert "FAKE" in result["unknown_genes"]
        assert len(result["unknown_genes"]) == 1


class TestNoValidGenes:
    """test_no_valid_genes — genes='FAKE1,FAKE2' → error returned."""

    def test_no_valid_genes(self, monkeypatch, tmp_path):
        ctx = _make_ctx(tmp_path)
        _patch_infra(monkeypatch, tmp_path)

        from biobank_agent.skills.vcf_annotation import vcf_annotation

        result = vcf_annotation(genes="FAKE1,FAKE2", ctx=ctx)

        assert "error" in result
        assert "No valid gene" in result["error"] or "Available" in result["error"]


class TestQcPropagation:
    """test_qc_propagation — pass_samples in custom_data filters samples when no sample_ids."""

    def test_qc_propagation(self, monkeypatch, tmp_path):
        pass_set = ["J001", "J002", "J003"]
        custom_data = {"vcf_qc_results": {"pass_samples": pass_set}}
        ctx = _make_ctx(tmp_path, custom_data=custom_data)

        used_paths = []

        def _fake_merge(vcf_paths, output_path, region=None, **kw):
            used_paths.extend(vcf_paths)
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
            return p

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
            lambda ctx: {s: f"/fake/vcf/{s}.vcf.gz" for s in ALL_SAMPLE_IDS},
        )
        tmp_dir = tmp_path / "vcf_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("biobank_agent.utils.bcftools.get_tmp_dir", lambda ctx: tmp_dir)
        monkeypatch.setattr("biobank_agent.utils.bcftools.merge_vcfs", _fake_merge)
        monkeypatch.setattr(
            "biobank_agent.skills.vcf_annotation._query_gene_variants", _noop_query
        )

        from biobank_agent.skills.vcf_annotation import vcf_annotation

        result = vcf_annotation(genes="TYR", ctx=ctx)

        assert "error" not in result
        # Only the 3 pass_samples VCFs should have been passed to merge
        unique_paths = set(used_paths)
        expected_paths = {f"/fake/vcf/{s}.vcf.gz" for s in pass_set}
        assert unique_paths == expected_paths


class TestSampleIdsOverrideQc:
    """test_sample_ids_override_qc — when sample_ids specified, QC filter is skipped."""

    def test_sample_ids_override_qc(self, monkeypatch, tmp_path):
        # QC pass_samples contains only J001; requested sample_ids contain S001 too
        custom_data = {"vcf_qc_results": {"pass_samples": ["J001"]}}
        ctx = _make_ctx(tmp_path, custom_data=custom_data)

        used_paths = []

        def _fake_merge(vcf_paths, output_path, region=None, **kw):
            used_paths.extend(vcf_paths)
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
            return p

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
            lambda ctx: {s: f"/fake/vcf/{s}.vcf.gz" for s in ALL_SAMPLE_IDS},
        )
        tmp_dir = tmp_path / "vcf_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("biobank_agent.utils.bcftools.get_tmp_dir", lambda ctx: tmp_dir)
        monkeypatch.setattr("biobank_agent.utils.bcftools.merge_vcfs", _fake_merge)
        monkeypatch.setattr(
            "biobank_agent.skills.vcf_annotation._query_gene_variants", _noop_query
        )

        from biobank_agent.skills.vcf_annotation import vcf_annotation

        # Specify both J001 and S001; QC would exclude S001 but sample_ids overrides
        result = vcf_annotation(genes="TYR", sample_ids="J001,S001", ctx=ctx)

        assert "error" not in result
        unique_paths = set(used_paths)
        assert "/fake/vcf/S001.vcf.gz" in unique_paths
        assert "/fake/vcf/J001.vcf.gz" in unique_paths


class TestNoVcfFiles:
    """test_no_vcf_files — empty sample_paths → error."""

    def test_no_vcf_files(self, monkeypatch, tmp_path):
        ctx = _make_ctx(tmp_path)

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
            lambda ctx: {},
        )
        tmp_dir = tmp_path / "vcf_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("biobank_agent.utils.bcftools.get_tmp_dir", lambda ctx: tmp_dir)

        from biobank_agent.skills.vcf_annotation import vcf_annotation

        result = vcf_annotation(genes="TYR", ctx=ctx)

        assert "error" in result
        assert "No VCF" in result["error"] or "vcf" in result["error"].lower()


class TestMergeFailSkipsChrom:
    """test_merge_fail_skips_chrom — merge raises → genes on that chrom are skipped."""

    def test_merge_fail_skips_chrom(self, monkeypatch, tmp_path):
        ctx = _make_ctx(tmp_path)
        _patch_infra(monkeypatch, tmp_path, raise_merge=True)

        queried = []

        def _tracking_query(vcf_path, gene, chrom, start, end, **kw):
            queried.append(gene)
            return []

        monkeypatch.setattr(
            "biobank_agent.skills.vcf_annotation._query_gene_variants", _tracking_query
        )

        from biobank_agent.skills.vcf_annotation import vcf_annotation

        result = vcf_annotation(genes="TYR", ctx=ctx)

        # No error at top level — merge failure is silent/skipped
        assert "error" not in result
        # _query_gene_variants should never have been called (merge failed)
        assert queried == []
        assert result["n_genes_with_variants"] == 0
        assert result["skipped_regions"][0]["region"].startswith("chr11:")
        assert result["skipped_regions"][0]["stage"] == "merge"
        assert "merge fail" in result["skipped_regions"][0]["reason"].lower()


# ---------------------------------------------------------------------------
# _query_gene_variants unit tests (cyvcf2.VCF patched directly)
# ---------------------------------------------------------------------------

class TestQueryGeneVariants:
    """Unit tests for _query_gene_variants with mocked cyvcf2."""

    def test_basic(self, monkeypatch):
        """test_query_gene_variants_basic — returns variant with correct carrier count."""
        from biobank_agent.skills.vcf_annotation import _query_gene_variants

        samples = ["S1", "S2", "S3"]
        # S1=het(1), S2=hom-ref(0), S3=hom-alt(2) → carriers = [S1, S3]
        v = FakeVariant(
            chrom="chr11", pos=88915000, ref="A", alt=["T"],
            qual=50.0, gt_types=np.array([1, 0, 2], dtype=np.int8),
        )

        class AnnotVCF(FakeVCF):
            def __call__(self, region=None):
                return iter(self._variants)

        monkeypatch.setattr(
            "cyvcf2.VCF",
            lambda path: AnnotVCF(samples=samples, variants=[v]),
        )

        result = _query_gene_variants("/f.vcf.gz", "TYR", "chr11", 88911696, 88921223)

        assert len(result) == 1
        assert result[0]["n_carriers"] == 2
        assert set(result[0]["carrier_samples"]) == {"S1", "S3"}
        assert result[0]["gene"] == "TYR"
        assert result[0]["ref"] == "A"
        assert result[0]["alt"] == "T"

    def test_min_qual_filter(self, monkeypatch):
        """test_query_gene_variants_min_qual — variant with QUAL < min_qual is filtered out."""
        from biobank_agent.skills.vcf_annotation import _query_gene_variants

        samples = ["S1", "S2"]
        # QUAL=10, min_qual=30 → should be filtered
        v_low = FakeVariant(
            chrom="chr11", pos=88912000, ref="C", alt=["G"],
            qual=10.0, gt_types=np.array([1, 0], dtype=np.int8),
        )
        # QUAL=50, min_qual=30 → should be included
        v_ok = FakeVariant(
            chrom="chr11", pos=88913000, ref="A", alt=["T"],
            qual=50.0, gt_types=np.array([1, 0], dtype=np.int8),
        )

        class AnnotVCF(FakeVCF):
            def __call__(self, region=None):
                return iter(self._variants)

        monkeypatch.setattr(
            "cyvcf2.VCF",
            lambda path: AnnotVCF(samples=samples, variants=[v_low, v_ok]),
        )

        result = _query_gene_variants(
            "/f.vcf.gz", "TYR", "chr11", 88911696, 88921223, min_qual=30.0
        )

        assert len(result) == 1
        assert result[0]["pos"] == 88913000

    def test_no_alt_skipped(self, monkeypatch):
        """test_query_gene_variants_no_alt — variant with ALT='.' is skipped."""
        from biobank_agent.skills.vcf_annotation import _query_gene_variants

        samples = ["S1", "S2"]
        v_no_alt = FakeVariant(
            chrom="chr11", pos=88914000, ref="A", alt=["."],
            qual=60.0, gt_types=np.array([1, 0], dtype=np.int8),
        )

        class AnnotVCF(FakeVCF):
            def __call__(self, region=None):
                return iter(self._variants)

        monkeypatch.setattr(
            "cyvcf2.VCF",
            lambda path: AnnotVCF(samples=samples, variants=[v_no_alt]),
        )

        result = _query_gene_variants("/f.vcf.gz", "TYR", "chr11", 88911696, 88921223)

        assert len(result) == 0

    def test_no_carriers_skipped(self, monkeypatch):
        """test_query_gene_variants_no_carriers — all hom-ref → no carriers → skipped."""
        from biobank_agent.skills.vcf_annotation import _query_gene_variants

        samples = ["S1", "S2", "S3"]
        # All gt_type=0 (hom-ref) → no carriers
        v_all_ref = FakeVariant(
            chrom="chr11", pos=88915000, ref="A", alt=["T"],
            qual=55.0, gt_types=np.array([0, 0, 0], dtype=np.int8),
        )

        class AnnotVCF(FakeVCF):
            def __call__(self, region=None):
                return iter(self._variants)

        monkeypatch.setattr(
            "cyvcf2.VCF",
            lambda path: AnnotVCF(samples=samples, variants=[v_all_ref]),
        )

        result = _query_gene_variants("/f.vcf.gz", "TYR", "chr11", 88911696, 88921223)

        assert len(result) == 0

    def test_max_variants_limit(self, monkeypatch):
        """Ensure max_variants cap is respected."""
        from biobank_agent.skills.vcf_annotation import _query_gene_variants

        samples = ["S1"]
        variants = [
            FakeVariant(
                chrom="chr11", pos=88911696 + i * 10, ref="A", alt=["T"],
                qual=50.0, gt_types=np.array([1], dtype=np.int8),
            )
            for i in range(20)
        ]

        class AnnotVCF(FakeVCF):
            def __call__(self, region=None):
                return iter(self._variants)

        monkeypatch.setattr(
            "cyvcf2.VCF",
            lambda path: AnnotVCF(samples=samples, variants=variants),
        )

        result = _query_gene_variants(
            "/f.vcf.gz", "TYR", "chr11", 88911696, 88921223, max_variants=5
        )

        assert len(result) == 5


# ---------------------------------------------------------------------------
# Gene summary / figure tests
# ---------------------------------------------------------------------------

class TestGeneSummaryCounts:
    """test_gene_summary_counts — n_snv + n_indel correctly categorised."""

    def test_gene_summary_counts(self, monkeypatch, tmp_path):
        ctx = _make_ctx(tmp_path)
        _patch_infra(monkeypatch, tmp_path)

        def _mixed_query(vcf_path, gene, chrom, start, end, **kw):
            if gene == "TYR":
                # 2 SNVs + 1 indel
                return [
                    {"chrom": chrom, "pos": start + 100, "ref": "A", "alt": "T",
                     "rsid": ".", "qual": 50.0, "gene": gene,
                     "n_carriers": 1, "carrier_samples": ["J001"], "gt_types": [1]},
                    {"chrom": chrom, "pos": start + 200, "ref": "C", "alt": "G",
                     "rsid": ".", "qual": 50.0, "gene": gene,
                     "n_carriers": 1, "carrier_samples": ["J001"], "gt_types": [1]},
                    {"chrom": chrom, "pos": start + 300, "ref": "ACGT", "alt": "A",
                     "rsid": ".", "qual": 40.0, "gene": gene,
                     "n_carriers": 1, "carrier_samples": ["S001"], "gt_types": [1]},
                ]
            return []

        monkeypatch.setattr(
            "biobank_agent.skills.vcf_annotation._query_gene_variants", _mixed_query
        )

        from biobank_agent.skills.vcf_annotation import vcf_annotation

        result = vcf_annotation(genes="TYR", ctx=ctx)

        assert "error" not in result
        tyr_summary = next(g for g in result["gene_summary"] if g["gene"] == "TYR")
        assert tyr_summary["n_snv"] == 2
        assert tyr_summary["n_indel"] == 1
        assert tyr_summary["n_variants"] == 3


class TestFigureGeneratedWithHits:
    """test_figure_generated_with_hits — genes with hits produce bar chart figure."""

    def test_figure_generated_with_hits(self, monkeypatch, tmp_path):
        ctx = _make_ctx(tmp_path)
        _patch_infra(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "biobank_agent.skills.vcf_annotation._query_gene_variants",
            _single_hit_query,
        )

        from biobank_agent.skills.vcf_annotation import vcf_annotation

        result = vcf_annotation(genes="TYR", ctx=ctx)

        assert "error" not in result
        assert result["n_genes_with_variants"] >= 1
        # At least one figure path returned
        assert len(result["figures"]) >= 1
        # Figures also stored on ctx.state
        assert len(ctx.state.figures) >= 1


class TestNoFigureWithoutHits:
    """test_no_figure_without_hits — no variants → no figure generated."""

    def test_no_figure_without_hits(self, monkeypatch, tmp_path):
        ctx = _make_ctx(tmp_path)
        _patch_infra(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "biobank_agent.skills.vcf_annotation._query_gene_variants", _noop_query
        )

        from biobank_agent.skills.vcf_annotation import vcf_annotation

        result = vcf_annotation(genes="TYR", ctx=ctx)

        assert "error" not in result
        assert result["figures"] == []
        assert ctx.state.figures == []


class TestStandardAnnotationDowngrade:
    """Standard annotation mode should pause before built-in coordinate fallback."""

    def test_standard_annotation_missing_pauses_without_user_consent(self, monkeypatch, tmp_path):
        ctx = _make_ctx(tmp_path, custom_data={"workflow_mode": "standard"})
        _patch_infra(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "biobank_agent.skills.vcf_annotation.wgs_environment_status",
            lambda: {
                "executables": {"snpEff": "", "vep": "", "table_annovar.pl": ""},
                "modules": {"standard_annotation": "PARTIAL"},
            },
        )

        from biobank_agent.skills.vcf_annotation import vcf_annotation

        result = vcf_annotation(genes="TYR", ctx=ctx)

        assert result["awaiting_user"] is True
        assert result["standard_annotation_downgrade"] is True
        assert result["annotation_mode"] == "standard_annotation_blocked"

    def test_standard_annotation_missing_can_continue_exploratory(self, monkeypatch, tmp_path):
        ctx = _make_ctx(tmp_path, custom_data={"workflow_mode": "standard"})
        _patch_infra(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "biobank_agent.skills.vcf_annotation.wgs_environment_status",
            lambda: {
                "executables": {"snpEff": "", "vep": "", "table_annovar.pl": ""},
                "modules": {"standard_annotation": "PARTIAL"},
            },
        )
        monkeypatch.setattr(
            "biobank_agent.skills.vcf_annotation._query_gene_variants", _noop_query
        )

        from biobank_agent.skills.vcf_annotation import vcf_annotation

        result = vcf_annotation(genes="TYR", allow_exploratory_fallback=True, ctx=ctx)

        assert "error" not in result
        assert result["annotation_mode"] == "built_in_hg38_candidate_gene_coordinates"


class TestAnnotationResultsInState:
    """test_annotation_results_in_state — custom_data['annotation_results'] is populated."""

    def test_annotation_results_in_state(self, monkeypatch, tmp_path):
        ctx = _make_ctx(tmp_path)
        _patch_infra(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "biobank_agent.skills.vcf_annotation._query_gene_variants",
            _single_hit_query,
        )

        from biobank_agent.skills.vcf_annotation import vcf_annotation

        result = vcf_annotation(genes="TYR,OCA2", ctx=ctx)

        assert "error" not in result
        ann = ctx.state.custom_data.get("annotation_results")
        assert ann is not None, "annotation_results not stored in custom_data"
        assert "gene_hits" in ann
        assert "gene_coords" in ann
        # Both requested genes should appear in gene_coords
        assert "TYR" in ann["gene_coords"]
        assert "OCA2" in ann["gene_coords"]
        # gene_hits maps gene → count; TYR has 1 hit from _single_hit_query
        assert ann["gene_hits"].get("TYR", 0) == 1
