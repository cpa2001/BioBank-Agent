"""Tests for WGS-related report generation in biobank_agent/skills/report.py."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _make_rec(skill_name, results=None, args=None):
    """Create a minimal execution record for report functions."""
    return SimpleNamespace(
        skill=skill_name,
        key_results=results or {},
        args=args or {},
    )


# ---------------------------------------------------------------------------
# _interpret_skill  (WGS branches)
# ---------------------------------------------------------------------------

class TestInterpretSkill:
    def test_interpret_vcf_qc(self):
        from biobank_agent.skills.report import _interpret_skill
        rec = _make_rec("vcf_qc", {
            "n_samples_analyzed": 28,
            "n_samples_pass": 26,
            "n_samples_fail": 2,
            "region": "chr22",
            "hwe_fail_count": 5,
            "cohort_stats": {"ti_tv_ratio": {"mean": 2.1}},
        })
        text = _interpret_skill(rec)
        assert "28" in text
        assert "26" in text
        assert "chr22" in text
        assert "HWE" in text
        assert "Ti/Tv" in text

    def test_interpret_vcf_qc_no_hwe(self):
        from biobank_agent.skills.report import _interpret_skill
        rec = _make_rec("vcf_qc", {
            "n_samples_analyzed": 10,
            "n_samples_pass": 10,
            "n_samples_fail": 0,
            "region": "full genome",
            "hwe_fail_count": None,
            "cohort_stats": {},
        })
        text = _interpret_skill(rec)
        assert "10" in text
        assert "HWE" not in text

    def test_interpret_vcf_pca(self):
        from biobank_agent.skills.report import _interpret_skill
        rec = _make_rec("vcf_pca", {
            "n_samples": 28,
            "n_snps_used": 5000,
            "n_components": 10,
            "cumulative_variance_pct": 45.2,
        })
        text = _interpret_skill(rec)
        assert "28" in text
        assert "5,000" in text
        assert "45.2" in text
        assert "variance" in text.lower()

    def test_interpret_vcf_kinship(self):
        from biobank_agent.skills.report import _interpret_skill
        rec = _make_rec("vcf_kinship", {
            "n_samples": 28,
            "n_related_pairs": 2,
            "n_snps_used": 10000,
            "kinship_summary": {"max_off_diagonal": 0.25},
        })
        text = _interpret_skill(rec)
        assert "28" in text
        assert "2 related" in text
        assert "0.2500" in text

    def test_interpret_vcf_association(self):
        from biobank_agent.skills.report import _interpret_skill
        rec = _make_rec("vcf_association", {
            "n_cases": 10,
            "n_controls": 6,
            "n_variants_tested": 5000,
            "n_significant_bonferroni": 3,
            "lambda_gc": 1.05,
            "case_group": "J",
            "control_group": "V",
            "power_warning": "Power is limited.",
        })
        text = _interpret_skill(rec)
        assert "J vs V" in text
        assert "5,000" in text
        assert "3" in text
        assert "1.050" in text
        assert "Power is limited." in text

    def test_interpret_vcf_association_with_lambda_warning(self):
        from biobank_agent.skills.report import _interpret_skill
        rec = _make_rec("vcf_association", {
            "n_cases": 10, "n_controls": 6,
            "n_variants_tested": 100, "n_significant_bonferroni": 0,
            "lambda_gc": 1.25, "case_group": "J", "control_group": "V",
            "lambda_gc_warning": "Substantial inflation detected.",
            "power_warning": "",
        })
        text = _interpret_skill(rec)
        assert "Substantial inflation" in text

    def test_interpret_vcf_annotation(self):
        from biobank_agent.skills.report import _interpret_skill
        rec = _make_rec("vcf_annotation", {
            "n_genes_queried": 20,
            "n_genes_with_variants": 15,
            "n_total_variants": 234,
        })
        text = _interpret_skill(rec)
        assert "20" in text
        assert "15" in text
        assert "234" in text

    def test_interpret_vcf_burden_test(self):
        from biobank_agent.skills.report import _interpret_skill
        rec = _make_rec("vcf_burden_test", {
            "n_genes_tested": 20,
            "n_genes_significant_burden": 2,
            "case_group": "J",
            "control_group": "V",
            "maf_threshold": 0.05,
            "top_gene": "TYR",
        })
        text = _interpret_skill(rec)
        assert "20" in text
        assert "2" in text
        assert "TYR" in text
        assert "J" in text

    def test_interpret_pathway_enrichment(self):
        from biobank_agent.skills.report import _interpret_skill
        rec = _make_rec("pathway_enrichment", {
            "n_input_genes": 10,
            "n_pathways_tested": 8,
            "n_pathways_significant": 2,
            "database": "vitiligo_pathways (built-in)",
            "results": [{
                "pathway": "Melanogenesis",
                "significant": True,
                "fold_enrichment": 15.5,
                "p_fdr": 0.002,
            }],
        })
        text = _interpret_skill(rec)
        assert "10" in text
        assert "8" in text
        assert "2" in text
        assert "Melanogenesis" in text

    def test_interpret_pathway_no_significant(self):
        from biobank_agent.skills.report import _interpret_skill
        rec = _make_rec("pathway_enrichment", {
            "n_input_genes": 5,
            "n_pathways_tested": 8,
            "n_pathways_significant": 0,
            "database": "vitiligo_pathways (built-in)",
            "results": [{"pathway": "A", "significant": False, "p_fdr": 0.5}],
        })
        text = _interpret_skill(rec)
        assert "0" in text


# ---------------------------------------------------------------------------
# _caption_for_figure  (WGS branches)
# ---------------------------------------------------------------------------

class TestCaptionForFigure:
    def _make_ctx(self):
        return SimpleNamespace(settings=SimpleNamespace(biobank_name="TestBank"))

    def test_caption_qc_scatter(self):
        from biobank_agent.skills.report import _build_figure_caption
        from pathlib import Path
        rec = _make_rec("vcf_qc", {})
        caption = _build_figure_caption(Path("/reports/vcf_qc_scatter.png"), rec, 1, self._make_ctx())
        assert "call rate" in caption.lower()

    def test_caption_qc_titv(self):
        from biobank_agent.skills.report import _build_figure_caption
        from pathlib import Path
        rec = _make_rec("vcf_qc", {})
        caption = _build_figure_caption(Path("/reports/vcf_qc_titv.png"), rec, 1, self._make_ctx())
        assert "Ti/Tv" in caption

    def test_caption_qc_generic(self):
        from biobank_agent.skills.report import _build_figure_caption
        from pathlib import Path
        rec = _make_rec("vcf_qc", {})
        caption = _build_figure_caption(Path("/reports/vcf_qc_other.png"), rec, 1, self._make_ctx())
        assert "quality control" in caption.lower() or "VCF" in caption

    def test_caption_pca_scree(self):
        from biobank_agent.skills.report import _build_figure_caption
        from pathlib import Path
        rec = _make_rec("vcf_pca", {})
        caption = _build_figure_caption(Path("/reports/vcf_pca_scree.png"), rec, 1, self._make_ctx())
        assert "scree" in caption.lower()

    def test_caption_pca_pc1_pc2(self):
        from biobank_agent.skills.report import _build_figure_caption
        from pathlib import Path
        rec = _make_rec("vcf_pca", {})
        caption = _build_figure_caption(Path("/reports/vcf_pca_pc1_pc2.png"), rec, 1, self._make_ctx())
        assert "PC1 vs PC2" in caption

    def test_caption_pca_pc1_pc3(self):
        from biobank_agent.skills.report import _build_figure_caption
        from pathlib import Path
        rec = _make_rec("vcf_pca", {})
        caption = _build_figure_caption(Path("/reports/vcf_pca_pc1_pc3.png"), rec, 1, self._make_ctx())
        assert "PC1 vs PC3" in caption

    def test_caption_kinship(self):
        from biobank_agent.skills.report import _build_figure_caption
        from pathlib import Path
        rec = _make_rec("vcf_kinship", {})
        caption = _build_figure_caption(Path("/reports/vcf_kinship_heatmap.png"), rec, 1, self._make_ctx())
        assert "kinship" in caption.lower() or "Kinship" in caption

    def test_caption_manhattan(self):
        from biobank_agent.skills.report import _build_figure_caption
        from pathlib import Path
        rec = _make_rec("vcf_association", {})
        caption = _build_figure_caption(Path("/reports/vcf_association_manhattan.png"), rec, 1, self._make_ctx())
        assert "Manhattan" in caption

    def test_caption_qq(self):
        from biobank_agent.skills.report import _build_figure_caption
        from pathlib import Path
        rec = _make_rec("vcf_association", {})
        caption = _build_figure_caption(Path("/reports/vcf_association_qq.png"), rec, 1, self._make_ctx())
        assert "QQ" in caption

    def test_caption_annotation(self):
        from biobank_agent.skills.report import _build_figure_caption
        from pathlib import Path
        rec = _make_rec("vcf_annotation", {})
        caption = _build_figure_caption(Path("/reports/vcf_annotation_gene_counts.png"), rec, 1, self._make_ctx())
        assert "gene" in caption.lower()

    def test_caption_burden(self):
        from biobank_agent.skills.report import _build_figure_caption
        from pathlib import Path
        rec = _make_rec("vcf_burden_test", {})
        caption = _build_figure_caption(Path("/reports/vcf_burden_test.png"), rec, 1, self._make_ctx())
        assert "burden" in caption.lower() or "Burden" in caption

    def test_caption_pathway(self):
        from biobank_agent.skills.report import _build_figure_caption
        from pathlib import Path
        rec = _make_rec("pathway_enrichment", {})
        caption = _build_figure_caption(Path("/reports/pathway_enrichment.png"), rec, 1, self._make_ctx())
        assert "enrichment" in caption.lower()


# ---------------------------------------------------------------------------
# _extract_key_findings  (WGS branches)
# ---------------------------------------------------------------------------

class TestExtractKeyFindings:
    def test_findings_vcf_qc(self):
        from biobank_agent.skills.report import _extract_key_findings
        rec = _make_rec("vcf_qc", {"n_samples_pass": 26, "n_samples_fail": 2})
        findings = _extract_key_findings([rec])
        assert any("26" in f and "2" in f for f in findings)

    def test_findings_association(self):
        from biobank_agent.skills.report import _extract_key_findings
        rec = _make_rec("vcf_association", {
            "lambda_gc": 1.05, "n_significant_bonferroni": 3,
        })
        findings = _extract_key_findings([rec])
        assert any("1.05" in f or "λGC" in f for f in findings)

    def test_findings_burden(self):
        from biobank_agent.skills.report import _extract_key_findings
        rec = _make_rec("vcf_burden_test", {
            "n_genes_significant_burden": 2, "top_gene": "TYR",
        })
        findings = _extract_key_findings([rec])
        assert any("TYR" in f for f in findings)

    def test_findings_pathway(self):
        from biobank_agent.skills.report import _extract_key_findings
        rec = _make_rec("pathway_enrichment", {
            "n_pathways_significant": 3,
            "results": [{"pathway": "Melanogenesis", "significant": True}],
        })
        findings = _extract_key_findings([rec])
        assert any("Melanogenesis" in f or "3 significant" in f for f in findings)
