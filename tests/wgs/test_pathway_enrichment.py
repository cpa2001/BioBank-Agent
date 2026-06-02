"""Comprehensive tests for biobank_agent/skills/pathway_enrichment.py."""

from __future__ import annotations

import pytest
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call(gene_list="", database="vitiligo_pathways", fdr_threshold=0.05,
          top_n=20, background_size=20000, *, ctx):
    from biobank_agent.skills.pathway_enrichment import pathway_enrichment
    return pathway_enrichment(
        gene_list=gene_list,
        database=database,
        fdr_threshold=fdr_threshold,
        top_n=top_n,
        background_size=background_size,
        ctx=ctx,
    )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestPathwayEnrichment:

    # ------------------------------------------------------------------
    # 1. Gene source: explicit gene_list
    # ------------------------------------------------------------------

    def test_explicit_gene_list(self, mock_ctx):
        """Comma-separated gene_list is parsed, stripped and uppercased."""
        result = _call(gene_list="TYR,OCA2,MC1R", ctx=mock_ctx)
        assert result["n_input_genes"] == 3
        assert "TYR" in result["input_genes"]
        assert "OCA2" in result["input_genes"]
        assert "MC1R" in result["input_genes"]

    def test_explicit_gene_list_lowercased_input(self, mock_ctx):
        """Gene symbols in lowercase must be uppercased before processing."""
        result = _call(gene_list="tyr, oca2 , mc1r", ctx=mock_ctx)
        assert result["n_input_genes"] == 3
        assert "TYR" in result["input_genes"]

    # ------------------------------------------------------------------
    # 2. Gene source: burden_results from ctx.state
    # ------------------------------------------------------------------

    def test_genes_from_burden_state(self, mock_ctx):
        """Genes with p_burden < 0.1 are extracted from burden_results."""
        mock_ctx.state.custom_data["burden_results"] = {
            "gene_results": [
                {"gene": "TYR",  "p_burden": 0.01},
                {"gene": "OCA2", "p_burden": 0.05},
                {"gene": "MC1R", "p_burden": 0.5},   # above threshold, excluded
            ]
        }
        result = _call(ctx=mock_ctx)
        assert result["n_input_genes"] == 2
        assert "TYR"  in result["input_genes"]
        assert "OCA2" in result["input_genes"]
        assert "MC1R" not in result["input_genes"]

    def test_genes_from_burden_state_uppercase(self, mock_ctx):
        """Gene names from burden_results are uppercased."""
        mock_ctx.state.custom_data["burden_results"] = {
            "gene_results": [
                {"gene": "tyr", "p_burden": 0.01},
            ]
        }
        result = _call(ctx=mock_ctx)
        assert "TYR" in result["input_genes"]

    # ------------------------------------------------------------------
    # 3. Gene source: annotation_results from ctx.state
    # ------------------------------------------------------------------

    def test_genes_from_annotation_state(self, mock_ctx):
        """Keys of annotation_results['gene_hits'] are used when burden empty."""
        mock_ctx.state.custom_data["annotation_results"] = {
            "gene_hits": {
                "MITF": {"count": 5},
                "SOX10": {"count": 3},
            }
        }
        result = _call(ctx=mock_ctx)
        assert result["n_input_genes"] == 2
        assert "MITF"  in result["input_genes"]
        assert "SOX10" in result["input_genes"]

    def test_annotation_state_ignored_when_burden_has_genes(self, mock_ctx):
        """burden_results takes priority over annotation_results."""
        mock_ctx.state.custom_data["burden_results"] = {
            "gene_results": [{"gene": "TYR", "p_burden": 0.01}]
        }
        mock_ctx.state.custom_data["annotation_results"] = {
            "gene_hits": {"MITF": {}, "SOX10": {}}
        }
        result = _call(ctx=mock_ctx)
        assert result["n_input_genes"] == 1
        assert "TYR" in result["input_genes"]

    # ------------------------------------------------------------------
    # 4. No genes → error
    # ------------------------------------------------------------------

    def test_no_genes_error(self, mock_ctx):
        """No gene_list and empty state returns an error dict."""
        result = _call(gene_list="", ctx=mock_ctx)
        assert "error" in result

    def test_no_genes_error_blank_list(self, mock_ctx):
        """A whitespace-only gene_list is treated as empty."""
        result = _call(gene_list="  ,  , ", ctx=mock_ctx)
        assert "error" in result

    # ------------------------------------------------------------------
    # 5–7. Database selection
    # ------------------------------------------------------------------

    def test_database_vitiligo(self, mock_ctx):
        """Default database tests exactly 8 vitiligo pathways."""
        result = _call(gene_list="TYR,MITF,SOX10,MC1R", ctx=mock_ctx)
        assert result["n_pathways_tested"] == 8

    def test_database_hallmark(self, mock_ctx):
        """'hallmark' database tests exactly 6 hallmark pathways."""
        result = _call(gene_list="TYR,TNF,IL6", database="hallmark", ctx=mock_ctx)
        assert result["n_pathways_tested"] == 6

    def test_database_all(self, mock_ctx):
        """'all' database tests 14 pathways (8 vitiligo + 6 hallmark)."""
        result = _call(gene_list="TYR,TNF,IFNG", database="all", ctx=mock_ctx)
        assert result["n_pathways_tested"] == 14

    # ------------------------------------------------------------------
    # 8. Hypergeometric p-value
    # ------------------------------------------------------------------

    def test_hypergeometric_test(self, mock_ctx):
        """Known overlap produces a p-value consistent with hypergeom.sf."""
        # TYR, OCA2, MC1R, MITF, SLC45A2 all appear in Melanogenesis (K=15)
        genes = "TYR,OCA2,MC1R,MITF,SLC45A2"
        result = _call(gene_list=genes, ctx=mock_ctx)
        melanogenesis = next(
            r for r in result["results"] if r["pathway"] == "Melanogenesis"
        )
        k = melanogenesis["overlap_count"]
        n = 5          # input gene count
        K = 15         # Melanogenesis pathway size
        N = 20000      # background
        expected_p = sp_stats.hypergeom.sf(k - 1, N, K, n)
        assert abs(melanogenesis["p_value"] - expected_p) < 1e-10

    # ------------------------------------------------------------------
    # 9. Zero overlap → p_value == 1.0
    # ------------------------------------------------------------------

    def test_no_overlap(self, mock_ctx):
        """A gene not in any pathway should give p_value=1.0 for all pathways."""
        result = _call(gene_list="FAKEGENE1,FAKEGENE2", ctx=mock_ctx)
        for r in result["results"]:
            assert r["p_value"] == 1.0

    # ------------------------------------------------------------------
    # 10. Fold enrichment formula: (k/n) / (K/N)
    # ------------------------------------------------------------------

    def test_fold_enrichment_computation(self, mock_ctx):
        """fold_enrichment = (k/n) / (K/N) rounded to 2 dp."""
        genes = "TYR,OCA2,MC1R"
        result = _call(gene_list=genes, ctx=mock_ctx)
        for r in result["results"]:
            n = 3
            K = r["pathway_size"]
            N = 20000
            k = r["overlap_count"]
            if n > 0 and K > 0 and N > 0:
                expected = round((k / n) / (K / N), 2)
            else:
                expected = 0
            assert r["fold_enrichment"] == expected

    # ------------------------------------------------------------------
    # 11. FDR correction present
    # ------------------------------------------------------------------

    def test_fdr_correction(self, mock_ctx):
        """Every result entry must contain a p_fdr key."""
        result = _call(gene_list="TYR,OCA2,MC1R", ctx=mock_ctx)
        for r in result["results"]:
            assert "p_fdr" in r
            assert 0.0 <= r["p_fdr"] <= 1.0

    # ------------------------------------------------------------------
    # 12. Significant flag when fdr < threshold
    # ------------------------------------------------------------------

    def test_significant_flagged(self, mock_ctx):
        """Results with p_fdr < fdr_threshold must have significant=True."""
        # Use a large set of known pathway genes to force low p-values
        genes = "TYR,OCA2,MC1R,MITF,SLC45A2,DCT,TYRP1,PMEL,RAB27A,MLANA,SOX10,PAX3"
        result = _call(gene_list=genes, fdr_threshold=0.05, ctx=mock_ctx)
        for r in result["results"]:
            if r["p_fdr"] < 0.05:
                assert r["significant"] == True  # noqa: E712  (numpy.bool_ safe)
            else:
                assert r["significant"] == False  # noqa: E712

    def test_significant_false_when_fdr_above_threshold(self, mock_ctx):
        """Results with p_fdr >= fdr_threshold must have significant=False."""
        # Genes with minimal overlap will yield high FDR
        result = _call(gene_list="FAKEGENE,TYR", fdr_threshold=0.01, ctx=mock_ctx)
        for r in result["results"]:
            if r["p_fdr"] >= 0.01:
                assert r["significant"] == False  # noqa: E712  (numpy.bool_ safe)

    # ------------------------------------------------------------------
    # 13. Results sorted ascending by p_value
    # ------------------------------------------------------------------

    def test_results_sorted_by_pvalue(self, mock_ctx):
        """Returned results list must be sorted in ascending p_value order."""
        result = _call(gene_list="TYR,OCA2,MC1R,MITF,SOX10,IFNG,TNF,TP53", ctx=mock_ctx)
        p_values = [r["p_value"] for r in result["results"]]
        assert p_values == sorted(p_values)

    # ------------------------------------------------------------------
    # 14. top_n limits the returned results
    # ------------------------------------------------------------------

    def test_top_n_limit(self, mock_ctx):
        """Only top_n pathway results are returned."""
        result = _call(gene_list="TYR,OCA2,MC1R,MITF", top_n=3, ctx=mock_ctx)
        assert len(result["results"]) == 3

    def test_top_n_larger_than_pathways(self, mock_ctx):
        """When top_n > n_pathways, all pathways are returned."""
        result = _call(gene_list="TYR", top_n=100, ctx=mock_ctx)
        assert len(result["results"]) == result["n_pathways_tested"]

    # ------------------------------------------------------------------
    # 15. Figure generated for pathways with p < 1.0
    # ------------------------------------------------------------------

    def test_figure_generated(self, mock_ctx):
        """At least one bar-chart figure is created when some p_value < 1.0."""
        # Use many strong pathway genes to ensure p < 1.0 for some pathways
        genes = "TYR,OCA2,MC1R,MITF,SLC45A2,DCT,TYRP1,PMEL,RAB27A,MLANA,SOX10,PAX3"
        result = _call(gene_list=genes, ctx=mock_ctx)
        # Confirm at least one pathway with p < 1.0 exists
        any_significant = any(r["p_value"] < 1.0 for r in result["results"])
        assert any_significant, "Precondition: need at least one pathway with p < 1.0"
        # Figure paths should be listed in the result and stored on ctx
        assert len(result["figures"]) >= 1
        assert len(mock_ctx.state.figures) >= 1

    # ------------------------------------------------------------------
    # 16. No figure when all p_values are 1.0
    # ------------------------------------------------------------------

    def test_no_figure_all_pval_1(self, mock_ctx):
        """No figure is produced when every pathway has p_value == 1.0."""
        result = _call(gene_list="FAKEGENE1,FAKEGENE2,FAKEGENE3", ctx=mock_ctx)
        # All overlaps are zero, so all p_values must be 1.0
        assert all(r["p_value"] == 1.0 for r in result["results"])
        assert result["figures"] == []
        assert mock_ctx.state.figures == []
