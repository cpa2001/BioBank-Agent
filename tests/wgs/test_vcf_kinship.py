"""Comprehensive tests for biobank_agent/skills/vcf_kinship.py."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tests.wgs.conftest import ALL_SAMPLE_IDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestVcfKinship:
    """Tests for the vcf_kinship skill."""

    def _setup_mocks(
        self,
        monkeypatch,
        mock_ctx,
        n_samples: int = 28,
        n_variants: int = 500,
        G_override=None,
    ):
        """Patch all external I/O so vcf_kinship runs without real files."""
        mapping = {s: f"/fake/{s}.vcf.gz" for s in ALL_SAMPLE_IDS[:n_samples]}

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
            lambda ctx: mapping,
        )
        monkeypatch.setattr(
            "biobank_agent.utils.bcftools.get_tmp_dir",
            lambda ctx: mock_ctx.report_dir / "tmp",
        )
        (mock_ctx.report_dir / "tmp").mkdir(parents=True, exist_ok=True)

        def fake_merge(vcf_paths, out, region=None):
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).touch()
            return Path(out)

        monkeypatch.setattr("biobank_agent.utils.bcftools.merge_vcfs", fake_merge)

        rng = np.random.default_rng(42)

        def fake_geno(merged, maf_min=0.05, max_variants=10000, snv_only=True, **kw):
            if G_override is not None:
                G = G_override
            else:
                G = rng.choice([0, 1, 2], size=(n_samples, n_variants)).astype(np.float64)
            vids = [f"chr22:{i * 100}:A:T" for i in range(G.shape[1])]
            return G, vids, list(mapping.keys())[: G.shape[0]]

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.build_genotype_matrix", fake_geno
        )

        return mapping

    # ------------------------------------------------------------------
    # 1. Fewer than 2 samples → error
    # ------------------------------------------------------------------
    def test_fewer_than_2_samples(self, monkeypatch, mock_ctx):
        """Only 1 sample in mapping → error returned immediately."""
        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
            lambda ctx: {"J001": "/fake/J001.vcf.gz"},
        )
        from biobank_agent.skills.vcf_kinship import vcf_kinship

        result = vcf_kinship(ctx=mock_ctx)
        assert "error" in result
        assert "2 samples" in result["error"] or "least 2" in result["error"]

    # ------------------------------------------------------------------
    # 2. Fewer than 10 variants → error
    # ------------------------------------------------------------------
    def test_fewer_than_10_variants(self, monkeypatch, mock_ctx):
        """build_genotype_matrix returns only 5 columns → error reported."""
        G_tiny = np.zeros((28, 5), dtype=np.float64)
        self._setup_mocks(monkeypatch, mock_ctx, G_override=G_tiny)

        from biobank_agent.skills.vcf_kinship import vcf_kinship

        result = vcf_kinship(ctx=mock_ctx)
        assert "error" in result
        assert "5" in result["error"]

    # ------------------------------------------------------------------
    # 3. QC propagation: pass_samples filters the mapping
    # ------------------------------------------------------------------
    def test_qc_propagation(self, monkeypatch, mock_ctx):
        """vcf_qc_results.pass_samples limits samples used."""
        pass_samples = ALL_SAMPLE_IDS[:5]  # only 5 of the 28
        mock_ctx.state.custom_data["vcf_qc_results"] = {
            "pass_samples": pass_samples,
        }

        merge_calls: list[dict] = []

        def tracking_merge(vcf_paths, out, region=None):
            merge_calls.append({"vcf_paths": list(vcf_paths)})
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).touch()
            return Path(out)

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
            lambda ctx: {s: f"/fake/{s}.vcf.gz" for s in ALL_SAMPLE_IDS},
        )
        monkeypatch.setattr(
            "biobank_agent.utils.bcftools.get_tmp_dir",
            lambda ctx: mock_ctx.report_dir / "tmp",
        )
        (mock_ctx.report_dir / "tmp").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr("biobank_agent.utils.bcftools.merge_vcfs", tracking_merge)

        rng = np.random.default_rng(0)

        def fake_geno(merged, maf_min=0.05, max_variants=10000, snv_only=True, **kw):
            G = rng.choice([0, 1, 2], size=(5, 500)).astype(np.float64)
            vids = [f"chr22:{i * 100}:A:T" for i in range(500)]
            return G, vids, pass_samples[:5]

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.build_genotype_matrix", fake_geno
        )

        from biobank_agent.skills.vcf_kinship import vcf_kinship

        result = vcf_kinship(ctx=mock_ctx)
        assert "error" not in result
        # Only the 5 pass_sample VCF paths should be fed to merge
        assert len(merge_calls) == 1
        for path in merge_calls[0]["vcf_paths"]:
            # Strip .vcf.gz: "/fake/J001.vcf.gz" → "J001"
            sample_id = Path(path).name.split(".")[0]
            assert sample_id in pass_samples

    # ------------------------------------------------------------------
    # 4. Basic output with 28 samples and 500 variants
    # ------------------------------------------------------------------
    def test_basic_kinship_output(self, monkeypatch, mock_ctx):
        """Valid run → correct output keys and sensible numeric ranges."""
        self._setup_mocks(monkeypatch, mock_ctx)

        from biobank_agent.skills.vcf_kinship import vcf_kinship

        result = vcf_kinship(ctx=mock_ctx)
        assert "error" not in result
        assert result["n_samples"] == 28
        assert result["n_snps_used"] == 500
        assert isinstance(result["n_related_pairs"], int)
        assert isinstance(result["related_pairs"], list)
        assert "kinship_summary" in result
        assert "figures" in result

    # ------------------------------------------------------------------
    # 5. Related pairs: inject known high kinship
    # ------------------------------------------------------------------
    def test_related_pairs_threshold(self, monkeypatch, mock_ctx):
        """Samples with identical genotypes → kinship > default threshold."""
        rng = np.random.default_rng(7)
        G = rng.choice([0, 1, 2], size=(28, 500)).astype(np.float64)
        # Make samples 0 and 1 identical
        G[1, :] = G[0, :]
        self._setup_mocks(monkeypatch, mock_ctx, G_override=G)

        from biobank_agent.skills.vcf_kinship import vcf_kinship

        result = vcf_kinship(kinship_threshold=0.1, ctx=mock_ctx)
        sample_pairs = [
            (p["sample1"], p["sample2"]) for p in result["related_pairs"]
        ]
        s0 = ALL_SAMPLE_IDS[0]
        s1 = ALL_SAMPLE_IDS[1]
        assert (s0, s1) in sample_pairs or any(
            (p["sample1"] == s0 and p["sample2"] == s1)
            or (p["sample1"] == s1 and p["sample2"] == s0)
            for p in result["related_pairs"]
        )

    # ------------------------------------------------------------------
    # 6. No related pairs when all samples are random
    # ------------------------------------------------------------------
    def test_no_related_pairs(self, monkeypatch, mock_ctx):
        """Independent random genotypes → no pair above a high threshold."""
        self._setup_mocks(monkeypatch, mock_ctx)

        from biobank_agent.skills.vcf_kinship import vcf_kinship

        # Use a very high threshold — random data won't exceed it
        result = vcf_kinship(kinship_threshold=0.99, ctx=mock_ctx)
        assert "error" not in result
        assert result["n_related_pairs"] == 0
        assert result["related_pairs"] == []

    # ------------------------------------------------------------------
    # 7. Relationship: identical / MZ twin (kinship > 0.354)
    # ------------------------------------------------------------------
    def test_relationship_identical(self, monkeypatch, mock_ctx):
        """Duplicate rows produce kinship > 0.354 → classified as identical/MZ twin.

        Using all 28 samples so the GRM column-normalisation is stable and the
        within-cohort kinship of the duplicate pair lands around 0.47.
        """
        rng = np.random.default_rng(11)
        n_s = 28
        G = rng.choice([0, 1, 2], size=(n_s, 500)).astype(np.float64)
        G[2, :] = G[0, :]  # exact duplicate → kinship ≈ 0.47

        self._setup_mocks(monkeypatch, mock_ctx, G_override=G)

        from biobank_agent.skills.vcf_kinship import vcf_kinship

        result = vcf_kinship(kinship_threshold=0.1, ctx=mock_ctx)
        assert "error" not in result
        # Find the pair involving the duplicated samples (indices 0 and 2)
        s0 = ALL_SAMPLE_IDS[0]
        s2 = ALL_SAMPLE_IDS[2]
        twin_pair = next(
            (
                p
                for p in result["related_pairs"]
                if {p["sample1"], p["sample2"]} == {s0, s2}
            ),
            None,
        )
        assert twin_pair is not None, "Expected identical pair not found"
        assert "identical" in twin_pair["relationship_estimate"]

    # ------------------------------------------------------------------
    # 8. Relationship: 1st degree (0.177 < kinship < 0.354)
    # ------------------------------------------------------------------
    def test_relationship_first_degree(self, monkeypatch, mock_ctx):
        """Construct a pair where kinship is in (0.177, 0.354) → 1st degree.

        With 28 samples and 50% genotype overlap between sample 0 and 1
        the GRM normalisation yields kinship ≈ 0.24 which sits in the
        1st-degree bucket.
        """
        rng = np.random.default_rng(99)
        n_s, n_v = 28, 1000
        G = rng.choice([0, 1, 2], size=(n_s, n_v)).astype(np.float64)
        base = G[0, :].copy()
        rand = rng.choice([0, 1, 2], size=n_v).astype(np.float64)
        mask = rng.random(n_v) < 0.5
        G[1, :] = np.where(mask, base, rand)

        # Verify analytically that the pair lands in the 1st-degree range
        G_norm = (G - G.mean(axis=0)) / (G.std(axis=0) + 1e-10)
        GRM = G_norm @ G_norm.T / n_v
        kinship_val = float(GRM[0, 1] / 2.0)
        assert 0.177 < kinship_val < 0.354, (
            f"Precondition failed: kinship {kinship_val:.4f} not in 1st-degree range"
        )

        self._setup_mocks(monkeypatch, mock_ctx, G_override=G)

        from biobank_agent.skills.vcf_kinship import vcf_kinship

        result = vcf_kinship(kinship_threshold=0.0, ctx=mock_ctx)
        assert "error" not in result

        s0 = ALL_SAMPLE_IDS[0]
        s1 = ALL_SAMPLE_IDS[1]
        pair = next(
            (
                p
                for p in result["related_pairs"]
                if {p["sample1"], p["sample2"]} == {s0, s1}
            ),
            None,
        )
        assert pair is not None, f"Pair ({s0}, {s1}) not found in related_pairs"
        assert "1st degree" in pair["relationship_estimate"]

    # ------------------------------------------------------------------
    # 9. Relationship: 2nd degree (0.0884 < kinship < 0.177)
    # ------------------------------------------------------------------
    def test_relationship_second_degree(self, monkeypatch, mock_ctx):
        """Construct a pair where kinship is in (0.0884, 0.177) → 2nd degree.

        With 28 samples and 25% genotype overlap between sample 0 and 1
        the GRM normalisation yields kinship ≈ 0.126 which sits in the
        2nd-degree bucket.
        """
        rng = np.random.default_rng(555)
        n_s, n_v = 28, 500
        G = rng.choice([0, 1, 2], size=(n_s, n_v)).astype(np.float64)
        base = G[0, :].copy()
        rand = rng.choice([0, 1, 2], size=n_v).astype(np.float64)
        frac = 0.25
        mask = rng.random(n_v) < frac
        G[1, :] = np.where(mask, base, rand)

        # Verify analytically that the pair lands in the 2nd-degree range
        G_norm = (G - G.mean(axis=0)) / (G.std(axis=0) + 1e-10)
        GRM = G_norm @ G_norm.T / n_v
        kinship_val = float(GRM[0, 1] / 2.0)
        assert 0.0884 < kinship_val < 0.177, (
            f"Precondition failed: kinship {kinship_val:.4f} not in 2nd-degree range"
        )

        self._setup_mocks(monkeypatch, mock_ctx, G_override=G)

        from biobank_agent.skills.vcf_kinship import vcf_kinship

        result = vcf_kinship(kinship_threshold=0.0, ctx=mock_ctx)
        assert "error" not in result

        s0 = ALL_SAMPLE_IDS[0]
        s1 = ALL_SAMPLE_IDS[1]
        pair = next(
            (
                p
                for p in result["related_pairs"]
                if {p["sample1"], p["sample2"]} == {s0, s1}
            ),
            None,
        )
        assert pair is not None, f"Pair ({s0}, {s1}) not found in related_pairs"
        assert "2nd degree" in pair["relationship_estimate"]

    # ------------------------------------------------------------------
    # 10. Kinship summary stats present and valid
    # ------------------------------------------------------------------
    def test_kinship_summary_stats(self, monkeypatch, mock_ctx):
        """mean/max/min off-diagonal fields exist and are finite floats."""
        self._setup_mocks(monkeypatch, mock_ctx)

        from biobank_agent.skills.vcf_kinship import vcf_kinship

        result = vcf_kinship(ctx=mock_ctx)
        ks = result["kinship_summary"]
        for key in ("mean_off_diagonal", "max_off_diagonal", "min_off_diagonal"):
            assert key in ks, f"Missing key: {key}"
            assert isinstance(ks[key], float), f"{key} is not float"
            assert np.isfinite(ks[key]), f"{key} is not finite"

    # ------------------------------------------------------------------
    # 11. Heatmap vmax is adaptive when off-diag max > 0.25
    # ------------------------------------------------------------------
    def test_heatmap_vmax_adaptive(self, monkeypatch, mock_ctx):
        """When two samples are identical the vmax should exceed 0.25."""
        rng = np.random.default_rng(3)
        G = rng.choice([0, 1, 2], size=(28, 500)).astype(np.float64)
        # Identical pair → kinship ≈ 0.5 → vmax must be > 0.25
        G[5, :] = G[0, :]
        self._setup_mocks(monkeypatch, mock_ctx, G_override=G)

        # Intercept imshow to capture vmax
        captured_vmax: list[float] = []
        import matplotlib.pyplot as plt

        original_imshow = None

        def patched_imshow(data, *args, vmax=None, **kwargs):
            captured_vmax.append(vmax)
            return original_imshow(data, *args, vmax=vmax, **kwargs)

        import matplotlib.axes

        original_imshow = matplotlib.axes.Axes.imshow
        monkeypatch.setattr(matplotlib.axes.Axes, "imshow", patched_imshow)

        from biobank_agent.skills.vcf_kinship import vcf_kinship

        result = vcf_kinship(ctx=mock_ctx)
        assert "error" not in result
        # vmax should be > 0.25 due to the high off-diagonal kinship from the twin pair
        assert any(v is not None and v > 0.25 for v in captured_vmax)

    # ------------------------------------------------------------------
    # 12. Phenotype labels include group when phenotype_group is present
    # ------------------------------------------------------------------
    def test_phenotype_labels(self, monkeypatch, mock_ctx):
        """Labels on heatmap contain '(X)' group suffix when phenotype_group exists."""
        self._setup_mocks(monkeypatch, mock_ctx)

        import matplotlib.axes

        captured_ylabels: list = []
        original_set_yticklabels = matplotlib.axes.Axes.set_yticklabels

        def patched_set_yticklabels(self_ax, labels, *args, **kwargs):
            captured_ylabels.extend(labels)
            return original_set_yticklabels(self_ax, labels, *args, **kwargs)

        monkeypatch.setattr(
            matplotlib.axes.Axes, "set_yticklabels", patched_set_yticklabels
        )

        from biobank_agent.skills.vcf_kinship import vcf_kinship

        result = vcf_kinship(ctx=mock_ctx)
        assert "error" not in result
        # At least one label should contain a parenthetical group like "(J)"
        assert any("(" in lbl and ")" in lbl for lbl in captured_ylabels)

    # ------------------------------------------------------------------
    # 13. No phenotype_group → labels are bare sample IDs
    # ------------------------------------------------------------------
    def test_no_phenotype_labels(self, monkeypatch, mock_ctx):
        """Without phenotype_group column labels are plain sample IDs."""
        self._setup_mocks(monkeypatch, mock_ctx)

        # Override dm.query to return a df without phenotype_group
        df_no_grp = pd.DataFrame({"sample_id": ALL_SAMPLE_IDS})
        mock_ctx.dm.query = lambda sql, *a, **kw: df_no_grp

        import matplotlib.axes

        captured_ylabels: list = []
        original_set_yticklabels = matplotlib.axes.Axes.set_yticklabels

        def patched_set_yticklabels(self_ax, labels, *args, **kwargs):
            captured_ylabels.extend(labels)
            return original_set_yticklabels(self_ax, labels, *args, **kwargs)

        monkeypatch.setattr(
            matplotlib.axes.Axes, "set_yticklabels", patched_set_yticklabels
        )

        from biobank_agent.skills.vcf_kinship import vcf_kinship

        result = vcf_kinship(ctx=mock_ctx)
        assert "error" not in result
        # Labels should not contain parenthetical suffixes
        assert not any("(" in lbl for lbl in captured_ylabels)
        # Should match raw sample IDs
        for lbl in captured_ylabels:
            assert lbl in ALL_SAMPLE_IDS

    # ------------------------------------------------------------------
    # 14. Figures are stored in ctx.state.figures
    # ------------------------------------------------------------------
    def test_figures_stored_in_state(self, monkeypatch, mock_ctx):
        """After successful run ctx.state.figures is extended with figure paths."""
        self._setup_mocks(monkeypatch, mock_ctx)

        from biobank_agent.skills.vcf_kinship import vcf_kinship

        initial_count = len(mock_ctx.state.figures)
        result = vcf_kinship(ctx=mock_ctx)
        assert "error" not in result
        assert len(mock_ctx.state.figures) > initial_count

    # ------------------------------------------------------------------
    # 15. Region is forwarded to merge_vcfs
    # ------------------------------------------------------------------
    def test_region_passed(self, monkeypatch, mock_ctx):
        """The region argument is passed through to merge_vcfs as-is."""
        merge_calls: list[dict] = []

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
            lambda ctx: {s: f"/fake/{s}.vcf.gz" for s in ALL_SAMPLE_IDS},
        )
        monkeypatch.setattr(
            "biobank_agent.utils.bcftools.get_tmp_dir",
            lambda ctx: mock_ctx.report_dir / "tmp",
        )
        (mock_ctx.report_dir / "tmp").mkdir(parents=True, exist_ok=True)

        def tracking_merge(vcf_paths, out, region=None):
            merge_calls.append({"region": region})
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).touch()
            return Path(out)

        monkeypatch.setattr("biobank_agent.utils.bcftools.merge_vcfs", tracking_merge)

        rng = np.random.default_rng(42)

        def fake_geno(merged, maf_min=0.05, max_variants=10000, snv_only=True, **kw):
            G = rng.choice([0, 1, 2], size=(28, 500)).astype(np.float64)
            vids = [f"chr22:{i * 100}:A:T" for i in range(500)]
            return G, vids, ALL_SAMPLE_IDS[:28]

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.build_genotype_matrix", fake_geno
        )

        from biobank_agent.skills.vcf_kinship import vcf_kinship

        result = vcf_kinship(region="chr22:1-5000000", ctx=mock_ctx)
        assert "error" not in result
        assert len(merge_calls) == 1
        assert merge_calls[0]["region"] == "chr22:1-5000000"

    # ------------------------------------------------------------------
    # Extra: empty region string is converted to None for merge_vcfs
    # ------------------------------------------------------------------
    def test_empty_region_becomes_none(self, monkeypatch, mock_ctx):
        """An empty string region is normalised to None before merge_vcfs."""
        merge_calls: list[dict] = []

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.get_sample_vcf_paths",
            lambda ctx: {s: f"/fake/{s}.vcf.gz" for s in ALL_SAMPLE_IDS},
        )
        monkeypatch.setattr(
            "biobank_agent.utils.bcftools.get_tmp_dir",
            lambda ctx: mock_ctx.report_dir / "tmp",
        )
        (mock_ctx.report_dir / "tmp").mkdir(parents=True, exist_ok=True)

        def tracking_merge(vcf_paths, out, region=None):
            merge_calls.append({"region": region})
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).touch()
            return Path(out)

        monkeypatch.setattr("biobank_agent.utils.bcftools.merge_vcfs", tracking_merge)

        rng = np.random.default_rng(42)

        def fake_geno(merged, maf_min=0.05, max_variants=10000, snv_only=True, **kw):
            G = rng.choice([0, 1, 2], size=(28, 500)).astype(np.float64)
            vids = [f"chr22:{i * 100}:A:T" for i in range(500)]
            return G, vids, ALL_SAMPLE_IDS[:28]

        monkeypatch.setattr(
            "biobank_agent.utils.vcf_genotypes.build_genotype_matrix", fake_geno
        )

        from biobank_agent.skills.vcf_kinship import vcf_kinship

        result = vcf_kinship(region="", ctx=mock_ctx)
        assert "error" not in result
        assert merge_calls[0]["region"] is None
