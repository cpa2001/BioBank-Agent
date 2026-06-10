"""Methodological-soundness reviewer.

Scientific correctness is not exit-code 0: a GWAS/PheWAS that reports raw p-values
with no multiple-testing correction, an association with n below any power, or an
effect estimate with no confidence interval is *wrong* even when it "ran". This
pure, dependency-free reviewer scans an analysis result (and/or its narrative) for
consensus statistical sins and returns flags the verification mesh can hard-block
on (``severity='block'``) or surface as advisories (``severity='advisory'``).
"""

from __future__ import annotations

from typing import Any

_CORRECTION_HINTS = (
    "fdr", "bonferroni", "benjamini", "hochberg", "q_value", "qvalue", "q_val",
    "p_adj", "padj", "p_adjusted", "adjusted_p", "p_corrected", "p_bonferroni",
    "p_fdr", "holm", "sidak",
)
_CI_HINTS = ("ci", "conf_int", "confidence_interval", "ci_low", "ci_lower", "ci95",
             "ci_95", "lower_ci", "l95", "u95", "ci_upper", "hr_ci", "or_ci")
_EFFECT_HINTS = ("odds_ratio", "beta", "hazard_ratio", "coef", "effect_size", "log_or")
_MIN_GROUP_N = 10
_MIN_TOTAL_N = 20

# Single-cell / spatial context markers. The omics-sin checks fire only when the evidence
# reads as single-cell/spatial work, so tabular GWAS/PheWAS evidence never trips them.
_OMICS_HINTS = (
    "single-cell", "single cell", "scrna", "scatac", "snrna", "anndata", "h5ad", "scanpy",
    "spatial", "stereo-seq", "stereo seq", "visium", "squidpy", "cell type", "cell-type",
    "umap", "leiden", "louvain", "spliced", "velocity", "pseudotime", "trajectory",
    "cellchat", "ligand-receptor", "ligand receptor", "niche", "deconvolution", "scenic",
)


def _walk_numbers(payload: Any, key_substr: str, limit: int = 100000) -> list[float]:
    out: list[float] = []

    def _rec(node: Any, depth: int = 0) -> None:
        if len(out) >= limit or depth > 6:
            return
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and key_substr in str(k).lower():
                    out.append(float(v))
                else:
                    _rec(v, depth + 1)
        elif isinstance(node, (list, tuple)):
            for v in node:
                _rec(v, depth + 1)

    _rec(payload)
    return out


def _has_key_hint(payload: Any, hints: tuple[str, ...], depth: int = 0) -> bool:
    if depth > 6:
        return False
    if isinstance(payload, dict):
        for k, v in payload.items():
            kl = str(k).lower()
            if any(h in kl for h in hints):
                return True
            if _has_key_hint(v, hints, depth + 1):
                return True
    elif isinstance(payload, (list, tuple)):
        return any(_has_key_hint(v, hints, depth + 1) for v in payload)
    return False


def review_methodology(payload: Any, *, text: str = "", n_test_threshold: int = 20) -> list[dict[str, Any]]:
    """Return methodology flags: ``[{issue, severity ('block'|'advisory'), detail}]``.
    Heuristic and conservative — only clear consensus errors are 'block'."""
    flags: list[dict[str, Any]] = []
    lower = (text or "").lower()

    # 1) Uncorrected multiple testing.
    p_like = [p for p in _walk_numbers(payload, "p") if 0.0 <= p <= 1.0]
    n_tests = len(p_like)
    n_tests_field = _walk_numbers(payload, "n_test")
    if n_tests_field:
        n_tests = max(n_tests, int(max(n_tests_field)))
    corrected = _has_key_hint(payload, _CORRECTION_HINTS) or any(
        h in lower for h in ("fdr", "bonferroni", "benjamini", "multiple testing", "multiple-testing", "q-value", "adjusted p")
    )
    if n_tests >= n_test_threshold and not corrected:
        flags.append({
            "issue": "uncorrected_multiple_testing", "severity": "block",
            "detail": f"{n_tests} tests reported with no multiple-testing correction. Apply BH-FDR or "
                      "Bonferroni before calling hits significant.",
        })

    # 2) Sample size below power.
    # n == 0 means "not reported", not "zero samples"; only a positive-but-small count
    # is genuinely underpowered.
    for key, label in (("n_cases", "cases"), ("n_controls", "controls"), ("n_case", "cases"), ("n_control", "controls")):
        small = [int(v) for v in _walk_numbers(payload, key) if 0 < v < _MIN_GROUP_N]
        if small:
            flags.append({
                "issue": "underpowered_group", "severity": "block" if min(small) < 5 else "advisory",
                "detail": f"{label} group n={min(small)} below a usable power floor (<{_MIN_GROUP_N}); "
                          "results from this arm are exploratory only.",
            })
            break
    totals = [int(t) for t in (_walk_numbers(payload, "n_samples") + _walk_numbers(payload, "n_total")) if t > 0]
    if totals and min(totals) < _MIN_TOTAL_N:
        flags.append({"issue": "underpowered_total", "severity": "advisory",
                      "detail": f"total n={min(totals)} (<{_MIN_TOTAL_N}) — interpret with caution."})

    # 3) Effect estimates without confidence intervals.
    has_effect = _has_key_hint(payload, _EFFECT_HINTS) or any(h in lower for h in ("odds ratio", "hazard ratio", "beta ="))
    has_ci = _has_key_hint(payload, _CI_HINTS) or any(h in lower for h in ("95% ci", "confidence interval", "ci:"))
    if has_effect and not has_ci:
        flags.append({"issue": "missing_confidence_interval", "severity": "advisory",
                      "detail": "Effect estimates reported without confidence intervals — add 95% CIs."})

    # 4) Genomic inflation without population-structure correction.
    lam = _walk_numbers(payload, "lambda_gc")
    if lam and max(lam) > 1.1:
        pc_corrected = _has_key_hint(payload, ("pc1", "pc_", "covar", "principal_component")) or any(
            h in lower for h in ("principal component", "pc covariate", "pca covariate", "mixed model", "mixed-model")
        )
        if not pc_corrected:
            flags.append({"issue": "unaddressed_stratification", "severity": "advisory",
                          "detail": f"lambda_gc={max(lam):.2f}>1.1 with no PC/mixed-model correction in evidence — "
                                    "add principal-component covariates (vcf_pca) or a mixed model."})
    # 5) Single-cell / spatial method sins. Text-driven and gated on omics context so they
    # never fire on tabular GWAS/PheWAS evidence (verified by the tabular-regression test).
    if _omics_context(lower):
        de_terms = ("differential expression", "marker gene", "rank_genes", "wilcoxon", " deg", "de analysis")
        de_aggregated = ("pseudobulk", "pseudo-bulk", "per-sample", "per sample", "per-donor",
                         "per donor", "sample-level", "mixed model", "mixed-model", "aggregate", "deseq2", "edger")
        if any(h in lower for h in de_terms) and not any(h in lower for h in de_aggregated):
            flags.append({
                "issue": "pseudoreplicated_de", "severity": "block",
                "detail": "single-cell differential expression appears to treat cells as independent "
                          "replicates; aggregate to pseudobulk per sample/donor (or use a mixed model) first.",
            })
        cluster_terms = ("cluster", "integrat", "umap", "leiden", "louvain")
        batch_terms = ("batch", "multiple samples", "multiple donors", "donors", "across samples")
        batch_corrected = ("harmony", "bbknn", "scvi", "scanorama", "combat", "regress out", "batch correct", "batch-correct")
        if any(h in lower for h in cluster_terms) and any(h in lower for h in batch_terms) \
                and not any(h in lower for h in batch_corrected):
            flags.append({
                "issue": "batch_confounded_clustering", "severity": "advisory",
                "detail": "clustering/integration spans multiple batches with no batch-correction method "
                          "named (Harmony/scVI/BBKNN/...); clusters may track batch, not biology.",
            })
        if "velocity" in lower and not any(h in lower for h in ("spliced", "unspliced", "loom", "velocyto", "scvelo", "kallisto", "kb-python")):
            flags.append({
                "issue": "velocity_without_splicing", "severity": "block",
                "detail": "RNA-velocity claim without spliced/unspliced counts; velocity from a neighbour "
                          "graph alone is invalid — provide spliced/unspliced layers (velocyto/scVelo).",
            })
        spatial_enrich = ("neighborhood enrichment", "neighbourhood enrichment", "co-occurrence", "co-localiz", "colocaliz", "niche enrichment", "spatial enrichment")
        spatial_null = ("permutation", "null model", "null distribution", "label-permut", "random", "monte carlo", "montecarlo", "z-score")
        if any(h in lower for h in spatial_enrich) and not any(h in lower for h in spatial_null):
            flags.append({
                "issue": "spatial_enrichment_no_null", "severity": "advisory",
                "detail": "spatial enrichment/co-occurrence reported with no permutation/null model; "
                          "report enrichment against a label-permuted null.",
            })

    # 6) Colocalization / MR across datasets without allele harmonization (statgen, not omics-
    # gated). Narrowed to explicit coloc/MR phrasing and surfaced as advisory to avoid false
    # hard-blocks on standard GWAS evidence that simply did not spell out "harmonized".
    coloc_terms = ("colocaliz", "colocalis", "coloc.abf", "coloc analysis", "two-sample mr", "two sample mr", "mendelian randomi")
    harmonized = ("harmoniz", "harmonis", "effect allele", "align allele", "allele align", "palindrom", "strand-flip", "strand flip")
    if any(h in lower for h in coloc_terms) and not any(h in lower for h in harmonized):
        flags.append({
            "issue": "coloc_unharmonized_alleles", "severity": "advisory",
            "detail": "colocalization/MR across datasets without explicit allele harmonization; harmonize "
                      "effect alleles and handle palindromic SNPs before coloc/MR.",
        })
    return flags


def methodology_blocks(flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [f for f in (flags or []) if f.get("severity") == "block"]


def _omics_context(text_lower: str) -> bool:
    """True when the evidence text reads as single-cell / spatial omics work."""
    return any(h in text_lower for h in _OMICS_HINTS)


def check_artifact(
    summary: Any,
    *,
    min_obs: int | None = None,
    min_vars: int | None = None,
    require_obs_keys: tuple[str, ...] = (),
    require_layers: tuple[str, ...] = (),
    require_keys: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Validate an AnnData-style JSON *summary* against shape requirements WITHOUT importing
    scanpy/anndata, so a synthesized/ingested skill's gate test can assert its postconditions
    on a tiny synthetic summary (``{n_obs, n_vars, obs_keys, var_keys, layers, ...}``) in the
    secret-stripped worktree. Returns the same ``[{issue, severity, detail}]`` flags."""
    if not isinstance(summary, dict):
        return [{"issue": "artifact_missing", "severity": "block",
                 "detail": "no AnnData-style summary dict was produced"}]

    def _num(v: Any) -> float | None:
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    flags: list[dict[str, Any]] = []
    n_obs, n_vars = _num(summary.get("n_obs")), _num(summary.get("n_vars"))
    if min_obs is not None and (n_obs is None or n_obs < min_obs):
        flags.append({"issue": "artifact_too_few_obs", "severity": "block",
                      "detail": f"n_obs={summary.get('n_obs')!r} below required {min_obs}"})
    if min_vars is not None and (n_vars is None or n_vars < min_vars):
        flags.append({"issue": "artifact_too_few_vars", "severity": "block",
                      "detail": f"n_vars={summary.get('n_vars')!r} below required {min_vars}"})
    obs_keys = set(summary.get("obs_keys") or [])
    for k in require_obs_keys:
        if k not in obs_keys:
            flags.append({"issue": "artifact_missing_obs_key", "severity": "block",
                          "detail": f"obs is missing required column '{k}'"})
    layers = set(summary.get("layers") or [])
    for k in require_layers:
        if k not in layers:
            flags.append({"issue": "artifact_missing_layer", "severity": "block",
                          "detail": f"missing required layer '{k}'"})
    for k in require_keys:
        if k not in summary:
            flags.append({"issue": "artifact_missing_key", "severity": "block",
                          "detail": f"summary is missing required key '{k}'"})
    return flags
