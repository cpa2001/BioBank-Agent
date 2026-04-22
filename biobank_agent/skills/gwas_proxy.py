"""GWAS-proxy analysis — phenotype-wide association study against disease status."""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)


@skill(
    name="gwas_proxy",
    description="Run phenotype-wide association study as a GWAS proxy. Tests all available "
                "biomarkers and phenotypes against a target disease. Produces Manhattan plot.",
    parameters={
        "icd10_code": {
            "type": "string",
            "description": "Target disease ICD10 code (e.g., 'E11', 'I25')",
        },
        "correction": {
            "type": "string",
            "description": "Multiple testing correction: 'bonferroni' or 'fdr'",
            "default": "fdr",
        },
        "min_effect_size": {
            "type": "number",
            "description": "Minimum absolute effect size (Cohen's d) to report",
            "default": 0.1,
        },
    },
    required=["icd10_code"],
)
def gwas_proxy(
    icd10_code: str,
    correction: str = "fdr",
    min_effect_size: float = 0.1,
    *,
    ctx=None,
) -> dict:
    """Run phenotype-wide association against a disease."""
    id_col = ctx.settings.subject_id_col

    from biobank_agent.data.cohort import build_cohort
    from biobank_agent.data.features import BIOMARKER_GROUPS

    # Build cohort
    try:
        cohort_df = build_cohort(icd10_code, ctx.dm)
    except Exception as e:
        return {"error": f"Failed to build cohort for {icd10_code}: {e}"}

    n_cases = int(cohort_df["label"].sum())
    n_controls = len(cohort_df) - n_cases

    if n_cases < 50:
        return {"error": f"Too few cases ({n_cases}) for GWAS-proxy. Need >= 50."}

    cases = cohort_df[cohort_df["label"] == 1][id_col].tolist()
    controls = cohort_df[cohort_df["label"] == 0][id_col].tolist()

    # Test all biomarker fields
    results = []
    all_fields = {}
    for group_name, field_ids in BIOMARKER_GROUPS.items():
        for fid in field_ids:
            all_fields[fid] = group_name

    tested = 0
    for fid, group in all_fields.items():
        try:
            df = ctx.dm.get_field(fid)
            if df is None or df.empty:
                continue

            col = [c for c in df.columns if c != id_col][0]
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=[col])

            case_vals = df[df[id_col].isin(cases)][col].values
            ctrl_vals = df[df[id_col].isin(controls)][col].values

            if len(case_vals) < 20 or len(ctrl_vals) < 20:
                continue

            # Mann-Whitney U test
            stat, p_value = stats.mannwhitneyu(case_vals, ctrl_vals, alternative="two-sided")

            # Cohen's d effect size
            pooled_std = np.sqrt(
                ((len(case_vals) - 1) * np.var(case_vals) +
                 (len(ctrl_vals) - 1) * np.var(ctrl_vals)) /
                (len(case_vals) + len(ctrl_vals) - 2)
            )
            cohens_d = (np.mean(case_vals) - np.mean(ctrl_vals)) / pooled_std if pooled_std > 0 else 0

            # Get field name from catalog
            field_name = fid
            if hasattr(ctx, "catalog"):
                info = ctx.catalog.fields.get(fid)
                if info:
                    field_name = info.get("title", fid)

            results.append({
                "field_id": fid,
                "field_name": field_name,
                "group": group,
                "p_value": float(p_value),
                "cohens_d": float(cohens_d),
                "abs_effect": abs(float(cohens_d)),
                "direction": "higher_in_cases" if np.mean(case_vals) > np.mean(ctrl_vals) else "lower_in_cases",
                "n_cases_tested": len(case_vals),
                "n_controls_tested": len(ctrl_vals),
                "mean_cases": float(np.mean(case_vals)),
                "mean_controls": float(np.mean(ctrl_vals)),
            })
            tested += 1

        except Exception as e:
            logger.debug("Failed to test field %s: %s", fid, e)
            continue

    if not results:
        return {"error": "No fields could be tested. Check data availability."}

    # Multiple testing correction
    p_values = np.array([r["p_value"] for r in results])

    if correction == "bonferroni":
        threshold = 0.05 / len(p_values)
        for i, r in enumerate(results):
            r["p_corrected"] = min(r["p_value"] * len(p_values), 1.0)
            r["significant"] = r["p_value"] < threshold
    else:  # FDR
        from scipy.stats import false_discovery_control
        try:
            # scipy >= 1.11
            rejected = false_discovery_control(p_values, axis=0)
            for i, r in enumerate(results):
                r["significant"] = bool(rejected[i])
                r["p_corrected"] = r["p_value"]  # FDR doesn't produce adjusted p
        except (ImportError, AttributeError):
            # Fallback: Benjamini-Hochberg manual
            sorted_idx = np.argsort(p_values)
            n = len(p_values)
            threshold_arr = np.arange(1, n + 1) / n * 0.05
            sorted_p = p_values[sorted_idx]
            rejected = sorted_p <= threshold_arr
            # Find largest k where p[k] <= k/n * alpha
            if rejected.any():
                max_k = np.max(np.where(rejected))
                for i, r in enumerate(results):
                    rank = np.searchsorted(sorted_idx, i)
                    r["significant"] = rank <= max_k
                    r["p_corrected"] = r["p_value"]
            else:
                for r in results:
                    r["significant"] = False
                    r["p_corrected"] = r["p_value"]

    # Sort by significance and effect size
    results.sort(key=lambda r: (-r["significant"], r["p_value"]))

    significant = [r for r in results if r["significant"] and r["abs_effect"] >= min_effect_size]

    # Generate Manhattan plot
    fig_paths = _plot_manhattan(results, icd10_code, correction, ctx)
    for p in fig_paths:
        ctx.state.figures.append(p)

    return {
        "icd10_code": icd10_code,
        "n_cases": n_cases,
        "n_controls": n_controls,
        "n_fields_tested": tested,
        "n_significant": len(significant),
        "correction": correction,
        "top_associations": significant[:20],
        "all_results_count": len(results),
        "figures": [str(p) for p in fig_paths],
    }


def _plot_manhattan(results, icd10_code, correction, ctx):
    """Generate Manhattan-style plot."""
    from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE

    fig, ax = nature_figure(width="double", height_ratio=0.5)
    if hasattr(ax, "__len__"):
        ax = ax.flat[0]

    # Assign x positions by group
    groups = sorted(set(r["group"] for r in results))
    group_colors = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(groups)}

    x_pos = 0
    x_ticks = []
    x_labels = []
    group_starts = {}

    for group in groups:
        group_results = [r for r in results if r["group"] == group]
        group_starts[group] = x_pos

        for r in group_results:
            neg_log_p = -np.log10(max(r["p_value"], 1e-300))
            color = group_colors[group]
            alpha = 0.9 if r["significant"] else 0.3
            size = 4 if r["significant"] else 2
            ax.scatter(x_pos, neg_log_p, c=color, s=size, alpha=alpha, edgecolors="none")

            # Label significant hits
            if r["significant"] and r["abs_effect"] >= 0.2:
                label = r["field_name"][:15] if len(r["field_name"]) > 15 else r["field_name"]
                ax.annotate(label, (x_pos, neg_log_p), fontsize=3.5,
                            rotation=45, ha="left", va="bottom")
            x_pos += 1

        mid = group_starts[group] + len(group_results) / 2
        x_ticks.append(mid)
        x_labels.append(group.replace("_", "\n"))

    # Significance line
    if correction == "bonferroni":
        threshold = -np.log10(0.05 / len(results))
    else:
        threshold = -np.log10(0.05)
    ax.axhline(threshold, color="red", linestyle="--", linewidth=0.5, alpha=0.7)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=4)
    ax.set_xlabel("Biomarker Group")
    ax.set_ylabel("$-\\log_{10}$(P-value)")
    ax.set_title(f"Phenotype-Wide Association: {icd10_code}")

    fig.tight_layout()
    return save_figure(fig, f"gwas_proxy_{icd10_code}", ctx.report_dir)
