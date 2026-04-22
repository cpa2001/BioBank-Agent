"""PheWAS — Phenome-Wide Association Study."""

import numpy as np
from biobank_agent.data.features import ALL_BIOMARKERS
from biobank_agent.registry import skill
from biobank_agent.utils.icd10 import icd10_name, icd10_chapter
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE
from biobank_agent.utils.stats import mann_whitney, benjamini_hochberg


@skill(
    name="phewas",
    description="Run a Phenome-Wide Association Study: test whether a biomarker is "
                "associated with each ICD10 disease code. Generates a Manhattan-style plot "
                "with FDR-corrected p-values.",
    parameters={
        "field_id": {
            "type": "string",
            "description": "Biomarker field ID (e.g. '30710' for CRP, '30740' for glucose)",
        },
        "min_cases": {
            "type": "integer",
            "description": "Minimum cases per disease to test (default 500)",
            "default": 500,
        },
    },
    required=["field_id"],
)
def phewas(field_id: str, min_cases: int = 500, *, ctx=None) -> dict:
    dm = ctx.dm
    diag_col = ctx.settings.diagnoses_code_col
    id_col = ctx.settings.subject_id_col
    biomarker_name = ALL_BIOMARKERS.get(field_id, f"Field {field_id}")

    # Get biomarker values
    col = f'"{field_id}-0.0"'
    biomarker_df = dm.query(f"SELECT {id_col}, {col} AS value FROM biomarkers WHERE {col} IS NOT NULL")

    # Get disease counts (top diseases with enough cases)
    disease_counts = dm.query(f"""
        SELECT LEFT({diag_col}, 3) AS code, COUNT(DISTINCT {id_col}) AS n
        FROM diagnoses
        WHERE {diag_col} IS NOT NULL
        GROUP BY code HAVING n >= {min_cases}
        ORDER BY n DESC
    """)

    # Test each disease
    results = []
    all_eids = set(biomarker_df[id_col].tolist())

    for _, row in disease_counts.iterrows():
        code = row["code"]
        # Get case eids for this disease
        case_df = dm.query(f"""
            SELECT DISTINCT {id_col} FROM diagnoses
            WHERE {diag_col} LIKE '{code}%'
        """)
        case_eids = set(case_df[id_col].tolist()) & all_eids
        control_eids = all_eids - case_eids

        if len(case_eids) < min_cases or len(control_eids) < 100:
            continue

        cases_vals = biomarker_df[biomarker_df[id_col].isin(case_eids)]["value"].values
        ctrls_vals = biomarker_df[biomarker_df[id_col].isin(control_eids)]["value"].values

        test = mann_whitney(cases_vals.astype(float), ctrls_vals.astype(float))
        chap_num, chap_name = icd10_chapter(code)

        results.append({
            "code": code,
            "name": icd10_name(code),
            "chapter": chap_num,
            "chapter_name": chap_name,
            "n_cases": len(case_eids),
            "p_value": test["p_value"],
            "effect_size": test["effect_size_r"],
            "cases_mean": float(np.nanmean(cases_vals)),
            "controls_mean": float(np.nanmean(ctrls_vals)),
        })

    if not results:
        return {"error": f"No diseases with >= {min_cases} cases found"}

    # FDR correction
    p_values = np.array([r["p_value"] for r in results])
    fdr = benjamini_hochberg(p_values)
    for i, r in enumerate(results):
        r["fdr_q"] = float(fdr[i])
        r["neg_log10_p"] = float(-np.log10(max(r["p_value"], 1e-300)))

    results.sort(key=lambda x: x["p_value"])

    # Manhattan plot
    fig, ax = nature_figure(width="double", height_ratio=0.4)

    # Color by chapter
    chapters = sorted(set(r["chapter"] for r in results))
    chap_colors = {ch: PALETTE[i % len(PALETTE)] for i, ch in enumerate(chapters)}

    x_pos = 0
    x_ticks = []
    x_labels = []
    for ch in chapters:
        ch_results = [r for r in results if r["chapter"] == ch]
        xs = list(range(x_pos, x_pos + len(ch_results)))
        ys = [r["neg_log10_p"] for r in ch_results]
        ax.scatter(xs, ys, c=chap_colors[ch], s=8, alpha=0.7, edgecolors="none")
        x_ticks.append(x_pos + len(ch_results) // 2)
        x_labels.append(ch)
        x_pos += len(ch_results) + 2

    # Significance line (Bonferroni)
    bonf_thresh = -np.log10(0.05 / len(results))
    ax.axhline(y=bonf_thresh, color="red", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.text(x_pos, bonf_thresh, " Bonferroni", fontsize=5, color="red", va="bottom")

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=5)
    ax.set_xlabel("ICD10 Chapter")
    ax.set_ylabel("-log₁₀(p)")
    ax.set_title(f"PheWAS: {biomarker_name} (field {field_id})")
    fig.tight_layout()
    paths = save_figure(fig, f"phewas_{field_id}", ctx.report_dir)
    ctx.state.figures.extend(paths)

    # Top associations
    significant = [r for r in results if r["fdr_q"] < 0.05]

    return {
        "biomarker": biomarker_name,
        "field_id": field_id,
        "n_diseases_tested": len(results),
        "n_significant_fdr05": len(significant),
        "bonferroni_threshold": round(bonf_thresh, 2),
        "top_associations": [
            {
                "code": r["code"],
                "name": r["name"],
                "p_value": f"{r['p_value']:.2e}",
                "fdr_q": f"{r['fdr_q']:.2e}",
                "effect_size": round(r["effect_size"], 3),
            }
            for r in results[:15]
        ],
        "figure": str(paths[0]),
    }
