"""Comorbidity network — disease co-occurrence analysis."""

import numpy as np
from biobank_agent.registry import skill
from biobank_agent.utils.icd10 import icd10_name
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE


@skill(
    name="comorbidity",
    description="Analyse disease comorbidity: which diseases commonly co-occur with a "
                "target disease? Computes odds ratios, generates a network/bar chart.",
    parameters={
        "icd10_code": {
            "type": "string",
            "description": "Target ICD10 code (e.g. 'E11')",
        },
        "top_n": {
            "type": "integer",
            "description": "Number of top comorbidities to show (default 15)",
            "default": 15,
        },
    },
    required=["icd10_code"],
)
def comorbidity(icd10_code: str, top_n: int = 15, *, ctx=None) -> dict:
    dm = ctx.dm
    diag_col = ctx.settings.diagnoses_code_col
    id_col = ctx.settings.subject_id_col

    # Get patients with target disease
    target_df = dm.query(f"""
        SELECT DISTINCT {id_col} FROM diagnoses
        WHERE {diag_col} LIKE '{icd10_code}%'
    """)
    target_eids = set(target_df[id_col].tolist())
    n_target = len(target_eids)

    # Get all diagnosis codes for these patients
    comorbid_df = dm.query(f"""
        SELECT LEFT({diag_col}, 3) AS code, COUNT(DISTINCT {id_col}) AS n_co
        FROM diagnoses
        WHERE {id_col} IN (SELECT DISTINCT {id_col} FROM diagnoses WHERE {diag_col} LIKE '{icd10_code}%')
          AND LEFT({diag_col}, 3) != '{icd10_code[:3]}'
          AND {diag_col} IS NOT NULL
        GROUP BY code
        HAVING n_co >= 100
        ORDER BY n_co DESC
    """)

    # Get total prevalence of each comorbid disease
    total_n = dm.count_subjects()
    results = []
    for _, row in comorbid_df.head(top_n * 2).iterrows():
        code = row["code"]
        n_co = int(row["n_co"])

        # Total with this disease
        total_with = dm.query(f"""
            SELECT COUNT(DISTINCT {id_col}) AS n FROM diagnoses
            WHERE {diag_col} LIKE '{code}%'
        """)["n"].iloc[0]

        # Odds ratio
        a = n_co                          # target + comorbid
        b = n_target - n_co               # target, no comorbid
        c = total_with - n_co             # comorbid, no target
        d = total_n - n_target - c        # neither

        if b <= 0 or c <= 0 or d <= 0:
            continue
        odds_ratio = (a * d) / (b * c)
        results.append({
            "code": code,
            "name": icd10_name(code),
            "n_comorbid": n_co,
            "prevalence_in_target": round(n_co / n_target * 100, 1),
            "prevalence_overall": round(total_with / total_n * 100, 1),
            "odds_ratio": round(float(odds_ratio), 2),
        })

    results.sort(key=lambda x: x["odds_ratio"], reverse=True)
    results = results[:top_n]

    # Bar chart of odds ratios
    fig, ax = nature_figure(width="single", height_ratio=0.04 * top_n)
    names = [f"{r['code']} {r['name']}" for r in reversed(results)]
    ors = [r["odds_ratio"] for r in reversed(results)]
    colors = [PALETTE[1] if o > 2 else PALETTE[0] for o in ors]
    ax.barh(range(len(names)), ors, color=colors, height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Odds Ratio")
    ax.axvline(x=1, color="grey", linestyle="--", linewidth=0.5)
    ax.set_title(f"Comorbidities of {icd10_code} {icd10_name(icd10_code)}")
    fig.tight_layout()
    paths = save_figure(fig, f"comorbidity_{icd10_code}", ctx.report_dir)
    ctx.state.figures.extend(paths)

    return {
        "target": f"{icd10_code} {icd10_name(icd10_code)}",
        "n_target_patients": n_target,
        "comorbidities": results,
        "figure": str(paths[0]),
    }
