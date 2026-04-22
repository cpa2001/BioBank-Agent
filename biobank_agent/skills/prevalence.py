"""Disease prevalence calculation from ICD10 diagnosis records."""

from biobank_agent.registry import skill
from biobank_agent.utils.icd10 import icd10_name
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE


@skill(
    name="prevalence",
    description="Calculate disease prevalence in the biobank cohort. Returns top N most common "
                "ICD10 3-character codes with patient counts and prevalence percentages. "
                "Generates a horizontal bar chart.",
    parameters={
        "top_n": {
            "type": "integer",
            "description": "Number of top diseases to show (default 20)",
            "default": 20,
        },
        "chapter_filter": {
            "type": "string",
            "description": "Optional ICD10 chapter letter to filter (e.g. 'E' for endocrine)",
            "default": "",
        },
    },
    required=[],
)
def prevalence(top_n: int = 20, chapter_filter: str = "", *, ctx=None) -> dict:
    dm = ctx.dm
    diag_col = ctx.settings.diagnoses_code_col
    id_col = ctx.settings.subject_id_col

    # Count unique patients per 3-char ICD10 code
    sql = f"""
        SELECT LEFT({diag_col}, 3) AS code, COUNT(DISTINCT {id_col}) AS n_patients
        FROM diagnoses
        WHERE {diag_col} IS NOT NULL AND {diag_col} != ''
    """
    params = []
    if chapter_filter:
        sql += f" AND {diag_col} LIKE ?"
        params.append(f"{chapter_filter.upper()}%")
    sql += " GROUP BY code ORDER BY n_patients DESC"
    sql += f" LIMIT {top_n}"

    df = dm.query(sql, params if params else None)
    total = dm.count_subjects()

    df["prevalence_pct"] = (df["n_patients"] / total * 100).round(2)
    df["disease_name"] = df["code"].apply(icd10_name)

    # Plot
    fig, ax = nature_figure(width="single", height_ratio=0.05 * top_n)
    y_pos = range(len(df) - 1, -1, -1)
    ax.barh(list(y_pos), df["n_patients"].values, color=PALETTE[0], height=0.7)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels([f"{r['code']} {r['disease_name']}" for _, r in df.iterrows()])
    ax.set_xlabel("Number of patients")
    bank_name = ctx.settings.biobank_name if ctx and hasattr(ctx, "settings") else "Biobank"
    ax.set_title(f"Top {top_n} diseases in {bank_name} (N={total:,})")

    # Add percentage labels
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["n_patients"], len(df) - 1 - i,
                f" {row['prevalence_pct']}%", va="center", fontsize=5)

    fig.tight_layout()
    paths = save_figure(fig, "prevalence", ctx.report_dir)
    ctx.state.figures.extend(paths)

    return {
        "total_subjects": total,
        "top_diseases": df[["code", "disease_name", "n_patients", "prevalence_pct"]].to_dict("records"),
        "figure": str(paths[0]),
    }
