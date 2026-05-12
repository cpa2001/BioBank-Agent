"""Automated scientific discovery pipeline.

Orchestrates: cohort building -> model training -> feature importance ->
PheWAS -> biomarker identification -> discovery report.
"""

import logging

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)


@skill(
    name="discover",
    description="Run automated scientific discovery pipeline for a disease: "
                "build cohort, train model, identify important features, run PheWAS, "
                "and identify disease-specific biomarkers. Produces comprehensive report.",
    parameters={
        "icd10_code": {
            "type": "string",
            "description": "Target disease ICD10 code (e.g., 'E11', 'I25', 'I10')",
        },
        "discovery_depth": {
            "type": "string",
            "description": "'quick' (features only), 'standard' (+ PheWAS), or 'deep' (+ literature)",
            "default": "standard",
        },
        "model_type": {
            "type": "string",
            "description": "Model type for training: 'auto', 'xgb', 'lgbm', or 'catboost'",
            "default": "auto",
            "enum": ["auto", "xgb", "lgbm", "catboost", "xgboost", "lightgbm"],
        },
    },
    required=["icd10_code"],
)
def discover(
    icd10_code: str,
    discovery_depth: str = "standard",
    model_type: str = "auto",
    *,
    ctx=None,
) -> dict:
    """Run automated scientific discovery pipeline."""
    steps_completed = []
    results = {"icd10_code": icd10_code, "depth": discovery_depth}
    all_figures = []

    # ── Step 1: Build cohort ────────────────────────────────
    logger.info("Discovery Step 1: Building cohort for %s", icd10_code)
    try:
        cohort_result = ctx.dm.registry_ref.execute(
            "cohort_summary", {"icd10_code": icd10_code}, ctx=ctx
        ) if hasattr(ctx.dm, "registry_ref") else _run_cohort(icd10_code, ctx)

        steps_completed.append("cohort_built")
        results["cohort"] = {
            "n_cases": cohort_result.get("n_cases", "?"),
            "n_controls": cohort_result.get("n_controls", "?"),
        }
    except Exception as e:
        logger.warning("Cohort building failed: %s", e)
        results["cohort"] = {"error": str(e)}

    # ── Step 2: Train model ─────────────────────────────────
    logger.info("Discovery Step 2: Training %s model", model_type)
    try:
        from biobank_agent.skills.train_model import train_model
        model_result = train_model(
            icd10_code=icd10_code,
            model_type=model_type,
            ctx=ctx,
        )
        if model_result.get("error"):
            raise RuntimeError(model_result["error"])
        trained_model_key = model_result.get("model_key")
        trained_model_type = model_result.get("selected_model_type") or model_result.get("model_type", model_type)
        steps_completed.append("model_trained")
        results["model"] = {
            "type": trained_model_type,
            "requested_type": model_type,
            "model_key": trained_model_key,
            "auc": model_result.get("mean_auc", model_result.get("auc_mean", model_result.get("auc"))),
            "n_features": model_result.get("n_features"),
            "selection_rationale": model_result.get("selection_rationale"),
            "fallback": model_result.get("fallback"),
        }
        if model_result.get("figures"):
            all_figures.extend(model_result["figures"])
    except Exception as e:
        logger.warning("Model training failed: %s", e)
        results["model"] = {"error": str(e)}

    # ── Step 3: Feature importance ──────────────────────────
    logger.info("Discovery Step 3: Feature importance analysis")
    try:
        from biobank_agent.skills.feature_importance import feature_importance
        model_key = results.get("model", {}).get("model_key") or f"{icd10_code}_{model_type}"
        fi_result = feature_importance(model_key=model_key, ctx=ctx)
        steps_completed.append("features_ranked")
        top_features = fi_result.get("top_features", [])[:10]
        results["top_features"] = top_features
        if fi_result.get("figures"):
            all_figures.extend(fi_result["figures"])
    except Exception as e:
        logger.warning("Feature importance failed: %s", e)
        results["top_features"] = {"error": str(e)}
        top_features = []

    # ── Step 4: PheWAS (if standard or deep) ────────────────
    if discovery_depth in ("standard", "deep"):
        logger.info("Discovery Step 4: PheWAS analysis")
        try:
            from biobank_agent.skills.gwas_proxy import gwas_proxy
            phewas_result = gwas_proxy(
                icd10_code=icd10_code,
                correction="fdr",
                ctx=ctx,
            )
            steps_completed.append("phewas_done")
            results["phewas"] = {
                "n_significant": phewas_result.get("n_significant"),
                "top_associations": phewas_result.get("top_associations", [])[:5],
            }
            if phewas_result.get("figures"):
                all_figures.extend(phewas_result["figures"])
        except Exception as e:
            logger.warning("PheWAS failed: %s", e)
            results["phewas"] = {"error": str(e)}

    # ── Step 5: Literature search (if deep) ─────────────────
    if discovery_depth == "deep":
        logger.info("Discovery Step 5: Literature search")
        try:
            from biobank_agent.skills.web_search import web_search
            lit_results = web_search(
                query=f"{icd10_code} biomarkers UK Biobank",
                max_results=5,
                ctx=ctx,
            )
            steps_completed.append("literature_searched")
            results["literature"] = lit_results.get("results", [])[:5]
        except Exception as e:
            logger.warning("Literature search failed (non-critical): %s", e)
            results["literature"] = {"note": "Web search unavailable", "error": str(e)}

    # ── Step 6: Synthesize findings ─────────────────────────
    discovery_summary = _synthesize_findings(icd10_code, results, top_features)
    results["summary"] = discovery_summary
    results["steps_completed"] = steps_completed
    results["figures"] = all_figures

    return results


def _run_cohort(icd10_code: str, ctx) -> dict:
    """Build cohort directly using data layer."""
    from biobank_agent.data.cohort import build_cohort

    df = build_cohort(icd10_code, ctx.dm)
    n_cases = int(df["label"].sum())
    n_controls = len(df) - n_cases

    # Store in session state
    ctx.state.cohorts[icd10_code] = df

    return {"n_cases": n_cases, "n_controls": n_controls, "n_total": len(df)}


def _synthesize_findings(icd10_code: str, results: dict, top_features: list) -> str:
    """Generate a synthesis of all discovery findings."""
    parts = [f"Discovery pipeline for {icd10_code}:"]

    # Cohort
    cohort = results.get("cohort", {})
    if "error" not in cohort:
        parts.append(
            f"- Cohort: {cohort.get('n_cases', '?')} cases, "
            f"{cohort.get('n_controls', '?')} controls"
        )

    # Model
    model = results.get("model", {})
    if "error" not in model:
        auc = model.get("auc")
        if auc:
            try:
                auc_f = float(auc)
                quality = (
                    "excellent" if auc_f > 0.9 else
                    "good" if auc_f > 0.8 else
                    "moderate" if auc_f > 0.7 else "limited"
                )
                parts.append(f"- Model: {model.get('type')} AUC={auc_f:.4f} ({quality})")
            except (TypeError, ValueError):
                parts.append(f"- Model: {model.get('type')} trained")

    # Top features
    if top_features and not isinstance(top_features, dict):
        features_str = ", ".join(str(f) for f in top_features[:5])
        parts.append(f"- Top biomarkers: {features_str}")

    # PheWAS
    phewas = results.get("phewas", {})
    if "error" not in phewas:
        n_sig = phewas.get("n_significant", 0)
        parts.append(f"- PheWAS: {n_sig} significant phenotype associations")

    # Literature
    lit = results.get("literature", {})
    if isinstance(lit, list) and lit:
        parts.append(f"- Literature: {len(lit)} relevant papers found")

    return "\n".join(parts)
