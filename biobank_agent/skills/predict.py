"""Patient-level prediction — predict disease risk using trained models."""

import logging

import numpy as np
import pandas as pd

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)


@skill(
    name="predict",
    description="Predict disease risk for individual patients or an unseen cohort using a "
                "previously trained model. Returns risk scores, risk categories, and confidence.",
    parameters={
        "model_key": {
            "type": "string",
            "description": "Key of trained model in session (e.g., 'E11:xgboost')",
        },
        "patient_eids": {
            "type": "string",
            "description": "Comma-separated participant EIDs, or 'unseen' for held-out test set",
            "default": "unseen",
        },
        "horizon_years": {
            "type": "integer",
            "description": "Prediction horizon in years (informational label, does not change model)",
            "default": 10,
        },
    },
    required=["model_key"],
)
def predict(
    model_key: str,
    patient_eids: str = "unseen",
    horizon_years: int = 10,
    *,
    ctx=None,
) -> dict:
    """Predict disease risk for individual patients."""
    # Retrieve trained model
    if model_key not in ctx.state.models:
        available = list(ctx.state.models.keys())
        return {"error": f"Model '{model_key}' not found. Available: {available}"}

    model = ctx.state.models[model_key]
    meta = ctx.state.model_metadata.get(model_key, {})

    # Get feature names from model metadata
    feature_names = meta.get("feature_names", [])
    if not feature_names:
        return {"error": "Model metadata missing feature_names. Retrain with latest agent version."}

    # Get patient data
    if patient_eids == "unseen":
        # Use held-out test set if available
        if ctx.state.feature_matrix is not None and ctx.state.labels is not None:
            X = ctx.state.feature_matrix
            y = ctx.state.labels
            # Use last 20% as test set (mirroring train_model split)
            n_test = max(1, len(X) // 5)
            X_pred = X.iloc[-n_test:]
            y_actual = y.iloc[-n_test:]
            eids = X_pred.index.tolist()
        else:
            return {"error": "No feature matrix in session. Run train_model first."}
    else:
        # Fetch data for specific EIDs
        eid_list = [e.strip() for e in patient_eids.split(",")]
        try:
            # Build feature matrix for specified EIDs
            X_pred = _fetch_patient_features(eid_list, feature_names, ctx)
            y_actual = None
            eids = eid_list
        except Exception as e:
            return {"error": f"Failed to fetch patient data: {e}"}

    # Ensure correct feature columns
    missing_cols = set(feature_names) - set(X_pred.columns)
    for col in missing_cols:
        X_pred[col] = np.nan
    X_pred = X_pred[feature_names]

    # Impute missing values (simple median fill)
    X_pred = X_pred.fillna(X_pred.median())

    # Predict probabilities
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_pred)[:, 1]
        else:
            probs = model.predict(X_pred)
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

    # Risk categories
    risk_categories = []
    for p in probs:
        if p >= 0.7:
            risk_categories.append("high")
        elif p >= 0.3:
            risk_categories.append("medium")
        else:
            risk_categories.append("low")

    # Build results
    predictions = []
    for i, eid in enumerate(eids[:50]):  # Limit output to 50
        pred = {
            "eid": str(eid),
            "risk_score": round(float(probs[i]), 4),
            "risk_category": risk_categories[i],
        }
        if y_actual is not None:
            pred["actual_label"] = int(y_actual.iloc[i])
        predictions.append(pred)

    # Generate risk distribution figure
    fig_paths = _plot_risk_distribution(probs, risk_categories, model_key, horizon_years, ctx)
    for p in fig_paths:
        ctx.state.figures.append(p)

    # Summary stats
    result = {
        "model_key": model_key,
        "n_patients": len(probs),
        "horizon_years": horizon_years,
        "mean_risk": round(float(np.mean(probs)), 4),
        "median_risk": round(float(np.median(probs)), 4),
        "high_risk_count": sum(1 for c in risk_categories if c == "high"),
        "medium_risk_count": sum(1 for c in risk_categories if c == "medium"),
        "low_risk_count": sum(1 for c in risk_categories if c == "low"),
        "predictions": predictions[:20],  # Return top 20 for display
        "figures": [str(p) for p in fig_paths],
    }

    if y_actual is not None:
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_actual, probs)
            result["test_auc"] = round(float(auc), 4)
        except Exception:
            pass

    return result


def _fetch_patient_features(
    eid_list: list[str],
    feature_names: list[str],
    ctx,
) -> pd.DataFrame:
    """Fetch biomarker data for specific patient EIDs."""
    # Use DataManager to query features
    from biobank_agent.data.features import BIOMARKER_GROUPS

    # Collect all field IDs needed
    field_ids = set()
    for group in BIOMARKER_GROUPS.values():
        for fid in group:
            col_name = f"{fid}-0.0"
            if col_name in feature_names:
                field_ids.add(fid)

    if not field_ids:
        # Fallback: try to parse field IDs from feature names
        for name in feature_names:
            if "-" in name:
                fid = name.split("-")[0]
                if fid.isdigit():
                    field_ids.add(fid)

    # Query from DuckDB
    eid_str = ", ".join(f"'{e}'" for e in eid_list)
    dfs = []
    for fid in field_ids:
        try:
            df = ctx.dm.get_field(fid)
            if df is not None:
                df = df[df["eid"].astype(str).isin(eid_list)]
                dfs.append(df.set_index("eid"))
        except Exception:
            continue

    if not dfs:
        raise ValueError("No features found for specified EIDs")

    result = pd.concat(dfs, axis=1)
    return result


def _plot_risk_distribution(probs, risk_categories, model_key, horizon_years, ctx):
    """Generate risk distribution and waterfall plots."""
    from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE

    fig, axes = nature_figure(nrows=1, ncols=2, width="double")

    # Left: Risk score histogram
    ax1 = axes[0]
    colors = [PALETTE[0] if c == "low" else PALETTE[3] if c == "medium" else PALETTE[1]
              for c in risk_categories]
    ax1.hist(probs, bins=30, color=PALETTE[0], edgecolor="white", linewidth=0.3, alpha=0.8)
    ax1.axvline(0.3, color=PALETTE[3], linestyle="--", linewidth=0.5, label="Medium threshold")
    ax1.axvline(0.7, color=PALETTE[1], linestyle="--", linewidth=0.5, label="High threshold")
    ax1.set_xlabel("Predicted Risk Score")
    ax1.set_ylabel("Count")
    ax1.set_title(f"{model_key} Risk Distribution")
    ax1.legend(fontsize=5)

    # Right: Top 20 patients waterfall
    ax2 = axes[1]
    sorted_idx = np.argsort(probs)[::-1][:20]
    top_probs = probs[sorted_idx]
    top_colors = [PALETTE[1] if p >= 0.7 else PALETTE[3] if p >= 0.3 else PALETTE[0]
                  for p in top_probs]
    ax2.barh(range(len(top_probs)), top_probs, color=top_colors, edgecolor="white", linewidth=0.3)
    ax2.set_xlabel("Risk Score")
    ax2.set_ylabel("Patient Rank")
    ax2.set_title(f"Top 20 Highest Risk ({horizon_years}-yr)")
    ax2.invert_yaxis()

    fig.tight_layout()
    return save_figure(fig, f"risk_prediction_{model_key.replace(':', '_')}", ctx.report_dir)
