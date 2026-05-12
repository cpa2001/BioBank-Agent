"""Feature importance — SHAP or tree-based importance."""

import numpy as np
from biobank_agent.data.features import ALL_BIOMARKERS
from biobank_agent.registry import skill
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE


@skill(
    name="feature_importance",
    description="Show feature importance for a trained model. Uses tree-based importance "
                "by default, or SHAP if available. Generates a horizontal bar chart of "
                "the top N most important features.",
    parameters={
        "model_key": {
            "type": "string",
            "description": "Model key from train_model (e.g. 'E11_xgb'). "
                           "If omitted, uses the most recently trained model.",
            "default": "",
        },
        "top_n": {
            "type": "integer",
            "description": "Number of top features to show (default 20)",
            "default": 20,
        },
        "method": {
            "type": "string",
            "description": "Importance method: 'tree' (default) or 'shap'",
            "default": "tree",
            "enum": ["tree", "shap"],
        },
    },
    required=[],
)
def feature_importance(model_key: str = "", top_n: int = 20,
                       method: str = "tree", *, ctx=None) -> dict:
    # Find model
    if not model_key:
        if not ctx.state.models:
            return {"error": "No trained model found. Run train_model first."}
        model_key = list(ctx.state.models.keys())[-1]

    if model_key not in ctx.state.models:
        return {"error": f"Model '{model_key}' not found. Available: {list(ctx.state.models.keys())}"}

    model = ctx.state.models[model_key]
    meta = ctx.state.model_metadata.get(model_key, {})
    feature_names = meta.get("feature_names", [])
    requested_method = method
    fallback_warning = ""

    if method == "shap" and ctx.state.feature_matrix is None:
        method = "tree"
        fallback_warning = "SHAP was requested but the stored feature matrix is unavailable; used tree-based importance instead."

    if method == "shap" and ctx.state.feature_matrix is not None:
        try:
            import shap
            X = ctx.state.feature_matrix
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.iloc[:1000])  # subsample for speed
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # positive class for binary
            shap_arr = np.array(shap_values)
            if shap_arr.ndim == 3:
                shap_arr = shap_arr[..., 1]  # (samples, features, classes) → positive class
            importance = np.abs(shap_arr).mean(axis=0)
            importance_type = "SHAP"
        except (ImportError, Exception) as exc:
            method = "tree"  # fallback
            fallback_warning = f"SHAP was requested but unavailable or failed; used tree-based importance instead ({exc})."

    if method == "tree":
        importance = model.feature_importances_
        importance_type = "Tree-based"

    # Map to names
    named_importance = {}
    for i, fid_col in enumerate(feature_names):
        if i >= len(importance):
            break
        fid = fid_col.split("-")[0] if "-" in fid_col else fid_col
        name = ALL_BIOMARKERS.get(fid, fid_col)
        named_importance[name] = float(importance[i])

    # Sort and get top N
    sorted_imp = sorted(named_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_imp[:top_n]

    # Plot
    fig, ax = nature_figure(width="single", height_ratio=0.04 * top_n)
    names = [f[0] for f in reversed(top_features)]
    values = [f[1] for f in reversed(top_features)]
    ax.barh(range(len(names)), values, color=PALETTE[0], height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel(f"{importance_type} importance")
    ax.set_title(f"Top {top_n} features — {model_key}")
    fig.tight_layout()
    paths = save_figure(fig, f"feature_importance_{model_key}", ctx.report_dir)
    ctx.state.figures.extend(paths)

    return {
        "model_key": model_key,
        "requested_method": requested_method,
        "effective_method": method,
        "importance_type": importance_type,
        "top_features": [{"feature": n, "importance": round(v, 4)} for n, v in top_features],
        "top_feature": top_features[0][0] if top_features else "",
        "warning": fallback_warning,
        "figure": str(paths[0]),
    }
