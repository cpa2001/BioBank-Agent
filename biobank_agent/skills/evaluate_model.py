"""ROC curve, PR curve, and evaluation metrics."""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from biobank_agent.registry import skill
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE


@skill(
    name="evaluate_model",
    description="Generate ROC and precision-recall curves for a trained model. "
                "Shows per-fold curves and the mean curve with AUC.",
    parameters={
        "model_key": {
            "type": "string",
            "description": "Model key (e.g. 'E11_xgb'). If omitted, uses most recent.",
            "default": "",
        },
    },
    required=[],
)
def evaluate_model(model_key: str = "", *, ctx=None) -> dict:
    if not model_key:
        if not ctx.state.models:
            return {"error": "No trained model. Run train_model first."}
        model_key = list(ctx.state.models.keys())[-1]

    if model_key not in ctx.state.models:
        return {"error": f"Model '{model_key}' not found."}

    X = ctx.state.feature_matrix
    y = ctx.state.labels
    if X is None or y is None:
        return {"error": "No feature matrix available. Run train_model first."}

    from sklearn.base import clone
    model = ctx.state.models[model_key]

    # Re-run CV to get fold-level predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_data = []
    pr_data = []
    aucs = []
    aps = []

    for train_idx, test_idx in cv.split(X, y):
        clf = clone(model)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_prob = clf.predict_proba(X.iloc[test_idx])[:, 1]

        fpr, tpr, _ = roc_curve(y.iloc[test_idx], y_prob)
        roc_auc = auc(fpr, tpr)
        roc_data.append((fpr, tpr, roc_auc))
        aucs.append(roc_auc)

        prec, rec, _ = precision_recall_curve(y.iloc[test_idx], y_prob)
        ap = average_precision_score(y.iloc[test_idx], y_prob)
        pr_data.append((rec, prec, ap))
        aps.append(ap)

    # Plot ROC + PR curves side by side
    fig, (ax1, ax2) = nature_figure(nrows=1, ncols=2, width="double")

    # ROC
    for i, (fpr, tpr, roc_auc) in enumerate(roc_data):
        ax1.plot(fpr, tpr, alpha=0.3, color=PALETTE[0], linewidth=0.5)
    # Mean ROC
    mean_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr, _ in roc_data], axis=0)
    mean_auc = np.mean(aucs)
    ax1.plot(mean_fpr, mean_tpr, color=PALETTE[0], linewidth=1.5,
             label=f"Mean AUC = {mean_auc:.3f}")
    ax1.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.5)
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(frameon=False, loc="lower right")

    # PR
    for i, (rec, prec, ap) in enumerate(pr_data):
        ax2.plot(rec, prec, alpha=0.3, color=PALETTE[1], linewidth=0.5)
    mean_ap = np.mean(aps)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"PR Curve (Mean AP = {mean_ap:.3f})")

    fig.suptitle(model_key, fontsize=8, y=1.02)
    fig.tight_layout()
    paths = save_figure(fig, f"roc_pr_{model_key}", ctx.report_dir)
    ctx.state.figures.extend(paths)

    return {
        "model_key": model_key,
        "mean_auc": round(mean_auc, 4),
        "std_auc": round(float(np.std(aucs)), 4),
        "mean_ap": round(mean_ap, 4),
        "fold_aucs": [round(a, 4) for a in aucs],
        "figure": str(paths[0]),
    }
