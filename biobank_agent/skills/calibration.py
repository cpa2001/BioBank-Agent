"""Calibration analysis — reliability diagram, ECE, MCE, Brier score."""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from biobank_agent.registry import skill
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE


@skill(
    name="calibration",
    description="Evaluate model calibration with a reliability diagram and metrics "
                "(ECE, MCE, Brier score). Shows how well predicted probabilities match "
                "observed frequencies.",
    parameters={
        "model_key": {
            "type": "string",
            "description": "Model key (e.g. 'E11_xgb'). Uses most recent if omitted.",
            "default": "",
        },
        "n_bins": {
            "type": "integer",
            "description": "Number of calibration bins (default 10)",
            "default": 10,
        },
    },
    required=[],
)
def calibration(model_key: str = "", n_bins: int = 10, *, ctx=None) -> dict:
    if not model_key:
        if not ctx.state.models:
            return {"error": "No trained model. Run train_model first."}
        model_key = list(ctx.state.models.keys())[-1]

    model = ctx.state.models.get(model_key)
    X = ctx.state.feature_matrix
    y = ctx.state.labels
    if model is None or X is None or y is None:
        return {"error": "No model/data. Run train_model first."}

    # Get predictions
    y_prob = model.predict_proba(X)[:, 1]

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=n_bins, strategy="uniform")

    # ECE and MCE
    bin_sizes = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    bin_weights = bin_sizes / len(y_prob)
    ece = float(np.sum(np.abs(prob_true - prob_pred) * bin_weights[:len(prob_true)]))
    mce = float(np.max(np.abs(prob_true - prob_pred)))
    brier = float(brier_score_loss(y, y_prob))

    # Plot reliability diagram
    fig, ax = nature_figure(width="single")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.5, label="Perfectly calibrated")
    ax.plot(prob_pred, prob_true, "o-", color=PALETTE[0], markersize=4, linewidth=1,
            label=f"Model (ECE={ece:.3f})")
    ax.fill_between(prob_pred, prob_pred, prob_true, alpha=0.15, color=PALETTE[0])
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration: {model_key}")
    ax.legend(frameon=False, fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    paths = save_figure(fig, f"calibration_{model_key}", ctx.report_dir)
    ctx.state.figures.extend(paths)

    return {
        "model_key": model_key,
        "ece": round(ece, 4),
        "mce": round(mce, 4),
        "brier_score": round(brier, 4),
        "n_bins": n_bins,
        "bin_data": [
            {"predicted": round(float(p), 3), "observed": round(float(t), 3)}
            for p, t in zip(prob_pred, prob_true)
        ],
        "figure": str(paths[0]),
    }
