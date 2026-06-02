"""Train a phenotype classifier — designed for small WGS cohorts without ICD10 data."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from biobank_agent.registry import skill
from biobank_agent.utils.plotting import nature_figure, save_figure, PALETTE

logger = logging.getLogger(__name__)


def _select_model(model_type: str):
    mt = model_type.lower().strip()
    if mt in ("auto", "", "logistic", "logreg", "lr"):
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
    if mt in ("rf", "random_forest"):
        return RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    if mt in ("svm", "svc"):
        return SVC(kernel="rbf", probability=True, random_state=42)
    return LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)


def _candidate_models(model_type: str, tune_hyperparameters: bool) -> list[tuple[str, object]]:
    """Return deterministic small-cohort model candidates."""
    mt = model_type.lower().strip()
    if not tune_hyperparameters:
        model = _select_model(mt)
        return [(f"{type(model).__name__}:default", model)]

    candidates: list[tuple[str, object]] = []
    if mt in ("auto", "", "logistic", "logreg", "lr"):
        for c_value in (0.1, 1.0, 10.0):
            candidates.append((
                f"LogisticRegression:C={c_value:g}",
                LogisticRegression(max_iter=2000, solver="lbfgs", C=c_value, random_state=42),
            ))
    if mt in ("auto", "", "rf", "random_forest"):
        depths = (3,) if mt in ("auto", "") else (2, 3, None)
        for depth in depths:
            label = "None" if depth is None else str(depth)
            candidates.append((
                f"RandomForestClassifier:max_depth={label}",
                RandomForestClassifier(
                    n_estimators=50,
                    max_depth=depth,
                    min_samples_leaf=2,
                    random_state=42,
                ),
            ))
    if mt in ("auto", "", "svm", "svc"):
        c_values = (1.0,) if mt in ("auto", "") else (0.5, 1.0, 2.0)
        for c_value in c_values:
            candidates.append((
                f"SVC:rbf:C={c_value:g}",
                SVC(kernel="rbf", C=c_value, gamma="scale", probability=True, random_state=42),
            ))
    if not candidates:
        model = _select_model(mt)
        candidates.append((f"{type(model).__name__}:default", model))
    return candidates


@skill(
    name="train_phenotype_model",
    description=(
        "Train a classifier to predict phenotype group (e.g., Juvenile/Senile/Vitiligo) "
        "from demographic and biomarker features. Designed for small cohorts: performs "
        "deterministic hyperparameter/model selection and uses Leave-One-Out CV when "
        "N < 50, stratified K-fold otherwise. "
        "Stores model in session state for feature_importance and embedding."
    ),
    parameters={
        "target": {
            "type": "string",
            "description": "Target column (default: phenotype_group)",
            "default": "phenotype_group",
        },
        "positive_class": {
            "type": "string",
            "description": "For binary classification: treat this class as positive, rest as negative. "
                           "Leave empty for multiclass.",
            "default": "",
        },
        "model_type": {
            "type": "string",
            "description": "Model: 'auto' (logistic), 'logistic', 'rf', 'svm'",
            "default": "auto",
            "enum": ["auto", "logistic", "rf", "svm"],
        },
        "features": {
            "type": "string",
            "description": "Comma-separated feature column names, or 'all_numeric' (default)",
            "default": "all_numeric",
        },
        "include_classes": {
            "type": "string",
            "description": "Comma-separated target classes to keep before training (empty = all classes)",
            "default": "",
        },
        "n_folds": {
            "type": "integer",
            "description": "CV folds. 0 = auto (LOOCV if N<50, else 5-fold)",
            "default": 0,
        },
        "tune_hyperparameters": {
            "type": "boolean",
            "description": "Compare deterministic small-cohort candidate hyperparameters/models before final fitting.",
            "default": True,
        },
    },
    required=[],
)
def train_phenotype_model(
    target: str = "phenotype_group",
    positive_class: str = "",
    model_type: str = "auto",
    features: str = "all_numeric",
    include_classes: str = "",
    n_folds: int = 0,
    tune_hyperparameters: bool = True,
    *,
    ctx=None,
) -> dict:
    dm = ctx.dm
    id_col = getattr(dm, "subject_id_col", ctx.settings.subject_id_col)

    df = dm.query("SELECT * FROM biomarkers")
    if df.empty:
        return {"error": "No data in biomarkers view."}

    if target not in df.columns:
        return {
            "error": f"Target column '{target}' not found.",
            "available_columns": [c for c in df.columns if c != id_col],
        }

    included_classes = [c.strip() for c in include_classes.split(",") if c.strip()]
    if included_classes:
        before = len(df)
        df = df[df[target].astype(str).isin(included_classes)].copy()
        if df.empty:
            return {
                "error": f"No rows remain after include_classes={include_classes!r}.",
                "available_classes": sorted(pd.Series(ctx.dm.query("SELECT * FROM biomarkers")[target]).dropna().astype(str).unique().tolist()),
            }
        if len(df) < before:
            logger.info("Filtered phenotype model data from %d to %d rows using include_classes=%s", before, len(df), included_classes)

    y_raw = df[target].copy()
    if y_raw.isnull().any():
        df = df.dropna(subset=[target])
        y_raw = df[target].copy()

    if positive_class:
        y_raw = (y_raw == positive_class).astype(int)
        class_names = [f"not_{positive_class}", positive_class]
    else:
        class_names = sorted(y_raw.unique().tolist())

    if len(set(y_raw.tolist())) < 2:
        return {"error": f"Need at least two target classes after filtering; observed {sorted(set(y_raw.tolist()))}."}

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    if features.strip().lower() in ("all_numeric", ""):
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != id_col and c != target
        ]
    else:
        feature_cols = [c.strip() for c in features.split(",") if c.strip() in df.columns]

    if not feature_cols:
        return {"error": "No numeric feature columns available for training."}

    X = df[feature_cols].copy()
    X = X.fillna(X.median())

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols, index=X.index)

    n_samples = len(X_scaled)
    if n_folds <= 0:
        if n_samples < 50:
            cv = LeaveOneOut()
            cv_name = "LOOCV"
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_name = "5-fold"
    else:
        if n_folds >= n_samples:
            cv = LeaveOneOut()
            cv_name = "LOOCV"
        else:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            cv_name = f"{n_folds}-fold"

    tuning_rows: list[dict] = []
    best_candidate: tuple[str, object, np.ndarray] | None = None
    best_score = (-np.inf, -np.inf, -np.inf)
    for label, candidate in _candidate_models(model_type, tune_hyperparameters):
        try:
            pred = cross_val_predict(clone(candidate), X_scaled, y, cv=cv)
            cand_acc = accuracy_score(y, pred)
            cand_bal_acc = balanced_accuracy_score(y, pred)
            cand_report = classification_report(
                y, pred, target_names=[str(c) for c in class_names], output_dict=True, zero_division=0
            )
            cand_macro_f1 = float(cand_report.get("macro avg", {}).get("f1-score", 0.0))
            row = {
                "candidate": label,
                "model_type": type(candidate).__name__,
                "accuracy": round(cand_acc, 4),
                "balanced_accuracy": round(cand_bal_acc, 4),
                "macro_f1": round(cand_macro_f1, 4),
                "status": "ok",
                "error": "",
            }
            score = (cand_bal_acc, cand_macro_f1, cand_acc)
            if score > best_score:
                best_score = score
                best_candidate = (label, candidate, pred)
        except Exception as exc:
            row = {
                "candidate": label,
                "model_type": type(candidate).__name__,
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "macro_f1": np.nan,
                "status": "failed",
                "error": str(exc)[:500],
            }
        tuning_rows.append(row)

    if best_candidate is None:
        return {
            "error": "All phenotype model candidates failed during cross-validation.",
            "tuning_results": tuning_rows,
        }

    best_label, best_model, y_pred = best_candidate
    model = clone(best_model)
    model_name = type(model).__name__

    acc = accuracy_score(y, y_pred)
    bal_acc = balanced_accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(
        y, y_pred, target_names=[str(c) for c in class_names], output_dict=True, zero_division=0
    )

    model.fit(X_scaled, y)

    model_key = f"phenotype_{target}_{model_name.lower()}"
    if hasattr(ctx, "state"):
        ctx.state.models[model_key] = model
        ctx.state.model_metadata[model_key] = {
            "feature_names": feature_cols,
            "class_names": class_names,
            "cv_name": cv_name,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "scaler": scaler,
            "label_encoder": le,
            "best_candidate": best_label,
            "tuning_results": tuning_rows,
        }
        ctx.state.feature_matrix = X_scaled
        ctx.state.labels = pd.Series(y, name="label")

    figures = []
    report_dir = getattr(ctx, "report_dir", None)
    if report_dir is None:
        report_dir = getattr(ctx.settings, "reports_dir", Path("./reports"))
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    tuning_table = report_dir / f"phenotype_model_tuning_{target}.tsv"
    pd.DataFrame(tuning_rows).to_csv(tuning_table, sep="\t", index=False)

    fig, ax = nature_figure(width="single")
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, fontsize=6, rotation=45, ha="right")
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({cv_name}, {model_name})")
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    paths = save_figure(fig, f"phenotype_cm_{target}", report_dir)
    figures.extend(paths)
    if hasattr(ctx, "state") and hasattr(ctx.state, "figures"):
        ctx.state.figures.extend(paths)

    tuning_fig, tuning_ax = nature_figure(width="single")
    tuning_df = pd.DataFrame([row for row in tuning_rows if row["status"] == "ok"])
    if not tuning_df.empty:
        plot_df = tuning_df.sort_values("balanced_accuracy", ascending=True)
        y_pos = np.arange(len(plot_df))
        tuning_ax.barh(y_pos, plot_df["balanced_accuracy"], color=PALETTE[0], alpha=0.85)
        tuning_ax.set_yticks(y_pos)
        tuning_ax.set_yticklabels(plot_df["candidate"], fontsize=5)
        tuning_ax.set_xlabel("Balanced accuracy")
        tuning_ax.set_xlim(0, 1)
        tuning_ax.set_title(f"Model Tuning ({cv_name})")
    else:
        tuning_ax.text(0.5, 0.5, "No successful candidates", ha="center", va="center")
        tuning_ax.set_axis_off()
    tuning_fig.tight_layout()
    tuning_paths = save_figure(tuning_fig, f"phenotype_model_tuning_{target}", report_dir)
    figures.extend(tuning_paths)
    if hasattr(ctx, "state") and hasattr(ctx.state, "figures"):
        ctx.state.figures.extend(tuning_paths)

    return {
        "model_key": model_key,
        "model_type": model_name,
        "best_candidate": best_label,
        "target": target,
        "positive_class": positive_class or None,
        "include_classes": included_classes,
        "n_classes": n_classes,
        "class_names": class_names,
        "n_samples": n_samples,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "cv_method": cv_name,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "tuning_table": str(tuning_table),
        "tuning_results": tuning_rows,
        "confusion_matrix": cm.tolist(),
        "per_class_report": {
            k: v for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "macro_f1": round(report.get("macro avg", {}).get("f1-score", 0), 4),
        "figures": [str(p) for p in figures],
    }
