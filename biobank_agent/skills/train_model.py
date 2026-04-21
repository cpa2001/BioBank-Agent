"""Train a predictive model (XGBoost, LightGBM, CatBoost) with cross-validation."""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score

from biobank_agent.data.cohort import build_cohort
from biobank_agent.data.features import ALL_BIOMARKERS, rename_columns
from biobank_agent.registry import skill
from biobank_agent.utils.icd10 import icd10_name


def _get_estimator(model_type: str):
    """Return a configured estimator."""
    if model_type == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, gamma=0.7,
            reg_alpha=0.5, eval_metric="logloss", random_state=42,
        )
    elif model_type == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=20, reg_alpha=0.5,
            reg_lambda=1.0, is_unbalance=True, random_state=42, verbose=-1,
        )
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=300, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=42, verbose=0,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'xgb', 'lgbm', or 'catboost'.")


@skill(
    name="train_model",
    description="Train a machine learning model to predict a disease from biomarkers. "
                "Performs 5-fold stratified cross-validation and reports AUC with 95% CI. "
                "The trained model is stored in session state for subsequent analysis.",
    parameters={
        "icd10_code": {
            "type": "string",
            "description": "ICD10 code prefix (e.g. 'E11')",
        },
        "model_type": {
            "type": "string",
            "description": "Model type: 'xgb' (XGBoost), 'lgbm' (LightGBM), or 'catboost'",
            "default": "xgb",
            "enum": ["xgb", "lgbm", "catboost"],
        },
        "n_folds": {
            "type": "integer",
            "description": "Number of CV folds (default 5)",
            "default": 5,
        },
    },
    required=["icd10_code"],
)
def train_model(icd10_code: str, model_type: str = "xgb", n_folds: int = 5, *, ctx=None) -> dict:
    dm = ctx.dm

    # Build or reuse cohort
    cohort_key = f"{icd10_code}_1:4"
    if cohort_key in ctx.state.cohorts:
        df = ctx.state.cohorts[cohort_key]
    else:
        df = build_cohort(dm, icd10_code, controls_ratio=4)
        ctx.state.cohorts[cohort_key] = df

    n_cases = int(df["label"].sum())
    if n_cases < 100:
        return {"error": f"Insufficient cases (n={n_cases}). Minimum 100 required."}

    # Prepare features
    feature_cols = [c for c in df.columns
                    if c not in ("eid", "label")
                    and df[c].dtype in ("float64", "float32", "int64", "int32")]

    X = df[feature_cols].copy()
    y = df["label"].copy()

    # Impute missing with median
    X = X.fillna(X.median())

    # Drop columns that are all-constant
    nunique = X.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    X = X.drop(columns=const_cols)

    # Train with cross-validation
    estimator = _get_estimator(model_type)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    scoring = {
        "auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }

    results = cross_validate(
        estimator, X, y, cv=cv, scoring=scoring,
        return_estimator=True, n_jobs=-1,
    )

    # Compute metrics with 95% CI (t-distribution, df=n_folds-1)
    from scipy.stats import t as t_dist
    auc_scores = results["test_auc"]
    auc_mean = float(np.mean(auc_scores))
    auc_std = float(np.std(auc_scores))
    t_crit = t_dist.ppf(0.975, df=n_folds - 1)
    auc_ci_low = float(auc_mean - t_crit * auc_std / np.sqrt(n_folds))
    auc_ci_high = float(auc_mean + t_crit * auc_std / np.sqrt(n_folds))

    # Store best model (by AUC) in state
    best_idx = int(np.argmax(auc_scores))
    model_key = f"{icd10_code}_{model_type}"
    ctx.state.models[model_key] = results["estimator"][best_idx]
    ctx.state.model_metadata[model_key] = {
        "icd10_code": icd10_code,
        "model_type": model_type,
        "auc": auc_mean,
        "n_cases": n_cases,
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
    }

    # Store feature matrix and labels for downstream skills
    ctx.state.feature_matrix = X
    ctx.state.labels = y

    return {
        "model_key": model_key,
        "disease": icd10_name(icd10_code),
        "model_type": model_type,
        "n_cases": n_cases,
        "n_controls": int((y == 0).sum()),
        "n_features": X.shape[1],
        "auc_mean": round(auc_mean, 4),
        "auc_95ci": f"[{auc_ci_low:.4f}, {auc_ci_high:.4f}]",
        "f1_mean": round(float(np.mean(results["test_f1"])), 4),
        "precision_mean": round(float(np.mean(results["test_precision"])), 4),
        "recall_mean": round(float(np.mean(results["test_recall"])), 4),
        "fold_aucs": [round(float(a), 4) for a in auc_scores],
    }
