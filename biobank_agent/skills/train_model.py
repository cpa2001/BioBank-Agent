"""Train a predictive model with cross-validation and optional model selection."""

import os

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate

from biobank_agent.data.cohort import build_cohort
from biobank_agent.registry import skill
from biobank_agent.utils.icd10 import icd10_name


for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")


BOOSTED_MODEL_TYPES = ("xgb", "lgbm", "catboost")
SKLEARN_MODEL_TYPES = ("sklearn_rf", "logistic")
ALL_MODEL_TYPES = (*BOOSTED_MODEL_TYPES, *SKLEARN_MODEL_TYPES)
MODEL_ALIASES = {
    "xgboost": "xgb",
    "lightgbm": "lgbm",
    "cat": "catboost",
    "random_forest": "sklearn_rf",
    "rf": "sklearn_rf",
    "logreg": "logistic",
    "logistic_regression": "logistic",
}
FALLBACK_MODEL_TYPE = "sklearn_rf"
SAFE_CV_N_JOBS = 1
SAFE_MODEL_THREADS = 1
AUTO_MAX_TRAIN_ROWS = 0  # 0 means use every eligible cohort row.
AUTO_HOLDOUT_THRESHOLD = 100_000
AUTO_LARGE_DATA_CANDIDATES = ("lgbm", "logistic")


def _emit_progress(ctx, phase: str, message: str, metadata: dict | None = None) -> None:
    callback = getattr(ctx, "emit_progress", None)
    if callable(callback):
        callback(phase, message, metadata or {})


def _normalize_model_type(model_type: str) -> str:
    """Normalize public model aliases while keeping legacy short names."""
    normalized = (model_type or "auto").strip().lower().replace("-", "_")
    normalized = MODEL_ALIASES.get(normalized, normalized)
    allowed = {"auto", *ALL_MODEL_TYPES}
    if normalized not in allowed:
        allowed_str = "', '".join(["auto", *ALL_MODEL_TYPES])
        raise ValueError(f"Unknown model type: {model_type}. Use '{allowed_str}'.")
    return normalized


def _get_estimator(model_type: str):
    """Return a configured estimator."""
    model_type = _normalize_model_type(model_type)
    if model_type == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, gamma=0.7,
            reg_alpha=0.5, eval_metric="logloss", random_state=42,
            n_jobs=SAFE_MODEL_THREADS, tree_method="hist",
        )
    elif model_type == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=60, max_depth=5, learning_rate=0.08,
            num_leaves=31, min_child_samples=20, reg_alpha=0.5,
            reg_lambda=1.0, is_unbalance=True, random_state=42, verbose=-1,
            n_jobs=SAFE_MODEL_THREADS, force_col_wise=True,
        )
    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=120, depth=6, learning_rate=0.06,
            auto_class_weights="Balanced", random_seed=42, verbose=0,
            thread_count=SAFE_MODEL_THREADS, allow_writing_files=False,
        )
    elif model_type == "sklearn_rf":
        return _get_fallback_estimator()
    elif model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                max_iter=1000,
                random_state=42,
            ),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'auto', 'xgb', 'lgbm', 'catboost', 'sklearn_rf', or 'logistic'.")


def _get_fallback_estimator():
    """Return a dependency-light fallback estimator for automatic selection."""
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=SAFE_MODEL_THREADS,
    )


def _get_large_data_estimator(model_type: str):
    """Return a faster estimator for full-row large-cohort model comparison."""
    model_type = _normalize_model_type(model_type)
    if model_type == "logistic":
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(with_mean=False),
            SGDClassifier(
                loss="log_loss",
                class_weight="balanced",
                max_iter=200,
                tol=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5,
                random_state=42,
            ),
        )
    if model_type == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=40,
            max_depth=4,
            learning_rate=0.10,
            num_leaves=15,
            min_child_samples=50,
            reg_alpha=0.5,
            reg_lambda=1.0,
            is_unbalance=True,
            random_state=42,
            verbose=-1,
            n_jobs=SAFE_MODEL_THREADS,
            force_col_wise=True,
        )
    return _get_estimator(model_type)


def _auto_candidate_order(n_samples: int, n_features: int, positive_rate: float) -> tuple[list[str], list[str]]:
    """Choose an evaluation order for auto mode; all candidates are still compared."""
    minority_rate = min(positive_rate, 1.0 - positive_rate)
    reasons = []
    if n_samples >= 5000 or n_features >= 64:
        order = ["lgbm", "xgb", "catboost", "sklearn_rf", "logistic"]
        reasons.append("larger or wider tabular cohort favours LightGBM as the first candidate")
    elif minority_rate < 0.20:
        order = ["lgbm", "catboost", "xgb", "sklearn_rf", "logistic"]
        reasons.append("class imbalance favours candidates with built-in balancing first")
    else:
        order = ["xgb", "lgbm", "catboost", "sklearn_rf", "logistic"]
        reasons.append("moderate balanced tabular cohort uses XGBoost as the first candidate")
    reasons.append("successful boosted-tree and sklearn baselines are ranked by identical stratified-CV mean AUC")
    return order, reasons


def _scoring() -> dict[str, str]:
    return {
        "auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
    }


def _summarize_cv(model_type: str, results: dict, n_folds: int, *, role: str = "candidate") -> dict:
    """Summarize cross-validation results without storing estimator objects."""
    from scipy.stats import t as t_dist

    auc_scores = np.asarray(results["test_auc"], dtype=float)
    auc_mean = float(np.mean(auc_scores))
    auc_std = float(np.std(auc_scores))
    if n_folds > 1:
        t_crit = float(t_dist.ppf(0.975, df=n_folds - 1))
        auc_ci_low = float(auc_mean - t_crit * auc_std / np.sqrt(n_folds))
        auc_ci_high = float(auc_mean + t_crit * auc_std / np.sqrt(n_folds))
    else:
        auc_ci_low = auc_mean
        auc_ci_high = auc_mean

    return {
        "model_type": model_type,
        "role": role,
        "status": "success",
        "auc_mean": auc_mean,
        "auc_std": auc_std,
        "auc_95ci": [auc_ci_low, auc_ci_high],
        "f1_mean": float(np.mean(results["test_f1"])),
        "precision_mean": float(np.mean(results["test_precision"])),
        "recall_mean": float(np.mean(results["test_recall"])),
        "fold_aucs": [float(a) for a in auc_scores],
    }


def _failed_candidate(model_type: str, exc: Exception, *, role: str = "candidate") -> dict:
    return {
        "model_type": model_type,
        "role": role,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _fit_candidate(model_type: str, X, y, cv, scoring: dict) -> tuple[dict, dict]:
    estimator = _get_estimator(model_type)
    results = cross_validate(
        estimator, X, y, cv=cv, scoring=scoring,
        return_estimator=True, n_jobs=SAFE_CV_N_JOBS,
    )
    return _summarize_cv(model_type, results, cv.n_splits), results


def _fit_fallback_candidate(X, y, cv, scoring: dict) -> tuple[dict, dict]:
    estimator = _get_fallback_estimator()
    results = cross_validate(
        estimator, X, y, cv=cv, scoring=scoring,
        return_estimator=True, n_jobs=SAFE_CV_N_JOBS,
    )
    return _summarize_cv(FALLBACK_MODEL_TYPE, results, cv.n_splits, role="fallback"), results


def _predict_scores(estimator, X):
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        return proba[:, 1] if getattr(proba, "ndim", 1) > 1 else proba
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return estimator.predict(X)


def _summarize_holdout(model_type: str, estimator, X_valid, y_valid, *, role: str = "candidate") -> tuple[dict, dict]:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    scores = np.asarray(_predict_scores(estimator, X_valid), dtype=float)
    if np.nanmin(scores) >= 0.0 and np.nanmax(scores) <= 1.0:
        y_pred = (scores >= 0.5).astype(int)
    else:
        y_pred = (scores >= 0.0).astype(int)
    auc = float(roc_auc_score(y_valid, scores))
    summary = {
        "model_type": model_type,
        "role": role,
        "status": "success",
        "evaluation_strategy": "stratified_holdout",
        "auc_mean": auc,
        "auc_std": 0.0,
        "auc_95ci": [auc, auc],
        "f1_mean": float(f1_score(y_valid, y_pred, zero_division=0)),
        "precision_mean": float(precision_score(y_valid, y_pred, zero_division=0)),
        "recall_mean": float(recall_score(y_valid, y_pred, zero_division=0)),
        "fold_aucs": [auc],
    }
    results = {
        "test_auc": np.asarray([summary["auc_mean"]], dtype=float),
        "test_f1": np.asarray([summary["f1_mean"]], dtype=float),
        "test_precision": np.asarray([summary["precision_mean"]], dtype=float),
        "test_recall": np.asarray([summary["recall_mean"]], dtype=float),
        "estimator": [estimator],
        "validation_y_true": np.asarray(y_valid, dtype=int),
        "validation_y_prob": scores,
        "validation_scope": "holdout_validation",
        "validation_n": int(len(y_valid)),
    }
    return summary, results


def _diagnostic_biomarker_leakage_features(icd10_code: str, feature_cols: list[str]) -> list[str]:
    """Return features that are endpoint-adjacent for common diabetes discrimination tasks."""
    if not str(icd10_code or "").upper().startswith("E11"):
        return []
    diabetes_markers = {
        "30740",  # glucose
        "30750",  # HbA1c
        "30760",  # HDL cholesterol
        "30780",  # LDL direct
        "30870",  # triglycerides
    }
    marker_terms = ("glucose", "hba1c", "glycated", "diabetes")
    hits: list[str] = []
    for col in feature_cols:
        base = str(col).split("-", 1)[0].lower()
        lower = str(col).lower()
        if base in diabetes_markers or any(term in lower for term in marker_terms):
            hits.append(str(col))
    return hits[:20]


def _fit_holdout_candidate(model_type: str, X_train, X_valid, y_train, y_valid, *, large_data: bool = False) -> tuple[dict, dict]:
    estimator = _get_large_data_estimator(model_type) if large_data else _get_estimator(model_type)
    estimator.fit(X_train, y_train)
    return _summarize_holdout(model_type, estimator, X_valid, y_valid)


def _cap_training_rows(X, y, max_rows: int = AUTO_MAX_TRAIN_ROWS):
    """Optionally bound training rows while preserving class balance.

    The default is no cap: the local UKB parquet data is already
    de-identified and analytical skills should use all eligible rows unless
    the user explicitly requests a smaller smoke-test sample.
    """
    if max_rows is None or int(max_rows) <= 0:
        return X, y, {
            "applied": False,
            "max_rows": 0,
            "n_rows_before": int(len(y)),
            "n_rows_after": int(len(y)),
            "reason": "full_dataset_default",
        }
    max_rows = int(max_rows)
    if len(y) <= max_rows:
        return X, y, {"applied": False, "max_rows": max_rows, "n_rows_before": int(len(y)), "n_rows_after": int(len(y))}

    from sklearn.model_selection import train_test_split

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=max_rows,
        stratify=y,
        random_state=42,
    )
    return X_sample, y_sample, {
        "applied": True,
        "max_rows": max_rows,
        "n_rows_before": int(len(y)),
        "n_rows_after": int(len(y_sample)),
    }


def _rank_successful_candidates(candidate_comparison: list[dict]) -> list[dict]:
    successful = [c for c in candidate_comparison if c.get("status") == "success"]
    ranked = sorted(successful, key=lambda c: c["auc_mean"], reverse=True)
    for rank, candidate in enumerate(ranked, start=1):
        candidate["rank"] = rank
    return ranked


def _selection_rationale(
    requested_model_type: str,
    selected_model_type: str,
    candidate_comparison: list[dict],
    heuristic_reasons: list[str],
    fallback_metadata: dict,
) -> str:
    if requested_model_type != "auto":
        return f"Used explicit model_type='{selected_model_type}'; automatic comparison was not run."

    if fallback_metadata.get("used"):
        return (
            f"Selected {selected_model_type} fallback because all automatic "
            "candidate models failed cross-validation."
        )

    winner = next(
        (c for c in candidate_comparison if c.get("model_type") == selected_model_type and c.get("status") == "success"),
        {},
    )
    failed = [c["model_type"] for c in candidate_comparison if c.get("status") == "failed"]
    auc = winner.get("auc_mean")
    strategy = winner.get("evaluation_strategy", "cross_validation")
    metric_label = "holdout AUC" if strategy == "stratified_holdout" else "mean CV AUC"
    reason = f"Selected {selected_model_type} because it had the highest {metric_label}"
    if auc is not None:
        reason += f" ({auc:.4f})"
    reason += "."
    if heuristic_reasons:
        reason += " Candidate order: " + "; ".join(heuristic_reasons) + "."
    if failed:
        reason += f" Failed candidates were excluded from ranking: {', '.join(failed)}."
    return reason


def _performance_grade(auc: float | None) -> str:
    if auc is None:
        return "unknown"
    if auc >= 0.90:
        return "excellent"
    if auc >= 0.80:
        return "good"
    if auc >= 0.70:
        return "moderate"
    return "limited"


def _optimization_next_steps(auc: float | None, selection_metadata: dict) -> list[str]:
    if auc is None:
        return ["Inspect candidate failures and verify endpoint/cohort construction before retrying."]
    if auc >= 0.80:
        return ["Prioritize calibration, feature importance, and external validation rather than further model searching."]
    failed = [
        c.get("model_type")
        for c in selection_metadata.get("candidate_comparison", [])
        if c.get("status") == "failed"
    ]
    steps = [
        "Review missingness, leakage, endpoint definition, and candidate feature coverage.",
        "Add richer longitudinal or lifestyle predictors only if they are available and valid before outcome/index date.",
        "Treat results as exploratory unless calibration and independent validation improve.",
    ]
    if failed:
        steps.insert(0, f"Investigate failed candidate backends before retrying: {', '.join(str(x) for x in failed)}.")
    return steps


def _fit_auto_model_holdout(X, y, n_folds: int, *, ctx=None) -> tuple[str, dict, dict]:
    from sklearn.model_selection import train_test_split

    positive_rate = float(np.mean(y))
    full_order, heuristic_reasons = _auto_candidate_order(len(y), X.shape[1], positive_rate)
    candidate_order = [candidate for candidate in AUTO_LARGE_DATA_CANDIDATES if candidate in full_order]
    heuristic_reasons = [
        *heuristic_reasons,
        (
            "large full-data cohort uses a single stratified holdout for auto model selection "
            "so every eligible row is used as either training or validation data without "
            "multiplying runtime by k-fold candidate CV"
        ),
    ]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )
    candidate_comparison: list[dict] = []
    results_by_type: dict[str, dict] = {}
    _emit_progress(
        ctx,
        "model-selection",
        f"large cohort: comparing {len(candidate_order)} candidate model(s) with a stratified holdout",
        {
            "candidate_order": candidate_order,
            "evaluation_strategy": "stratified_holdout",
            "n_rows": int(len(y)),
            "n_train": int(len(y_train)),
            "n_validation": int(len(y_valid)),
            "requested_n_folds": int(n_folds),
        },
    )
    for idx, candidate_type in enumerate(candidate_order, start=1):
        try:
            _emit_progress(
                ctx,
                "model-selection",
                f"candidate {idx}/{len(candidate_order)}: {candidate_type} holdout training started",
                {"candidate": candidate_type, "candidate_index": idx, "n_candidates": len(candidate_order)},
            )
            summary, results = _fit_holdout_candidate(candidate_type, X_train, X_valid, y_train, y_valid, large_data=True)
            candidate_comparison.append(summary)
            results_by_type[candidate_type] = results
            auc = summary.get("auc_mean")
            auc_text = f"AUC {auc:.4f}" if isinstance(auc, (int, float)) else "AUC unavailable"
            _emit_progress(
                ctx,
                "model-selection",
                f"candidate {candidate_type} completed on holdout ({auc_text})",
                {"candidate": candidate_type, "summary": summary},
            )
        except Exception as exc:
            candidate_comparison.append(_failed_candidate(candidate_type, exc))
            _emit_progress(
                ctx,
                "model-selection",
                f"candidate {candidate_type} failed: {exc}",
                {"candidate": candidate_type, "error": str(exc)},
            )

    ranked = _rank_successful_candidates(candidate_comparison)
    fallback_metadata = {
        "used": False,
        "model_type": FALLBACK_MODEL_TYPE,
        "reason": "",
    }
    if not ranked:
        fallback_metadata["used"] = True
        fallback_metadata["reason"] = "no automatic candidate completed holdout evaluation"
        try:
            estimator = _get_fallback_estimator()
            estimator.fit(X_train, y_train)
            summary, results = _summarize_holdout(FALLBACK_MODEL_TYPE, estimator, X_valid, y_valid, role="fallback")
            candidate_comparison.append(summary)
            results_by_type[FALLBACK_MODEL_TYPE] = results
            ranked = _rank_successful_candidates(candidate_comparison)
        except Exception as exc:
            failed = _failed_candidate(FALLBACK_MODEL_TYPE, exc, role="fallback")
            candidate_comparison.append(failed)
            fallback_metadata["error"] = failed["error"]
            fallback_metadata["error_type"] = failed["error_type"]
            return "", {}, {
                "mode": "auto",
                "requested_model_type": "auto",
                "selected_model_type": None,
                "selection_metric": "auc_mean",
                "candidate_order": candidate_order,
                "candidate_comparison": candidate_comparison,
                "rationale": "Automatic model selection failed; no candidate or fallback model completed.",
                "fallback": fallback_metadata,
                "evaluation_strategy": "stratified_holdout",
            }

    selected_model_type = ranked[0]["model_type"]
    selection_metadata = {
        "mode": "auto",
        "requested_model_type": "auto",
        "selected_model_type": selected_model_type,
        "selection_metric": "auc_mean",
        "candidate_order": candidate_order,
        "heuristic_reasons": heuristic_reasons,
        "candidate_comparison": candidate_comparison,
        "rationale": _selection_rationale(
            "auto", selected_model_type, candidate_comparison, heuristic_reasons, fallback_metadata
        ),
        "fallback": fallback_metadata,
        "evaluation_strategy": "stratified_holdout",
        "requested_n_folds": int(n_folds),
        "n_train": int(len(y_train)),
        "n_validation": int(len(y_valid)),
        "full_dataset_rows_used": int(len(y)),
    }
    _emit_progress(
        ctx,
        "model-selection",
        f"selected {selected_model_type} by holdout AUC",
        {"selected_model_type": selected_model_type, "ranked": ranked},
    )
    return selected_model_type, results_by_type[selected_model_type], selection_metadata


def _fit_auto_model(X, y, n_folds: int, *, ctx=None) -> tuple[str, dict, dict]:
    if len(y) >= AUTO_HOLDOUT_THRESHOLD:
        return _fit_auto_model_holdout(X, y, n_folds, ctx=ctx)

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scoring = _scoring()
    positive_rate = float(np.mean(y))
    candidate_order, heuristic_reasons = _auto_candidate_order(len(y), X.shape[1], positive_rate)
    candidate_comparison: list[dict] = []
    cv_results_by_type: dict[str, dict] = {}

    _emit_progress(
        ctx,
        "model-selection",
        f"comparing {len(candidate_order)} candidate model(s) with {n_folds}-fold CV",
        {"candidate_order": candidate_order, "n_folds": n_folds},
    )
    for idx, candidate_type in enumerate(candidate_order, start=1):
        try:
            _emit_progress(
                ctx,
                "model-selection",
                f"candidate {idx}/{len(candidate_order)}: {candidate_type} cross-validation started",
                {"candidate": candidate_type, "candidate_index": idx, "n_candidates": len(candidate_order)},
            )
            summary, results = _fit_candidate(candidate_type, X, y, cv, scoring)
            candidate_comparison.append(summary)
            cv_results_by_type[candidate_type] = results
            auc = summary.get("auc_mean")
            auc_text = f"AUC {auc:.4f}" if isinstance(auc, (int, float)) else "AUC unavailable"
            _emit_progress(
                ctx,
                "model-selection",
                f"candidate {candidate_type} completed ({auc_text})",
                {"candidate": candidate_type, "summary": summary},
            )
        except Exception as exc:
            candidate_comparison.append(_failed_candidate(candidate_type, exc))
            _emit_progress(
                ctx,
                "model-selection",
                f"candidate {candidate_type} failed: {exc}",
                {"candidate": candidate_type, "error": str(exc)},
            )

    ranked = _rank_successful_candidates(candidate_comparison)
    fallback_metadata = {
        "used": False,
        "model_type": FALLBACK_MODEL_TYPE,
        "reason": "",
    }

    if not ranked:
        fallback_metadata["used"] = True
        fallback_metadata["reason"] = "no automatic candidate completed cross-validation"
        try:
            _emit_progress(
                ctx,
                "model-selection",
                f"all auto candidates failed; trying fallback {FALLBACK_MODEL_TYPE}",
                {"fallback": FALLBACK_MODEL_TYPE},
            )
            summary, results = _fit_fallback_candidate(X, y, cv, scoring)
            candidate_comparison.append(summary)
            cv_results_by_type[FALLBACK_MODEL_TYPE] = results
            ranked = _rank_successful_candidates(candidate_comparison)
        except Exception as exc:
            failed = _failed_candidate(FALLBACK_MODEL_TYPE, exc, role="fallback")
            candidate_comparison.append(failed)
            fallback_metadata["error"] = failed["error"]
            fallback_metadata["error_type"] = failed["error_type"]
            return "", {}, {
                "mode": "auto",
                "requested_model_type": "auto",
                "selected_model_type": None,
                "selection_metric": "auc_mean",
                "candidate_order": candidate_order,
                "candidate_comparison": candidate_comparison,
                "rationale": "Automatic model selection failed; no candidate or fallback model completed.",
                "fallback": fallback_metadata,
            }

    selected_model_type = ranked[0]["model_type"]
    _emit_progress(
        ctx,
        "model-selection",
        f"selected {selected_model_type} by mean CV AUC",
        {"selected_model_type": selected_model_type, "ranked": ranked},
    )
    selection_metadata = {
        "mode": "auto",
        "requested_model_type": "auto",
        "selected_model_type": selected_model_type,
        "selection_metric": "auc_mean",
        "candidate_order": candidate_order,
        "heuristic_reasons": heuristic_reasons,
        "candidate_comparison": candidate_comparison,
        "rationale": _selection_rationale(
            "auto", selected_model_type, candidate_comparison, heuristic_reasons, fallback_metadata
        ),
        "fallback": fallback_metadata,
    }
    return selected_model_type, cv_results_by_type[selected_model_type], selection_metadata


def _fit_explicit_model(model_type: str, X, y, n_folds: int) -> tuple[str, dict, dict]:
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    summary, results = _fit_candidate(model_type, X, y, cv, _scoring())
    selection_metadata = {
        "mode": "explicit",
        "requested_model_type": model_type,
        "selected_model_type": model_type,
        "selection_metric": "auc_mean",
        "candidate_order": [model_type],
        "candidate_comparison": [summary],
        "rationale": _selection_rationale(model_type, model_type, [summary], [], {"used": False}),
        "fallback": {"used": False, "model_type": FALLBACK_MODEL_TYPE, "reason": "explicit model requested"},
    }
    return model_type, results, selection_metadata


@skill(
    name="train_model",
    description="Train a machine learning model to predict a disease from biomarkers. "
                "With model_type='auto', compares boosted-tree and sklearn baseline candidates by cross-validated "
                "AUC on small cohorts or stratified holdout AUC on large full-data cohorts. "
                "Reports AUC, uncertainty metadata and model-selection rationale. "
                "The trained model is stored in session state for subsequent analysis. "
                "After training, consider running statistical_review to check for issues, "
                "and feature_importance to identify top predictors.",
    parameters={
        "icd10_code": {
            "type": "string",
            "description": "ICD10 code prefix (e.g. 'E11')",
        },
        "model_type": {
            "type": "string",
            "description": "Model type: 'auto', boosted trees ('xgb', 'lgbm', 'catboost'), or baselines ('sklearn_rf', 'logistic')",
            "default": "auto",
            "enum": ["auto", "xgb", "lgbm", "catboost", "xgboost", "lightgbm", "sklearn_rf", "logistic", "rf", "logreg"],
        },
        "n_folds": {
            "type": "integer",
            "description": "Number of CV folds (default 5)",
            "default": 5,
        },
        "max_train_rows": {
            "type": "integer",
            "description": "Optional cap for smoke tests. Use 0 or omit to train on all eligible cohort rows.",
            "default": 0,
        },
        "controls_ratio": {
            "type": "integer",
            "description": "Controls per case when building the cohort. Use 0 or omit to include all eligible controls.",
            "default": 0,
        },
    },
    required=["icd10_code"],
)
def train_model(
    icd10_code: str,
    model_type: str = "auto",
    n_folds: int = 5,
    max_train_rows: int = 0,
    controls_ratio: int = 0,
    *,
    ctx=None,
) -> dict:
    requested_model_type = _normalize_model_type(model_type)
    dm = ctx.dm
    id_col = getattr(dm, "subject_id_col", getattr(ctx.settings, "subject_id_col", "eid"))
    _emit_progress(ctx, "setup", f"building or loading cohort for {icd10_code}", {"icd10_code": icd10_code})

    # Build or reuse cohort
    cohort_key = f"{icd10_code}_1:{controls_ratio or 'all'}"
    if cohort_key in ctx.state.cohorts:
        df = ctx.state.cohorts[cohort_key]
    else:
        df = build_cohort(dm, icd10_code, controls_ratio=controls_ratio)
        ctx.state.cohorts[cohort_key] = df

    n_cases = int(df["label"].sum())
    if n_cases < 100:
        return {"error": f"Insufficient cases (n={n_cases}). Minimum 100 required."}
    _emit_progress(
        ctx,
        "cohort",
        f"cohort has {n_cases} cases and {int((df['label'] == 0).sum())} controls",
        {"n_cases": n_cases, "n_controls": int((df["label"] == 0).sum())},
    )

    # Prepare features
    feature_cols = [c for c in df.columns
                    if c not in (id_col, "label")
                    and df[c].dtype in ("float64", "float32", "int64", "int32")]

    X = df[feature_cols].copy()
    y = df["label"].copy()
    cohort_n_cases = n_cases
    cohort_n_controls = int((y == 0).sum())

    # Impute missing with median
    X = X.fillna(X.median())

    # Drop columns that are all-constant
    nunique = X.nunique()
    const_cols = nunique[nunique <= 1].index.tolist()
    X = X.drop(columns=const_cols)

    if max_train_rows in (None, 0):
        max_train_rows = int(getattr(ctx.settings, "max_train_rows_default", 0) or 0)
    X, y, training_sample = _cap_training_rows(X, y, max_rows=max_train_rows)
    training_n_cases = int(y.sum())
    training_n_controls = int((y == 0).sum())
    _emit_progress(
        ctx,
        "features",
        f"training matrix {len(y)} rows x {X.shape[1]} features",
        {"n_rows": int(len(y)), "n_features": int(X.shape[1]), "training_sample": training_sample},
    )

    # Train with cross-validation. Explicit model requests preserve the legacy
    # single-estimator path; auto mode wraps that path in candidate comparison.
    if requested_model_type == "auto":
        selected_model_type, results, selection_metadata = _fit_auto_model(X, y, n_folds, ctx=ctx)
        if not selected_model_type:
            return {
                "error": "Automatic model selection failed. No candidate completed cross-validation.",
                "requested_model_type": "auto",
                "candidate_comparison": selection_metadata["candidate_comparison"],
                "selection_rationale": selection_metadata["rationale"],
                "fallback": selection_metadata["fallback"],
                "model_selection": selection_metadata,
            }
    else:
        _emit_progress(
            ctx,
            "training",
            f"training explicit {requested_model_type} model with {n_folds}-fold CV",
            {"model_type": requested_model_type, "n_folds": n_folds},
        )
        selected_model_type, results, selection_metadata = _fit_explicit_model(
            requested_model_type, X, y, n_folds
        )

    selected_summary = next(
        c for c in selection_metadata["candidate_comparison"]
        if c.get("model_type") == selected_model_type and c.get("status") == "success"
    )
    auc_scores = np.asarray(results["test_auc"], dtype=float)
    auc_mean = selected_summary["auc_mean"]
    auc_ci_low, auc_ci_high = selected_summary["auc_95ci"]
    performance_grade = _performance_grade(auc_mean)
    optimization_next_steps = _optimization_next_steps(auc_mean, selection_metadata)

    # Store best model (by AUC) in state
    best_idx = int(np.argmax(auc_scores))
    model_key = f"{icd10_code}_{selected_model_type}"
    ctx.state.models[model_key] = results["estimator"][best_idx]
    diagnostic_leakage_features = _diagnostic_biomarker_leakage_features(icd10_code, list(X.columns))
    analysis_design = "prevalent_or_ever_diagnosed_case_control"
    incident_risk_supported = False
    evaluation_bundle = {}
    if results.get("validation_y_true") is not None and results.get("validation_y_prob") is not None:
        evaluation_bundle = {
            "evaluation_scope": str(results.get("validation_scope") or "holdout_validation"),
            "y_true": results.get("validation_y_true"),
            "y_prob": results.get("validation_y_prob"),
            "n": int(results.get("validation_n") or len(results.get("validation_y_true"))),
            "selection_strategy": selection_metadata.get("evaluation_strategy", ""),
        }
        try:
            ctx.state.custom_data.setdefault("model_evaluation", {})[model_key] = evaluation_bundle
        except Exception:
            pass
    ctx.state.model_metadata[model_key] = {
        "icd10_code": icd10_code,
        "model_type": selected_model_type,
        "requested_model_type": requested_model_type,
        "auc": auc_mean,
        "auc_mean": auc_mean,
        "mean_auc": auc_mean,
        "n_cases": cohort_n_cases,
        "n_controls": cohort_n_controls,
        "training_n_cases": training_n_cases,
        "training_n_controls": training_n_controls,
        "training_sample": training_sample,
        "controls_ratio": controls_ratio,
        "controls_sampling_applied": bool(controls_ratio and controls_ratio > 0),
        "n_features": X.shape[1],
        "feature_names": list(X.columns),
        "model_selection": selection_metadata,
        "evaluation_strategy": selection_metadata.get("evaluation_strategy", "cross_validation"),
        "analysis_design": analysis_design,
        "prediction_target": f"ever_diagnosed_{icd10_code}_discrimination",
        "incident_risk_supported": incident_risk_supported,
        "diagnostic_biomarker_leakage_risk": bool(diagnostic_leakage_features),
        "diagnostic_biomarker_leakage_features": diagnostic_leakage_features,
        "performance_grade": performance_grade,
        "optimization_next_steps": optimization_next_steps,
    }

    # Store feature matrix and labels for downstream skills
    ctx.state.feature_matrix = X
    ctx.state.labels = y

    return {
        "model_key": model_key,
        "disease": icd10_name(icd10_code),
        "model_type": selected_model_type,
        "requested_model_type": requested_model_type,
        "selected_model_type": selected_model_type,
        "n_cases": cohort_n_cases,
        "n_controls": cohort_n_controls,
        "training_n_cases": training_n_cases,
        "training_n_controls": training_n_controls,
        "training_sample": training_sample,
        "controls_ratio": controls_ratio,
        "controls_sampling_applied": bool(controls_ratio and controls_ratio > 0),
        "n_features": X.shape[1],
        "analysis_design": analysis_design,
        "prediction_target": f"ever_diagnosed_{icd10_code}_discrimination",
        "incident_risk_supported": incident_risk_supported,
        "evaluation_strategy": selection_metadata.get("evaluation_strategy", "cross_validation"),
        "calibration_preferred_scope": evaluation_bundle.get("evaluation_scope", "cross_validated_or_in_sample"),
        "diagnostic_biomarker_leakage_risk": bool(diagnostic_leakage_features),
        "diagnostic_biomarker_leakage_features": diagnostic_leakage_features,
        "auc_mean": round(auc_mean, 4),
        "mean_auc": round(auc_mean, 4),
        "auc": round(auc_mean, 4),
        "auc_95ci": f"[{auc_ci_low:.4f}, {auc_ci_high:.4f}]",
        "f1_mean": round(float(np.mean(results["test_f1"])), 4),
        "precision_mean": round(float(np.mean(results["test_precision"])), 4),
        "recall_mean": round(float(np.mean(results["test_recall"])), 4),
        "fold_aucs": [round(float(a), 4) for a in auc_scores],
        "candidate_comparison": selection_metadata["candidate_comparison"],
        "selection_rationale": selection_metadata["rationale"],
        "performance_grade": performance_grade,
        "optimization_status": (
            "good_enough_for_internal_reporting"
            if auc_mean >= 0.80 else
            "exploratory_needs_diagnostics"
        ),
        "optimization_next_steps": optimization_next_steps,
        "fallback": selection_metadata["fallback"],
        "model_selection": selection_metadata,
    }
