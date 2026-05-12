"""Compare a paper-replication target against local UKB results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from biobank_agent.registry import skill


def _records(ctx) -> list[Any]:
    state = getattr(ctx, "state", None)
    return list(getattr(state, "records", []) or [])


def _latest(records: list[Any], skill: str) -> Any | None:
    matches = [rec for rec in records if getattr(rec, "skill", "") == skill]
    return matches[-1] if matches else None


def _result(record: Any | None) -> dict[str, Any]:
    if record is None:
        return {}
    value = getattr(record, "key_results", {}) or {}
    return value if isinstance(value, dict) else {}


def _arg(record: Any | None, key: str, default: Any = "") -> Any:
    if record is None:
        return default
    args = getattr(record, "args", {}) or {}
    return args.get(key, default)


def _read_paper_access(read_paper_result: dict[str, Any]) -> str:
    if not read_paper_result:
        return "not_recorded"
    if read_paper_result.get("error"):
        return "unavailable"
    summary = read_paper_result.get("summary") or {}
    if summary.get("access_warning"):
        return "abstract_or_partial_text"
    if read_paper_result.get("text_truncated"):
        return "full_or_long_text_truncated_for_context"
    if read_paper_result.get("paper_text"):
        return "text_available"
    return "metadata_only"


def _paper_spec(rep_result: dict[str, Any], read_result: dict[str, Any]) -> dict[str, Any]:
    spec = dict(rep_result.get("proposed_spec") or {})
    if not spec:
        summary = read_result.get("summary") or {}
        spec = {
            "source": summary.get("doi") or summary.get("pdf_path") or "",
            "design": "",
            "outcomes": [],
            "icd10_code": "E11",
            "biomarkers": [],
        }
    if not spec.get("icd10_code"):
        spec["icd10_code"] = "E11"
    return spec


def _top_features(feature_result: dict[str, Any], limit: int = 10) -> list[str]:
    features = feature_result.get("top_features") or []
    names: list[str] = []
    for item in features[:limit]:
        if isinstance(item, dict):
            name = item.get("feature")
            if name:
                names.append(str(name))
        elif item:
            names.append(str(item))
    return names


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    numeric = _as_float(value)
    if numeric is None:
        return None
    return int(numeric)


def _session_figure_paths(ctx, records: list[Any]) -> list[str]:
    """Collect local figure artifacts from state and per-record figure paths."""
    paths: list[str] = []
    state = getattr(ctx, "state", None)
    for item in getattr(state, "figures", []) or []:
        if item and str(item) not in paths:
            paths.append(str(item))
    for rec in records:
        for item in getattr(rec, "figure_paths", []) or []:
            if item and str(item) not in paths:
                paths.append(str(item))
    return paths


def _target_keywords(caption: str) -> list[str]:
    lower = str(caption or "").lower()
    keywords: list[str] = []
    if any(term in lower for term in ("roc", "auc", "discrimination", "precision", "recall", "pr curve")):
        keywords.extend(["roc", "pr", "auc"])
    if "calibration" in lower:
        keywords.append("calibration")
    if any(term in lower for term in ("feature", "importance", "biomarker", "predictor")):
        keywords.extend(["feature", "importance"])
    if any(term in lower for term in ("missing", "missingness")):
        keywords.append("missing")
    if any(term in lower for term in ("cohort", "baseline", "age", "characteristic")):
        keywords.extend(["cohort", "age", "missing"])
    return keywords


def _choose_local_figure(caption: str, figure_paths: list[str], used: set[str]) -> str:
    candidates = [path for path in figure_paths if path not in used]
    if not candidates:
        return ""
    keywords = _target_keywords(caption)
    for keyword in keywords:
        for path in candidates:
            if keyword in Path(path).name.lower():
                used.add(path)
                return path
    # If the paper caption is too generic, still attach the first local figure
    # so the final report can show what was actually generated.
    path = candidates[0]
    used.add(path)
    return path


def _resolved_table_figure_diff(
    targets: list[dict[str, Any]],
    *,
    figure_paths: list[str],
    table_artifact: str,
) -> list[dict[str, Any]]:
    """Attach executed local artifacts to paper table/figure targets."""
    resolved: list[dict[str, Any]] = []
    used_figures: set[str] = set()
    for target in targets:
        row = dict(target)
        target_type = str(row.get("target_type", "")).lower()
        if target_type == "figure":
            artifact = _choose_local_figure(str(row.get("paper_caption", "")), figure_paths, used_figures)
            if artifact:
                row["local_artifact"] = artifact
                row["status"] = "local_artifact_available"
                row["limitation"] = (
                    "A local figure artifact was generated and linked for qualitative comparison. "
                    "Pixel-level difference requires the original paper figure image, which may not be available."
                )
            else:
                row["status"] = row.get("status") or "local_artifact_missing"
                row["limitation"] = row.get("limitation") or "No local figure artifact matched this paper target."
        elif target_type == "table":
            row["local_artifact"] = table_artifact
            row["status"] = "local_summary_table_available"
            row["limitation"] = (
                "The local comparison matrix is available. Exact numeric table reproduction "
                "requires the paper's original row/column definitions and executed matching tables."
            )
        elif target_type == "paper_outputs" and figure_paths:
            row["local_artifact"] = figure_paths[0]
            row["status"] = "local_artifacts_available"
            row["limitation"] = (
                "The source text did not expose explicit table/figure captions, but local "
                "analysis artifacts were generated for report inspection."
            )
        resolved.append(row)
    return resolved


def _status(exact: bool = False, approximated: bool = False, unavailable: bool = False) -> str:
    if exact:
        return "exact"
    if approximated:
        return "approximation"
    if unavailable:
        return "unavailable"
    return "not_recorded"


def _acceptance_gate(
    gate: str,
    status: str,
    observed: Any,
    threshold: str,
    evidence: str,
    message: str,
) -> dict[str, Any]:
    return {
        "gate": gate,
        "status": status,
        "observed": observed,
        "threshold": threshold,
        "evidence": evidence,
        "message": message,
    }


def _acceptance_verdict(gates: list[dict[str, Any]], *, overall_status: str) -> str:
    statuses = {str(g.get("status", "")).upper() for g in gates}
    if "FAIL" in statuses:
        return "FAIL"
    if statuses & {"PARTIAL", "WARN"}:
        return "PASS_WITH_LIMITATIONS"
    if overall_status == "exact_replication_candidate":
        return "EXACT_CANDIDATE"
    return "PASS_WITH_LIMITATIONS"


def _build_acceptance_gates(
    *,
    paper_access: str,
    cohort_result: dict[str, Any],
    train_result: dict[str, Any],
    eval_result: dict[str, Any],
    cal_result: dict[str, Any],
    top_features: list[str],
    n_targets: int,
    n_matched: int,
    min_cases: int,
    min_auc: float,
    max_calibration_ece: float,
    require_figure_match: bool,
) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    if paper_access in {"text_available", "full_or_long_text_truncated_for_context"}:
        gates.append(_acceptance_gate(
            "paper_access",
            "PASS",
            paper_access,
            "paper text or long extracted text available",
            "read_paper",
            "Paper text was available enough to ground the comparison.",
        ))
    elif paper_access == "abstract_or_partial_text":
        gates.append(_acceptance_gate(
            "paper_access",
            "PARTIAL",
            paper_access,
            "paper text or long extracted text available",
            "read_paper",
            "Only abstract or partial text was available; method matching must remain limited.",
        ))
    else:
        gates.append(_acceptance_gate(
            "paper_access",
            "FAIL",
            paper_access,
            "paper text or long extracted text available",
            "read_paper",
            "Paper text was unavailable or not recorded.",
        ))

    n_cases = _as_int(cohort_result.get("n_cases"))
    n_controls = _as_int(cohort_result.get("n_controls"))
    count_ok = n_cases is not None and n_cases >= min_cases
    if n_controls is not None:
        count_ok = count_ok and n_controls >= min_cases
    gates.append(_acceptance_gate(
        "cohort_count",
        "PASS" if count_ok else "FAIL",
        {"n_cases": n_cases, "n_controls": n_controls},
        f"n_cases >= {min_cases}" + (f" and n_controls >= {min_cases}" if n_controls is not None else ""),
        "cohort_summary",
        "Cohort counts are adequate for a local modelling slice." if count_ok else "Cohort counts are missing or below the configured threshold.",
    ))

    auc = _as_float(train_result.get("auc", eval_result.get("mean_auc")))
    gates.append(_acceptance_gate(
        "model_auc",
        "PASS" if auc is not None and auc >= min_auc else "FAIL",
        auc if auc is not None else "not_recorded",
        f"AUC >= {min_auc:g}",
        "train_model/evaluate_model",
        "Discrimination met the configured minimum." if auc is not None and auc >= min_auc else "Discrimination was missing or below the configured minimum.",
    ))

    ece = _as_float(cal_result.get("ece"))
    gates.append(_acceptance_gate(
        "calibration_ece",
        "PASS" if ece is not None and ece <= max_calibration_ece else "FAIL",
        ece if ece is not None else "not_recorded",
        f"ECE <= {max_calibration_ece:g}",
        "calibration",
        "Calibration met the configured maximum ECE." if ece is not None and ece <= max_calibration_ece else "Calibration was missing or above the configured maximum ECE.",
    ))

    gates.append(_acceptance_gate(
        "feature_importance",
        "PASS" if top_features else "FAIL",
        len(top_features),
        "at least one local important feature",
        "feature_importance",
        "Local feature ranking is available." if top_features else "No local feature ranking was recorded.",
    ))

    if n_targets <= 0:
        figure_status = "PARTIAL"
        figure_message = "No explicit paper table/figure targets were extracted; local figures can only support qualitative inspection."
    elif n_matched >= n_targets:
        figure_status = "PASS"
        figure_message = "All extracted paper table/figure targets have local artifacts for inspection."
    elif n_matched > 0:
        figure_status = "PARTIAL" if require_figure_match else "WARN"
        figure_message = "Only some extracted paper table/figure targets have local artifacts."
    else:
        figure_status = "FAIL" if require_figure_match else "PARTIAL"
        figure_message = "No extracted paper table/figure targets were matched to local artifacts."
    gates.append(_acceptance_gate(
        "figure_artifacts",
        figure_status,
        {"targets": n_targets, "matched": n_matched},
        "all extracted targets matched" if require_figure_match else "local figure matching attempted",
        "paper_replication_compare",
        figure_message,
    ))
    return gates


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Paper Replication Comparison",
        "",
        "## Summary",
        "",
        f"- Paper access: {payload['paper_access']}",
        f"- Local endpoint: {payload['local_endpoint']}",
        f"- Overall status: {payload['overall_status']}",
        f"- Acceptance verdict: {payload.get('acceptance_summary', {}).get('verdict', 'not_recorded')}",
        "",
        "## Comparison Matrix",
        "",
        "| Dimension | Paper target | Local UKB result | Status | Evidence |",
        "|-----------|--------------|------------------|--------|----------|",
    ]
    for row in payload["comparison_rows"]:
        lines.append(
            "| {dimension} | {paper_target} | {local_result} | {status} | {evidence} |".format(
                **{k: str(v).replace("|", "\\|") for k, v in row.items()}
            )
        )
    gates = payload.get("acceptance_gates") or []
    if gates:
        lines.extend([
            "",
            "## Acceptance Gates",
            "",
            "| Gate | Status | Observed | Threshold | Evidence | Message |",
            "|------|--------|----------|-----------|----------|---------|",
        ])
        for gate in gates:
            lines.append(
                "| {gate} | {status} | {observed} | {threshold} | {evidence} | {message} |".format(
                    **{k: str(v).replace("|", "\\|") for k, v in gate.items()}
                )
            )
    lines.extend(["", "## Required Report Language", ""])
    for item in payload["report_requirements"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Table/Figure Diff Targets", ""])
    for row in payload.get("table_figure_diff", []):
        lines.append(
            f"- **{row.get('target_type', 'output')}** `{row.get('status', 'unknown')}`: "
            f"{row.get('paper_caption', '')}"
        )
        if row.get("local_artifact"):
            lines.append(f"  - Local artifact: `{row.get('local_artifact')}`")
        lines.append(f"  - Method: {row.get('comparison_method', '')}")
        lines.append(f"  - Limitation: {row.get('limitation', '')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@skill(
    name="paper_replication_compare",
    description=(
        "Compare paper replication targets with local UKB outputs. Writes a "
        "comparison matrix for exact, approximate, and unavailable elements so "
        "the final report cannot claim full replication without evidence."
    ),
    parameters={
        "scope": {
            "type": "string",
            "description": "Comparison scope. Use 'session' for the current plan.",
            "default": "session",
        },
        "output_dir": {
            "type": "string",
            "description": "Optional output directory. Defaults to the active report directory.",
            "default": "",
        },
        "min_cases": {
            "type": "integer",
            "description": "Minimum case and control count required for the local modelling slice.",
            "default": 100,
        },
        "min_auc": {
            "type": "number",
            "description": "Minimum acceptable local model AUC for replication acceptance.",
            "default": 0.55,
        },
        "max_calibration_ece": {
            "type": "number",
            "description": "Maximum acceptable expected calibration error.",
            "default": 0.2,
        },
        "require_figure_match": {
            "type": "boolean",
            "description": "Whether unmatched extracted paper table/figure targets fail acceptance.",
            "default": True,
        },
    },
    required=[],
)
def paper_replication_compare(
    scope: str = "session",
    output_dir: str = "",
    min_cases: int = 100,
    min_auc: float = 0.55,
    max_calibration_ece: float = 0.2,
    require_figure_match: bool = True,
    *,
    ctx=None,
) -> dict:
    records = _records(ctx)
    replicate = _latest(records, "replicate_paper")
    read_paper = _latest(records, "read_paper")
    cohort = _latest(records, "cohort_summary")
    missing = _latest(records, "missing_data")
    train = _latest(records, "train_model")
    evaluate = _latest(records, "evaluate_model")
    calibration = _latest(records, "calibration")
    features = _latest(records, "feature_importance")

    rep_result = _result(replicate)
    read_result = _result(read_paper)
    spec = _paper_spec(rep_result, read_result)
    table_figure_diff = list(rep_result.get("table_figure_diff") or [])
    icd10_code = str(spec.get("icd10_code") or _arg(train, "icd10_code", "E11") or "E11")
    local_endpoint = f"UKB ICD-10 {icd10_code} diagnosis-centered cohort"

    cohort_result = _result(cohort)
    missing_result = _result(missing)
    train_result = _result(train)
    eval_result = _result(evaluate)
    cal_result = _result(calibration)
    feat_result = _result(features)

    paper_access = _read_paper_access(read_result)
    source = spec.get("source") or (read_result.get("summary") or {}).get("doi") or "paper target"
    paper_outcomes = spec.get("outcomes") or []
    paper_biomarkers = spec.get("biomarkers") or []
    top_features = _top_features(feat_result)

    comparison_rows = [
        {
            "dimension": "Paper access",
            "paper_target": source,
            "local_result": paper_access,
            "status": _status(exact=paper_access in {"text_available", "full_or_long_text_truncated_for_context"}, approximated=paper_access == "abstract_or_partial_text", unavailable=paper_access in {"unavailable", "metadata_only", "not_recorded"}),
            "evidence": "read_paper",
        },
        {
            "dimension": "Endpoint",
            "paper_target": "; ".join(str(x) for x in paper_outcomes) or "not extracted",
            "local_result": local_endpoint,
            "status": _status(approximated=True),
            "evidence": f"cohort_summary n_cases={cohort_result.get('n_cases', 'not recorded')}",
        },
        {
            "dimension": "Feature set",
            "paper_target": ", ".join(str(x) for x in paper_biomarkers) or "not extracted",
            "local_result": f"{train_result.get('n_features', 'not recorded')} local model features; missingness={missing_result.get('overall_missing_pct', 'not recorded')}%",
            "status": _status(approximated=True),
            "evidence": "field_search, missing_data, train_model",
        },
        {
            "dimension": "Model evaluation",
            "paper_target": "paper disease-prediction performance",
            "local_result": (
                f"{train_result.get('selected_model_type', train_result.get('model_type', 'model'))} "
                f"AUC={train_result.get('auc', eval_result.get('mean_auc', 'not recorded'))}; "
                f"calibration ECE={cal_result.get('ece', 'not recorded')}"
            ),
            "status": _status(approximated=bool(train_result)),
            "evidence": "train_model, evaluate_model, calibration",
        },
        {
            "dimension": "Biomarker interpretation",
            "paper_target": "paper feature importance or biomarker ranking",
            "local_result": ", ".join(top_features[:5]) if top_features else "not recorded",
            "status": _status(approximated=bool(top_features), unavailable=not top_features),
            "evidence": "feature_importance",
        },
    ]

    report_dir = Path(output_dir).expanduser() if output_dir else Path(getattr(ctx, "report_dir", "."))
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "paper_replication_comparison.json"
    md_path = report_dir / "paper_replication_comparison.md"
    table_figure_diff = _resolved_table_figure_diff(
        table_figure_diff,
        figure_paths=_session_figure_paths(ctx, records),
        table_artifact=str(md_path),
    )
    matched_artifacts = [row for row in table_figure_diff if row.get("local_artifact")]

    unavailable = [row for row in comparison_rows if row["status"] == "unavailable"]
    approximations = [row for row in comparison_rows if row["status"] == "approximation"]
    overall_status = "partial_replication" if approximations or unavailable else "exact_replication_candidate"
    acceptance_gates = _build_acceptance_gates(
        paper_access=paper_access,
        cohort_result=cohort_result,
        train_result=train_result,
        eval_result=eval_result,
        cal_result=cal_result,
        top_features=top_features,
        n_targets=len(table_figure_diff),
        n_matched=len(matched_artifacts),
        min_cases=max(1, int(min_cases or 100)),
        min_auc=float(min_auc if min_auc is not None else 0.55),
        max_calibration_ece=float(max_calibration_ece if max_calibration_ece is not None else 0.2),
        require_figure_match=bool(require_figure_match),
    )
    acceptance_summary = {
        "verdict": _acceptance_verdict(acceptance_gates, overall_status=overall_status),
        "passed": sum(1 for gate in acceptance_gates if gate["status"] == "PASS"),
        "partial": sum(1 for gate in acceptance_gates if gate["status"] in {"PARTIAL", "WARN"}),
        "failed": sum(1 for gate in acceptance_gates if gate["status"] == "FAIL"),
        "thresholds": {
            "min_cases": max(1, int(min_cases or 100)),
            "min_auc": float(min_auc if min_auc is not None else 0.55),
            "max_calibration_ece": float(max_calibration_ece if max_calibration_ece is not None else 0.2),
            "require_figure_match": bool(require_figure_match),
        },
    }
    report_requirements = [
        "State that this is a closest feasible UKB-only slice, not full paper replication, unless all dimensions are exact.",
        f"Report replication acceptance as {acceptance_summary['verdict']} and include every failed or partial gate.",
        "Report paper access limitations before comparing results.",
        "Separate paper method targets from local UKB endpoint, feature set and model diagnostics.",
        "Include model discrimination, calibration, missingness and feature-importance evidence.",
        "Preserve observational and non-causal claim boundaries.",
    ]
    payload = {
        "scope": scope,
        "paper_access": paper_access,
        "source": source,
        "local_endpoint": local_endpoint,
        "overall_status": overall_status,
        "n_approximated_dimensions": len(approximations),
        "n_unavailable_dimensions": len(unavailable),
        "comparison_rows": comparison_rows,
        "table_figure_diff": table_figure_diff,
        "n_table_figure_targets": len(table_figure_diff),
        "n_local_artifacts_matched": len(matched_artifacts),
        "table_figure_diff_status": "partial" if matched_artifacts else "not_matched",
        "acceptance_gates": acceptance_gates,
        "acceptance_summary": acceptance_summary,
        "report_requirements": report_requirements,
        "top_local_features": top_features,
    }

    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _write_markdown(md_path, payload)
    payload["artifacts"] = {
        "comparison_json": str(json_path),
        "comparison_markdown": str(md_path),
    }
    return payload
