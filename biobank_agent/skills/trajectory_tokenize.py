"""HealthFormer-style trajectory tokenization skill."""

import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from biobank_agent.data.trajectory import TrajectoryTokenizer
from biobank_agent.data.loader import _quote_ident
from biobank_agent.registry import skill


@dataclass(frozen=True)
class _TrajectoryFieldSpec:
    semantic: str
    label: str
    value_type: str = "continuous"
    unit: str = ""


_DEFAULT_TRAJECTORY_FIELDS = [
    _TrajectoryFieldSpec("bmi", "bmi", unit="kg/m2"),
    _TrajectoryFieldSpec("hba1c", "hba1c", unit="mmol/mol"),
    _TrajectoryFieldSpec("glucose", "glucose", unit="mmol/L"),
    _TrajectoryFieldSpec("systolic_bp", "systolic_bp", unit="mmHg"),
    _TrajectoryFieldSpec("diastolic_bp", "diastolic_bp", unit="mmHg"),
    _TrajectoryFieldSpec("ldl_cholesterol", "ldl_cholesterol", unit="mmol/L"),
    _TrajectoryFieldSpec("hdl_cholesterol", "hdl_cholesterol", unit="mmol/L"),
]


def _instance_timestamp(instance: int) -> str:
    """Map UKB-like assessment instances to deterministic pseudo-dates.

    The tokenization layer needs ordered timestamps. Many local UKB extracts
    expose repeated assessment instances but not exact visit dates in the same
    wide biomarker table. We therefore use auditable synthetic anchors and mark
    ``trajectory_time_source`` accordingly so reports cannot overstate temporal
    precision.
    """
    year = 2010 + max(0, int(instance)) * 4
    return f"{year:04d}-01-01"


def _native_instance(column: str, names: list[str]) -> int | None:
    escaped = "|".join(re.escape(n.lower()) for n in names if n)
    if not escaped:
        return None
    patterns = [
        rf"^(?:{escaped})[_-]?(?:visit|wave|time|instance|round|t|v)[_-]?(\d+)$",
        rf"^(?:{escaped})[_-](\d+)$",
    ]
    lower = column.lower()
    for pattern in patterns:
        match = re.match(pattern, lower)
        if match:
            return int(match.group(1))
    return 0 if lower in set(names) else None


def _discover_trajectory_columns(dm: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Discover repeated biomarker columns and return extraction metadata."""
    try:
        columns = list(dm.list_parquet_columns())
    except Exception as exc:
        return [], {
            "source": "data_manager",
            "status": "NO_COLUMNS",
            "blocking_reasons": [f"Could not list biomarker columns: {exc}"],
        }

    id_col = getattr(dm, "subject_id_col", getattr(getattr(dm, "settings", None), "subject_id_col", "eid"))
    column_set = set(columns)
    discovered: list[dict[str, Any]] = []

    for spec in _DEFAULT_TRAJECTORY_FIELDS:
        try:
            resolved = str(dm.resolve_field_id(spec.semantic))
        except Exception:
            resolved = spec.semantic
        names = [spec.semantic.lower(), spec.label.lower(), resolved.lower()]
        seen: set[str] = set()

        # UKB-style repeated assessment columns: <field>-<instance>.<array>
        ukb_pattern = re.compile(rf"^{re.escape(resolved)}-(\d+)\.(\d+)$")
        for column in columns:
            match = ukb_pattern.match(str(column))
            if not match:
                continue
            instance = int(match.group(1))
            array = int(match.group(2))
            if array != 0:
                continue
            discovered.append({
                "column": column,
                "modality": spec.label,
                "instance": instance,
                "value_type": spec.value_type,
                "unit": spec.unit,
                "time_source": "ukb_assessment_instance_synthetic_date",
            })
            seen.add(column)

        # Native HPP/CKB-style columns: bmi, bmi_visit1, hba1c_t2, ...
        for column in columns:
            if column == id_col or column in seen:
                continue
            instance = _native_instance(str(column), names)
            if instance is None:
                continue
            discovered.append({
                "column": column,
                "modality": spec.label,
                "instance": instance,
                "value_type": spec.value_type,
                "unit": spec.unit,
                "time_source": "native_column_instance_synthetic_date",
            })
            seen.add(column)

        # Fall back to DataManager.field_column for a single baseline column.
        if not seen:
            try:
                physical = dm.field_column(resolved, source="biomarkers")
            except Exception:
                physical = ""
            if physical and physical in column_set:
                discovered.append({
                    "column": physical,
                    "modality": spec.label,
                    "instance": 0,
                    "value_type": spec.value_type,
                    "unit": spec.unit,
                    "time_source": "single_baseline_column",
                })

    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for item in discovered:
        unique[(item["column"], item["modality"])] = item
    discovered = sorted(unique.values(), key=lambda x: (x["modality"], x["instance"], x["column"]))

    instances = {int(item["instance"]) for item in discovered}
    modality_instances: dict[str, set[int]] = {}
    for item in discovered:
        modality_instances.setdefault(item["modality"], set()).add(int(item["instance"]))
    repeated_modalities = sorted(m for m, inst in modality_instances.items() if len(inst) >= 2)
    support = "multi_timepoint" if repeated_modalities or len(instances) >= 2 else "single_timepoint"
    return discovered, {
        "source": "data_manager",
        "status": "READY" if discovered else "NO_MATCHING_FIELDS",
        "subject_id_col": id_col,
        "columns_used": [item["column"] for item in discovered],
        "modalities_used": sorted({item["modality"] for item in discovered}),
        "instances_used": sorted(instances),
        "repeated_modalities": repeated_modalities,
        "longitudinal_support": support,
        "trajectory_time_source": (
            "synthetic_assessment_instance_dates"
            if discovered
            else "none"
        ),
    }


def _rows_from_data_manager(dm: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    discovered, summary = _discover_trajectory_columns(dm)
    if not discovered:
        return [], summary
    id_col = str(summary.get("subject_id_col") or "eid")
    select_cols = [f"{_quote_ident(id_col)} AS {_quote_ident(id_col)}"]
    for item in discovered:
        column = str(item["column"])
        select_cols.append(f"{_quote_ident(column)} AS {_quote_ident(column)}")
    try:
        df = dm.query(f"SELECT {', '.join(select_cols)} FROM biomarkers")
    except Exception as exc:
        summary = dict(summary)
        summary["status"] = "QUERY_FAILED"
        summary["blocking_reasons"] = [f"Could not extract biomarker rows: {exc}"]
        return [], summary

    rows: list[dict[str, Any]] = []
    for item in discovered:
        column = item["column"]
        if column not in df.columns:
            continue
        values = df[[id_col, column]].dropna()
        for participant_id, value in values.itertuples(index=False, name=None):
            rows.append({
                "participant_id": str(participant_id),
                "timestamp": _instance_timestamp(int(item["instance"])),
                "modality": item["modality"],
                "value": value,
                "value_type": item["value_type"],
                "unit": item["unit"],
                "source_column": column,
                "time_source": item["time_source"],
            })

    summary = dict(summary)
    summary["n_rows_extracted"] = len(rows)
    if not rows:
        summary["status"] = "NO_NON_NULL_VALUES"
        summary["blocking_reasons"] = ["Matching trajectory columns were found but all extracted values were null."]
    return rows, summary


@skill(
    name="trajectory_tokenize",
    description=(
        "Tokenize longitudinal multimodal participant measurements into HealthFormer-style "
        "time-ordered sequences. If no rows are supplied, attempts to extract repeated "
        "assessment biomarker columns from the active DataManager. This prepares "
        "cohort-aligned evaluation data; it does not train a model."
    ),
    parameters={
        "rows_json": {
            "type": "string",
            "description": (
                "JSON list of rows with participant_id, timestamp, modality, value, optional "
                "value_type/unit/sleep. If empty, uses ctx.state.custom_data['trajectory_rows'] "
                "or auto-extracts repeated biomarker columns from ctx.dm."
            ),
            "default": "",
        },
        "max_bins": {
            "type": "integer",
            "description": "Maximum quantile bins per continuous modality",
            "default": 20,
        },
        "target_modality": {
            "type": "string",
            "description": "Optional future query modality to include in the output",
            "default": "",
        },
        "target_timestamp": {
            "type": "string",
            "description": "Optional future query timestamp for target_modality",
            "default": "",
        },
    },
    required=[],
)
def trajectory_tokenize(
    rows_json: str = "",
    max_bins: int = 20,
    target_modality: str = "",
    target_timestamp: str = "",
    *,
    ctx=None,
) -> dict:
    rows = None
    extraction_summary = {
        "source": "explicit_rows_json" if rows_json else "none",
        "status": "NOT_ATTEMPTED",
        "longitudinal_support": "unknown",
        "trajectory_time_source": "provided_timestamps" if rows_json else "none",
    }
    if rows_json:
        rows = json.loads(rows_json)
    elif ctx is not None:
        rows = getattr(getattr(ctx, "state", None), "custom_data", {}).get("trajectory_rows")
        if rows:
            extraction_summary = {
                "source": "ctx.state.custom_data.trajectory_rows",
                "status": "READY",
                "longitudinal_support": "provided_rows",
                "trajectory_time_source": "provided_timestamps",
                "n_rows_extracted": len(rows),
            }
    if not rows and ctx is not None and getattr(ctx, "dm", None) is not None:
        rows, extraction_summary = _rows_from_data_manager(ctx.dm)
    if not rows:
        return {
            "status": "PARTIAL",
            "n_participants": 0,
            "n_tokens": 0,
            "vocab_size": 0,
            "modalities": [],
            "missingness_by_modality": {},
            "first_sequence": {},
            "future_query": None,
            "future_query_error": "",
            "blocking_reasons": [
                "No longitudinal trajectory rows were supplied in rows_json or ctx.state.custom_data['trajectory_rows'], "
                "and no repeated biomarker columns could be auto-extracted from the active DataManager."
            ] + list(extraction_summary.get("blocking_reasons", []) or []),
            "expected_columns": ["participant_id", "timestamp", "modality", "value"],
            "auto_extraction": extraction_summary,
            "longitudinal_support": extraction_summary.get("longitudinal_support", "unknown"),
            "trajectory_time_source": extraction_summary.get("trajectory_time_source", "none"),
        }

    df = pd.DataFrame(rows)
    tokenizer = TrajectoryTokenizer(max_bins=max_bins)
    dataset = tokenizer.fit(df).transform(df)
    future_query = None
    future_query_error = ""
    if target_modality and target_timestamp:
        try:
            future_query = tokenizer.build_future_query(target_modality, target_timestamp)
        except ValueError as exc:
            future_query_error = str(exc)

    if ctx is not None and hasattr(getattr(ctx, "memory", None), "upsert_node"):
        node_id = f"trajectory:{dataset.n_participants}:{dataset.n_tokens}"
        ctx.memory.upsert_node(
            "trajectory_dataset",
            node_id,
            payload={
                "n_participants": dataset.n_participants,
                "n_tokens": dataset.n_tokens,
                "vocab_size": dataset.vocab_size,
                "modalities": list(dataset.vocab.keys()),
                "missingness_by_modality": dataset.missingness_by_modality,
                "longitudinal_support": extraction_summary.get("longitudinal_support", "unknown"),
                "trajectory_time_source": extraction_summary.get("trajectory_time_source", "provided_timestamps"),
            },
            score=1.0,
        )

    longitudinal_support = str(extraction_summary.get("longitudinal_support", "unknown"))
    status = "READY" if dataset.n_tokens else "PARTIAL"
    if dataset.n_tokens and longitudinal_support == "single_timepoint":
        status = "PARTIAL"

    return {
        "n_participants": dataset.n_participants,
        "n_tokens": dataset.n_tokens,
        "vocab_size": dataset.vocab_size,
        "modalities": list(dataset.vocab.keys()),
        "missingness_by_modality": dataset.missingness_by_modality,
        "first_sequence": dataset.sequences[0].to_dict() if dataset.sequences else {},
        "future_query": future_query,
        "future_query_error": future_query_error,
        "status": status,
        "auto_extraction": extraction_summary,
        "longitudinal_support": longitudinal_support,
        "trajectory_time_source": extraction_summary.get("trajectory_time_source", "provided_timestamps"),
    }
