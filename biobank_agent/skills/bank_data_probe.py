"""Probe whether the active BankAdapter is wired into the data path."""

from __future__ import annotations

from typing import Any

from biobank_agent.data.loader import _quote_ident
from biobank_agent.registry import skill


def _view_exists(dm: Any, view_name: str) -> bool:
    try:
        dm.conn.execute(f"SELECT 1 FROM {_quote_ident(view_name)} LIMIT 1")
        return True
    except Exception:
        return False


def _case_count_probe(dm: Any, icd10_code: str) -> dict[str, Any]:
    id_col = getattr(dm, "subject_id_col", getattr(getattr(dm, "settings", None), "subject_id_col", "eid"))
    diag_col = getattr(dm, "diagnoses_code_col", getattr(getattr(dm, "settings", None), "diagnoses_code_col", "diag_icd10"))
    normalized = dm.normalize_icd_code(icd10_code) if hasattr(dm, "normalize_icd_code") else str(icd10_code).upper().replace(".", "")
    if not _view_exists(dm, "diagnoses"):
        return {
            "icd10_code": icd10_code,
            "normalized_code": normalized,
            "status": "NO_DIAGNOSES_VIEW",
            "n_case_subjects": None,
        }
    where_sql, params = dm.code_prefix_filter("diagnoses", normalized, diag_col) if hasattr(dm, "code_prefix_filter") else (
        f"{_quote_ident(diag_col)} LIKE ?",
        [f"{normalized}%"],
    )
    if not where_sql:
        return {
            "icd10_code": icd10_code,
            "normalized_code": normalized,
            "status": "NO_CODE_COLUMN",
            "n_case_subjects": None,
        }
    row = dm.conn.execute(
        f"SELECT COUNT(DISTINCT {_quote_ident(id_col)}) FROM diagnoses WHERE {where_sql}",
        params,
    ).fetchone()
    return {
        "icd10_code": icd10_code,
        "normalized_code": normalized,
        "status": "READY",
        "where_sql": where_sql,
        "param_count": len(params),
        "n_case_subjects": int(row[0]) if row else 0,
    }


def _field_probe(dm: Any, fields: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    biomarkers_exists = _view_exists(dm, "biomarkers")
    for field in fields:
        item: dict[str, Any] = {"query": field}
        try:
            resolved = dm.resolve_field_id(field) if hasattr(dm, "resolve_field_id") else field
            item["resolved"] = resolved
            if not biomarkers_exists:
                item["status"] = "NO_BIOMARKERS_VIEW"
            else:
                item["source"] = dm.field_source(resolved) if hasattr(dm, "field_source") else "unknown"
                item["column"] = dm.field_column(resolved, source="biomarkers") if hasattr(dm, "field_column") else resolved
                item["status"] = "READY"
        except Exception as exc:
            item["status"] = "MISSING"
            item["error"] = str(exc)
        out.append(item)
    return out


@skill(
    name="bank_data_probe",
    description=(
        "Check that the active UKB/HPP/CKB/RAP bank adapter is wired into the "
        "DataManager, diagnosis filters, biomarker field resolution, and full-data "
        "subject counts. Returns aggregate metadata only, never raw participant rows."
    ),
    parameters={
        "icd10_code": {
            "type": "string",
            "description": "ICD10 code prefix used to probe diagnosis filtering, e.g. E11",
            "default": "E11",
        },
        "probe_fields": {
            "type": "string",
            "description": "Comma-separated semantic or physical biomarker fields to resolve",
            "default": "hba1c,bmi,glucose",
        },
    },
    required=[],
)
def bank_data_probe(
    icd10_code: str = "E11",
    probe_fields: str = "hba1c,bmi,glucose",
    *,
    ctx=None,
) -> dict:
    dm = getattr(ctx, "dm", None)
    if dm is None:
        return {"error": "No DataManager is available in context.", "status": "ERROR"}

    adapter = getattr(dm, "bank_adapter", None)
    fields = [item.strip() for item in str(probe_fields or "").split(",") if item.strip()]
    views = {
        "biomarkers": _view_exists(dm, "biomarkers"),
        "diagnoses": _view_exists(dm, "diagnoses"),
        "deaths": _view_exists(dm, "deaths"),
    }
    warnings: list[str] = []
    n_subjects = None
    try:
        n_subjects = int(dm.count_subjects()) if views["biomarkers"] else None
    except Exception as exc:
        warnings.append(f"count_subjects failed: {exc}")

    diagnosis = _case_count_probe(dm, icd10_code)
    fields_out = _field_probe(dm, fields)

    if not views["biomarkers"]:
        warnings.append("No biomarkers view is registered for the active bank.")
    if not views["diagnoses"]:
        warnings.append("No diagnoses view is registered for the active bank.")
    if diagnosis.get("status") == "READY" and int(diagnosis.get("n_case_subjects") or 0) == 0:
        warnings.append(f"No cases found for {icd10_code}; check code mapping or data coverage.")
    missing_fields = [item["query"] for item in fields_out if item.get("status") != "READY"]
    if missing_fields:
        warnings.append("Unresolved biomarker probes: " + ", ".join(missing_fields))

    remote = bool(getattr(adapter, "is_remote", False))
    ready = views["biomarkers"] and views["diagnoses"] and diagnosis.get("status") == "READY" and not missing_fields
    return {
        "status": "READY" if ready else "REMOTE_READY" if remote else "PARTIAL",
        "bank_id": getattr(dm, "bank_id", ""),
        "display_name": getattr(adapter, "display_name", getattr(dm, "bank_id", "")),
        "is_remote": remote,
        "subject_id_col": getattr(dm, "subject_id_col", ""),
        "diagnoses_code_col": getattr(dm, "diagnoses_code_col", ""),
        "deaths_code_col": getattr(dm, "deaths_code_col", ""),
        "views": views,
        "n_subjects": n_subjects,
        "diagnosis_probe": diagnosis,
        "field_probe": fields_out,
        "warnings": warnings,
    }
