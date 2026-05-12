"""Credential-gated readiness probes for configured biobank data paths."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from biobank_agent.config import Settings
from biobank_agent.data.loader import DataManager
from biobank_agent.domain.banks import canonical_bank_id
from biobank_agent.registry import skill
from biobank_agent.skills.bank_data_probe import bank_data_probe


_LOCAL_DATA_ENV = {
    "ukb": ("BIOBANK_UKB_DATA_DIR", "UKB_PARQUET_DIR", "DATA_DIR"),
    "hpp": ("BIOBANK_HPP_DATA_DIR", "HPP_DATA_DIR"),
    "ckb": ("BIOBANK_CKB_DATA_DIR", "CKB_DATA_DIR"),
}

_RAP_CREDENTIAL_ENV = (
    "BIOBANK_RAP_ENABLED",
    "DX_PROJECT_CONTEXT_ID",
    "DX_WORKSPACE_ID",
    "DNANEXUS_PROJECT_ID",
)


def _env_path(bank_id: str) -> tuple[Path | None, str]:
    for env_name in _LOCAL_DATA_ENV.get(bank_id, ()):
        value = os.getenv(env_name, "").strip()
        if value:
            return Path(value).expanduser(), env_name
    return None, ""


def _path_has_bank_files(bank_id: str, path: Path) -> bool:
    expected = {
        "ukb": ("ukb.parquet", "hesin_diag.parquet"),
        "hpp": ("hpp_biomarkers.parquet", "hpp_diagnoses.parquet"),
        "ckb": ("ckb_biomarkers.parquet", "ckb_diagnoses.parquet"),
    }.get(bank_id, ())
    return bool(expected) and all((path / name).exists() for name in expected)


def _base_settings(ctx: Any, bank_id: str, data_dir: Path, report_dir: Path) -> Settings:
    current = getattr(ctx, "settings", None)
    return Settings(
        data_dir=data_dir,
        raw_dir=data_dir,
        reports_dir=report_dir,
        memory_dir=getattr(current, "memory_dir", Path.home() / ".biobank_agent"),
        plans_dir=getattr(current, "plans_dir", Path("./plans")),
        bank_id=bank_id,
        max_train_rows_default=0,
        default_analysis_sample_size=0,
    )


def _run_probe_for_settings(settings: Settings, report_dir: Path, icd10_code: str, probe_fields: str) -> dict[str, Any]:
    dm = DataManager(settings)
    probe_ctx = SimpleNamespace(
        dm=dm,
        settings=settings,
        report_dir=report_dir,
        emit_progress=lambda *args, **kwargs: None,
    )
    return bank_data_probe(icd10_code=icd10_code, probe_fields=probe_fields, ctx=probe_ctx)


def _scrub_probe(probe: dict[str, Any]) -> dict[str, Any]:
    """Keep readiness output aggregate-only and JSON-serializable."""
    out = json.loads(json.dumps(probe, default=str))
    out.pop("first_rows", None)
    out.pop("raw_rows", None)
    return out


def _rap_probe(ctx: Any, report_dir: Path, icd10_code: str, probe_fields: str) -> dict[str, Any]:
    present = [name for name in _RAP_CREDENTIAL_ENV if os.getenv(name, "").strip()]
    current_dm = getattr(ctx, "dm", None)
    if current_dm is not None and canonical_bank_id(getattr(current_dm, "bank_id", "")) == "ukb_rap":
        probe = bank_data_probe(icd10_code=icd10_code, probe_fields=probe_fields, ctx=ctx)
        probe["credential_envs_present"] = present
        return _scrub_probe(probe)
    if not present:
        return {
            "status": "SKIPPED_NO_CONFIG",
            "bank_id": "ukb_rap",
            "display_name": "UK Biobank Research Analysis Platform",
            "is_remote": True,
            "credential_envs_present": [],
            "required_any_env": list(_RAP_CREDENTIAL_ENV),
            "warnings": ["No RAP credential/context environment variables are configured."],
        }

    settings = _base_settings(ctx, "ukb_rap", Path.home() / ".biobank_agent" / "rap_cache", report_dir)
    probe = _run_probe_for_settings(settings, report_dir, icd10_code, probe_fields)
    probe["credential_envs_present"] = present
    try:
        __import__("dxpy")
        probe["dxpy_available"] = True
    except Exception:
        probe["dxpy_available"] = False
        probe.setdefault("warnings", []).append("dxpy is not installed; RAP probe is credential-ready but cannot execute DNAnexus queries yet.")
    return _scrub_probe(probe)


def _local_probe(
    ctx: Any,
    bank_id: str,
    report_dir: Path,
    icd10_code: str,
    probe_fields: str,
    use_current_context: bool,
) -> dict[str, Any]:
    current_dm = getattr(ctx, "dm", None)
    if use_current_context and current_dm is not None and canonical_bank_id(getattr(current_dm, "bank_id", "")) == bank_id:
        probe = bank_data_probe(icd10_code=icd10_code, probe_fields=probe_fields, ctx=ctx)
        probe["source"] = "current_context"
        return _scrub_probe(probe)

    data_dir, env_name = _env_path(bank_id)
    if data_dir is None and bank_id == "ukb":
        current_settings = getattr(ctx, "settings", None)
        configured = getattr(current_settings, "data_dir", None)
        if configured:
            data_dir = Path(configured).expanduser()
            env_name = "settings.data_dir"
    if data_dir is None:
        return {
            "status": "SKIPPED_NO_CONFIG",
            "bank_id": bank_id,
            "is_remote": False,
            "data_dir": "",
            "required_any_env": list(_LOCAL_DATA_ENV.get(bank_id, ())),
            "warnings": [f"No configured data directory environment variable for {bank_id}."],
        }
    if not data_dir.exists():
        return {
            "status": "SKIPPED_PATH_MISSING",
            "bank_id": bank_id,
            "is_remote": False,
            "data_dir": str(data_dir),
            "source_env": env_name,
            "warnings": [f"Configured data directory does not exist: {data_dir}"],
        }
    if not _path_has_bank_files(bank_id, data_dir):
        return {
            "status": "SKIPPED_INCOMPLETE_DATA",
            "bank_id": bank_id,
            "is_remote": False,
            "data_dir": str(data_dir),
            "source_env": env_name,
            "warnings": [f"Configured directory is missing required {bank_id} parquet files."],
        }

    settings = _base_settings(ctx, bank_id, data_dir, report_dir)
    probe = _run_probe_for_settings(settings, report_dir, icd10_code, probe_fields)
    probe["source"] = f"env:{env_name}"
    probe["data_dir"] = str(data_dir)
    return _scrub_probe(probe)


def _overall_status(results: list[dict[str, Any]]) -> str:
    statuses = [str(item.get("status", "")).upper() for item in results]
    if not statuses:
        return "ERROR"
    if any(status == "ERROR" for status in statuses):
        return "ERROR"
    if all(status in {"READY", "REMOTE_READY"} for status in statuses):
        return "READY"
    if any(status in {"READY", "REMOTE_READY", "PARTIAL"} for status in statuses):
        return "PARTIAL"
    return "SKIPPED"


def _write_artifacts(report_dir: Path, payload: dict[str, Any]) -> tuple[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"bank_readiness_{stamp}.json"
    md_path = report_dir / f"bank_readiness_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Bank Data Readiness",
        "",
        f"Generated: {payload['generated_at']}",
        f"Overall status: **{payload['status']}**",
        "",
        "| Bank | Status | Subjects | Cases | Source | Warnings |",
        "|---|---:|---:|---:|---|---|",
    ]
    for item in payload["banks"]:
        diag = item.get("diagnosis_probe") or {}
        warnings = "; ".join(item.get("warnings") or [])
        source = item.get("source") or item.get("source_env") or ",".join(item.get("credential_envs_present") or [])
        lines.append(
            "| {bank} | {status} | {subjects} | {cases} | {source} | {warnings} |".format(
                bank=item.get("bank_id", ""),
                status=item.get("status", ""),
                subjects=item.get("n_subjects", ""),
                cases=diag.get("n_case_subjects", ""),
                source=source,
                warnings=warnings.replace("|", "/"),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(json_path), str(md_path)


@skill(
    name="bank_data_readiness",
    description=(
        "Run credential-gated aggregate readiness probes for UKB/HPP/CKB/RAP data paths. "
        "Uses current context or configured environment data directories, writes JSON/Markdown "
        "readiness artifacts, and skips cleanly when credentials or paths are absent."
    ),
    parameters={
        "banks": {
            "type": "string",
            "description": "Comma-separated banks to probe, e.g. ukb,hpp,ckb,ukb_rap",
            "default": "ukb,hpp,ckb,ukb_rap",
        },
        "icd10_code": {
            "type": "string",
            "description": "ICD10 code prefix used for diagnosis readiness checks",
            "default": "E11",
        },
        "probe_fields": {
            "type": "string",
            "description": "Comma-separated biomarker semantic fields to resolve",
            "default": "hba1c,bmi,glucose",
        },
        "output_dir": {
            "type": "string",
            "description": "Optional output directory for readiness artifacts; defaults to report_dir",
            "default": "",
        },
        "use_current_context": {
            "type": "boolean",
            "description": "Use the active agent DataManager when its bank matches a requested bank",
            "default": True,
        },
    },
    required=[],
)
def bank_data_readiness(
    banks: str = "ukb,hpp,ckb,ukb_rap",
    icd10_code: str = "E11",
    probe_fields: str = "hba1c,bmi,glucose",
    output_dir: str = "",
    use_current_context: bool = True,
    *,
    ctx=None,
) -> dict:
    if ctx is None:
        return {"error": "No agent context is available.", "status": "ERROR"}

    report_dir = Path(output_dir).expanduser() if output_dir else Path(getattr(ctx, "report_dir", getattr(ctx.settings, "reports_dir", "./reports")))
    requested = [canonical_bank_id(item.strip()) for item in str(banks or "").split(",") if item.strip()]
    if not requested:
        return {"error": "No banks requested.", "status": "ERROR"}

    results: list[dict[str, Any]] = []
    for bank_id in requested:
        try:
            if bank_id == "ukb_rap":
                result = _rap_probe(ctx, report_dir, icd10_code, probe_fields)
            elif bank_id in _LOCAL_DATA_ENV:
                result = _local_probe(ctx, bank_id, report_dir, icd10_code, probe_fields, bool(use_current_context))
            else:
                result = {"status": "ERROR", "bank_id": bank_id, "warnings": [f"Unknown bank id: {bank_id}"]}
        except Exception as exc:
            result = {"status": "ERROR", "bank_id": bank_id, "error": str(exc), "warnings": [str(exc)]}
        results.append(_scrub_probe(result))

    payload = {
        "artifact_type": "bank_data_readiness",
        "schema_version": 1,
        "generated_by": "bank_data_readiness",
        "status": _overall_status(results),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "icd10_code": icd10_code,
        "probe_fields": [item.strip() for item in str(probe_fields or "").split(",") if item.strip()],
        "banks": results,
    }
    json_path, md_path = _write_artifacts(report_dir, payload)
    payload["artifact_json"] = json_path
    payload["artifact_markdown"] = md_path
    return payload
