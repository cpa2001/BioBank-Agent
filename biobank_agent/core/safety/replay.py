"""``/replay`` skill / slash-command implementation.

Loads a stored ``ReproducibilityHarness`` checkpoint by ``provenance_id``
or ``context_hash``, recomputes the live data fingerprint, and either
replays the skill (when fingerprints match) or refuses with an
explanation of what drifted.

Per the M1 contract agreed with Codex:

* Default is **warn-and-execute**: if the fingerprint diverges in a
  recoverable way (row count drift but identical schema/cohorts) we
  surface a warning and proceed, marking the result with
  ``replay.drift = "warn"``.

* The user / caller can pass ``allow_data_drift=False`` (strict mode)
  which refuses to execute on any divergence.

* For severe divergence (table schema changed, cohorts disappeared) we
  refuse regardless of the flag and return a structured diff.

This module exposes the implementation as a plain function so both the
CLI ``/replay`` command and the existing skill registry can call it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from .data_fingerprint import (
    DataFingerprint,
    diff_fingerprints,
    fingerprint as compute_fingerprint,
)

logger = logging.getLogger(__name__)


_SEVERE_DRIFT_KEYS = ("bank_id", "tables", "cohorts")


def load_checkpoint(
    checkpoint_dir: Path,
    provenance_id: str,
    *,
    legacy_provenances: Optional[list] = None,
) -> Optional[dict[str, Any]]:
    """Load a stored ``ReproducibilityHarness`` checkpoint by id.

    Accepts three id forms (Codex review MAJOR #4 — the project has
    two id systems and CLI users shouldn't have to pick the right
    one):

    1. Full SHA-256 ``context_hash`` produced by
       ``ReproducibilityHarness.checkpoint`` (canonical).
    2. Any prefix (>= 6 chars) of that hash.
    3. The 8-char MD5 ``Provenance.make_id`` produced by the legacy
       loop in ``state.py:63``. We use ``legacy_provenances`` to
       resolve the matching ``context_hash`` if provided.
    """
    if not provenance_id:
        return None
    pid = provenance_id.strip()
    candidates: list[Path] = []
    full = checkpoint_dir / f"{pid}.json"
    if full.exists():
        candidates.append(full)
    if len(pid) >= 6 and not candidates:
        for path in checkpoint_dir.glob(f"{pid}*.json"):
            candidates.append(path)
    # 8-char Provenance.make_id resolution.
    if not candidates and len(pid) == 8 and legacy_provenances:
        for prov in legacy_provenances:
            try:
                if str(getattr(prov, "provenance_id", "")) == pid:
                    target_hash = getattr(prov, "context_hash", "") or getattr(
                        prov, "result_hash", ""
                    )
                    if target_hash:
                        match = checkpoint_dir / f"{target_hash}.json"
                        if match.exists():
                            candidates.append(match)
                            break
            except Exception:  # pragma: no cover
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    try:
        return json.loads(candidates[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("failed to parse checkpoint %s: %s", candidates[0], e)
        return None


def _fingerprint_from_checkpoint(payload: dict[str, Any]) -> Optional[DataFingerprint]:
    """Extract the stored DataFingerprint from a checkpoint payload."""
    inputs = payload.get("inputs") or {}
    if not isinstance(inputs, dict):
        return None
    fp_dict = inputs.get("fingerprint")
    if not isinstance(fp_dict, dict):
        return None
    try:
        from .data_fingerprint import TableSnapshot
        tables = [TableSnapshot(**t) for t in fp_dict.get("tables", [])]
        return DataFingerprint(
            bank_id=fp_dict.get("bank_id", ""),
            bank_config_hash=fp_dict.get("bank_config_hash", ""),
            duckdb_schema_hash=fp_dict.get("duckdb_schema_hash", ""),
            tables=tables,
            cohort_ids=list(fp_dict.get("cohort_ids", [])),
            inputs_hash=fp_dict.get("inputs_hash", ""),
            fingerprint_hash=fp_dict.get("fingerprint_hash", ""),
        )
    except Exception as e:
        logger.warning("could not deserialise fingerprint: %s", e)
        return None


def _classify_drift(diff: dict[str, Any]) -> str:
    """Return one of ``"none" | "soft" | "severe"`` for the diff."""
    if not diff:
        return "none"
    severe_keys = set(diff.keys()) & set(_SEVERE_DRIFT_KEYS)
    if severe_keys:
        # Even within these, row_count_changed is "soft" — keep severity
        # as soft if all table changes are row count only.
        if "tables" in severe_keys:
            tbl_changes = diff.get("tables") or {}
            severe_table_change = any(
                v not in (None, "row_count_changed")
                and not (isinstance(v, str) and v.startswith("row_count_changed"))
                for v in tbl_changes.values()
            )
            other_severe = severe_keys - {"tables"}
            if not severe_table_change and not other_severe:
                return "soft"
        return "severe"
    return "soft"


def replay(
    *,
    provenance_id: str,
    legacy_agent: Any,
    allow_data_drift: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Replay a previously checkpointed skill execution.

    Parameters
    ----------
    provenance_id:
        ``context_hash`` (or unique prefix) returned by the original
        ``ReproducibilityHarness.checkpoint`` call. Must exist on disk
        in ``settings.reports_dir / "checkpoints"``.
    legacy_agent:
        The legacy ``biobank_agent.agent.Agent`` providing settings,
        registry, DuckDB connection, etc.
    allow_data_drift:
        When True (default), soft drift produces a warning but the
        skill is still executed. When False, *any* drift triggers a
        refusal with the diff so the caller can investigate.
    dry_run:
        When True, never executes the skill; only returns the diff.

    Returns
    -------
    A dict with at minimum ``status``, ``drift``, and ``diff``. When
    the replay actually executes, also includes ``result``.
    """
    settings = legacy_agent.settings
    checkpoint_dir = settings.reports_dir / "checkpoints"
    payload = load_checkpoint(
        checkpoint_dir,
        provenance_id,
        legacy_provenances=getattr(legacy_agent.state, "provenances", None),
    )
    if payload is None:
        return {
            "status": "not_found",
            "provenance_id": provenance_id,
            "error": f"No checkpoint matching '{provenance_id}' under {checkpoint_dir}",
        }

    skill_name = str(payload.get("skill_name") or "")
    inputs_blob = payload.get("inputs") or {}
    skill_args = (
        inputs_blob.get("args")
        if isinstance(inputs_blob, dict) and isinstance(inputs_blob.get("args"), dict)
        else (inputs_blob if isinstance(inputs_blob, dict) else {})
    )

    saved_fp = _fingerprint_from_checkpoint(payload)
    current_fp = compute_fingerprint(
        settings=legacy_agent.settings,
        duckdb_conn=getattr(legacy_agent.dm, "conn", None),
        state=legacy_agent.state,
        inputs=dict(skill_args or {}),
    )
    if saved_fp is None:
        # Codex review BLOCKER #3: missing saved fingerprint must refuse,
        # not be treated as soft drift. The whole point of /replay is to
        # prove data identity — without a saved fingerprint we cannot.
        return {
            "status": "refused",
            "provenance_id": provenance_id,
            "skill": skill_name,
            "args": skill_args,
            "drift": "missing_fingerprint",
            "diff": {"missing_saved_fingerprint": True},
            "saved_fingerprint": None,
            "current_fingerprint": current_fp.to_dict(),
            "error": (
                "checkpoint has no stored data fingerprint; cannot prove "
                "data identity. Re-run the skill once the AsyncAgent "
                "reproducibility hook has been active to record a "
                "fingerprint, then /replay will be able to verify drift."
            ),
        }
    diff = diff_fingerprints(saved_fp, current_fp)
    drift_class = _classify_drift(diff)

    response: dict[str, Any] = {
        "status": "ok",
        "provenance_id": provenance_id,
        "skill": skill_name,
        "args": skill_args,
        "drift": drift_class,
        "diff": diff,
        "saved_fingerprint": asdict(saved_fp) if saved_fp else None,
        "current_fingerprint": current_fp.to_dict(),
    }

    if drift_class == "severe":
        response["status"] = "refused"
        response["error"] = "data fingerprint drift exceeds replay safety threshold"
        return response

    if drift_class == "soft" and not allow_data_drift:
        response["status"] = "refused"
        response["error"] = "soft drift detected; pass allow_data_drift=True to proceed"
        return response

    if dry_run or not skill_name:
        response["status"] = "dry_run" if dry_run else "no_skill_in_checkpoint"
        return response

    # Replay via the legacy executor — gives us reflexion, audit, etc.
    try:
        report_dir = settings.reports_dir / f"replay_{provenance_id[:8]}"
        report_dir.mkdir(parents=True, exist_ok=True)
        execution = legacy_agent._execute_skill_and_record(
            skill_name,
            dict(skill_args or {}),
            report_dir,
            allow_retry=False,
        )
    except Exception as e:
        response["status"] = "execution_failed"
        response["error"] = str(e)
        return response

    response["result"] = execution.get("result")
    response["is_error"] = execution.get("is_error", False)
    response["report_dir"] = str(report_dir)
    if drift_class == "soft":
        response["status"] = "completed_with_warning"
    return response


__all__ = ["replay", "load_checkpoint"]
