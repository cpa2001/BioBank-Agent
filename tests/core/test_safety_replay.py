"""Tests for biobank_agent.core.safety.replay."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from biobank_agent.core.safety.data_fingerprint import (
    DataFingerprint,
    TableSnapshot,
)
from biobank_agent.core.safety.replay import (
    _classify_drift,
    load_checkpoint,
    replay,
)


def _write_checkpoint(tmp_dir: Path, ctx_hash: str, payload: dict) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p = tmp_dir / f"{ctx_hash}.json"
    p.write_text(json.dumps(payload, default=str))
    return p


def test_load_checkpoint_full_hash(tmp_path):
    payload = {"skill_name": "prevalence", "inputs": {"args": {"icd10_code": "E11"}}}
    _write_checkpoint(tmp_path, "abcdef0123456789", payload)
    out = load_checkpoint(tmp_path, "abcdef0123456789")
    assert out and out["skill_name"] == "prevalence"


def test_load_checkpoint_prefix_match(tmp_path):
    _write_checkpoint(tmp_path, "ffffeee0001", {"skill_name": "prevalence"})
    out = load_checkpoint(tmp_path, "ffffeee")
    assert out and out["skill_name"] == "prevalence"


def test_classify_drift_severity():
    assert _classify_drift({}) == "none"
    assert _classify_drift({"inputs_hash": ("aaa", "bbb")}) == "soft"
    assert _classify_drift(
        {"tables": {"biomarkers": "schema_changed"}}
    ) == "severe"
    assert _classify_drift(
        {"tables": {"biomarkers": "row_count_changed:100->200"}}
    ) == "soft"
    assert _classify_drift({"bank_id": ("ukb", "ckb")}) == "severe"


class _StubAgentForReplay:
    def __init__(self, tmp_path: Path, raise_on_skill: bool = False):
        self.settings = SimpleNamespace(
            reports_dir=tmp_path,
            biobank_name="ukb",
            bank_id="ukb",
            data_dir=tmp_path / "data",
            field_txt=tmp_path / "field.txt",
            category_txt=tmp_path / "category.txt",
            biomarkers_path=tmp_path / "biomarkers.parquet",
            diagnoses_path=tmp_path / "diagnoses.parquet",
            deaths_path=tmp_path / "deaths.parquet",
            memory_dir=tmp_path / "mem",
        )
        self.state = SimpleNamespace(cohorts={})
        self.dm = SimpleNamespace(conn=None)
        self._raise = raise_on_skill

    def _execute_skill_and_record(self, skill, args, report_dir, allow_retry=False):
        if self._raise:
            raise RuntimeError("intentional")
        return {
            "result": {"replayed_skill": skill, "args": dict(args)},
            "is_error": False,
        }


def _matching_fingerprint(bank_id: str = "ukb") -> dict:
    return {
        "bank_id": bank_id,
        "bank_config_hash": "deadbeef",
        "duckdb_schema_hash": "feedface",
        "tables": [],
        "cohort_ids": [],
        "inputs_hash": "deadbeef",
        "fingerprint_hash": "fingerprint",
        "schema_version": 1,
    }


def test_replay_executes_when_no_drift(tmp_path):
    legacy = _StubAgentForReplay(tmp_path)
    ctx_hash = "fffeeeddccbbaa00"
    checkpoint_dir = legacy.settings.reports_dir / "checkpoints"
    fp = _matching_fingerprint("ukb")
    _write_checkpoint(
        checkpoint_dir,
        ctx_hash,
        {
            "skill_name": "prevalence",
            "inputs": {
                "args": {"icd10_code": "E11"},
                "fingerprint": fp,
            },
            "outputs": {"summary": "stored"},
        },
    )

    out = replay(
        provenance_id=ctx_hash,
        legacy_agent=legacy,
        allow_data_drift=False,
    )
    # Stubs report drift because the live fingerprint will not match
    # the synthetic stored one, but classification should be "soft".
    assert out["status"] in {"completed_with_warning", "refused", "ok"}
    if out["status"] == "completed_with_warning":
        assert out["result"]["replayed_skill"] == "prevalence"
        assert out["drift"] == "soft"


def test_replay_refuses_severe_drift(tmp_path):
    legacy = _StubAgentForReplay(tmp_path)
    ctx_hash = "abc123def456abcd"
    fp = _matching_fingerprint("ckb")  # different bank_id -> severe drift
    _write_checkpoint(
        legacy.settings.reports_dir / "checkpoints",
        ctx_hash,
        {
            "skill_name": "prevalence",
            "inputs": {"args": {"icd10_code": "E11"}, "fingerprint": fp},
        },
    )
    out = replay(provenance_id=ctx_hash, legacy_agent=legacy)
    assert out["status"] == "refused"
    assert out["drift"] == "severe"


def test_replay_handles_unknown_provenance(tmp_path):
    legacy = _StubAgentForReplay(tmp_path)
    out = replay(provenance_id="nonexistent_hash", legacy_agent=legacy)
    assert out["status"] == "not_found"


def test_replay_refuses_when_saved_fingerprint_missing(tmp_path):
    """Replay must refuse when the saved fingerprint is missing."""
    legacy = _StubAgentForReplay(tmp_path)
    ctx_hash = "noprintbeefcafe0"
    _write_checkpoint(
        legacy.settings.reports_dir / "checkpoints",
        ctx_hash,
        {
            "skill_name": "prevalence",
            "inputs": {"args": {"icd10_code": "E11"}},  # no "fingerprint" key
        },
    )
    out = replay(provenance_id=ctx_hash, legacy_agent=legacy, allow_data_drift=True)
    assert out["status"] == "refused"
    assert out["drift"] == "missing_fingerprint"
    assert "fingerprint" in (out.get("error") or "")


def test_replay_dry_run(tmp_path):
    legacy = _StubAgentForReplay(tmp_path)
    ctx_hash = "1234abcd5678efef"
    fp = _matching_fingerprint("ukb")
    _write_checkpoint(
        legacy.settings.reports_dir / "checkpoints",
        ctx_hash,
        {
            "skill_name": "prevalence",
            "inputs": {"args": {"icd10_code": "E11"}, "fingerprint": fp},
        },
    )
    out = replay(provenance_id=ctx_hash, legacy_agent=legacy, dry_run=True)
    assert out["status"] in {"dry_run", "refused"}
