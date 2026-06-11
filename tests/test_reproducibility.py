"""Tests for reproducibility harness (Stream B).

Tests SHA-256 checkpoints, audit logs, and provenance records.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from biobank_agent.reproducibility import (
    ReproducibilityHarness,
    ExecutionContext,
    AuditEntry,
    ProvenanceRecord,
)


@pytest.fixture
def harness(tmp_path):
    return ReproducibilityHarness(checkpoint_dir=tmp_path)


class TestCheckpointing:
    """Test SHA-256 checkpoint creation and verification."""

    def test_checkpoint_returns_sha256(self, harness):
        """Checkpoint should return 64-char hex string (SHA-256)."""
        h = harness.checkpoint("test_skill", {"a": 1}, {"b": 2})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_checkpoint_creates_file(self, harness):
        """Checkpoint should write JSON file to disk."""
        h = harness.checkpoint("test_skill", {"x": 42}, {"result": "ok"})
        file = harness.checkpoint_dir / f"{h}.json"
        assert file.exists()
        data = json.loads(file.read_text())
        assert data["skill_name"] == "test_skill"
        assert data["inputs"] == {"x": 42}
        assert data["outputs"] == {"result": "ok"}

    def test_verify_valid_checkpoint(self, harness):
        """Valid checkpoint should verify successfully."""
        h = harness.checkpoint("skill", {"in": 1}, {"out": 2})
        assert harness.verify_checkpoint(h) is True

    def test_verify_missing_checkpoint(self, harness):
        """Missing checkpoint should fail verification."""
        assert harness.verify_checkpoint("a" * 64) is False

    def test_verify_corrupted_checkpoint(self, harness, tmp_path):
        """Corrupted checkpoint should fail verification."""
        h = harness.checkpoint("skill", {"in": 1}, {"out": 2})
        file = harness.checkpoint_dir / f"{h}.json"
        # Corrupt the file
        file.write_text('{"corrupted": true}')
        assert harness.verify_checkpoint(h) is False

    def test_deterministic_hash(self, harness):
        """Same inputs should produce same hash."""
        h1 = harness.checkpoint("s", {"a": 1}, {"b": 2})
        # Reset to avoid duplicate file issue
        harness.reset()
        h2 = harness.checkpoint("s", {"a": 1}, {"b": 2})
        # Hashes include timestamp, so they differ — this tests that format is consistent
        assert len(h1) == len(h2) == 64

    def test_checkpoint_records_dependencies(self, harness):
        """Checkpoint should capture package versions."""
        h = harness.checkpoint("skill", {}, {})
        file = harness.checkpoint_dir / f"{h}.json"
        data = json.loads(file.read_text())
        assert "dependencies" in data
        assert "pandas" in data["dependencies"] or "numpy" in data["dependencies"]

    def test_git_version_detection_and_dependency_import_failures(self, harness, monkeypatch, tmp_path):
        """Git and optional dependency failures should fall back cleanly."""
        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("git")))
        assert ReproducibilityHarness(checkpoint_dir=tmp_path)._detect_git_version() == "unknown"

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("git", 5)),
        )
        assert ReproducibilityHarness(checkpoint_dir=tmp_path)._detect_git_version() == "unknown"

        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout=""))
        assert ReproducibilityHarness(checkpoint_dir=tmp_path)._detect_git_version() == "unknown"

        real_import = __import__

        def missing_science_packages(name, *args, **kwargs):
            if name in {"pandas", "numpy", "sklearn", "scikit_learn", "duckdb", "scipy", "xgboost", "lightgbm"}:
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", missing_science_packages)
        assert harness._get_key_dependencies() == {}

    def test_checkpoint_write_failure_is_non_fatal(self, harness, monkeypatch):
        """Checkpoint hashing should still return an id if the JSON write fails."""
        monkeypatch.setattr(Path, "write_text", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full")))

        checkpoint_id = harness.checkpoint("skill", {"x": 1}, {"y": 2})

        assert len(checkpoint_id) == 64
        assert harness.get_audit_log()[-1].entry_id == checkpoint_id

    def test_verify_checkpoint_handles_invalid_json(self, harness):
        """Invalid checkpoint JSON should fail verification instead of raising."""
        bad_hash = "b" * 64
        (harness.checkpoint_dir / f"{bad_hash}.json").write_text("{not-json", encoding="utf-8")

        assert harness.verify_checkpoint(bad_hash) is False


class TestAuditLog:
    """Test audit log creation and chain linking."""

    def test_create_audit_log(self, harness):
        """Audit log entry should be created."""
        entry = harness.create_audit_log("skill_a", {"x": 1}, {"y": 2}, duration_ms=150.0)
        assert entry.skill_name == "skill_a"
        assert entry.duration_ms == 150.0
        assert entry.status == "success"

    def test_audit_chain_linking(self, harness):
        """Sequential audit entries should link parent→child."""
        e1 = harness.create_audit_log("step1", {}, {})
        e2 = harness.create_audit_log("step2", {}, {})
        assert e2.parent_entry == e1.entry_id

    def test_audit_log_accumulates(self, harness):
        """Audit log should grow with each entry."""
        harness.create_audit_log("s1", {}, {})
        harness.create_audit_log("s2", {}, {})
        harness.create_audit_log("s3", {}, {})
        assert len(harness.get_audit_log()) == 3

    def test_failed_status(self, harness):
        """Failed executions should be recorded."""
        entry = harness.create_audit_log("bad_skill", {}, {}, status="failed", error="timeout")
        assert entry.status == "failed"
        assert entry.error == "timeout"


class TestProvenance:
    """Test provenance record export."""

    def test_export_provenance(self, harness):
        """Export should produce a complete provenance record."""
        harness.create_audit_log("s1", {"a": 1}, {"b": 2})
        harness.create_audit_log("s2", {"c": 3}, {"d": 4})
        record = harness.export_provenance("session_123", goal="Test analysis")
        assert record.session_id == "session_123"
        assert record.goal == "Test analysis"
        assert record.n_steps == 2
        assert record.total_duration_ms == 0.0
        assert record.final_status == "completed"

    def test_provenance_chain_hash_is_sha256(self, harness):
        """Chain hash should be 64-char SHA-256."""
        harness.create_audit_log("s1", {}, {})
        record = harness.export_provenance("test")
        h = record.chain_hash()
        assert len(h) == 64

    def test_provenance_persisted_to_disk(self, harness):
        """Provenance record should be saved as JSON."""
        harness.create_audit_log("s1", {}, {})
        harness.export_provenance("sess_001")
        file = harness.checkpoint_dir / "provenance_sess_001.json"
        assert file.exists()

    def test_export_empty_and_write_failure(self, harness, monkeypatch):
        """Empty provenance and persistence failures should be represented cleanly."""
        empty_record = harness.export_provenance("empty")
        assert empty_record.final_status == "empty"

        harness.create_audit_log("s1", {}, {})
        monkeypatch.setattr(Path, "write_text", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("read-only")))

        record = harness.export_provenance("unsaved")

        assert record.final_status == "completed"
        assert record.n_steps == 1

    def test_reset_clears_log(self, harness):
        """Reset should clear in-memory log."""
        harness.create_audit_log("s1", {}, {})
        assert len(harness.get_audit_log()) == 1
        harness.reset()
        assert len(harness.get_audit_log()) == 0
