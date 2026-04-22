"""Memory system tests — all 7 tiers + corruption recovery."""

import json
import pytest
import tempfile
from pathlib import Path

from biobank_agent.memory import (
    LongTermMemory,
    DomainMemory,
    UserMemory,
    SessionSearch,
)
from biobank_agent.state import AnalysisRecord


@pytest.fixture
def mem_dir(tmp_path):
    return tmp_path / "memory"


class TestLongTermMemory:
    """Tier 3: Long-term JSON memory."""

    def test_create_fresh(self, mem_dir):
        ltm = LongTermMemory(mem_dir)
        assert mem_dir.exists()  # directory created
        # memory.json created lazily on first save
        ltm.save_pipeline("init", [])
        assert (mem_dir / "memory.json").exists()

    def test_save_and_reload(self, mem_dir):
        ltm = LongTermMemory(mem_dir)
        ltm.save_pipeline("test_pipe", [{"skill": "prevalence", "args": {}}])
        ltm2 = LongTermMemory(mem_dir)
        assert "test_pipe" in ltm2.list_pipelines()

    def test_model_config_best_auc(self, mem_dir):
        ltm = LongTermMemory(mem_dir)
        ltm.remember_model_config("E11", "xgb", {"n_estimators": 200}, 0.85)
        assert ltm.best_auc("E11", "xgb") == 0.85
        ltm.remember_model_config("E11", "xgb", {"n_estimators": 300}, 0.80)
        assert ltm.best_auc("E11", "xgb") == 0.85  # shouldn't downgrade
        ltm.remember_model_config("E11", "xgb", {"n_estimators": 300}, 0.90)
        assert ltm.best_auc("E11", "xgb") == 0.90  # should upgrade

    def test_field_usage(self, mem_dir):
        ltm = LongTermMemory(mem_dir)
        for _ in range(5):
            ltm.record_field_usage("30600")
        ltm.record_field_usage("21001")
        top = ltm.most_used_fields(2)
        assert top[0][0] == "30600"
        assert top[0][1] == 5

    def test_empty_summary(self, mem_dir):
        ltm = LongTermMemory(mem_dir)
        assert ltm.summary() == ""

    def test_nonempty_summary(self, mem_dir):
        ltm = LongTermMemory(mem_dir)
        ltm.save_pipeline("pipe1", [{"skill": "x"}])
        s = ltm.summary()
        assert "pipe1" in s

    def test_corruption_recovery(self, mem_dir):
        mem_dir.mkdir(parents=True, exist_ok=True)
        (mem_dir / "memory.json").write_text("NOT VALID JSON{{{")
        ltm = LongTermMemory(mem_dir)
        assert ltm.list_pipelines() == []  # fresh start


class TestErrorCatalog:
    """Tier 4: Error tracking and suggestions."""

    def test_record_and_retrieve_error(self, mem_dir):
        ltm = LongTermMemory(mem_dir)
        ltm.record_error("ValueError", "bad input", "train_model")
        ltm.record_error("ValueError", "bad input", "train_model")
        errors = ltm.most_common_errors(5)
        assert len(errors) >= 1
        assert errors[0]["error_type"] == "ValueError"
        assert errors[0]["count"] == 2

    def test_error_suggestions(self, mem_dir):
        ltm = LongTermMemory(mem_dir)
        ltm.record_error("ValueError", "test", "skill1", suggested_fix="try X")
        suggestions = ltm.get_error_suggestions("ValueError", "skill1")
        assert "try X" in suggestions

    def test_empty_suggestions(self, mem_dir):
        ltm = LongTermMemory(mem_dir)
        assert ltm.get_error_suggestions("Unknown", "unknown") == []


class TestDomainMemory:
    """Tier 5: Domain knowledge."""

    def test_create_fresh(self, tmp_path):
        dm = DomainMemory(tmp_path / "domain.md")
        assert (tmp_path / "domain.md").exists()

    def test_append_finding(self, tmp_path):
        dm = DomainMemory(tmp_path / "domain.md")
        dm.append_finding("Test Finding", "This is a test body.")
        content = dm.read()
        assert "Test Finding" in content
        assert "This is a test body." in content

    def test_idempotent_append(self, tmp_path):
        dm = DomainMemory(tmp_path / "domain.md")
        dm.append_finding("Finding", "Body text")
        dm.append_finding("Finding", "Body text")
        assert dm.read().count("Body text") == 1

    def test_summary_empty(self, tmp_path):
        dm = DomainMemory(tmp_path / "domain.md")
        assert dm.summary() == ""

    def test_summary_truncation(self, tmp_path):
        dm = DomainMemory(tmp_path / "domain.md")
        for i in range(50):
            dm.append_finding(f"Finding {i}", f"Body {i} " * 50)
        s = dm.summary(max_chars=500)
        assert len(s) <= 600  # Allow some overhead


class TestUserMemory:
    """Tier 6: Researcher preferences."""

    def test_create_fresh(self, tmp_path):
        um = UserMemory(tmp_path / "user.md")
        assert (tmp_path / "user.md").exists()

    def test_upsert_preference(self, tmp_path):
        um = UserMemory(tmp_path / "user.md")
        um.upsert_preference("preferred_model", "xgboost")
        content = um.read()
        assert "preferred_model" in content
        assert "xgboost" in content

    def test_upsert_replaces(self, tmp_path):
        um = UserMemory(tmp_path / "user.md")
        um.upsert_preference("key1", "value1")
        um.upsert_preference("key1", "value2")
        content = um.read()
        assert "value2" in content
        assert content.count("**key1:**") == 1

    def test_summary_empty(self, tmp_path):
        um = UserMemory(tmp_path / "user.md")
        assert um.summary() == ""


class TestSessionSearch:
    """Tier 7: Episodic cross-session recall via FTS5."""

    def test_create_fresh(self, tmp_path):
        ss = SessionSearch(tmp_path / "sessions.db")
        assert (tmp_path / "sessions.db").exists()
        ss.close()

    def test_index_and_search(self, tmp_path):
        ss = SessionSearch(tmp_path / "sessions.db")
        records = [
            AnalysisRecord(
                timestamp="2024-01-01",
                skill="prevalence",
                args={"icd10_code": "E11"},
                key_results={"total_subjects": 500000},
                figure_paths=[],
            ),
        ]
        ss.index_turn("session1", "What are common diabetes codes?", records)
        results = ss.search("diabetes")
        assert len(results) >= 1
        assert results[0]["session_id"] == "session1"
        ss.close()

    def test_search_empty(self, tmp_path):
        ss = SessionSearch(tmp_path / "sessions.db")
        results = ss.search("nonexistent query xyz")
        assert results == []
        ss.close()

    def test_index_empty_records(self, tmp_path):
        ss = SessionSearch(tmp_path / "sessions.db")
        ss.index_turn("session2", "test query", [])
        # Should not raise
        ss.close()
