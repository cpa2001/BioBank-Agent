"""Tests for learned difficulty estimator (Stream D).

Tests fallback behavior, training, and outcome recording.
"""

import json

import pytest
import numpy as np
from biobank_agent.difficulty import DifficultyEstimator, OutcomeRecord
from biobank_agent.complexity import Strategy


@pytest.fixture
def estimator(tmp_path):
    return DifficultyEstimator(history_path=tmp_path, min_history=5)


class TestDifficultyFallback:
    """Test heuristic fallback when untrained."""

    def test_untrained_uses_heuristic(self, estimator):
        """Untrained estimator should delegate to classify_complexity."""
        result = estimator.estimate("What is the prevalence of E11?")
        assert result.source == "heuristic"
        assert result.strategy == Strategy.SINGLE
        assert not estimator.is_trained

    def test_complex_query_routes_higher(self, estimator):
        """Complex query should score higher even in heuristic mode."""
        result = estimator.estimate(
            "Compare multiple models debate end-to-end pipeline workflow systematic comprehensive"
        )
        assert result.heuristic_score > 0.3

    def test_confidence_high_for_simple(self, estimator):
        """Simple queries should have high confidence."""
        result = estimator.estimate("hello")
        assert result.confidence >= 0.7


class TestOutcomeRecording:
    """Test that outcomes are properly recorded and persisted."""

    def test_record_outcome_increments_history(self, estimator):
        """Recording an outcome should grow the history."""
        assert estimator.n_history == 0
        estimator.record_outcome("test query", "single", True)
        assert estimator.n_history == 1

    def test_record_outcome_persists_to_disk(self, estimator):
        """Outcomes should be written to JSONL file."""
        estimator.record_outcome("test query", "single", True)
        history_file = estimator.history_path / "outcomes.jsonl"
        assert history_file.exists()
        lines = history_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["query"] == "test query"
        assert data["strategy_used"] == "single"
        assert data["succeeded"] is True

    def test_reload_history_on_init(self, tmp_path):
        """New estimator should load existing history from disk."""
        # Write some history
        est1 = DifficultyEstimator(history_path=tmp_path, min_history=100)
        est1.record_outcome("q1", "single", True)
        est1.record_outcome("q2", "debate", False)

        # Create new instance — should load history
        est2 = DifficultyEstimator(history_path=tmp_path, min_history=100)
        assert est2.n_history == 2

    def test_load_history_skips_blank_lines_and_trains_when_ready(self, tmp_path, monkeypatch):
        history_file = tmp_path / "outcomes.jsonl"
        history_file.write_text(
            "\n"
            + json.dumps({"query": "q1", "strategy_used": "single", "succeeded": True})
            + "\n",
            encoding="utf-8",
        )
        calls = []

        monkeypatch.setattr(DifficultyEstimator, "_train", lambda self: calls.append(self.n_history) or {"ok": 1})

        est = DifficultyEstimator(history_path=tmp_path, min_history=1)

        assert est.n_history == 1
        assert calls == [1]


class TestTraining:
    """Test MLP training when enough data is collected."""

    def test_trains_after_min_history(self, tmp_path):
        """Should auto-train after min_history examples."""
        est = DifficultyEstimator(history_path=tmp_path, min_history=5)

        # Record enough diverse outcomes (need variety for MLP to train)
        queries = [
            "What is the prevalence of E11?",
            "Compare XGBoost vs LightGBM for diabetes prediction end-to-end pipeline",
            "Show me top 10 biomarkers",
            "Run comprehensive systematic GWAS analysis with debate review",
            "List available fields",
            "Train model for I21 with blood biochemistry multiple steps workflow",
            "Hello",
        ]
        strategies = ["single", "debate", "single", "supervisor", "single", "ensemble", "single"]
        for q, s in zip(queries, strategies):
            est.record_outcome(q, s, True)

        # Should have triggered training (RETRAIN_INTERVAL=25, but min_history=5 reached)
        # Force retrain since auto-train triggers on RETRAIN_INTERVAL
        metrics = est.retrain()
        assert est.is_trained
        assert metrics.get("train_accuracy", 0) > 0

    def test_manual_retrain(self, tmp_path):
        """Manual retrain should work after enough data."""
        est = DifficultyEstimator(history_path=tmp_path, min_history=5)
        for i in range(10):
            strategy = "debate" if i % 3 == 0 else "single"
            est.record_outcome(f"complex analysis query {i} pipeline workflow", strategy, i % 2 == 0)

        metrics = est.retrain()
        assert "train_accuracy" in metrics
        assert metrics["train_accuracy"] > 0.0

    def test_insufficient_data_returns_status(self, estimator):
        """Retrain with too few examples should report insufficient data."""
        estimator.record_outcome("one query", "single", True)
        result = estimator.retrain()
        assert result["status"] == "insufficient_data"

    def test_auto_retrain_and_sklearn_unavailable_paths(self, tmp_path, monkeypatch):
        """Auto-retrain should trigger on interval; missing sklearn is reported cleanly."""
        est = DifficultyEstimator(history_path=tmp_path / "auto", min_history=1)
        calls = []

        monkeypatch.setattr(DifficultyEstimator, "RETRAIN_INTERVAL", 1)
        monkeypatch.setattr(est, "_train", lambda: calls.append("trained") or {"ok": 1})

        est.record_outcome("pipeline workflow", "supervisor", True)

        assert calls == ["trained"]

        blocked = DifficultyEstimator(history_path=tmp_path / "blocked", min_history=1)
        original_import = __import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("sklearn"):
                raise ImportError("sklearn missing")
            return original_import(name, *args, **kwargs)

        blocked._history.append(OutcomeRecord("q", "single", True))
        monkeypatch.setattr("builtins.__import__", fake_import)
        assert blocked._train() == {"status": "sklearn_unavailable"}


class FakeVectorizer:
    def __init__(self, error=None):
        self.error = error

    def transform(self, queries):
        if self.error:
            raise self.error
        return queries


class FakeModel:
    def __init__(self, probs):
        self.probs = np.array([probs])

    def predict_proba(self, features):
        return self.probs


def trained_estimator(tmp_path, probs, blend_weight=0.6, vectorizer=None):
    est = DifficultyEstimator(history_path=tmp_path, min_history=1, blend_weight=blend_weight)
    est._history = [OutcomeRecord("seed", "single", True)]
    est._is_trained = True
    est._vectorizer = vectorizer or FakeVectorizer()
    est._model = FakeModel(probs)
    return est


class TestLearnedEstimateBranches:
    def test_learned_agreement_uses_blended_source(self, tmp_path):
        est = trained_estimator(tmp_path, [0.9, 0.05, 0.03, 0.02])

        result = est.estimate("hello")

        assert result.strategy is Strategy.SINGLE
        assert result.source == "blended_agree"
        assert result.learned_score == pytest.approx(0.9)
        assert len(result.factors["learned_probs"]) == 4

    def test_learned_override_and_heuristic_override_disagreement(self, tmp_path):
        learned = trained_estimator(tmp_path / "learned", [0.05, 0.05, 0.9, 0.0], blend_weight=0.9)
        heuristic = trained_estimator(tmp_path / "heuristic", [0.05, 0.05, 0.6, 0.3], blend_weight=0.2)

        learned_result = learned.estimate("hello")
        heuristic_result = heuristic.estimate("hello")

        assert learned_result.strategy is Strategy.DEBATE
        assert learned_result.source == "learned_override"
        assert heuristic_result.strategy is Strategy.SINGLE
        assert heuristic_result.source == "heuristic_override"

    def test_learned_failure_falls_back_to_heuristic(self, tmp_path):
        est = trained_estimator(
            tmp_path,
            [0.0, 0.0, 1.0, 0.0],
            vectorizer=FakeVectorizer(error=RuntimeError("vectorizer failed")),
        )

        result = est.estimate("hello")

        assert result.source == "heuristic"
        assert result.strategy is Strategy.SINGLE


class TestStats:
    """Test statistics reporting."""

    def test_stats_empty(self, estimator):
        """Stats on empty estimator."""
        stats = estimator.stats()
        assert stats["n_history"] == 0
        assert stats["is_trained"] is False
        assert stats["overall_success_rate"] == 0.0

    def test_stats_with_data(self, estimator):
        """Stats after recording some outcomes."""
        estimator.record_outcome("q1", "single", True)
        estimator.record_outcome("q2", "debate", False)
        stats = estimator.stats()
        assert stats["n_history"] == 2
        assert stats["overall_success_rate"] == 0.5
        assert stats["strategy_distribution"] == {"single": 1, "debate": 1}


def test_load_and_save_history_error_paths(tmp_path, monkeypatch):
    bad_history = tmp_path / "bad"
    bad_history.mkdir()
    (bad_history / "outcomes.jsonl").write_text("{bad json\n", encoding="utf-8")

    est = DifficultyEstimator(history_path=bad_history, min_history=10)
    assert est.n_history == 0

    def raising_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", raising_open)
    est._save_record(OutcomeRecord("query", "single", True))
