"""Learned difficulty estimation — adaptive routing based on query history.

Based on DAAO (arXiv:2509.11079): Difficulty-Aware Agentic Orchestration.
Uses a lightweight MLP on TF-IDF features trained from (query, strategy, success)
history triples. Falls back to existing heuristic (complexity.py) when
insufficient training data exists (<MIN_HISTORY examples).

Architecture decision (per Codex review): WRAPS classify_complexity, never replaces.
The heuristic is always run as baseline; when enough data is collected, the learned
estimator blends with the heuristic for better routing decisions.

Usage
-----
    from biobank_agent.difficulty import DifficultyEstimator

    estimator = DifficultyEstimator(history_path=Path("./memory/difficulty"))
    strategy, confidence = estimator.estimate(query, records)
    # After execution:
    estimator.record_outcome(query, strategy_used, succeeded=True)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .state import AnalysisRecord

from .complexity import Strategy, ComplexityScore, classify_complexity

logger = logging.getLogger(__name__)


@dataclass
class DifficultyEstimate:
    """Result of learned difficulty estimation."""
    strategy: Strategy
    confidence: float          # 0.0-1.0 confidence in this routing
    source: str                # "heuristic" | "learned" | "blended"
    heuristic_score: float     # Raw score from classify_complexity
    learned_score: Optional[float] = None  # Score from MLP (if trained)
    factors: dict = field(default_factory=dict)

    @classmethod
    def from_complexity(cls, cs: ComplexityScore) -> DifficultyEstimate:
        """Wrap a raw ComplexityScore as a DifficultyEstimate."""
        return cls(
            strategy=cs.strategy,
            confidence=abs(cs.score - 0.5) * 2,  # high confidence near extremes (0 or 1)
            source="heuristic",
            heuristic_score=cs.score,
            factors=cs.factors,
        )


@dataclass
class OutcomeRecord:
    """Record of a query routing outcome for training."""
    query: str
    strategy_used: str
    succeeded: bool
    timestamp: float = field(default_factory=time.time)
    score: float = 0.0         # heuristic score at time of routing
    execution_time_ms: float = 0.0


class DifficultyEstimator:
    """Learned difficulty-aware routing with graceful heuristic fallback.

    Training pipeline:
      1. Collect (query, strategy, success) triples via record_outcome()
      2. When history >= MIN_HISTORY, train MLP on TF-IDF features
      3. Blend learned predictions with heuristic scores

    The estimator self-improves as more queries are processed.
    """

    MIN_HISTORY = 50     # Minimum examples before trusting MLP
    RETRAIN_INTERVAL = 25  # Retrain every N new examples
    BLEND_WEIGHT = 0.6     # Weight of learned model vs heuristic (0=all heuristic, 1=all learned)

    STRATEGY_LABELS = ["single", "ensemble", "debate", "supervisor"]

    def __init__(
        self,
        history_path: Optional[Path] = None,
        min_history: int = 50,
        blend_weight: float = 0.6,
    ) -> None:
        self.history_path = history_path or Path("./memory/difficulty")
        self.history_path.mkdir(parents=True, exist_ok=True)
        self.min_history = min_history
        self.blend_weight = blend_weight

        self._history: list[OutcomeRecord] = []
        self._model = None
        self._vectorizer = None
        self._is_trained = False
        self._since_last_train = 0

        # Load existing history
        self._load_history()

    def _load_history(self) -> None:
        """Load training history from disk."""
        history_file = self.history_path / "outcomes.jsonl"
        if history_file.exists():
            try:
                with open(history_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            self._history.append(OutcomeRecord(**data))
                logger.debug("Loaded %d difficulty history records", len(self._history))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load difficulty history: %s", e)

        # Try to train if we have enough data
        if len(self._history) >= self.min_history:
            self._train()

    def _save_record(self, record: OutcomeRecord) -> None:
        """Append a single outcome record to disk."""
        history_file = self.history_path / "outcomes.jsonl"
        try:
            with open(history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "query": record.query,
                    "strategy_used": record.strategy_used,
                    "succeeded": record.succeeded,
                    "timestamp": record.timestamp,
                    "score": record.score,
                    "execution_time_ms": record.execution_time_ms,
                }) + "\n")
        except OSError as e:
            logger.warning("Failed to save difficulty record: %s", e)

    def estimate(
        self,
        query: str,
        records: Optional[list[AnalysisRecord]] = None,
    ) -> DifficultyEstimate:
        """Estimate query difficulty and recommend routing strategy.

        Always runs the heuristic. If trained, blends with learned model.
        """
        # Always run heuristic (zero-cost baseline)
        heuristic = classify_complexity(query, records)

        if not self._is_trained or len(self._history) < self.min_history:
            return DifficultyEstimate.from_complexity(heuristic)

        # Learned prediction
        try:
            features = self._vectorizer.transform([query])
            probs = self._model.predict_proba(features)[0]
            learned_idx = int(np.argmax(probs))
            learned_confidence = float(probs[learned_idx])
            learned_strategy = Strategy(self.STRATEGY_LABELS[learned_idx])

            # Blend: if learned and heuristic agree, high confidence
            if learned_strategy == heuristic.strategy:
                final_strategy = learned_strategy
                final_confidence = max(learned_confidence, 1.0 - abs(heuristic.score - 0.5))
                source = "blended_agree"
            else:
                # Disagreement: weight by blend factor
                if learned_confidence * self.blend_weight > (1 - self.blend_weight):
                    final_strategy = learned_strategy
                    final_confidence = learned_confidence * self.blend_weight
                    source = "learned_override"
                else:
                    final_strategy = heuristic.strategy
                    final_confidence = 1.0 - abs(heuristic.score - 0.5)
                    source = "heuristic_override"

            return DifficultyEstimate(
                strategy=final_strategy,
                confidence=final_confidence,
                source=source,
                heuristic_score=heuristic.score,
                learned_score=learned_confidence,
                factors={**heuristic.factors, "learned_probs": probs.tolist()},
            )

        except Exception as e:
            logger.warning("Learned estimation failed, falling back to heuristic: %s", e)
            return DifficultyEstimate.from_complexity(heuristic)

    def record_outcome(
        self,
        query: str,
        strategy_used: str,
        succeeded: bool,
        execution_time_ms: float = 0.0,
        heuristic_score: float = 0.0,
    ) -> None:
        """Record a routing outcome for future training.

        Call this after each agent execution to build training data.
        """
        record = OutcomeRecord(
            query=query,
            strategy_used=strategy_used,
            succeeded=succeeded,
            score=heuristic_score,
            execution_time_ms=execution_time_ms,
        )
        self._history.append(record)
        self._save_record(record)
        self._since_last_train += 1

        # Auto-retrain periodically
        if (
            len(self._history) >= self.min_history
            and self._since_last_train >= self.RETRAIN_INTERVAL
        ):
            self._train()

    def _train(self) -> dict[str, float]:
        """Train the MLP on collected history.

        Returns training metrics (accuracy on training set).
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.neural_network import MLPClassifier
        except ImportError:
            logger.warning("scikit-learn not available for difficulty training")
            return {"status": "sklearn_unavailable"}

        # Prepare training data
        # Only use successful outcomes to learn correct routing
        queries = [r.query for r in self._history]
        # Label: optimal strategy = strategy that succeeded
        labels = []
        for r in self._history:
            if r.succeeded:
                labels.append(r.strategy_used)
            else:
                # Failed strategy → label as needing upgrade
                idx = self.STRATEGY_LABELS.index(r.strategy_used) if r.strategy_used in self.STRATEGY_LABELS else 0
                next_idx = min(idx + 1, len(self.STRATEGY_LABELS) - 1)
                labels.append(self.STRATEGY_LABELS[next_idx])

        # TF-IDF vectorization
        self._vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words="english",
        )
        X = self._vectorizer.fit_transform(queries)

        # Encode labels
        label_to_idx = {l: i for i, l in enumerate(self.STRATEGY_LABELS)}
        y = np.array([label_to_idx.get(l, 0) for l in labels])

        # Train MLP
        self._model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        )
        self._model.fit(X, y)
        self._is_trained = True
        self._since_last_train = 0

        # Compute training accuracy
        train_acc = float(self._model.score(X, y))
        logger.info(
            "Difficulty estimator trained on %d examples (train_acc=%.3f)",
            len(queries), train_acc,
        )

        return {"train_accuracy": train_acc, "n_examples": len(queries)}

    def retrain(self) -> dict[str, float]:
        """Manually trigger retraining."""
        if len(self._history) < self.min_history:
            return {"status": "insufficient_data", "n_examples": len(self._history), "min_required": self.min_history}
        return self._train()

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def n_history(self) -> int:
        return len(self._history)

    def stats(self) -> dict[str, Any]:
        """Return estimator statistics."""
        from collections import Counter
        strategy_counts = Counter(r.strategy_used for r in self._history)
        success_rate = (
            sum(1 for r in self._history if r.succeeded) / len(self._history)
            if self._history else 0.0
        )
        return {
            "n_history": len(self._history),
            "is_trained": self._is_trained,
            "strategy_distribution": dict(strategy_counts),
            "overall_success_rate": round(success_rate, 3),
            "min_history_threshold": self.min_history,
        }
