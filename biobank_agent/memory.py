"""3-tier memory system for Biobank Agent.

Tier 1 (short-term): Current session messages — managed by agent.py
Tier 2 (mid-term): AnalysisRecord log — managed by state.py
Tier 3 (long-term): Persisted configs, pipelines, field usage — this file
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LongTermMemory:
    """Persistent memory across sessions.

    Stores:
    - Best model configs per disease (hyperparameter memory)
    - Saved analysis pipelines (macro recording)
    - Field usage statistics (frequently queried fields)
    """

    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.memory_dir / "memory.json"
        self._data = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Corrupted memory file, starting fresh")
        return {
            "model_configs": {},
            "pipelines": {},
            "field_usage": {},
            "metadata": {"created": datetime.now().isoformat(), "version": "0.1.0"},
        }

    def _save(self) -> None:
        self._path.write_text(json.dumps(self._data, indent=2, default=str))

    # ── Tier 1: Hyperparameter Memory ─────────────────────

    def remember_model_config(
        self, icd10_code: str, model_type: str, config: dict, auc: float
    ) -> None:
        """Save best model config for a disease if it beats the current best."""
        key = f"{icd10_code}:{model_type}"
        existing = self._data["model_configs"].get(key, {})
        if auc > existing.get("auc", 0):
            self._data["model_configs"][key] = {
                "config": config,
                "auc": auc,
                "date": datetime.now().isoformat(),
            }
            self._save()
            logger.info("Saved best config for %s: AUC=%.4f", key, auc)

    def recall_model_config(self, icd10_code: str, model_type: str) -> Optional[dict]:
        """Retrieve best known config for a disease."""
        key = f"{icd10_code}:{model_type}"
        entry = self._data["model_configs"].get(key)
        return entry["config"] if entry else None

    def best_auc(self, icd10_code: str, model_type: str) -> float:
        key = f"{icd10_code}:{model_type}"
        entry = self._data["model_configs"].get(key)
        return entry.get("auc", 0.0) if entry else 0.0

    # ── Tier 2: Pipeline Macros ───────────────────────────

    def save_pipeline(self, name: str, steps: list[dict]) -> None:
        """Record a named analysis pipeline for replay."""
        self._data["pipelines"][name] = {
            "steps": steps,
            "saved": datetime.now().isoformat(),
        }
        self._save()

    def get_pipeline(self, name: str) -> Optional[list[dict]]:
        entry = self._data["pipelines"].get(name)
        return entry["steps"] if entry else None

    def list_pipelines(self) -> list[str]:
        return list(self._data["pipelines"].keys())

    # ── Tier 3: Field Usage Stats ─────────────────────────

    def record_field_usage(self, field_id: str) -> None:
        """Track which fields are queried most often."""
        counts = self._data["field_usage"]
        counts[field_id] = counts.get(field_id, 0) + 1
        # Save periodically (every 10 increments of any field)
        if sum(counts.values()) % 10 == 0:
            self._save()

    def most_used_fields(self, top_n: int = 20) -> list[tuple[str, int]]:
        counts = self._data["field_usage"]
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # ── Summary for system prompt injection ───────────────

    def summary(self) -> str:
        configs = len(self._data["model_configs"])
        pipelines = len(self._data["pipelines"])
        fields = len(self._data["field_usage"])
        if configs == 0 and pipelines == 0:
            return ""
        parts = []
        if configs > 0:
            parts.append(f"{configs} saved model configs")
        if pipelines > 0:
            parts.append(f"{pipelines} saved pipelines: {', '.join(self.list_pipelines())}")
        if fields > 0:
            top = self.most_used_fields(5)
            parts.append(f"top fields: {', '.join(f[0] for f in top)}")
        return "Long-term memory: " + "; ".join(parts)

    # ── Tier 4: Error Catalog & Suggestions ───────────────
    
    def record_error(
        self, 
        error_type: str, 
        error_message: str, 
        skill_name: str,
        suggested_fix: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> None:
        """Record an error with its context and suggested fix.
        
        Builds error catalog for pattern recognition and auto-suggestion.
        """
        if "errors" not in self._data:
            self._data["errors"] = {}
        
        error_key = f"{error_type}:{skill_name}"
        if error_key not in self._data["errors"]:
            self._data["errors"][error_key] = {
                "count": 0,
                "last_seen": None,
                "messages": [],
                "suggested_fixes": [],
                "contexts": [],
            }
        
        entry = self._data["errors"][error_key]
        entry["count"] += 1
        entry["last_seen"] = datetime.now().isoformat()
        
        # Keep last 5 unique messages
        if error_message not in entry["messages"]:
            entry["messages"].append(error_message)
            entry["messages"] = entry["messages"][-5:]
        
        # Keep last 5 unique suggestions
        if suggested_fix and suggested_fix not in entry["suggested_fixes"]:
            entry["suggested_fixes"].append(suggested_fix)
            entry["suggested_fixes"] = entry["suggested_fixes"][-5:]
        
        # Keep last 5 contexts
        if context:
            entry["contexts"].append(context)
            entry["contexts"] = entry["contexts"][-5:]
        
        # Save periodically
        if entry["count"] % 3 == 0:
            self._save()
    
    def get_error_suggestions(self, error_type: str, skill_name: str) -> list[str]:
        """Get suggested fixes for a known error pattern.
        
        Returns list of previously successful fixes for this error+skill combo.
        """
        if "errors" not in self._data:
            return []
        
        error_key = f"{error_type}:{skill_name}"
        entry = self._data["errors"].get(error_key)
        return entry["suggested_fixes"] if entry else []
    
    def most_common_errors(self, top_n: int = 10) -> list[dict]:
        """Get most frequently occurring errors across all skills.
        
        Returns list of dicts with error info sorted by frequency.
        """
        if "errors" not in self._data:
            return []
        
        errors = []
        for key, entry in self._data["errors"].items():
            error_type, skill_name = key.split(":", 1)
            errors.append({
                "error_type": error_type,
                "skill": skill_name,
                "count": entry["count"],
                "last_seen": entry["last_seen"],
                "suggested_fixes": entry["suggested_fixes"],
            })
        
        return sorted(errors, key=lambda x: x["count"], reverse=True)[:top_n]
