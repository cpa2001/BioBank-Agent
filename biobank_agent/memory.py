"""Multi-tier memory system for Biobank Agent.

Tier 1 (short-term): Current session messages — managed by agent.py
Tier 2 (mid-term): AnalysisRecord log — managed by state.py
Tier 3 (long-term): Persisted configs, pipelines, field usage — this file
Tier 4 (error catalog): Error patterns + suggested fixes — this file
Tier 5 (domain): Accumulated biobank knowledge — domain.md (prose)
Tier 6 (user): Researcher preferences — user.md (prose)
Tier 7 (episodic): Cross-session recall — sessions.db (SQLite FTS5)
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

        # Tier 5-7: Enhanced memory layers
        self.domain = DomainMemory(self.memory_dir / "domain.md")
        self.user = UserMemory(self.memory_dir / "user.md")
        self.sessions = SessionSearch(self.memory_dir / "sessions.db")

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
        """Atomic write: write to .tmp then rename to avoid corruption."""
        tmp_path = self._path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(self._data, indent=2, default=str))
        tmp_path.replace(self._path)

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


# ── Tier 5: Domain Knowledge Memory ──────────────────────────


class DomainMemory:
    """Persistent biobank knowledge — variable relationships, confounders, findings.

    Stored as a Markdown file for human readability and direct injection
    into the LLM system prompt. Follows the Hermes-agent pattern of
    file-backed declarative memory.
    """

    _HEADER = "# Biobank Domain Knowledge\n\n"

    def __init__(self, path: Path) -> None:
        self._path = path
        if not self._path.exists():
            self._path.write_text(self._HEADER, encoding="utf-8")

    def read(self) -> str:
        return self._path.read_text(encoding="utf-8")

    def append_finding(self, heading: str, body: str) -> None:
        """Append a dated entry. Idempotent — skips if body already present."""
        existing = self.read()
        if body.strip() in existing:
            return
        entry = (
            f"\n### {heading} ({datetime.now().strftime('%Y-%m-%d')})\n"
            f"{body.strip()}\n"
        )
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(entry)
        logger.debug("Domain memory: added '%s'", heading)

    def summary(self, max_chars: int = 2000) -> str:
        """Return recent domain knowledge, truncated to max_chars."""
        text = self.read()
        if len(text) <= len(self._HEADER) + 5:
            return ""
        if len(text) <= max_chars:
            return text
        header_end = text.find("\n\n") + 2
        tail = text[-(max_chars - header_end):]
        return text[:header_end] + "...[earlier entries truncated]...\n" + tail


# ── Tier 6: User Profile Memory ──────────────────────────────


class UserMemory:
    """Researcher preferences — inferred from session patterns.

    Stored as Markdown with key-value entries that can be upserted.
    """

    _HEADER = "# Researcher Profile\n\n"

    def __init__(self, path: Path) -> None:
        self._path = path
        if not self._path.exists():
            self._path.write_text(self._HEADER, encoding="utf-8")

    def read(self) -> str:
        return self._path.read_text(encoding="utf-8")

    def upsert_preference(self, key: str, value: str) -> None:
        """Insert or replace a keyed preference line."""
        existing = self.read()
        marker = f"**{key}:**"
        new_line = f"{marker} {value}"
        if marker in existing:
            # Replace the entire line containing the marker
            lines = existing.split("\n")
            for i, line in enumerate(lines):
                if marker in line:
                    lines[i] = new_line
                    break
            self._path.write_text("\n".join(lines), encoding="utf-8")
        else:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(f"\n{new_line}\n")
        logger.debug("User memory: upserted '%s'", key)

    def summary(self) -> str:
        text = self.read()
        return text if len(text) > len(self._HEADER) + 5 else ""


# ── Tier 7: Episodic Session Search ──────────────────────────


class SessionSearch:
    """Cross-session recall via SQLite FTS5.

    Indexes past analysis sessions for BM25-ranked keyword search.
    Enables "what did we find about diabetes last time?" queries.
    """

    def __init__(self, db_path: Path) -> None:
        import sqlite3
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._setup()

    def _setup(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions_meta (
                session_id TEXT PRIMARY KEY,
                timestamp  TEXT NOT NULL,
                n_records  INTEGER DEFAULT 0
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
                session_id UNINDEXED,
                user_query,
                skills_used,
                diseases,
                key_findings,
                tokenize = 'porter ascii'
            );
        """)
        self._conn.commit()

    def index_turn(
        self,
        session_id: str,
        user_query: str,
        records: list,
    ) -> None:
        """Index one agent turn for later recall."""
        if not records:
            return

        skills_used = " ".join(sorted({r.skill for r in records if r.skill != "think"}))

        diseases = " ".join(
            str(v) for r in records
            for k, v in r.args.items()
            if k in ("icd10_code", "icd_code", "icd10", "disease_code", "code")
        )

        key_findings = " ".join(
            f"{k}={v}"
            for r in records
            for k, v in r.key_results.items()
            if not isinstance(v, (dict, list)) and k != "error"
        )

        self._conn.execute(
            "INSERT OR REPLACE INTO sessions_meta VALUES (?, ?, ?)",
            (session_id, datetime.now().isoformat(), len(records)),
        )
        self._conn.execute(
            "DELETE FROM sessions_fts WHERE session_id = ?", (session_id,)
        )
        self._conn.execute(
            "INSERT INTO sessions_fts VALUES (?, ?, ?, ?, ?)",
            (session_id, user_query, skills_used, diseases, key_findings),
        )
        self._conn.commit()

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """BM25-ranked search over past sessions."""
        try:
            rows = self._conn.execute(
                """SELECT s.session_id, m.timestamp, m.n_records,
                          s.user_query, s.diseases, s.key_findings,
                          rank
                   FROM sessions_fts s
                   JOIN sessions_meta m USING (session_id)
                   WHERE sessions_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (query, limit),
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning("Session search failed: %s", e)
            return []

    def close(self) -> None:
        self._conn.close()
