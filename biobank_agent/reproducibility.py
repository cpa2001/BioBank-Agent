"""Reproducibility harness — SHA-256 checkpoints, audit logs, provenance records.

Pattern adopted from NeuroClaw (arXiv:2604.24696, CUHK AIM Group):
  - SHA-256 hash of execution context (inputs + code version + environment)
  - Structured checkpoint at each DAG node completion
  - Audit log with full provenance chain for scientific reproducibility

Every skill execution gets an immutable record that allows any researcher
to verify: "Given these inputs and this code version, I should get these outputs."

Usage
-----
    from biobank_agent.reproducibility import ReproducibilityHarness

    harness = ReproducibilityHarness(checkpoint_dir=Path("./checkpoints"))
    ctx_hash = harness.checkpoint(skill_name="train_model", inputs={...}, outputs={...})
    assert harness.verify_checkpoint(ctx_hash)
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExecutionContext:
    """Complete execution context for reproducibility."""
    skill_name: str
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    python_version: str = field(default_factory=lambda: sys.version)
    platform_info: str = field(default_factory=lambda: platform.platform())
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_ms: float = 0.0
    code_version: str = ""  # git commit hash if available
    dependencies: dict[str, str] = field(default_factory=dict)  # package → version


@dataclass
class AuditEntry:
    """A single entry in the provenance audit log."""
    entry_id: str          # SHA-256 of the execution context
    skill_name: str
    timestamp_utc: str
    inputs_hash: str       # SHA-256 of inputs only
    outputs_hash: str      # SHA-256 of outputs only
    context_hash: str      # SHA-256 of full context (inputs + env + code)
    duration_ms: float
    status: str = "success"  # "success" | "failed" | "partial"
    error: str = ""
    parent_entry: str = ""  # links to previous step in pipeline


@dataclass
class ProvenanceRecord:
    """Full provenance chain for a session/pipeline."""
    session_id: str
    created_utc: str
    entries: list[AuditEntry] = field(default_factory=list)
    goal: str = ""
    final_status: str = "in_progress"

    @property
    def n_steps(self) -> int:
        return len(self.entries)

    @property
    def total_duration_ms(self) -> float:
        return sum(e.duration_ms for e in self.entries)

    def chain_hash(self) -> str:
        """Compute SHA-256 hash of the entire provenance chain."""
        content = json.dumps(
            [asdict(e) for e in self.entries],
            sort_keys=True, default=str,
        )
        return hashlib.sha256(content.encode()).hexdigest()


class ReproducibilityHarness:
    """Checkpoint and audit system for reproducible scientific computation.

    Provides:
      1. Deterministic hashing of execution contexts
      2. Persistent checkpoint storage (JSON)
      3. Verification that a checkpoint matches stored hash
      4. Audit log with parent-child chain linking
      5. Export to structured provenance records
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Path] = None,
        code_version: str = "",
    ) -> None:
        self.checkpoint_dir = checkpoint_dir or Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.code_version = code_version or self._detect_git_version()
        self._audit_log: list[AuditEntry] = []
        self._last_entry_id: str = ""

    @staticmethod
    def _detect_git_version() -> str:
        """Try to get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        return "unknown"

    @staticmethod
    def _compute_hash(data: Any) -> str:
        """Compute SHA-256 hash of arbitrary data."""
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_key_dependencies(self) -> dict[str, str]:
        """Get versions of key scientific packages."""
        deps = {}
        for pkg in ("pandas", "numpy", "scikit-learn", "duckdb", "scipy", "xgboost", "lightgbm"):
            try:
                mod = __import__(pkg.replace("-", "_"))
                deps[pkg] = getattr(mod, "__version__", "unknown")
            except ImportError:
                pass
        return deps

    def checkpoint(
        self,
        skill_name: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        duration_ms: float = 0.0,
        status: str = "success",
        error: str = "",
    ) -> str:
        """Create an immutable checkpoint of a skill execution.

        Returns the SHA-256 hash of the execution context.
        """
        context = ExecutionContext(
            skill_name=skill_name,
            inputs=inputs,
            outputs=outputs,
            duration_ms=duration_ms,
            code_version=self.code_version,
            dependencies=self._get_key_dependencies(),
        )

        # Compute hashes
        inputs_hash = self._compute_hash(inputs)
        outputs_hash = self._compute_hash(outputs)
        context_hash = self._compute_hash(asdict(context))

        # Create audit entry
        entry = AuditEntry(
            entry_id=context_hash,
            skill_name=skill_name,
            timestamp_utc=context.timestamp_utc,
            inputs_hash=inputs_hash,
            outputs_hash=outputs_hash,
            context_hash=context_hash,
            duration_ms=duration_ms,
            status=status,
            error=error,
            parent_entry=self._last_entry_id,
        )
        self._audit_log.append(entry)
        self._last_entry_id = context_hash

        # Persist checkpoint
        checkpoint_file = self.checkpoint_dir / f"{context_hash}.json"
        try:
            checkpoint_file.write_text(
                json.dumps(asdict(context), indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("Failed to write checkpoint %s: %s", context_hash[:8], e)

        logger.debug(
            "Checkpoint %s for skill '%s' (inputs=%s, outputs=%s)",
            context_hash[:8], skill_name, inputs_hash[:8], outputs_hash[:8],
        )
        return context_hash

    def verify_checkpoint(self, context_hash: str) -> bool:
        """Verify that a stored checkpoint matches its hash.

        Returns True if the checkpoint file exists and its content
        hashes to the expected value.
        """
        checkpoint_file = self.checkpoint_dir / f"{context_hash}.json"
        if not checkpoint_file.exists():
            logger.warning("Checkpoint %s not found", context_hash[:8])
            return False

        try:
            content = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            computed = self._compute_hash(content)
            if computed == context_hash:
                return True
            else:
                logger.warning(
                    "Checkpoint %s integrity failure: stored hash ≠ computed hash (%s)",
                    context_hash[:8], computed[:8],
                )
                return False
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Checkpoint %s verification error: %s", context_hash[:8], e)
            return False

    def create_audit_log(
        self,
        skill_name: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        duration_ms: float = 0.0,
        status: str = "success",
        error: str = "",
    ) -> AuditEntry:
        """Create an audit entry without persisting a full checkpoint.

        Lighter-weight than checkpoint() — for quick operations.
        """
        inputs_hash = self._compute_hash(inputs)
        outputs_hash = self._compute_hash(outputs)
        context_hash = self._compute_hash({
            "skill": skill_name, "inputs": inputs,
            "outputs": outputs, "code": self.code_version,
        })

        entry = AuditEntry(
            entry_id=context_hash,
            skill_name=skill_name,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            inputs_hash=inputs_hash,
            outputs_hash=outputs_hash,
            context_hash=context_hash,
            duration_ms=duration_ms,
            status=status,
            error=error,
            parent_entry=self._last_entry_id,
        )
        self._audit_log.append(entry)
        self._last_entry_id = context_hash
        return entry

    def export_provenance(self, session_id: str, goal: str = "") -> ProvenanceRecord:
        """Export the full provenance record for the current session."""
        record = ProvenanceRecord(
            session_id=session_id,
            created_utc=datetime.now(timezone.utc).isoformat(),
            entries=list(self._audit_log),
            goal=goal,
            final_status="completed" if self._audit_log else "empty",
        )

        # Persist the provenance record
        provenance_file = self.checkpoint_dir / f"provenance_{session_id}.json"
        try:
            provenance_file.write_text(
                json.dumps(asdict(record), indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("Failed to write provenance record: %s", e)

        return record

    def get_audit_log(self) -> list[AuditEntry]:
        """Return the current audit log."""
        return list(self._audit_log)

    def reset(self) -> None:
        """Clear the in-memory audit log (checkpoints on disk are preserved)."""
        self._audit_log.clear()
        self._last_entry_id = ""
