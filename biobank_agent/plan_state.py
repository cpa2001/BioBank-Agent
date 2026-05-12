"""Plan state persistence for crash recovery.

Serializes plan execution state to JSON so that interrupted plans
can be resumed after a CLI restart.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .planner import LongHorizonPlan, PlanState

logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_FILE = ".plan_checkpoint.json"


def _json_safe(value: Any, *, _depth: int = 0) -> Any:
    """Convert execution state to plain JSON without runtime objects."""
    if _depth > 8:
        return "<truncated>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 2000 else value[:2000] + "...<truncated>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if key_str == "ctx":
                continue
            safe[key_str] = _json_safe(item, _depth=_depth + 1)
        return safe
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, _depth=_depth + 1) for item in list(value)[:200]]
    return f"<{type(value).__name__}>"


@dataclass
class PlanCheckpoint:
    """Serializable snapshot of plan execution state.

    Saved to disk during execution so that interrupted plans
    can offer a resume option on next startup.
    """

    goal: str
    plan_data: dict = field(default_factory=dict)
    state: str = "INACTIVE"
    revision: int = 1
    execution_log: list[dict] = field(default_factory=list)
    repair_log: list[dict] = field(default_factory=list)
    report_dir: str = ""
    created_at: str = ""
    updated_at: str = ""

    def save(self, path: Path) -> None:
        """Atomically save checkpoint to disk."""
        self.updated_at = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = self.updated_at

        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_json_safe(asdict(self)), indent=2))
        tmp.rename(path)
        logger.debug("Plan checkpoint saved: %s", path)

    @classmethod
    def load(cls, path: Path) -> "PlanCheckpoint | None":
        """Load checkpoint from disk. Returns None if not found or invalid."""
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except Exception as e:
            logger.warning("Failed to load plan checkpoint: %s", e)
            return None

    @classmethod
    def from_plan_mode(cls, plan_mode: Any) -> "PlanCheckpoint":
        """Create checkpoint from current PlanMode state."""
        from .progress import StepResult

        plan_data = _json_safe(plan_mode.plan.to_dict() if plan_mode.plan else {})
        execution_log = [
            {
                "step_id": r.step_id,
                "step_description": r.step_description,
                "skill": r.skill,
                "success": r.success,
                "result": _json_safe(r.result),
                "error": r.error,
                "duration_s": r.duration_s,
            }
            for r in (plan_mode.execution_log or [])
        ]

        return cls(
            goal=plan_mode.goal,
            plan_data=plan_data,
            state=plan_mode.state.value if isinstance(plan_mode.state, PlanState) else str(plan_mode.state),
            revision=plan_mode.revision,
            execution_log=execution_log,
            repair_log=_json_safe(getattr(plan_mode, "repair_log", []) or []),
            report_dir=str(getattr(plan_mode, "report_dir", "") or ""),
        )

    def restore_to_plan_mode(self, plan_mode: Any) -> bool:
        """Restore checkpoint state into a PlanMode instance.

        Returns True if restoration was successful.

        Note: EXECUTING and APPROVED states are mapped to PAUSED on restore,
        since the executor is not running after a crash/restart. The user
        can then /plan-resume to continue execution.
        """
        try:
            from .progress import StepResult

            plan_mode.goal = self.goal
            plan_mode.revision = self.revision
            plan_mode.plan = LongHorizonPlan.from_dict(self.plan_data) if self.plan_data else None

            # Restore state — map non-resumable states to PAUSED
            try:
                restored_state = PlanState(self.state)
            except ValueError:
                restored_state = PlanState.REVIEW

            # EXECUTING/APPROVED after a crash means execution was interrupted
            # → restore as PAUSED so user can /plan-resume
            if restored_state in (PlanState.EXECUTING, PlanState.APPROVED):
                plan_mode.state = PlanState.PAUSED
                plan_mode.pause_reason = "Restored from interrupted session"
            else:
                plan_mode.state = restored_state

            # Restore execution log
            plan_mode.execution_log = [
                StepResult(
                    step_id=r.get("step_id", ""),
                    step_description=r.get("step_description", ""),
                    skill=r.get("skill", ""),
                    success=r.get("success", False),
                    result=r.get("result", {}) or {},
                    error=r.get("error", ""),
                    duration_s=r.get("duration_s", 0.0),
                )
                for r in self.execution_log
            ]
            plan_mode.repair_log = list(self.repair_log or [])
            plan_mode.report_dir = self.report_dir or ""

            logger.info("Plan checkpoint restored: %s (state=%s→%s)",
                        self.goal[:40], self.state, plan_mode.state.value)
            return True

        except Exception as e:
            logger.warning("Failed to restore plan checkpoint: %s", e)
            return False

    @staticmethod
    def clear(path: Path) -> None:
        """Remove checkpoint file."""
        if path.exists():
            path.unlink()
            logger.debug("Plan checkpoint cleared: %s", path)
