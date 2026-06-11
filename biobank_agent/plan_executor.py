"""Plan execution engine — step-by-step plan execution with progress tracking.

Connects the LongHorizonPlan DAG to the skill registry, executing steps
in dependency order with live Rich progress display.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from rich.console import Console

from .plan_state import PlanCheckpoint
from .planner import LongHorizonPlan, PlanMode, PlanState, PlanStep
from .progress import PlanProgressDisplay, StepResult

if TYPE_CHECKING:
    from .registry import SkillRegistry

logger = logging.getLogger(__name__)


def _result_error_message(value: Any, *, _depth: int = 0) -> str:
    """Return the first non-empty error message found in a skill result."""
    if _depth > 6:
        return ""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "error" and item:
                return str(item)
            nested = _result_error_message(item, _depth=_depth + 1)
            if nested:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _result_error_message(item, _depth=_depth + 1)
            if nested:
                return nested
    return ""


def _path_exists_nonempty(path_value: Any) -> bool:
    if not path_value:
        return False
    try:
        path = Path(str(path_value)).expanduser()
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _report_artifact_error(step: PlanStep, result: dict) -> str:
    """Require report steps to land the files they claim to produce."""
    if step.skill != "generate_report":
        return ""

    result_format = str(result.get("format") or step.args.get("format") or "report").strip().lower()
    dual_requested = result_format in {"dual", "both", "paired"}
    paired = result.get("paired_outputs") if isinstance(result.get("paired_outputs"), dict) else {}

    required = {
        "markdown": result.get("markdown"),
        "html": result.get("html"),
    }
    if dual_requested:
        required.update({
            "markdown_with_css": result.get("markdown_with_css"),
            "paired_outputs.technical_markdown": paired.get("technical_markdown"),
            "paired_outputs.nature_markdown": paired.get("nature_markdown"),
            "paired_outputs.nature_markdown_with_css": paired.get("nature_markdown_with_css"),
            "paired_outputs.nature_html": paired.get("nature_html"),
        })

    missing = [label for label, path in required.items() if not _path_exists_nonempty(path)]
    if missing:
        return f"generate_report did not produce required artifact(s): {', '.join(missing)}"
    broken_links = result.get("broken_figure_links") or []
    if broken_links:
        return "generate_report produced broken linked artifact(s): " + ", ".join(str(x) for x in broken_links[:8])
    if result.get("polished") is not True:
        return "generate_report did not pass through academic_report_polisher"
    polish_payloads: list[dict] = []
    for key in ("polisher", "polisher_css"):
        value = result.get(key)
        if isinstance(value, dict):
            if value.get("quality_checks"):
                polish_payloads.append(value)
            else:
                polish_payloads.extend(v for v in value.values() if isinstance(v, dict) and v.get("quality_checks"))
    if not polish_payloads:
        return "generate_report did not return report polishing quality metadata"
    for payload in polish_payloads:
        checks = payload.get("quality_checks") or {}
        if checks.get("figure_links_resolvable") is False:
            missing = checks.get("missing_figure_links") or []
            return "academic_report_polisher found broken figure link(s): " + ", ".join(str(x) for x in missing[:8])
        if checks.get("no_forbidden_main_body_patterns") is False:
            patterns = checks.get("forbidden_main_body_patterns") or []
            return "academic_report_polisher found machine-output pattern(s): " + ", ".join(str(x) for x in patterns[:8])
    return ""


def _report_artifact_metadata(plan_mode: PlanMode) -> dict[str, Any]:
    """Return a compact report-artifact summary for UI/event consumers."""
    report_dir_value = getattr(plan_mode, "report_dir", "") or ""
    if not report_dir_value:
        return {}
    try:
        report_dir = Path(str(report_dir_value)).expanduser()
    except TypeError:
        return {}
    metadata: dict[str, Any] = {"report_dir": str(report_dir)}
    if not report_dir.exists() or not report_dir.is_dir():
        metadata["report_artifacts"] = []
        return metadata

    priority = {
        "report.md",
        "report_technical.md",
        "report_nature.md",
        "_report_with_css.md",
        "_report_nature_with_css.md",
        "report.html",
        "report_nature.html",
    }
    files = [
        path for path in sorted(report_dir.iterdir())
        if path.is_file() and (path.name in priority or path.suffix.lower() in {".svg", ".png", ".pdf"})
    ]
    metadata["report_artifacts"] = [str(path) for path in files[:24]]
    return metadata


def _goal_explicitly_declines_report(goal: str) -> bool:
    lower = (goal or "").lower()
    return any(
        phrase in lower
        for phrase in (
            "no final report required",
            "no final report",
            "no report required",
            "do not generate report",
            "don't generate report",
            "without a report",
            "without final report",
            "不追求最终报告",
            "不需要最终报告",
            "不要生成报告",
        )
    )


def _goal_requests_report(goal: str) -> bool:
    if _goal_explicitly_declines_report(goal):
        return False
    lower = (goal or "").lower()
    return any(token in lower for token in ("report", "paper", "manuscript", "报告", "论文"))


def _quality_gate_error(step: PlanStep, result: dict) -> str:
    """Return an execution-blocking quality issue for a nominally successful skill."""
    if result.get("requires_repair"):
        warnings = result.get("warnings") or []
        detail = "; ".join(str(w) for w in warnings[:2]) if isinstance(warnings, list) else str(warnings)
        return detail or f"{step.skill} requested plan repair before continuing."
    if step.skill == "field_search" and result.get("total") == 0:
        return "field_search returned zero fields; broaden or repair the catalogue query before downstream analysis."
    return ""


class PlanExecutionError(Exception):
    """Raised when a plan step fails during execution."""

    def __init__(self, step: PlanStep, error: Exception) -> None:
        self.step = step
        self.original_error = error
        super().__init__(f"Step '{step.description}' failed: {error}")


@dataclass
class PlanRepairAction:
    """A bounded self-repair action proposed during plan execution."""

    action: str
    reason: str = ""
    args: dict = field(default_factory=dict)
    steps: list[Any] = field(default_factory=list)
    skill: dict = field(default_factory=dict)
    replacement_step: dict = field(default_factory=dict)


class PlanExecutor:
    """Execute an approved plan step-by-step with progress tracking.

    Connects LongHorizonPlan → skill registry with:
    - Live progress display (Rich panels)
    - Failure handling (pause on error)
    - Pause/resume support
    - Execution logging

    Usage::

        executor = PlanExecutor(plan_mode, registry, ctx_builder)
        executor.execute()  # Blocks until done, paused, or failed
    """

    def __init__(
        self,
        plan_mode: PlanMode,
        registry: "SkillRegistry",
        ctx_builder: Callable[[], Any],
        skill_executor: Callable[[str, dict], dict] | None = None,
        console: Console | None = None,
        checkpoint_dir: Path | None = None,
        on_log_update: Callable[[list[StepResult]], None] | None = None,
        repair_strategy: Callable[[PlanStep, StepResult, LongHorizonPlan, dict], dict | PlanRepairAction | None] | None = None,
        goal_acceptance_checker: Callable[[LongHorizonPlan, list[StepResult]], dict | None] | None = None,
        refresh_tools: Callable[[], None] | None = None,
        repair_budget_per_step: int = 3,
        repair_budget_total: int = 8,
        goal_acceptance_enabled: bool = True,
        event_sink: Callable[[str, str, str, str, dict | None], None] | None = None,
    ) -> None:
        self.plan_mode = plan_mode
        self.registry = registry
        self.ctx_builder = ctx_builder
        self.skill_executor = skill_executor
        self.console = console or Console()
        self.progress: PlanProgressDisplay | None = None
        self._checkpoint_path = (checkpoint_dir / ".plan_checkpoint.json") if checkpoint_dir else None
        self.on_log_update = on_log_update
        self.repair_strategy = repair_strategy
        self.goal_acceptance_checker = goal_acceptance_checker
        self.refresh_tools = refresh_tools
        self.repair_budget_per_step = max(0, int(repair_budget_per_step))
        self.repair_budget_total = max(0, int(repair_budget_total))
        self.goal_acceptance_enabled = bool(goal_acceptance_enabled)
        self.event_sink = event_sink
        self._repair_attempts_by_step: dict[str, int] = {}
        self._repair_attempts_total = 0

    def execute(self) -> list[StepResult]:
        """Execute the approved plan with progress tracking.

        Returns:
            List of StepResult for completed steps

        The executor will stop and return on:
        - All steps completed → state becomes DONE
        - Step failure → state becomes PAUSED
        - External pause request → state becomes PAUSED
        """
        plan = self.plan_mode.plan
        if not plan:
            return []

        self.progress = PlanProgressDisplay(plan, self.console)
        validation_error = self._current_plan_validation_error()
        if validation_error:
            self.plan_mode.set_executing()
            self.plan_mode.pause(f"Plan validation failed: {validation_error}")
            self._emit("Validation", "biobank", "failed", validation_error)
            self._save_checkpoint()
            self._notify_log_update()
            self.progress.show_step_failure(
                "Plan validation",
                validation_error,
                choices=self._validation_failure_choices(),
            )
            return self.plan_mode.execution_log

        self._emit("Execution", "biobank", "running", f"starting {plan.total_steps} plan steps")

        # Transition to EXECUTING
        self.plan_mode.set_executing()
        self._save_checkpoint()
        self._notify_log_update()

        # Initialize progress display
        # Set status for already-completed steps (resume case)
        for step in plan.steps:
            if step.is_done:
                self.progress._step_status[step.id] = step.status

        try:
            self.progress.start_execution_display()
            while True:
                self._run_loop(plan)
                if self.plan_mode.state != PlanState.EXECUTING:
                    break
                if plan.remaining_steps():
                    break
                if self._ensure_goal_accepted(plan):
                    break
        finally:
            self.progress.stop_display()

        # Check if all done
        if not plan.remaining_steps() and self.plan_mode.state == PlanState.EXECUTING:
            self.plan_mode.complete()
            warnings = self._completion_warnings()
            try:
                setattr(self.plan_mode, "completion_warnings", warnings)
            except Exception:
                pass
            if warnings:
                self._emit(
                    "Report",
                    "biobank",
                    "warning",
                    f"completed with {len(warnings)} critical warning(s)",
                    _report_artifact_metadata(self.plan_mode),
                )
            else:
                self._emit(
                    "Report",
                    "biobank",
                    "success",
                    "all required plan steps completed",
                    _report_artifact_metadata(self.plan_mode),
                )
            self.progress.show_report(self.plan_mode.execution_log)
            # Clear checkpoint on successful completion
            self._clear_checkpoint()
            self._notify_log_update()

        return self.plan_mode.execution_log

    def _run_loop(self, plan: LongHorizonPlan) -> None:
        """Main execution loop — process steps until done or interrupted.

        Detects stalls (no runnable steps but remaining work) and reports
        them to the user rather than exiting silently.
        """
        max_idle_iterations = 0  # Guard against infinite loops
        while True:
            # Check if paused externally
            if self.plan_mode.state == PlanState.PAUSED:
                return

            # Get next runnable steps
            runnable = plan.next_runnable()
            if not runnable:
                # Check for stall: remaining steps exist but none are runnable
                remaining = plan.remaining_steps()
                if remaining:
                    validation_error = self._current_plan_validation_error()
                    if validation_error:
                        self.progress.stop_display()
                        self.plan_mode.pause(f"Plan validation failed: {validation_error}")
                        choices = self._validation_failure_choices()
                        self._emit(
                            "Validation",
                            "biobank",
                            "failed",
                            validation_error,
                            {"choices": choices, "pause_reason": self.plan_mode.pause_reason},
                        )
                        self._save_checkpoint()
                        self._notify_log_update()
                        self.progress.show_step_failure(
                            "Plan validation",
                            validation_error,
                            choices=choices,
                        )
                        return
                    # Stall detected — all remaining steps are blocked
                    max_idle_iterations += 1
                    if max_idle_iterations > 1:
                        # Definitely stalled, not just a transient state
                        self.progress.stop_display()
                        stall_msg = self._stall_message(plan, remaining)
                        self.plan_mode.pause(f"Stall: {stall_msg}")
                        choices = self._stall_choices(remaining)
                        self._emit(
                            "Repair",
                            "biobank",
                            "failed",
                            "plan stalled with blocked steps",
                            {
                                "choices": choices,
                                "blocked_steps": [s.id for s in remaining],
                                "pause_reason": self.plan_mode.pause_reason,
                            },
                        )
                        self._save_checkpoint()
                        self._notify_log_update()
                        self.progress.show_step_failure(
                            "Plan stalled — blocked steps remain",
                            stall_msg,
                            choices=choices,
                        )
                        return
                    # Give one more iteration in case mark_done was just processed
                    continue
                # Truly no more steps to run
                return

            max_idle_iterations = 0  # Reset stall guard on progress

            for step in runnable:
                # Re-check pause between steps
                if self.plan_mode.state == PlanState.PAUSED:
                    return

                # Execute the step, allowing bounded self-repair before failing.
                result = self._execute_step_with_repair(plan, step)
                if result is None:
                    self._save_checkpoint()
                    self._notify_log_update()
                    break

                self.plan_mode.execution_log.append(result)

                if result.success:
                    # Save checkpoint after each successful step
                    self._save_checkpoint()
                    self._notify_log_update()
                else:
                    # Step failed — pause and let user decide
                    self.progress.stop_display()
                    plan.mark_failed(step.id, result.error)
                    self.plan_mode.pause(f"Step failed: {step.description}")
                    choices = self._step_failure_choices(step)
                    self._emit(
                        "Repair",
                        step.skill,
                        "failed",
                        result.error,
                        {
                            "step_id": step.id,
                            "choices": choices,
                            "pause_reason": self.plan_mode.pause_reason,
                        },
                    )
                    self._save_checkpoint()
                    self._notify_log_update()
                    self.progress.show_step_failure(
                        step.description,
                        result.error,
                        choices=choices,
                    )
                    return

    def _execute_step_with_repair(self, plan: LongHorizonPlan, step: PlanStep) -> StepResult | None:
        """Execute a step and apply bounded repair before declaring failure.

        Returns None when repair changed the plan graph and the main loop should
        re-evaluate runnable steps.
        """
        last_result = self._execute_step(step)
        if last_result.success:
            return last_result

        while self._can_repair(step.id):
            action = self._propose_repair(plan, step, last_result)
            if action is None:
                return last_result

            self._record_repair(step, last_result, action)
            self._emit(
                "Repair",
                step.skill,
                "running",
                action.reason or action.action,
                {
                    "step_id": step.id,
                    "action": action.action,
                    "attempt_for_step": self._repair_attempts_by_step.get(step.id, 0),
                    "attempt_total": self._repair_attempts_total,
                },
            )
            outcome = self._apply_repair_action(plan, step, action)
            self._emit(
                "Repair",
                step.skill,
                "success" if outcome in {"retry", "defer"} else "warning",
                f"{action.action} -> {outcome}",
                {
                    "step_id": step.id,
                    "action": action.action,
                    "outcome": outcome,
                    "attempt_for_step": self._repair_attempts_by_step.get(step.id, 0),
                    "attempt_total": self._repair_attempts_total,
                },
            )
            self._save_checkpoint()
            self._notify_log_update()

            if outcome == "retry":
                last_result = self._execute_step(step)
                if last_result.success:
                    return last_result
                continue
            if outcome == "defer":
                return None
            if outcome == "pause":
                return last_result

            return last_result

        return last_result

    def _can_repair(self, step_id: str) -> bool:
        return (
            self._repair_attempts_total < self.repair_budget_total
            and self._repair_attempts_by_step.get(step_id, 0) < self.repair_budget_per_step
        )

    def _propose_repair(
        self,
        plan: LongHorizonPlan,
        step: PlanStep,
        result: StepResult,
    ) -> PlanRepairAction | None:
        context = {
            "attempt_for_step": self._repair_attempts_by_step.get(step.id, 0) + 1,
            "attempt_total": self._repair_attempts_total + 1,
            "repair_budget_per_step": self.repair_budget_per_step,
            "repair_budget_total": self.repair_budget_total,
        }
        raw = None
        if self.repair_strategy:
            try:
                raw = self.repair_strategy(step, result, plan, context)
            except Exception as e:
                logger.warning("Plan repair strategy failed: %s", e)

        action = self._normalize_repair_action(raw)
        if action:
            return action

        # Built-in conservative repair: report goals should use dual output.
        if step.skill == "generate_report" and "required artifact" in (result.error or ""):
            if str(step.args.get("format", "")).lower() != "dual":
                return PlanRepairAction(
                    action="retry_args",
                    reason="Report artifact check failed; retrying as dual report output.",
                    args={**step.args, "format": "dual"},
                )
        if step.skill == "field_search":
            suggestions = []
            if isinstance(result.result, dict):
                suggestions = list(result.result.get("suggested_queries") or [])
                needs_field_retry = (
                    bool(result.result.get("requires_repair"))
                    or result.result.get("total") == 0
                    or "zero fields" in (result.error or "")
                )
            else:
                needs_field_retry = "zero fields" in (result.error or "")
            if suggestions:
                return PlanRepairAction(
                    action="retry_args",
                    reason="Catalogue phrase returned zero hits; retrying with the first concrete suggested field query.",
                    args={**step.args, "query": str(suggestions[0])},
                ) if needs_field_retry else None
        return None

    def _normalize_repair_action(self, raw: Any) -> PlanRepairAction | None:
        if raw is None:
            return None
        if isinstance(raw, PlanRepairAction):
            return raw if raw.action else None
        if not isinstance(raw, dict):
            return None
        action = str(raw.get("action") or raw.get("type") or "").strip()
        if not action or action in {"none", "noop"}:
            return None
        return PlanRepairAction(
            action=action,
            reason=str(raw.get("reason") or raw.get("message") or ""),
            args=dict(raw.get("args") or {}),
            steps=list(raw.get("steps") or []),
            skill=dict(raw.get("skill") or {}),
            replacement_step=dict(raw.get("replacement_step") or {}),
        )

    def _record_repair(self, step: PlanStep | None, result: StepResult | None, action: PlanRepairAction) -> None:
        step_id = step.id if step else "__goal__"
        self._repair_attempts_total += 1
        self._repair_attempts_by_step[step_id] = self._repair_attempts_by_step.get(step_id, 0) + 1
        entry = {
            "step_id": step_id,
            "skill": step.skill if step else "",
            "error": result.error if result else "",
            "action": action.action,
            "reason": action.reason,
            "attempt_for_step": self._repair_attempts_by_step[step_id],
            "attempt_total": self._repair_attempts_total,
        }
        try:
            self.plan_mode.repair_log.append(entry)
        except Exception:
            pass

    def _apply_repair_action(
        self,
        plan: LongHorizonPlan,
        step: PlanStep | None,
        action: PlanRepairAction,
    ) -> str:
        kind = action.action
        if kind == "retry_args":
            if step is None:
                return "pause"
            previous_args = dict(step.args or {})
            step.args = dict(action.args or step.args or {})
            step.status = "pending"
            step.error = ""
            validation_error = self._current_plan_validation_error()
            if validation_error:
                logger.warning("Retry args produced invalid plan: %s", validation_error)
                step.args = previous_args
                return "pause"
            self.progress.update_step(step.id, "pending", f"Repair: {action.reason}") if self.progress else None
            return "retry"

        if kind == "insert_prerequisite_steps":
            if not action.steps:
                return "pause"
            inserted = self._insert_repair_steps(plan, step, action.steps)
            return "defer" if inserted else "pause"

        if kind == "create_custom_skill":
            outcome = self._create_custom_skill(action)
            if outcome != "retry":
                return outcome
            if step is not None:
                if action.replacement_step:
                    self._update_step_from_dict(step, action.replacement_step)
                elif action.args:
                    step.args = dict(action.args)
                step.status = "pending"
                step.error = ""
            return "retry"

        if kind == "replan_remaining":
            if not action.steps:
                return "pause"
            return "defer" if self._replace_remaining_steps(plan, action.steps) else "pause"

        if kind == "pause_with_trace":
            return "pause"

        return "pause"

    def _create_custom_skill(self, action: PlanRepairAction) -> str:
        skill_args = dict(action.skill or {})
        if not skill_args:
            skill_args = dict(action.args or {})
        required = {"name", "description", "parameters", "code_body"}
        if not required.issubset(skill_args):
            logger.warning("create_custom_skill repair missing required fields: %s", sorted(required - set(skill_args)))
            return "pause"

        result, is_error, error_msg = self._run_skill_for_repair("create_skill", skill_args)
        if is_error or error_msg or result.get("status") != "success" or not result.get("activated"):
            logger.warning("create_custom_skill repair failed: %s", error_msg or result)
            return "pause"

        try:
            if self.refresh_tools:
                self.refresh_tools()
            elif hasattr(self.plan_mode, "refresh_tools"):
                self.plan_mode.refresh_tools(self.registry)
        except Exception as e:
            logger.warning("Tool refresh after skill creation failed: %s", e)
            return "pause"
        return "retry"

    def _run_skill_for_repair(self, skill: str, args: dict) -> tuple[dict, bool, str]:
        try:
            if self.skill_executor:
                execution = self.skill_executor(skill, dict(args))
                if isinstance(execution, dict):
                    result = execution.get("result", execution)
                    result_dict = result if isinstance(result, dict) else {"output": str(result)}
                    is_error = bool(execution.get("is_error", False))
                    return result_dict, is_error, _result_error_message(result_dict)
                return {"output": str(execution)}, False, ""
            ctx = self.ctx_builder()
            result = self.registry.execute(skill, dict(args), ctx=ctx)
            result_dict = result if isinstance(result, dict) else {"output": str(result)}
            return result_dict, False, _result_error_message(result_dict)
        except Exception as e:
            return {"error": str(e)}, True, str(e)

    def _insert_repair_steps(
        self,
        plan: LongHorizonPlan,
        step: PlanStep | None,
        raw_steps: list[Any],
    ) -> bool:
        new_steps = [self._coerce_step(s, plan) for s in raw_steps]
        new_steps = [s for s in new_steps if s is not None]
        if not new_steps:
            return False

        snapshot = plan.to_dict()
        if step is None:
            plan.steps.extend(new_steps)
        else:
            try:
                index = plan.steps.index(step)
            except ValueError:
                index = len(plan.steps)
            plan.steps[index:index] = new_steps
            new_ids = [s.id for s in new_steps]
            step.depends_on = list(dict.fromkeys([*step.depends_on, *new_ids]))
            step.status = "pending"
            step.error = ""

        if hasattr(self.plan_mode, "validate_current_plan"):
            issues = self.plan_mode.validate_current_plan()
            if issues:
                logger.warning("Repair inserted schema-invalid steps: %s", self.plan_mode.validation_summary())
                self.plan_mode.plan = LongHorizonPlan.from_dict(snapshot)
                self.plan_mode.validate_current_plan()
                return False
        return True

    def _replace_remaining_steps(self, plan: LongHorizonPlan, raw_steps: list[Any]) -> bool:
        completed = [s for s in plan.steps if s.is_done]
        new_steps = [self._coerce_step(s, plan) for s in raw_steps]
        new_steps = [s for s in new_steps if s is not None]
        if not new_steps:
            return False
        snapshot = plan.to_dict()
        plan.steps = completed + new_steps
        if hasattr(self.plan_mode, "validate_current_plan"):
            issues = self.plan_mode.validate_current_plan()
            if issues:
                logger.warning("Replanned steps failed validation: %s", self.plan_mode.validation_summary())
                self.plan_mode.plan = LongHorizonPlan.from_dict(snapshot)
                self.plan_mode.validate_current_plan()
                return False
        return True

    def _coerce_step(self, value: Any, plan: LongHorizonPlan) -> PlanStep | None:
        if isinstance(value, PlanStep):
            return value
        if not isinstance(value, dict):
            return None
        step_id = str(value.get("id") or self._next_step_id(plan))
        existing = {s.id for s in plan.steps}
        if step_id in existing:
            step_id = self._next_step_id(plan)
        return PlanStep(
            id=step_id,
            skill=str(value.get("skill") or "think"),
            args=dict(value.get("args") or {}),
            description=str(value.get("description") or value.get("skill") or "Repair step"),
            depends_on=list(value.get("depends_on") or []),
            can_parallelize=bool(value.get("can_parallelize", False)),
            criticality=str(value.get("criticality") or "required"),
            repair_hints=list(value.get("repair_hints") or []),
        )

    def _next_step_id(self, plan: LongHorizonPlan) -> str:
        existing = {s.id for s in plan.steps}
        i = len(plan.steps) + 1
        while f"r{i}" in existing or f"s{i}" in existing:
            i += 1
        return f"r{i}"

    def _update_step_from_dict(self, step: PlanStep, data: dict) -> None:
        if "skill" in data:
            step.skill = str(data["skill"])
        if "args" in data:
            step.args = dict(data.get("args") or {})
        if "description" in data:
            step.description = str(data["description"])
        if "depends_on" in data:
            step.depends_on = list(data.get("depends_on") or [])
        if "criticality" in data:
            step.criticality = str(data.get("criticality") or "required")
        if "repair_hints" in data:
            step.repair_hints = list(data.get("repair_hints") or [])

    def _ensure_goal_accepted(self, plan: LongHorizonPlan) -> bool:
        if not self.goal_acceptance_enabled:
            return True

        verdict = self._goal_acceptance_checker_result(plan)
        if verdict.get("accepted", True):
            try:
                self.plan_mode.repair_log.append({
                    "step_id": "__goal__",
                    "action": "goal_acceptance",
                    "accepted": True,
                    "reason": verdict.get("reason", "Goal accepted."),
                })
            except Exception:
                pass
            return True

        action = self._normalize_repair_action(verdict.get("repair_action"))
        if action and self._can_repair("__goal__"):
            self._record_repair(None, StepResult("__goal__", "Goal acceptance", success=False, error=verdict.get("reason", "")), action)
            outcome = self._apply_repair_action(plan, None, action)
            self._save_checkpoint()
            self._notify_log_update()
            if outcome == "defer":
                return False

        self.progress.stop_display() if self.progress else None
        reason = verdict.get("reason", "Goal acceptance failed.")
        self.plan_mode.pause(f"Goal acceptance failed: {reason}")
        self._emit("Repair", "goal_acceptance", "failed", reason)
        self._save_checkpoint()
        self._notify_log_update()
        self.progress.show_step_failure("Goal acceptance", reason) if self.progress else None
        return False

    def _goal_acceptance_checker_result(self, plan: LongHorizonPlan) -> dict:
        if self.goal_acceptance_checker:
            try:
                result = self.goal_acceptance_checker(plan, self.plan_mode.execution_log)
                if isinstance(result, dict):
                    return result
            except Exception as e:
                return {"accepted": False, "reason": f"Goal acceptance checker failed: {e}"}

        goal = self.plan_mode.goal or plan.goal or ""
        needs_report = _goal_requests_report(goal)
        has_report_step = any(s.skill == "generate_report" for s in plan.steps)
        if not needs_report and not has_report_step:
            return {"accepted": True, "reason": "No report output requested."}

        successful_report = any(
            r.success
            and r.skill == "generate_report"
            and isinstance(r.result, dict)
            and r.result.get("markdown")
            and not _report_artifact_error(
                next((s for s in plan.steps if s.id == r.step_id), PlanStep(r.step_id, "generate_report")),
                r.result,
            )
            for r in self.plan_mode.execution_log
        )
        if successful_report:
            return {"accepted": True, "reason": "Report artifact generated."}

        completed_ids = [s.id for s in plan.steps if s.is_done]
        next_id = len(plan.steps) + 1
        return {
            "accepted": False,
            "reason": "Goal requested a report, but no successful generate_report artifact was recorded.",
            "repair_action": {
                "action": "insert_prerequisite_steps",
                "reason": "Add governed dual-report completion steps.",
                "steps": [
                    {"id": f"r{next_id}", "skill": "statistical_review", "args": {"scope": "session"}, "description": "Review statistical validity", "depends_on": completed_ids},
                    {"id": f"r{next_id + 1}", "skill": "safety_check", "args": {"scope": "session"}, "description": "Check privacy and safety", "depends_on": [f"r{next_id}"]},
                    {"id": f"r{next_id + 2}", "skill": "world_model_audit", "args": {"task": self.plan_mode.goal or plan.goal}, "description": "Audit final claims", "depends_on": [f"r{next_id + 1}"]},
                    {"id": f"r{next_id + 3}", "skill": "generate_report", "args": {"title": self.plan_mode.goal or "Biobank Analysis Report", "format": "dual"}, "description": "Generate final dual report", "depends_on": [f"r{next_id + 2}"]},
                ],
            },
        }

    def _execute_step(self, step: PlanStep) -> StepResult:
        """Execute a single plan step via the skill registry.

        Args:
            step: The PlanStep to execute

        Returns:
            StepResult with success/failure info
        """
        assert self.progress is not None

        # Update display
        step.status = "running"
        self.progress.update_step(step.id, "running", step.description)
        self._emit("Execution", step.skill, "running", step.description, {"step_id": step.id})

        def step_progress_callback(event: dict) -> None:
            phase = str(event.get("phase") or step.skill).strip()
            message = str(event.get("message") or "").strip()
            detail = f"{step.skill}: {phase}"
            if message:
                detail += f" - {message}"
            self.progress.update_activity(detail)

        try:
            setattr(self.plan_mode, "_active_step_progress", step_progress_callback)
        except Exception:
            pass

        start_time = time.time()

        try:
            step_args = dict(step.args or {})
            is_error = False
            if self.skill_executor is not None:
                execution = self.skill_executor(step.skill, dict(step_args))
                if isinstance(execution, dict):
                    is_error = bool(execution.get("is_error", False))
                    result = execution.get("result", execution)
                else:
                    result = execution
            else:
                # Build fresh context for this step
                ctx = self.ctx_builder()
                try:
                    setattr(ctx, "emit_progress", lambda phase="", message="", metadata=None: step_progress_callback({
                        "phase": phase,
                        "message": message,
                        "metadata": metadata or {},
                    }))
                except Exception:
                    pass
                result = self.registry.execute(step.skill, dict(step_args), ctx=ctx)
            duration = time.time() - start_time

            result_dict = result if isinstance(result, dict) else {"output": str(result)}
            error_msg = _result_error_message(result_dict)
            if is_error and not error_msg:
                error_msg = "Skill executor returned is_error=True."
            if not error_msg:
                error_msg = _report_artifact_error(step, result_dict)
            if not error_msg:
                error_msg = _quality_gate_error(step, result_dict)
            if error_msg:
                self.progress.update_step(step.id, "failed")
                self._emit("Execution", step.skill, "failed", error_msg, {"step_id": step.id})
                return StepResult(
                    step_id=step.id,
                    step_description=step.description,
                    skill=step.skill,
                    success=False,
                    result=result_dict,
                    error=error_msg,
                    duration_s=duration,
                )

            # Mark success
            self.plan_mode.plan.mark_done(step.id, result_dict)
            self.progress.update_step(step.id, "done")
            self._emit("Execution", step.skill, "success", step.description, {"step_id": step.id})

            return StepResult(
                step_id=step.id,
                step_description=step.description,
                skill=step.skill,
                success=True,
                result=result_dict,
                duration_s=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            self.progress.update_step(step.id, "failed")
            self._emit("Execution", step.skill, "failed", str(e), {"step_id": step.id})

            logger.error("Step %s (%s) failed: %s", step.id, step.skill, e)

            return StepResult(
                step_id=step.id,
                step_description=step.description,
                skill=step.skill,
                success=False,
                error=str(e),
                duration_s=duration,
            )
        finally:
            try:
                if getattr(self.plan_mode, "_active_step_progress", None) is step_progress_callback:
                    delattr(self.plan_mode, "_active_step_progress")
            except Exception:
                pass

    def resume(self) -> list[StepResult]:
        """Resume execution after pause.

        Returns:
            List of new StepResult from resumed execution
        """
        if self.plan_mode.state != PlanState.PAUSED:
            return []

        self.plan_mode.resume()
        self._save_checkpoint()
        self._notify_log_update()
        return self.execute()

    def _save_checkpoint(self) -> None:
        """Persist current plan execution state for crash recovery."""
        if not self._checkpoint_path:
            return
        try:
            PlanCheckpoint.from_plan_mode(self.plan_mode).save(self._checkpoint_path)
        except Exception as e:
            logger.warning("Failed to save plan checkpoint: %s", e)

    def _clear_checkpoint(self) -> None:
        """Remove checkpoint after successful completion or explicit exit."""
        if not self._checkpoint_path:
            return
        try:
            PlanCheckpoint.clear(self._checkpoint_path)
        except Exception as e:
            logger.warning("Failed to clear plan checkpoint: %s", e)

    def _notify_log_update(self) -> None:
        if not self.on_log_update:
            return
        try:
            self.on_log_update(list(self.plan_mode.execution_log))
        except Exception as e:
            logger.debug("Plan execution-log callback failed: %s", e)

    def _emit(
        self,
        phase: str,
        actor: str,
        status: str,
        message: str,
        metadata: dict | None = None,
    ) -> None:
        if not self.event_sink:
            return
        try:
            self.event_sink(phase, actor, status, message, metadata or {})
        except Exception as e:
            logger.debug("Plan event callback failed: %s", e)

    def _current_plan_validation_error(self) -> str:
        if not hasattr(self.plan_mode, "validate_current_plan"):
            return ""
        try:
            issues = self.plan_mode.validate_current_plan()
        except Exception as e:
            return f"Plan validation failed internally: {e}"
        if not issues:
            return ""
        if hasattr(self.plan_mode, "validation_summary"):
            return str(self.plan_mode.validation_summary())
        return "\n".join(f"- {issue}" for issue in issues)

    def _stall_message(self, plan: LongHorizonPlan, remaining: list[PlanStep]) -> str:
        done_ids = {s.id for s in plan.steps if s.is_done}
        status_by_id = {s.id: s.status for s in plan.steps}
        lines = [f"{len(remaining)} steps are blocked and cannot proceed."]
        for step in remaining[:8]:
            missing = [dep for dep in step.depends_on if dep not in done_ids]
            if missing:
                dep_text = ", ".join(f"{dep}({status_by_id.get(dep, 'missing')})" for dep in missing)
                lines.append(f"- {step.id} {step.description}: waiting for {dep_text}")
            else:
                lines.append(f"- {step.id} {step.description}: no runnable reason identified")
        if len(remaining) > 8:
            lines.append(f"- ... {len(remaining) - 8} more blocked step(s)")
        return "\n".join(lines)

    def _validation_failure_choices(self) -> list[dict]:
        return [
            {
                "key": "A",
                "title": "Auto-repair the dependency graph",
                "detail": "Fix missing dependencies, cycles, or schema-invalid steps, then review the repaired plan.",
            },
            {
                "key": "B",
                "title": "Ask Codex/Claude/Gemini for a validation diagnosis",
                "detail": "Collect external advice before applying a graph repair.",
            },
            {
                "key": "C",
                "title": "Edit plan manually",
                "detail": "Type repair instructions, then approve the validated plan.",
            },
            {"key": "N", "title": "Abort", "detail": "Stop execution and keep current artifacts."},
        ]

    def _completion_warnings(self) -> list[str]:
        """Surface critical guardrail verdicts even when report generation succeeds."""
        warnings: list[str] = []
        guardrail_skills = {"statistical_review", "safety_check", "world_model_audit"}
        critical_tokens = ("critical", "fail", "failed", "unsafe", "blocked", "not_approved")
        for result in self.plan_mode.execution_log:
            if result.skill not in guardrail_skills or not isinstance(result.result, dict):
                continue
            text_bits = []
            for key in (
                "overall_assessment",
                "overall",
                "status",
                "safety_status",
                "verdict",
                "risk_level",
            ):
                value = result.result.get(key)
                if value:
                    text_bits.append(str(value))
            text = " ".join(text_bits).lower()
            if any(token in text for token in critical_tokens):
                warnings.append(f"{result.skill}: {'; '.join(text_bits) or 'critical guardrail finding'}")
        return warnings

    def _step_failure_choices(self, step: PlanStep) -> list[dict]:
        choices = [
            {
                "key": "A",
                "title": "Auto-repair or re-plan this step",
                "detail": "Use built-in repair, create/modify a skill when useful, then retry from the failed step.",
            },
            {
                "key": "B",
                "title": "Ask Codex/Claude/Gemini for repair advice",
                "detail": "Collect external diagnosis before deciding whether to edit tools or re-plan.",
            },
            {
                "key": "C",
                "title": "Resume after explicit fix",
                "detail": "Continue only after you type repair instructions or approve an edited plan.",
            },
        ]
        if getattr(step, "criticality", "required") != "required":
            choices.append({
                "key": "S",
                "title": f"Skip optional step {step.id}",
                "detail": "Only valid for optional or diagnostic steps; required scientific/report gates cannot be skipped.",
            })
        choices.append({"key": "N", "title": "Abort", "detail": "Stop execution and keep current artifacts."})
        return choices

    def _stall_choices(self, remaining: list[PlanStep]) -> list[dict]:
        optional = [s.id for s in remaining if getattr(s, "criticality", "required") != "required"]
        detail = "Required blocked steps need repair or re-planning before the final report is valid."
        if optional:
            detail += f" Optional blocked steps that may be skipped explicitly: {', '.join(optional[:5])}."
        return [
            {
                "key": "A",
                "title": "Auto-repair blocked dependencies",
                "detail": "Agent inspects missing prerequisites, updates the remaining graph, and resumes when valid.",
            },
            {
                "key": "B",
                "title": "Ask Codex/Claude/Gemini for a stall diagnosis",
                "detail": "Runs external review hooks against the blocked trajectory before modifying the plan.",
            },
            {
                "key": "C",
                "title": "Continue after manual repair",
                "detail": detail,
            },
            {"key": "N", "title": "Abort", "detail": "Stop execution and keep current artifacts."},
        ]
