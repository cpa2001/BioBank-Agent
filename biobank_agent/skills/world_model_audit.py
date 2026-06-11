"""World-model prediction audit skill."""

from __future__ import annotations

from typing import Any

from biobank_agent.registry import skill
from biobank_agent.world_model import (
    audit_world_model_prediction,
    record_world_model_card_to_action_graph,
)


def _coerce_positive_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed > 0 else 0


def _latest_trajectory_result(ctx: Any) -> dict[str, Any] | None:
    state = getattr(ctx, "state", None)
    records = list(getattr(state, "records", []) or [])
    for rec in reversed(records):
        if getattr(rec, "skill", "") != "trajectory_tokenize":
            continue
        result = getattr(rec, "key_results", {}) or {}
        if isinstance(result, dict) and "error" not in result:
            return result
    return None


def _session_trajectory_context(
    *,
    ctx: Any,
    available_tokens: int,
    modalities: list[str],
) -> tuple[int, list[str], dict[str, Any]]:
    """Use prior trajectory_tokenize evidence when the planner did not know it yet."""

    if ctx is None:
        return available_tokens, modalities, {}
    trajectory = _latest_trajectory_result(ctx)
    if not trajectory:
        return available_tokens, modalities, {}

    inferred: dict[str, Any] = {}
    if available_tokens <= 0:
        n_tokens = _coerce_positive_int(trajectory.get("n_tokens"))
        if n_tokens:
            available_tokens = n_tokens
            inferred["available_tokens"] = "trajectory_tokenize.n_tokens"
            inferred["trajectory_status"] = trajectory.get("status", "")
            inferred["trajectory_time_source"] = trajectory.get("trajectory_time_source", "")

    if not modalities:
        trajectory_modalities = trajectory.get("modalities") or []
        if isinstance(trajectory_modalities, str):
            trajectory_modalities = [trajectory_modalities]
        if isinstance(trajectory_modalities, (list, tuple)):
            modalities = [str(m).strip() for m in trajectory_modalities if str(m).strip()]
            if modalities:
                inferred["input_modalities"] = "trajectory_tokenize.modalities"

    return available_tokens, modalities, inferred


@skill(
    name="world_model_audit",
    description=(
        "Audit a HealthFormer-like physiological world-model prediction or intervention simulation. "
        "Classifies whether output can support an association-conditioned forecast, target-trial estimate, "
        "causal hypothesis, or must be blocked."
    ),
    parameters={
        "task": {
            "type": "string",
            "description": "Prediction or simulation task being audited",
        },
        "simulation_type": {
            "type": "string",
            "description": "association_conditioned_forecast, target_trial_emulation, or causal_effect_estimate",
            "default": "association_conditioned_forecast",
        },
        "input_modalities": {
            "type": "string",
            "description": "Comma-separated available modalities, e.g. blood,bmi,cgm,sleep",
            "default": "",
        },
        "available_tokens": {
            "type": "integer",
            "description": "Number of participant trajectory tokens available",
            "default": 0,
        },
        "training_distribution_coverage": {
            "type": "number",
            "description": "0-1 estimate of coverage by training/evaluation distribution",
            "default": 0.0,
        },
        "calibration_status": {
            "type": "string",
            "description": "calibrated, pass, unknown, failed, etc.",
            "default": "unknown",
        },
        "external_validation_status": {
            "type": "string",
            "description": "externally_validated, trial_validated, replicated, not_validated",
            "default": "not_validated",
        },
    },
    required=["task"],
)
def world_model_audit(
    task: str,
    simulation_type: str = "association_conditioned_forecast",
    input_modalities: str = "",
    available_tokens: int = 0,
    training_distribution_coverage: float = 0.0,
    calibration_status: str = "unknown",
    external_validation_status: str = "not_validated",
    *,
    ctx=None,
) -> dict:
    modalities = [m.strip() for m in input_modalities.split(",") if m.strip()]
    available_tokens, modalities, inferred = _session_trajectory_context(
        ctx=ctx,
        available_tokens=available_tokens,
        modalities=modalities,
    )
    card = audit_world_model_prediction(
        task=task,
        simulation_type=simulation_type,
        input_modalities=modalities,
        available_tokens=available_tokens,
        training_distribution_coverage=training_distribution_coverage,
        calibration_status=calibration_status,
        external_validation_status=external_validation_status,
    )
    if ctx is not None:
        record_world_model_card_to_action_graph(getattr(ctx, "memory", None), card)
    result = card.to_dict()
    if inferred:
        result["session_context_inferred"] = inferred
    return result
