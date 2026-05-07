"""World-model prediction audit skill."""

from biobank_agent.registry import skill
from biobank_agent.world_model import (
    audit_world_model_prediction,
    record_world_model_card_to_action_graph,
)


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
    return card.to_dict()
