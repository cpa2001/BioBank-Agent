"""Audit layer for physiological world-model predictions and simulations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorldModelEvidenceCard:
    """Evidence and safety card for a world-model prediction."""

    model_name: str
    task: str
    simulation_type: str
    input_modalities: list[str]
    available_tokens: int
    training_distribution_coverage: float
    calibration_status: str
    external_validation_status: str
    ood_flags: list[str] = field(default_factory=list)
    allowed_claim_type: str = "association_conditioned_forecast"
    safety_status: str = "PARTIAL"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_world_model_prediction(
    *,
    model_name: str = "HealthFormer-like physiological world model",
    task: str,
    simulation_type: str = "association_conditioned_forecast",
    input_modalities: list[str] | None = None,
    available_tokens: int = 0,
    training_distribution_coverage: float = 0.0,
    calibration_status: str = "unknown",
    external_validation_status: str = "not_validated",
) -> WorldModelEvidenceCard:
    """Classify what kind of claim a world-model output is allowed to support."""

    input_modalities = input_modalities or []
    sim = simulation_type.lower().strip()
    calibration = calibration_status.lower().strip()
    external = external_validation_status.lower().strip()
    reasons: list[str] = []
    ood_flags: list[str] = []

    if available_tokens < 20:
        ood_flags.append("sparse_context")
        reasons.append("available_tokens < 20; participant context is sparse")
    if training_distribution_coverage < 0.5:
        ood_flags.append("low_training_coverage")
        reasons.append("training distribution coverage < 0.5")
    if calibration not in {"calibrated", "pass", "validated"}:
        reasons.append("calibration status is not PASS/calibrated")
    if external not in {"externally_validated", "trial_validated", "replicated"}:
        reasons.append("external validation is missing or incomplete")

    if sim == "causal_effect_estimate":
        if external in {"trial_validated", "replicated"} and calibration in {"calibrated", "pass", "validated"}:
            allowed = "causal_hypothesis"
            reasons.append("causal wording still requires design-specific assumptions")
        else:
            allowed = "association_conditioned_forecast"
            reasons.append("causal effect estimate requested without sufficient trial/target-trial support")
    elif sim == "target_trial_emulation":
        allowed = "target_trial_emulation_estimate"
        if external not in {"externally_validated", "trial_validated", "replicated"}:
            reasons.append("target-trial emulation requires protocol and validation evidence")
    else:
        allowed = "association_conditioned_forecast"

    if not reasons and not ood_flags:
        safety = "PASS"
    elif "causal effect estimate requested without sufficient trial/target-trial support" in reasons:
        safety = "FAIL"
    else:
        safety = "PARTIAL"

    return WorldModelEvidenceCard(
        model_name=model_name,
        task=task,
        simulation_type=sim,
        input_modalities=input_modalities,
        available_tokens=available_tokens,
        training_distribution_coverage=float(training_distribution_coverage),
        calibration_status=calibration_status,
        external_validation_status=external_validation_status,
        ood_flags=ood_flags,
        allowed_claim_type=allowed,
        safety_status=safety,
        reasons=reasons,
    )


def record_world_model_card_to_action_graph(memory: Any, card: WorldModelEvidenceCard) -> None:
    """Persist a world-model audit card as evidence."""
    if memory is None or not hasattr(memory, "upsert_node"):
        return
    card_id = f"{card.model_name}:{card.task}:{card.simulation_type}".replace(" ", "_")
    memory.upsert_node("world_model_audit", card_id, payload=card.to_dict(), score=1.0)
    memory.upsert_node("result", card.task, payload={"task": card.task}, score=0.7)
    memory.link_nodes(
        "world_model_audit",
        card_id,
        "result",
        card.task,
        relation="audits_prediction",
        weight=0.9,
        evidence={"safety_status": card.safety_status, "allowed_claim_type": card.allowed_claim_type},
    )
