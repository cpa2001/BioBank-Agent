"""Hypothesis lifecycle tracking — propose, test, evaluate scientific hypotheses.

Manages hypotheses through: PROPOSED → TESTING → SUPPORTED/REFUTED/INCONCLUSIVE.
Each hypothesis is linked to analysis evidence and confidence scores.
"""

import hashlib
import logging
from datetime import datetime

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)

_VALID_STATUSES = {"testing", "supported", "refuted", "inconclusive"}


def _generate_id(statement: str) -> str:
    """Short hash ID from hypothesis statement."""
    return hashlib.md5(statement.encode()).hexdigest()[:8]


@skill(
    name="hypothesis",
    description="Track scientific hypotheses through their lifecycle. "
                "Actions: 'propose' (create new), 'list' (show all), "
                "'update' (change status with evidence), 'evaluate' (assess all). "
                "Use this to maintain a structured research narrative.",
    parameters={
        "action": {
            "type": "string",
            "description": "'propose', 'list', 'update', or 'evaluate'",
        },
        "statement": {
            "type": "string",
            "description": "Hypothesis statement (for 'propose') or ID (for 'update')",
            "default": "",
        },
        "status": {
            "type": "string",
            "description": "New status for 'update': 'testing', 'supported', 'refuted', 'inconclusive'",
            "default": "",
        },
        "evidence": {
            "type": "string",
            "description": "Evidence text or analysis reference for 'update'",
            "default": "",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence score 0-1 for 'update'",
            "default": 0.5,
        },
    },
    required=["action"],
)
def hypothesis(
    action: str,
    statement: str = "",
    status: str = "",
    evidence: str = "",
    confidence: float = 0.5,
    *,
    ctx=None,
) -> dict:
    """Manage scientific hypotheses through their lifecycle."""
    if ctx is None or not hasattr(ctx, "state") or not hasattr(ctx.state, "custom_data"):
        return {"error": "A session context with state.custom_data is required."}

    # Store hypotheses in session custom_data
    if "hypotheses" not in ctx.state.custom_data:
        ctx.state.custom_data["hypotheses"] = {}

    hypotheses = ctx.state.custom_data["hypotheses"]

    if action == "propose":
        if not statement:
            return {"error": "Provide a hypothesis statement."}
        h_id = _generate_id(statement)
        if h_id in hypotheses:
            return {"error": f"Hypothesis already exists: {h_id}", "hypothesis": hypotheses[h_id]}
        hypotheses[h_id] = {
            "id": h_id,
            "statement": statement,
            "status": "PROPOSED",
            "proposed_at": datetime.now().isoformat(),
            "evidence": [],
            "confidence": 0.5,
        }
        logger.info("Hypothesis proposed: %s — %s", h_id, statement[:60])
        return {
            "action": "proposed",
            "hypothesis": hypotheses[h_id],
            "message": f"Hypothesis {h_id} proposed. Next: test it with data.",
        }

    elif action == "list":
        if not hypotheses:
            return {"hypotheses": [], "message": "No hypotheses tracked yet."}
        return {
            "n_hypotheses": len(hypotheses),
            "hypotheses": list(hypotheses.values()),
            "summary": {
                s: sum(1 for h in hypotheses.values() if h["status"] == s)
                for s in ("PROPOSED", "TESTING", "SUPPORTED", "REFUTED", "INCONCLUSIVE")
            },
        }

    elif action == "update":
        h_id = statement  # reuse statement field for ID
        if h_id not in hypotheses:
            # Try matching by partial statement
            matches = [k for k, v in hypotheses.items() if statement.lower() in v["statement"].lower()]
            if len(matches) == 1:
                h_id = matches[0]
            else:
                return {"error": f"Hypothesis '{h_id}' not found. Use 'list' to see IDs."}

        h = hypotheses[h_id]
        if status:
            if status.lower() not in _VALID_STATUSES:
                return {
                    "error": (
                        "Unknown status. Use 'testing', 'supported', "
                        "'refuted', or 'inconclusive'."
                    )
                }
            h["status"] = status.upper()
        if evidence:
            h["evidence"].append({
                "text": evidence,
                "timestamp": datetime.now().isoformat(),
                "confidence": confidence,
            })
        h["confidence"] = confidence
        h["updated_at"] = datetime.now().isoformat()

        logger.info("Hypothesis %s updated: %s (confidence=%.2f)", h_id, h["status"], confidence)
        return {
            "action": "updated",
            "hypothesis": h,
        }

    elif action == "evaluate":
        if not hypotheses:
            return {"message": "No hypotheses to evaluate."}

        evaluations = []
        for h in hypotheses.values():
            n_evidence = len(h["evidence"])
            eval_result = {
                "id": h["id"],
                "statement": h["statement"],
                "status": h["status"],
                "confidence": h["confidence"],
                "n_evidence": n_evidence,
            }

            if h["status"] == "PROPOSED" and n_evidence == 0:
                eval_result["recommendation"] = "Not yet tested. Design an analysis to test this."
            elif h["status"] == "TESTING" and n_evidence > 0:
                avg_conf = sum(e["confidence"] for e in h["evidence"]) / n_evidence
                if avg_conf > 0.7:
                    eval_result["recommendation"] = "Evidence is accumulating. Consider marking as SUPPORTED."
                elif avg_conf < 0.3:
                    eval_result["recommendation"] = "Evidence is weak. Consider marking as REFUTED."
                else:
                    eval_result["recommendation"] = "Evidence is mixed. Gather more data."
            evaluations.append(eval_result)

        return {
            "action": "evaluate",
            "n_hypotheses": len(evaluations),
            "evaluations": evaluations,
        }

    else:
        return {"error": f"Unknown action: {action}. Use 'propose', 'list', 'update', or 'evaluate'."}
