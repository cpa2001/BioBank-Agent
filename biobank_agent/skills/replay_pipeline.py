"""Replay a saved analysis pipeline with optional parameter overrides.

This skill enables the agent to re-execute previously saved workflows, optionally
with modified parameters. This is critical for self-evolution and macro recording.
"""

import json
import logging
from typing import Optional

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)


@skill(
    name="replay_pipeline",
    description="Replay a saved analysis pipeline with optional parameter overrides. "
                "Requires pipeline name (e.g., 'diabetes_analysis'). "
                "Parameter overrides as JSON (e.g., '{\"top_n\": 50}'). "
                "Returns execution trace with results from each step.",
    parameters={
        "pipeline_name": {
            "type": "string",
            "description": "Name of saved pipeline (view available via list_pipelines)",
        },
        "overrides": {
            "type": "string",
            "description": "JSON dict of parameter overrides for pipeline steps (default: '{}')",
            "default": "",
        },
    },
    required=["pipeline_name"],
)
def replay_pipeline(pipeline_name: str, overrides: str = "", *, ctx=None) -> dict:
    """Execute a saved pipeline with optional parameter overrides.
    
    Parameters
    ----------
    pipeline_name : str
        Name of the saved pipeline to replay
    overrides : str, optional
        JSON string of parameter overrides (e.g., '{"top_n": 50}')
        Overrides apply to all steps in the pipeline
    ctx : SkillContext
        Agent context (dm, registry, state, etc.)
    
    Returns
    -------
    dict
        {
            "pipeline": pipeline_name,
            "steps_executed": int,
            "overrides_applied": dict,
            "results": [
                {"skill": skill_name, "args": {...}, "result": {...}},
                ...
            ],
            "total_execution_time": float seconds,
        }
    """
    memory = ctx.state.memory
    registry = ctx.registry
    
    # Retrieve pipeline definition
    pipeline_steps = memory.get_pipeline(pipeline_name)
    if pipeline_steps is None:
        available = memory.list_pipelines()
        return {
            "error": f"Pipeline '{pipeline_name}' not found",
            "available_pipelines": available,
        }
    
    # Parse overrides
    try:
        overrides_dict = json.loads(overrides) if overrides else {}
    except json.JSONDecodeError as e:
        return {
            "error": f"Invalid JSON in overrides: {e}",
            "overrides_received": overrides,
        }
    
    if not isinstance(overrides_dict, dict):
        return {
            "error": "Overrides must be a JSON object (dict)",
            "received_type": type(overrides_dict).__name__,
        }
    
    logger.info("Replaying pipeline: %s with overrides: %s", pipeline_name, overrides_dict)
    
    # Execute pipeline steps
    results = []
    errors = []
    import time
    start_time = time.time()
    
    for i, step in enumerate(pipeline_steps):
        skill_name = step.get("skill")
        step_args = step.get("args", {})
        
        # Apply parameter overrides
        merged_args = {**step_args, **overrides_dict}
        
        logger.info("Step %d: executing %s with args %s", i + 1, skill_name, merged_args)
        
        try:
            # Execute the skill
            result = registry.execute(skill_name, merged_args, ctx=ctx)
            results.append({
                "step": i + 1,
                "skill": skill_name,
                "args": merged_args,
                "status": "success",
                "result": result,
            })
        except Exception as e:
            error_msg = str(e)
            logger.error("Step %d failed: %s", i + 1, error_msg)
            results.append({
                "step": i + 1,
                "skill": skill_name,
                "args": merged_args,
                "status": "error",
                "error": error_msg,
            })
            errors.append(error_msg)
    
    elapsed = time.time() - start_time
    
    # Record this macro execution in session state
    ctx.state.records.append({
        "timestamp": time.time(),
        "skill": "replay_pipeline",
        "args": {"pipeline_name": pipeline_name, "overrides": overrides},
        "result_summary": {
            "steps_executed": len(results),
            "steps_succeeded": len([r for r in results if r["status"] == "success"]),
            "steps_failed": len(errors),
        },
    })
    
    return {
        "pipeline": pipeline_name,
        "steps_executed": len(results),
        "steps_succeeded": len([r for r in results if r["status"] == "success"]),
        "steps_failed": len(errors),
        "overrides_applied": overrides_dict,
        "execution_time_seconds": round(elapsed, 2),
        "results": results,
        "errors": errors if errors else None,
    }
