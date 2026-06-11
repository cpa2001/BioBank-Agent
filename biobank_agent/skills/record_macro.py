"""Record the current session as a replayable macro (pipeline).

This skill saves all executed analyses as a named pipeline that can be replayed later.
Supports filtering by record index to capture specific portions of the session.
"""

import json
import logging
from typing import Optional

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)


@skill(
    name="record_macro",
    description="Save current session as a replayable macro (pipeline). "
                "Records all executed skills and their parameters from the session. "
                "Use 'from_record_idx' to capture only recent analyses. "
                "Replayed via 'replay_pipeline' skill.",
    parameters={
        "name": {
            "type": "string",
            "description": "Name for this macro (e.g., 'diabetes_analysis_v1')",
        },
        "from_record_idx": {
            "type": "integer",
            "description": "Include only records from this index onward (default: 0 = all)",
            "default": 0,
        },
    },
    required=["name"],
)
def record_macro(name: str, from_record_idx: int = 0, *, ctx=None) -> dict:
    """Record session analyses as a replayable pipeline.
    
    Parameters
    ----------
    name : str
        Unique name for this macro (used to replay it later)
    from_record_idx : int, optional
        Start recording from this record index (default 0 = all records)
        Use this to capture only recent analyses
    ctx : SkillContext
        Agent context containing session state
    
    Returns
    -------
    dict
        {
            "saved": macro_name,
            "steps": number_of_steps,
            "records_included": record_range,
            "skills_recorded": [list of skill names],
        }
    """
    state = ctx.state
    memory = state.memory
    
    # Get records from the requested index onward
    records = state.records[from_record_idx:]
    
    if not records:
        return {
            "error": "No records to save",
            "total_records": len(state.records),
            "from_record_idx": from_record_idx,
        }
    
    # Filter out non-executable steps (like "think", session metadata, etc.)
    # Only keep skill executions with arguments
    steps = []
    recorded_skills = {}
    
    for record in records:
        skill_name = record.get("skill")
        args = record.get("args", {})
        
        # Skip non-skill records or records without args
        if not skill_name or skill_name == "think" or not args:
            continue
        
        # Skip meta skills (pipeline manipulation, macro recording)
        if skill_name in ("record_macro", "replay_pipeline", "list_pipelines"):
            logger.debug("Skipping meta-skill: %s", skill_name)
            continue
        
        step = {
            "skill": skill_name,
            "args": args,
        }
        steps.append(step)
        recorded_skills[skill_name] = recorded_skills.get(skill_name, 0) + 1
        
        logger.debug("Recording step: %s with args %s", skill_name, args)
    
    if not steps:
        return {
            "error": "No executable skills found in records",
            "total_records": len(records),
            "from_record_idx": from_record_idx,
        }
    
    # Save the pipeline to long-term memory
    try:
        memory.save_pipeline(name, steps)
        logger.info("Saved macro '%s' with %d steps", name, len(steps))
    except Exception as e:
        return {
            "error": f"Failed to save macro: {e}",
            "name": name,
        }
    
    # Record this macro recording in the session state
    ctx.state.records.append({
        "timestamp": len(ctx.state.records),  # simple sequence number
        "skill": "record_macro",
        "args": {"name": name, "from_record_idx": from_record_idx},
        "result_summary": {
            "steps_recorded": len(steps),
            "skills_included": list(recorded_skills.keys()),
        },
    })
    
    return {
        "saved": name,
        "steps": len(steps),
        "record_range": f"{from_record_idx} to {from_record_idx + len(records) - 1}",
        "total_session_records": len(state.records),
        "skills_recorded": recorded_skills,
        "steps_details": steps,
    }
