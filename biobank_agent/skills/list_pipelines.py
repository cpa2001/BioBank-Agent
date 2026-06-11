"""List and inspect saved analysis pipelines.

Provides visibility into available macros and their step details for
documentation and replay planning.
"""

import logging
from typing import Optional

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)


@skill(
    name="list_pipelines",
    description="List all saved analysis pipelines (macros) with optional details. "
                "Shows pipeline names and steps. Use 'replay_pipeline' to execute.",
    parameters={
        "show_details": {
            "type": "string",
            "description": "Show detailed step information ('true' or 'false', default: 'false')",
            "default": "false",
        },
    },
    required=[],
)
def list_pipelines(show_details: str = "false", *, ctx=None) -> dict:
    """List all saved pipelines and optionally show their structure.
    
    Parameters
    ----------
    show_details : str, optional
        If 'true', include full step details including parameters
    ctx : SkillContext
        Agent context containing memory
    
    Returns
    -------
    dict
        {
            "pipelines": [list of names],
            "count": number of pipelines,
            "details": {...} if show_details='true'
        }
    """
    memory = ctx.state.memory
    
    pipeline_names = memory.list_pipelines()
    
    if not pipeline_names:
        return {
            "pipelines": [],
            "count": 0,
            "message": "No saved pipelines yet. Use 'record_macro' to create one.",
        }
    
    show_details_bool = show_details.lower() in ("true", "1", "yes")
    
    result = {
        "pipelines": pipeline_names,
        "count": len(pipeline_names),
    }
    
    if show_details_bool:
        details = {}
        for name in pipeline_names:
            pipeline = memory.get_pipeline(name)
            if pipeline:
                details[name] = {
                    "steps": len(pipeline),
                    "steps_detail": [
                        {
                            "skill": step.get("skill"),
                            "args_keys": list(step.get("args", {}).keys()),
                        }
                        for step in pipeline
                    ],
                }
        result["details"] = details
        logger.info("Listed %d pipelines with details", len(pipeline_names))
    else:
        logger.info("Listed %d pipelines", len(pipeline_names))
    
    return result
