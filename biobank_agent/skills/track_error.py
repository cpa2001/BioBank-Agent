"""Track and suggest fixes for errors from failed skills.

Uses long-term memory to build error catalog and provide intelligent suggestions
based on previously successful resolutions.
"""

import logging
from typing import Optional

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)


@skill(
    name="track_error",
    description="Record an error event with context and suggested fix. "
                "Builds error catalog for pattern recognition. "
                "Returns error tracking ID and any previously known fixes.",
    parameters={
        "error_type": {
            "type": "string",
            "description": "Exception type name (e.g., 'ValueError', 'MemoryError')",
        },
        "error_message": {
            "type": "string",
            "description": "Full error message from the exception",
        },
        "skill_name": {
            "type": "string",
            "description": "Name of the skill that failed",
        },
        "suggested_fix": {
            "type": "string",
            "description": "Optional: suggested action to resolve the error",
            "default": "",
        },
        "context": {
            "type": "object",
            "description": "Optional: contextual info (e.g., parameters, data size)",
            "default": {},
        },
    },
    required=["error_type", "error_message", "skill_name"],
)
def track_error(
    error_type: str,
    error_message: str,
    skill_name: str,
    suggested_fix: str = "",
    context: Optional[dict] = None,
    *,
    ctx=None,
) -> dict:
    """Track an error in the long-term memory catalog.
    
    This skill:
    1. Records the error with its context
    2. Looks up any previously known fixes
    3. Returns suggestions for recovery
    4. Enables pattern recognition for common failure modes
    """
    
    if context is None:
        context = {}
    
    # Record error in long-term memory
    memory = ctx.state.memory
    memory.record_error(
        error_type=error_type,
        error_message=error_message,
        skill_name=skill_name,
        suggested_fix=suggested_fix if suggested_fix else None,
        context=context,
    )
    
    # Look up previously successful fixes
    known_fixes = memory.get_error_suggestions(error_type, skill_name)
    
    # Get error frequency
    error_count = 0
    error_key = f"{error_type}:{skill_name}"
    if "errors" in memory._data:
        error_count = memory._data["errors"].get(error_key, {}).get("count", 0)
    
    logger.warning(
        f"Tracked error: {error_type} in {skill_name} "
        f"({error_count} occurrences). Message: {error_message}"
    )
    
    if known_fixes:
        logger.info(f"Found {len(known_fixes)} previously successful fixes")
    
    return {
        "status": "tracked",
        "error_type": error_type,
        "skill": skill_name,
        "error_message": error_message,
        "occurrence_count": error_count,
        "error_key": error_key,
        "user_suggested_fix": suggested_fix if suggested_fix else None,
        "known_fixes": known_fixes,
        "context": context,
    }


@skill(
    name="list_errors",
    description="List most common errors across all skills with suggested fixes. "
                "Shows error patterns and recovery strategies.",
    parameters={
        "top_n": {
            "type": "integer",
            "description": "Number of top errors to show (default 10)",
            "default": 10,
        },
    },
    required=[],
)
def list_errors(top_n: int = 10, *, ctx=None) -> dict:
    """List most frequently occurring errors with their fixes.
    
    Provides insights into:
    - Most common failure modes
    - Which skills are most error-prone
    - Best known fixes for each error pattern
    """
    
    memory = ctx.state.memory
    
    # Get top errors
    top_errors = memory.most_common_errors(top_n)
    
    if not top_errors:
        return {
            "status": "no_errors",
            "message": "No errors recorded yet",
            "total_errors": 0,
        }
    
    # Format for readability
    formatted_errors = []
    for error in top_errors:
        formatted_errors.append({
            "error": f"{error['error_type']} in {error['skill']}",
            "occurrences": error["count"],
            "last_seen": error["last_seen"],
            "suggested_fixes": error["suggested_fixes"],
        })
    
    logger.info(f"Listed top {len(formatted_errors)} errors")
    
    return {
        "status": "success",
        "total_unique_errors": len(formatted_errors),
        "top_errors": formatted_errors,
    }
