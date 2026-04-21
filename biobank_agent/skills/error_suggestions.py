"""Intelligent error suggestion engine for skill recovery.

Analyzes errors and provides multi-strategy recovery suggestions:
- Parameter adjustment (reduce data size, simplify models)
- Resource management (reduce memory footprint, parallelize)
- Data validation (check input formats, verify data availability)
- Fallback strategies (alternative algorithms, graceful degradation)
"""

import logging
from typing import Optional

from biobank_agent.registry import skill

logger = logging.getLogger(__name__)


def _suggest_fixes_for_error(error_type: str, error_message: str, skill_name: str) -> list[str]:
    """Generate intelligent fix suggestions based on error type and context.
    
    Returns prioritized list of suggested actions.
    """
    error_msg_lower = error_message.lower()
    fixes = []
    
    # Memory errors
    if error_type == "MemoryError" or "memory" in error_msg_lower:
        fixes.extend([
            "Reduce sample size (pass lower value to sample_size parameter)",
            "Reduce n_repeats or n_folds for cross-validation",
            "Process data in chunks instead of all at once",
            "Use @retry_on_error decorator to reduce complexity on retry",
        ])
    
    # Timeout errors
    elif error_type == "TimeoutError" or "timeout" in error_msg_lower:
        fixes.extend([
            "Increase timeout duration if available",
            "Reduce data size or model complexity",
            "Check network connectivity and server status",
            "Split analysis into smaller sub-tasks",
        ])
    
    # Not found / Key errors
    elif error_type in ("KeyError", "ValueError") or "not found" in error_msg_lower:
        fixes.extend([
            f"Verify the field/column exists in the data",
            f"Check data was loaded correctly (biomarkers/diagnoses table)",
            f"Use a different ICD10 code or field ID",
            f"Inspect available columns: call list_fields() or get_data_summary()",
        ])
    
    # Attribute errors (API mismatch)
    elif error_type == "AttributeError":
        fixes.extend([
            "Check if the skill has the expected method/attribute",
            "Verify library version matches requirements",
            "Try importing the module manually to check for import errors",
        ])
    
    # I/O errors
    elif error_type in ("IOError", "OSError") or "io" in error_msg_lower or "file" in error_msg_lower:
        fixes.extend([
            "Check file/directory exists and is readable",
            "Verify disk space is available",
            "Check file permissions",
            "Try specifying absolute path instead of relative",
        ])
    
    # Runtime errors
    elif error_type == "RuntimeError":
        fixes.extend([
            "Check input data is valid and properly formatted",
            "Try with simpler parameters (fewer features, smaller model)",
            "Check if required dependencies are installed",
        ])
    
    # Type errors
    elif error_type == "TypeError":
        fixes.extend([
            "Verify parameter types match expected types",
            "Check function signature and argument order",
            "Ensure required keyword arguments are provided",
        ])
    
    # Import errors
    elif error_type == "ImportError" or "import" in error_msg_lower:
        fixes.extend([
            "Install missing package: pip install <package_name>",
            "Check Python environment has required dependencies",
            "Verify package versions are compatible",
        ])
    
    # Generic fixes for unknown errors
    if not fixes:
        fixes.extend([
            "Review error message for clues about root cause",
            "Check skill documentation and parameters",
            "Try with simpler inputs (fewer subjects, fewer features)",
            "Check if data is correctly loaded",
        ])
    
    return fixes


@skill(
    name="suggest_error_fix",
    description="Get intelligent suggestions for recovering from an error. "
                "Analyzes error type, message, and context to suggest recovery strategies: "
                "parameter adjustment, resource management, data validation, fallback strategies.",
    parameters={
        "error_type": {
            "type": "string",
            "description": "Exception type name (e.g., 'MemoryError', 'KeyError')",
        },
        "error_message": {
            "type": "string",
            "description": "Full error message from the exception",
        },
        "skill_name": {
            "type": "string",
            "description": "Name of the skill that failed",
        },
        "current_parameters": {
            "type": "object",
            "description": "Current parameters used by the skill",
            "default": {},
        },
    },
    required=["error_type", "error_message", "skill_name"],
)
def suggest_error_fix(
    error_type: str,
    error_message: str,
    skill_name: str,
    current_parameters: Optional[dict] = None,
    *,
    ctx=None,
) -> dict:
    """Generate intelligent suggestions for recovering from an error.
    
    This skill:
    1. Analyzes the error type and message
    2. Suggests multiple recovery strategies
    3. Recommends parameter adjustments if applicable
    4. Returns both generic fixes and context-specific actions
    """
    
    if current_parameters is None:
        current_parameters = {}
    
    # Get suggestions
    generic_fixes = _suggest_fixes_for_error(error_type, error_message, skill_name)
    
    # Get fixes from error history
    memory = ctx.state.memory
    known_fixes = memory.get_error_suggestions(error_type, skill_name)
    
    # Suggest parameter mutations if applicable
    parameter_mutations = {}
    if error_type == "MemoryError" or "memory" in error_message.lower():
        # Suggest reducing computational complexity
        if "n_folds" in current_parameters:
            parameter_mutations["n_folds"] = max(2, current_parameters["n_folds"] - 1)
        if "sample_size" in current_parameters:
            parameter_mutations["sample_size"] = max(10, current_parameters["sample_size"] // 2)
        if "n_repeats" in current_parameters:
            parameter_mutations["n_repeats"] = max(1, current_parameters["n_repeats"] - 1)
        if "top_n" in current_parameters:
            parameter_mutations["top_n"] = min(current_parameters["top_n"], 10)
    
    logger.info(
        f"Suggested fixes for {error_type} in {skill_name}: "
        f"{len(generic_fixes)} strategies, {len(known_fixes)} from history"
    )
    
    # Combine and deduplicate suggestions
    all_fixes = []
    seen = set()
    
    # Prioritize known fixes (already worked)
    for fix in known_fixes:
        if fix and fix not in seen:
            all_fixes.append({"source": "from_history", "suggestion": fix})
            seen.add(fix)
    
    # Then add generic fixes
    for fix in generic_fixes:
        if fix and fix not in seen:
            all_fixes.append({"source": "generic", "suggestion": fix})
            seen.add(fix)
    
    return {
        "status": "success",
        "error_type": error_type,
        "skill": skill_name,
        "total_suggestions": len(all_fixes),
        "suggestions": all_fixes,
        "suggested_parameter_mutations": parameter_mutations,
        "retry_recommended": error_type not in ("ValueError", "KeyError", "AttributeError", "TypeError"),
        "message": f"Error '{error_type}' - {len(all_fixes)} recovery strategies available"
    }
