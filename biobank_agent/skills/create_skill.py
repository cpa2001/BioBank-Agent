"""Create a new skill from code with AST validation and user approval workflow.

Generates validated Python skill code and saves it to a temporary location for
manual review before activation.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from biobank_agent.registry import skill
from biobank_agent.skills.generator import SkillGenerator

logger = logging.getLogger(__name__)


@skill(
    name="create_skill",
    description="Create a new analysis skill from code with AST validation. "
                "Code is validated for safety (allowed imports, forbidden calls blocked). "
                "Generated skill saved to reports/generated_skills/ for manual review. "
                "Move to biobank_agent/skills/ and restart to activate.",
    parameters={
        "name": {
            "type": "string",
            "description": "Skill name (alphanumeric + underscore, e.g. 'custom_analysis')",
        },
        "description": {
            "type": "string",
            "description": "One-line skill description for documentation",
        },
        "parameters": {
            "type": "string",
            "description": "JSON dict of parameters, e.g. '{\"top_n\": {\"type\": \"integer\", \"description\": \"Number of items\"}, "
                           "\"code_mode\": {\"type\": \"string\", \"description\": \"...\", \"default\": \"strict\"}}'",
        },
        "code_body": {
            "type": "string",
            "description": "Python code implementing the skill (will be indented into function body)",
        },
    },
    required=["name", "description", "parameters", "code_body"],
)
def create_skill(
    name: str,
    description: str,
    parameters: str,
    code_body: str,
    *,
    ctx=None,
) -> dict:
    """Create a new skill from code with validation.
    
    Parameters
    ----------
    name : str
        Skill name (must be valid Python identifier)
    description : str
        Human-readable description
    parameters : str
        JSON string defining skill parameters
    code_body : str
        Python code implementing the skill
    ctx : SkillContext
        Agent context (settings, etc.)
    
    Returns
    -------
    dict
        {
            "status": "success" | "validation_failed",
            "generated_skill_path": str (if successful),
            "code": str (full generated code),
            "validation_errors": [str] (if failed),
            "message": str
        }
    """
    
    # Validate skill name
    if not name.isidentifier():
        return {
            "status": "validation_failed",
            "error": f"Invalid skill name: '{name}' (must be valid Python identifier)",
        }
    
    # Parse parameters JSON
    try:
        params_dict = json.loads(parameters)
        if not isinstance(params_dict, dict):
            raise ValueError("Parameters must be a JSON object")
    except json.JSONDecodeError as e:
        return {
            "status": "validation_failed",
            "error": f"Invalid JSON in parameters: {e}",
        }
    except ValueError as e:
        return {
            "status": "validation_failed",
            "error": str(e),
        }
    
    # Validate code
    is_valid, validation_msg = SkillGenerator.validate_code(code_body)
    if not is_valid:
        errors = SkillGenerator.get_validation_errors(code_body)
        return {
            "status": "validation_failed",
            "error": f"Code validation failed: {validation_msg}",
            "validation_errors": errors,
        }
    
    # Generate skill code
    try:
        skill_code = SkillGenerator.template(name, description, params_dict, code_body)
    except Exception as e:
        return {
            "status": "validation_failed",
            "error": f"Failed to generate skill code: {e}",
        }
    
    # Save to temporary location for review
    try:
        temp_skill_dir = ctx.settings.reports_dir / "generated_skills"
        temp_skill_dir.mkdir(parents=True, exist_ok=True)
        
        skill_path = temp_skill_dir / f"{name}.py"
        skill_path.write_text(skill_code)
        
        logger.info("Generated skill saved to: %s", skill_path)
    except Exception as e:
        return {
            "status": "validation_failed",
            "error": f"Failed to save skill file: {e}",
        }
    
    return {
        "status": "success",
        "generated_skill_path": str(skill_path),
        "code": skill_code,
        "message": (
            f"✓ Skill '{name}' generated and saved.\n\n"
            f"Next steps:\n"
            f"1. Review the code at: {skill_path}\n"
            f"2. If approved, move to: biobank_agent/skills/{name}.py\n"
            f"3. Restart the agent to import and use the skill\n\n"
            f"Generated skill is INACTIVE until moved to biobank_agent/skills/"
        ),
    }
