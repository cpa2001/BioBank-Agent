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
        "activate": {
            "type": "boolean",
            "description": "Request activation into custom_skills after validation. Requires explicit approval in ctx.",
            "default": False,
        },
        "approved": {
            "type": "boolean",
            "description": "True only after explicit user approval for this generated skill activation.",
            "default": False,
        },
    },
    required=["name", "description", "parameters", "code_body"],
)
def create_skill(
    name: str,
    description: str,
    parameters: str,
    code_body: str,
    activate: bool = False,
    approved: bool = False,
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
    
    if ctx is None or not getattr(ctx, "settings", None):
        return {
            "status": "validation_failed",
            "error": "create_skill requires an agent ctx with settings.",
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

    # Activation is deliberately opt-in. Generated code is first-class output,
    # but turning it into an executable tool requires explicit user approval
    # represented by both call args and a trusted ctx flag.
    activated = False
    active_path = None
    activation_allowed = bool(
        activate
        and approved
        and (
            getattr(ctx, "allow_skill_activation", False)
            or getattr(getattr(ctx, "state", None), "custom_data", {}).get("allow_skill_activation")
        )
    )
    if activation_allowed:
        try:
            custom_dir = ctx.settings.custom_skills_dir
            custom_dir.mkdir(parents=True, exist_ok=True)
            active_path = custom_dir / f"{name}.py"
            active_path.write_text(skill_code)

            # Hot-reload into the running registry
            from biobank_agent.registry import discover_custom_skills, get_registry
            n_loaded = discover_custom_skills(custom_dir)
            registry = get_registry()
            if n_loaded > 0 or name in registry:
                activated = True
                logger.info("Skill '%s' hot-loaded into registry", name)
        except Exception as e:
            logger.warning("Activation failed (skill still saved for manual review): %s", e)

    if activated:
        message = (
            f"Skill '{name}' generated, validated, and activated.\n\n"
            f"The skill is now available in this session — no restart needed.\n"
            f"Source: {skill_path}\n"
            f"Active: {active_path}"
        )
    else:
        message = (
            f"Skill '{name}' generated and saved for review.\n\n"
            f"Next steps:\n"
            f"1. Review the code at: {skill_path}\n"
            f"2. If approved, run an activation flow that sets activate=true "
            f"and approved=true with ctx.allow_skill_activation.\n\n"
            f"Generated skill is INACTIVE until explicitly activated."
        )

    schema = None
    try:
        from biobank_agent.registry import get_registry
        schema = next(
            (s for s in get_registry().tool_schemas() if s.get("function", {}).get("name") == name),
            None,
        )
    except Exception:
        schema = None

    return {
        "status": "success",
        "skill_name": name,
        "generated_skill_path": str(skill_path),
        "active_path": str(active_path) if activated and active_path else None,
        "activated": activated,
        "schema": schema,
        "activation_error": "" if activated else "Skill saved but not activated.",
        "code": skill_code,
        "message": message,
    }
