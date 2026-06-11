"""AST-based skill code generation with safety validation.

Provides tools for generating new biobank analysis skills with strict safety checks:
- Imports are whitelisted (numpy, pandas, scipy, sklearn, xgboost, ...); the @skill
  decorator and the gated ``shell_exec`` seam are allowed by exact name only.
- Forbidden calls are blocked in both bare-name (exec, eval, open, system) and
  attribute (subprocess.run, os.system) form.
- Private attribute access is prevented.
- Generated code is linted and returned for manual review.

These rules are the same invariant enforced at apply time (create_skill) and at load time
(``discover_custom_skills`` runs ``validate_code`` before ``exec_module``).
"""

import ast
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SkillGenerator:
    """Generate and validate new skill code with AST safety checks."""
    
    # Whitelist of allowed imports
    ALLOWED_IMPORTS = {
        "numpy", "np", "pandas", "pd", "scipy", "sklearn",
        "xgboost", "xgb", "lightgbm", "lgb", "catboost",
        "json", "math", "re", "time", "datetime", "pathlib",
        "logging", "collections", "itertools", "functools", "typing",
    }

    # Exact ``from <module> import <name>`` allowances for modules outside ALLOWED_IMPORTS
    # whose specific names a skill legitimately needs. Two only: the @skill decorator that
    # every skill must import, and the gated shell-out seam — the one sanctioned way to run
    # heavy bioinformatics tools (scanpy/plink/...) is to shell out via ``shell_exec``, never
    # to import them or raw ``subprocess`` in-process. Importing any other name still fails.
    EXACT_FROM_IMPORTS = {
        "biobank_agent.registry": {"skill"},
        "biobank_agent.skills.local_exec": {"shell_exec"},
    }
    
    # Whitelist of safe builtins
    ALLOWED_BUILTINS = {
        "len", "range", "int", "float", "str", "list", "dict",
        "tuple", "set", "bool", "abs", "sum", "min", "max",
        "sorted", "reversed", "enumerate", "zip", "map", "filter",
        "any", "all", "round", "print", "isinstance", "type",
        "callable", "hasattr", "getattr", "setattr", "dir",
    }
    
    # Forbidden function/class calls (bare-name form, e.g. ``exec(...)``, ``open(...)``)
    FORBIDDEN_CALLS = {
        "exec", "eval", "compile", "__import__",
        "open", "input", "file",
        "system", "popen", "call", "run",
    }

    # Forbidden attribute-form callees (e.g. ``subprocess.run(...)``, ``os.system(...)``).
    # The bare-name FORBIDDEN_CALLS check never sees these — ``node.func`` is an Attribute,
    # not a Name — so a skill that obtained such a module would slip through. The shell-out
    # seam is invoked as a bare name (``shell_exec(...)``), so no exception is needed here.
    FORBIDDEN_CALL_ATTRS = {"run", "call", "system", "popen", "open"}

    # Forbidden attributes (anything starting with _)
    FORBIDDEN_ATTRIBUTES = {"_", "__"}
    
    @staticmethod
    def validate_code(code: str) -> Tuple[bool, str]:
        """Parse code and check for forbidden patterns.
        
        Parameters
        ----------
        code : str
            Python code to validate
        
        Returns
        -------
        tuple[bool, str]
            (is_valid, message) where message explains any validation errors
        """
        # Check basic syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Walk AST and validate
        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]  # Check root module
                    if module not in SkillGenerator.ALLOWED_IMPORTS:
                        return False, f"Forbidden import: {alias.name}"
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.split('.')[0] in SkillGenerator.ALLOWED_IMPORTS:
                    pass
                elif module in SkillGenerator.EXACT_FROM_IMPORTS:
                    allowed = SkillGenerator.EXACT_FROM_IMPORTS[module]
                    for alias in node.names:
                        if alias.name not in allowed:
                            return False, f"Forbidden import: from {module} import {alias.name}"
                else:
                    return False, f"Forbidden import: from {module}"

            # Check function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in SkillGenerator.FORBIDDEN_CALLS:
                        return False, f"Forbidden call: {func_name}()"
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in SkillGenerator.FORBIDDEN_CALL_ATTRS:
                        return False, f"Forbidden call: .{node.func.attr}()"
            
            # Check attribute access
            elif isinstance(node, ast.Attribute):
                attr = node.attr
                if attr.startswith("_"):
                    return False, f"Private attribute access not allowed: .{attr}"
        
        logger.info("Code validation passed")
        return True, "OK"
    
    @staticmethod
    def template(
        name: str,
        description: str,
        parameters: Dict[str, Dict[str, Any]],
        code_body: str,
    ) -> str:
        """Generate @skill decorated function.
        
        Parameters
        ----------
        name : str
            Skill name (alphanumeric + underscore)
        description : str
            Human-readable skill description
        parameters : dict[str, dict]
            Parameter definitions:
            {
                "param_name": {
                    "type": "string|integer|...",
                    "description": "...",
                    "default": value (optional)
                }
            }
        code_body : str
            Implementation code (indentation preserved)
        
        Returns
        -------
        str
            Complete @skill decorated Python function
        """
        # Build parameters dict string
        params_list = []
        for param_name, param_info in parameters.items():
            type_str = param_info.get("type", "string")
            desc_str = param_info.get("description", "")
            default = param_info.get("default")
            
            # Escape quotes in description
            desc_str = desc_str.replace('"', '\\"')
            
            param_def = f'"{param_name}": {{"type": "{type_str}", "description": "{desc_str}"}}'
            if default is not None:
                # Handle default values
                if isinstance(default, str):
                    param_def += f', "default": "{default}"'
                else:
                    param_def += f', "default": {default}'
            
            params_list.append(param_def)
        
        params_str = ", ".join(params_list)
        
        # Build required parameters list
        required_list = []
        for param_name, param_info in parameters.items():
            if "default" not in param_info:
                required_list.append(f'"{param_name}"')
        
        required_str = ", ".join(required_list)
        
        # Build function signature
        param_names = ", ".join(parameters.keys())
        
        # Indent code body
        indented_body = "\n".join(f"    {line}" for line in code_body.split("\n"))
        
        # Generate full skill code
        skill_code = f'''"""Auto-generated skill: {description}"""

from biobank_agent.registry import skill


@skill(
    name="{name}",
    description="{description}",
    parameters={{{params_str}}},
    required=[{required_str}],
)
def {name}({param_names}, *, ctx=None) -> dict:
{indented_body}
    return {{"status": "success"}}
'''
        
        return skill_code
    
    @staticmethod
    def get_validation_errors(code: str) -> list[str]:
        """Get detailed list of all validation issues in code.
        
        Parameters
        ----------
        code : str
            Python code to check
        
        Returns
        -------
        list[str]
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Syntax check
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [f"Syntax error: {e}"]
        
        # Collect all validation issues
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split('.')[0]
                    if module not in SkillGenerator.ALLOWED_IMPORTS:
                        errors.append(f"Line {node.lineno}: Forbidden import '{alias.name}'")
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.split('.')[0] in SkillGenerator.ALLOWED_IMPORTS:
                    pass
                elif module in SkillGenerator.EXACT_FROM_IMPORTS:
                    allowed = SkillGenerator.EXACT_FROM_IMPORTS[module]
                    for alias in node.names:
                        if alias.name not in allowed:
                            errors.append(f"Line {node.lineno}: Forbidden import 'from {module} import {alias.name}'")
                else:
                    errors.append(f"Line {node.lineno}: Forbidden import 'from {module}'")

            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in SkillGenerator.FORBIDDEN_CALLS:
                        errors.append(f"Line {node.lineno}: Forbidden call '{func_name}()'")
                    # Also check for getattr access to private attributes
                    if func_name == "getattr":
                        # Check if second argument is a string starting with "_"
                        if len(node.args) >= 2:
                            arg = node.args[1]
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                                if arg.value.startswith("_"):
                                    errors.append(f"Line {node.lineno}: Private attribute access via getattr()")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in SkillGenerator.FORBIDDEN_CALL_ATTRS:
                        errors.append(f"Line {node.lineno}: Forbidden call '.{node.func.attr}()'")
            
            elif isinstance(node, ast.Attribute):
                attr = node.attr
                if attr.startswith("_"):
                    errors.append(f"Line {node.lineno}: Private attribute access '.{attr}'")
        
        return errors
