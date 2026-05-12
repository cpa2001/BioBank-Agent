"""Environment repair skill for bounded runtime dependency fixes."""

from __future__ import annotations

import subprocess
import sys
import importlib.util

from biobank_agent.registry import skill


@skill(
    name="environment_repair",
    description=(
        "Diagnose or apply bounded Python environment repairs during plan execution. "
        "Use for missing or deprecated packages after asking for confirmation."
    ),
    parameters={
        "action": {
            "type": "string",
            "description": "Repair action: diagnose, pip_install, or package_hint",
            "default": "diagnose",
        },
        "package": {
            "type": "string",
            "description": "Package spec for pip_install or package_hint, e.g. ddgs>=9.14.2",
            "default": "",
        },
        "reason": {
            "type": "string",
            "description": "Why this repair is needed",
            "default": "",
        },
        "confirmed": {
            "type": "boolean",
            "description": "Must be true before mutating the Python environment",
            "default": False,
        },
    },
)
def environment_repair(
    action: str = "diagnose",
    package: str = "",
    reason: str = "",
    confirmed: bool = False,
    *,
    ctx=None,
) -> dict:
    """Diagnose or apply a bounded environment repair."""
    action = (action or "diagnose").strip().lower()
    package = (package or "").strip()

    if action in {"diagnose", "package_hint"}:
        hints = []
        packages = {
            "ddgs": importlib.util.find_spec("ddgs") is not None,
            "duckduckgo_search": importlib.util.find_spec("duckduckgo_search") is not None,
        }
        if "duckduckgo" in package.lower() or "ddgs" in package.lower() or "duckduckgo" in reason.lower():
            hints.append("duckduckgo_search was renamed to ddgs; prefer package spec ddgs>=9.14.2.")
            if not packages["ddgs"]:
                hints.append("The active Python environment cannot import ddgs; install ddgs to avoid legacy fallback warnings.")
        return {
            "status": "needs_confirmation" if action == "package_hint" else "success",
            "action": action,
            "package": package,
            "reason": reason,
            "packages": packages,
            "hints": hints,
            "next_action": "Call environment_repair(action='pip_install', confirmed=true) only after user approval.",
        }

    if action != "pip_install":
        return {
            "status": "error",
            "error": f"Unsupported repair action: {action}",
            "allowed_actions": ["diagnose", "package_hint", "pip_install"],
        }
    if not package:
        return {"status": "error", "error": "package is required for pip_install."}
    if not confirmed:
        return {
            "status": "requires_confirmation",
            "package": package,
            "reason": reason,
            "message": "Environment mutation is disabled until confirmed=true is provided.",
        }

    cmd = [sys.executable, "-m", "pip", "install", package]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return {
        "status": "success" if proc.returncode == 0 else "error",
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }
