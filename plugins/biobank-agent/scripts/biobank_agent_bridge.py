#!/usr/bin/env python3
"""Small bridge script used by Codex/Claude plugins for Biobank Agent."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _json(data: dict) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _prepare_runtime_env() -> None:
    """Use writable cache dirs when plugins run in read-only agent sandboxes."""
    cache_root = Path(tempfile.gettempdir()) / "biobank_agent_plugin_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)


def status_cmd(_args: argparse.Namespace) -> int:
    _prepare_runtime_env()
    root = _repo_root()
    sys.path.insert(0, str(root))
    from biobank_agent.registry import autodiscover_skills, get_registry

    with contextlib.redirect_stderr(io.StringIO()):
        autodiscover_skills()
    registry = get_registry()
    _json({
        "repo_root": str(root),
        "pyproject": str(root / "pyproject.toml"),
        "skills_registered": len(registry),
        "skills": [s["name"] for s in registry.list_skills()],
        "skills_preview": [s["name"] for s in registry.list_skills()[:20]],
    })
    return 0


def external_status_cmd(_args: argparse.Namespace) -> int:
    _prepare_runtime_env()
    root = _repo_root()
    sys.path.insert(0, str(root))
    from biobank_agent.external_agents import ExternalAgentRunner

    _json(ExternalAgentRunner(workspace=root, timeout_s=30).status())
    return 0


def pytest_cmd(args: argparse.Namespace) -> int:
    root = _repo_root()
    command = [sys.executable, "-m", "pytest", *args.pytest_args]
    if not args.pytest_args:
        command.extend(["tests", "-q", "-m", "not integration"])
    proc = subprocess.run(command, cwd=str(root), text=True)
    return int(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Biobank Agent plugin bridge")
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Show registered skill/runtime status")
    p_status.set_defaults(func=status_cmd)

    p_external = sub.add_parser("external-status", help="Show local Codex/Claude availability")
    p_external.set_defaults(func=external_status_cmd)

    p_test = sub.add_parser("test", help="Run pytest with optional args")
    p_test.add_argument("pytest_args", nargs=argparse.REMAINDER)
    p_test.set_defaults(func=pytest_cmd)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
