"""里程碑1: lightweight consistency self-check — docs/pyproject vs the live registries.

Guards against the skill/command counts and the version string drifting out of sync
(the exact defect 里程碑1 fixed). Deterministic and offline: it reads the same registries
the CLI builds, so adding a skill or command fails this test until the README is updated.
"""

from __future__ import annotations

import re
from pathlib import Path

from biobank_agent.cli.commands.registry import build_core_registry
from biobank_agent.registry import _registry, autodiscover_skills

_ROOT = Path(__file__).resolve().parents[1]


def _runtime_skill_count() -> int:
    autodiscover_skills()
    return len(getattr(_registry, "_schemas", {}))


def _runtime_command_count() -> int:
    return len(build_core_registry())


def test_readme_skill_count_matches_registry() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"\*\*(\d+) registered skills\*\*", readme)
    assert match, "README must state '**N registered skills**'"
    assert int(match.group(1)) == _runtime_skill_count()


def test_readme_command_count_matches_registry() -> None:
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"\*\*(\d+) slash commands\*\*", readme)
    assert match, "README must state '**N slash commands**'"
    assert int(match.group(1)) == _runtime_command_count()


def _pyproject_version() -> str:
    text = (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    assert match, "pyproject.toml must define a project version"
    return match.group(1)


def test_pyproject_version_is_v31_rc() -> None:
    assert _pyproject_version().startswith("3.1"), "version should be on the v3.1 RC line"


def test_package_version_matches_pyproject() -> None:
    import biobank_agent

    assert biobank_agent.__version__ == _pyproject_version()
