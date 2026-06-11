"""Invoke external coding-agent CLIs (codex / claude-code / gemini) as plugins.

Detection plus safe, non-interactive invocation. The process runner is injected, so the
orchestration is unit-testable offline with no real CLI or network. Anything an external
agent returns is UNTRUSTED: callers that act on its output (e.g. apply a produced diff)
must route through the review-branch apply gate in ``runtime.self_evolve``. This module
only launches the process and hands back its captured text.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

# Env keys whose name hints at a credential are removed before the environment is handed to a
# third-party agent process, so it cannot read the LLM API key, tokens, etc.
_SECRET_ENV_TOKENS = ("secret", "token", "key", "auth", "api", "password", "passwd", "credential")
_MAX_OUTPUT_CHARS = 200_000  # cap captured stdout so a runaway agent cannot exhaust memory/logs


@dataclass(frozen=True)
class ExternalAgentSpec:
    """How to invoke one external coding-agent CLI non-interactively.

    ``argv_template`` is a tuple of argv tokens; the literal token ``"{prompt}"`` is replaced
    by the prompt as a *single* argv element — it is never passed through a shell.
    """

    name: str
    argv_template: tuple[str, ...]

    def build_argv(self, prompt: str) -> list[str]:
        return [prompt if tok == "{prompt}" else tok for tok in self.argv_template]


# Known non-interactive invocations. Conservative defaults; a caller may pass ``overrides`` if a
# CLI's flags change. ``codex exec`` and ``claude -p`` / ``gemini -p`` run a single prompt and exit.
DEFAULT_SPECS: dict[str, ExternalAgentSpec] = {
    "codex": ExternalAgentSpec("codex", ("codex", "exec", "{prompt}")),
    "claude": ExternalAgentSpec("claude", ("claude", "-p", "{prompt}")),
    "claude-code": ExternalAgentSpec("claude-code", ("claude", "-p", "{prompt}")),
    "gemini": ExternalAgentSpec("gemini", ("gemini", "-p", "{prompt}")),
}


@dataclass
class ExternalAgentResult:
    name: str
    ok: bool
    text: str = ""
    error: str = ""
    returncode: Optional[int] = None


def _safe_env() -> dict[str, str]:
    """A copy of the environment with credential-bearing keys removed, so a third-party CLI
    cannot read secrets. PATH and the like are preserved so the binary still resolves."""
    return {
        k: v
        for k, v in os.environ.items()
        if not any(tok in k.lower() for tok in _SECRET_ENV_TOKENS)
    }


def parse_agent_list(value) -> list[str]:
    """Normalize a config value like ``"codex,claude,gemini"`` (or a list) into agent names."""
    items = list(value) if isinstance(value, (list, tuple)) else str(value or "").split(",")
    return [s.strip() for s in items if str(s).strip()]


def spec_for(name: str, *, overrides: Optional[dict] = None) -> Optional[ExternalAgentSpec]:
    if overrides and name in overrides:
        return overrides[name]
    return DEFAULT_SPECS.get(name)


def detect_available(
    names: Sequence[str], *, which: Callable[[str], Optional[str]] = shutil.which, overrides=None
) -> list[ExternalAgentSpec]:
    """Specs for the requested agents whose CLI binary resolves on PATH. ``which`` is injected so
    tests can simulate availability without the real binaries."""
    found: list[ExternalAgentSpec] = []
    for name in names:
        spec = spec_for(name, overrides=overrides)
        if spec is None or not spec.argv_template:
            continue
        if which(spec.argv_template[0]):
            found.append(spec)
    return found


# A runner takes (argv, cwd, timeout_s, env) and returns (returncode, stdout, stderr).
Runner = Callable[[list[str], str, float, dict], "tuple[int, str, str]"]


def _subprocess_runner(argv: list[str], cwd: str, timeout_s: float, env: dict) -> "tuple[int, str, str]":
    proc = subprocess.run(
        argv, cwd=cwd or None, env=env, text=True, capture_output=True, check=False, timeout=timeout_s
    )
    return proc.returncode, proc.stdout or "", proc.stderr or ""


def run_external_agent(
    spec: ExternalAgentSpec,
    prompt: str,
    *,
    cwd: str = "",
    timeout_s: float = 180.0,
    runner: Runner = _subprocess_runner,
) -> ExternalAgentResult:
    """Invoke one external coding-agent CLI with ``prompt`` (passed as a single argv element — no
    shell). Returns its captured stdout. Defensive: a timeout, a non-zero exit, or an OS error all
    map to ``ok=False`` with a reason rather than raising."""
    argv = spec.build_argv(prompt)
    try:
        rc, out, err = runner(argv, cwd, float(timeout_s), _safe_env())
    except subprocess.TimeoutExpired:
        return ExternalAgentResult(spec.name, ok=False, error=f"timed out after {timeout_s}s")
    except FileNotFoundError:
        return ExternalAgentResult(spec.name, ok=False, error=f"{spec.argv_template[0]} not found on PATH")
    except Exception as exc:  # pragma: no cover - defensive
        return ExternalAgentResult(spec.name, ok=False, error=str(exc))
    text = (out or "")[:_MAX_OUTPUT_CHARS]
    if rc != 0:
        return ExternalAgentResult(spec.name, ok=False, text=text, error=(err or "")[:2000], returncode=rc)
    return ExternalAgentResult(spec.name, ok=True, text=text, returncode=rc)


__all__ = [
    "ExternalAgentSpec",
    "ExternalAgentResult",
    "DEFAULT_SPECS",
    "detect_available",
    "run_external_agent",
    "parse_agent_list",
    "spec_for",
]
