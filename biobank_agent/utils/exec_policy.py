"""Single source of truth for command timeouts and background decisions.

Bioinformatics tools (plink/gatk/bcftools/vep/...) run far longer than a generic
shell command; a one-size 120s/1800s timeout either kills real work or blocks the
agent. These pure helpers resolve a timeout from (explicit override > per-tool
auto-scale > configured default) and decide whether a command should run as a
detached background job. No I/O, so every execution layer can import them.

Convention used across the codebase: a caller-supplied ``timeout`` of ``0`` (or a
negative value / ``None``) means "auto" — let the policy choose. A positive value
is honored verbatim (clamped to the hard ceiling for foreground runs). This keeps
existing call sites that pass explicit positive timeouts unchanged.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_TIMEOUT_S = 120
HARD_CEILING_S = 86400  # 1 day — clamp explicit foreground values

# Per-tool wall-clock baselines (seconds), matched against argv[0] basename.
LONG_TOOL_TIMEOUTS: dict[str, int] = {
    "plink": 7200, "plink2": 7200, "gatk": 14400, "gatk4": 14400,
    "bcftools": 7200, "vcftools": 3600, "tabix": 1800, "bgzip": 1800,
    "samtools": 3600, "bwa": 14400, "bowtie2": 14400, "star": 14400,
    "vep": 14400, "snpeff": 14400, "snpsift": 7200, "annovar": 14400,
    "table_annovar.pl": 14400, "rscript": 7200, "gseapy": 3600,
    "regenie": 14400, "saige": 14400, "king": 7200, "gcta": 7200,
    "bgenix": 3600, "qctool": 7200, "vep.pl": 14400,
}


def _setting(settings: Any, attr: str, default: Any) -> Any:
    if settings is None:
        return default
    value = getattr(settings, attr, default)
    return default if value is None else value


def _basename_key(token: str) -> str:
    token = str(token or "").strip().strip('"').strip("'")
    return os.path.basename(token).lower() if token else ""


def tool_key(argv: list[str] | None = None, command: str | None = None) -> str:
    """Best-effort program name from an argv list or a shell command string."""
    if argv:
        return _basename_key(argv[0])
    if command:
        for part in str(command).strip().split():
            # skip leading env assignments like FOO=bar before the program
            head = part.split("/", 1)[0]
            if "=" in head and not part.startswith(("/", "./", "~")):
                continue
            return _basename_key(part)
    return ""


def is_long_tool(argv: list[str] | None = None, command: str | None = None) -> bool:
    return tool_key(argv, command) in LONG_TOOL_TIMEOUTS


def resolve_timeout(
    *,
    argv: list[str] | None = None,
    command: str | None = None,
    override: Any = None,
    settings: Any = None,
    background: bool = False,
) -> int | None:
    """Resolve the wall-clock timeout in seconds, or None (uncapped).

    Precedence: positive ``override`` > per-tool auto-scale > configured default.
    ``override`` of 0 or None means "auto" (scale to the tool baseline / default);
    a NEGATIVE override requests "no cap" and resolves to None for a background job
    when ``exec_allow_no_timeout`` is set. Foreground values are always clamped to
    ``exec_hard_ceiling_s`` and never uncapped.
    """
    ceiling = int(_setting(settings, "exec_hard_ceiling_s", HARD_CEILING_S))
    allow_no_cap = bool(_setting(settings, "exec_allow_no_timeout", True))

    if override is not None:
        try:
            ov = int(override)
        except (TypeError, ValueError):
            ov = 0
        if ov > 0:
            return ov if background else min(ov, ceiling)
        # ov < 0 == explicit "no cap" request — only honored for background jobs.
        # ov == 0 means "auto"; fall through to per-tool / default resolution.
        if ov < 0 and background and allow_no_cap:
            return None

    key = tool_key(argv, command)
    if key in LONG_TOOL_TIMEOUTS:
        base = max(int(_setting(settings, "exec_long_tool_timeout_s", 0)), LONG_TOOL_TIMEOUTS[key])
        return base if background else min(base, ceiling)
    default = max(1, int(_setting(settings, "exec_default_timeout_s", DEFAULT_TIMEOUT_S)))
    return default if background else min(default, ceiling)


def should_background(
    *,
    argv: list[str] | None = None,
    command: str | None = None,
    expected_s: Any = None,
    settings: Any = None,
    interactive: bool = True,
    explicit: Any = None,
) -> bool:
    """Decide whether a command should run as a detached background job.

    Precedence: explicit flag > expected duration >= threshold > long-tool
    heuristic. The heuristic only auto-backgrounds when NOT interactive, so an
    interactive user keeps watching long tools live unless they opt in.
    """
    if explicit is not None:
        return bool(explicit)
    threshold = int(_setting(settings, "exec_background_threshold_s", 600))
    if expected_s is not None:
        try:
            if int(expected_s) >= threshold:
                return True
        except (TypeError, ValueError):
            pass
    if is_long_tool(argv=argv, command=command):
        if not interactive and bool(_setting(settings, "exec_auto_background_long_tools", True)):
            return True
    return False
