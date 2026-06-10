"""Reproducibility receipt.

A scientific result is only trustworthy if someone else can re-run it. A receipt
bundles everything needed to reproduce one analysis: the code version (git SHA),
the interpreter + key package versions, fingerprints of every input file
(path+size+mtime+sha256), the random seeds, the command/skill, and its parameters.
It is emitted as a machine-readable ``receipt.json`` next to the outputs and as a
markdown block folded into the report.

Pure-ish + dependency-free (stdlib only); ``git_revision``/``data_fingerprint`` are
best-effort and never raise, so attaching a receipt can't break an analysis.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "biobank-receipt/1"
_DEFAULT_PACKAGES = ("pandas", "numpy", "scipy", "statsmodels", "scikit-learn", "matplotlib")


def git_revision(cwd: str | Path | None = None) -> str:
    """Best-effort git commit SHA of the working tree; '' if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(cwd) if cwd else None,
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return ""


def data_fingerprint(path: str | Path, *, hash_max_bytes: int = 8_000_000) -> dict[str, Any]:
    """Fingerprint one input file: path/exists/size/mtime + sha256 (full for small
    files, of the first 1 MiB for large ones). Never raises."""
    p = Path(path)
    info: dict[str, Any] = {"path": str(p)}
    try:
        if not p.exists() or not p.is_file():
            info["exists"] = False
            return info
        st = p.stat()
        info.update(exists=True, size_bytes=st.st_size, mtime=int(st.st_mtime))
        h = hashlib.sha256()
        with p.open("rb") as fh:
            if st.st_size <= hash_max_bytes:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
                info["sha256"] = h.hexdigest()
            else:
                h.update(fh.read(1 << 20))
                info["sha256_head_1mib"] = h.hexdigest()
    except Exception:
        pass
    return info


def package_versions(packages: tuple[str, ...] = _DEFAULT_PACKAGES) -> dict[str, str]:
    import importlib.metadata as md

    out: dict[str, str] = {}
    for name in packages:
        try:
            out[name] = md.version(name)
        except Exception:
            continue
    return out


def build_receipt(
    *,
    objective: str = "",
    skill: str = "",
    command: str = "",
    params: dict[str, Any] | None = None,
    data_paths: list[str | Path] | None = None,
    seeds: dict[str, Any] | None = None,
    timestamp: float | None = None,
    cwd: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a reproducibility receipt for one analysis. ``timestamp`` is taken
    verbatim when provided (deterministic for tests/replay), else wall-clock now."""
    return {
        "schema": SCHEMA,
        "timestamp": float(timestamp) if timestamp is not None else time.time(),
        "objective": objective,
        "skill": skill,
        "command": command,
        "params": dict(params or {}),
        "seeds": dict(seeds or {}),
        "code_version": git_revision(cwd),
        "python_version": sys.version.split()[0],
        "packages": package_versions(),
        "data": [data_fingerprint(p) for p in (data_paths or [])],
        "extra": dict(extra or {}),
    }


def format_receipt_markdown(receipt: dict[str, Any]) -> str:
    """Render a receipt as a compact markdown block for a report."""
    lines = ["### Reproducibility receipt", ""]
    code = receipt.get("code_version") or "(uncommitted / unknown)"
    lines.append(f"- **Code version (git):** `{code}`")
    lines.append(f"- **Python:** {receipt.get('python_version', '?')}")
    if receipt.get("command"):
        lines.append(f"- **Command:** `{receipt['command']}`")
    if receipt.get("seeds"):
        lines.append(f"- **Seeds:** {json.dumps(receipt['seeds'], sort_keys=True)}")
    pkgs = receipt.get("packages") or {}
    if pkgs:
        lines.append("- **Key packages:** " + ", ".join(f"{k}={v}" for k, v in sorted(pkgs.items())))
    data = receipt.get("data") or []
    if data:
        lines.append("- **Input files:**")
        for d in data:
            if d.get("exists"):
                digest = d.get("sha256") or d.get("sha256_head_1mib") or ""
                short = (digest[:12] + "…") if digest else "(no hash)"
                lines.append(f"    - `{d.get('path')}` — {d.get('size_bytes', '?')} bytes, sha256 {short}")
            else:
                lines.append(f"    - `{d.get('path')}` — (missing at receipt time)")
    return "\n".join(lines)


def write_receipt(receipt: dict[str, Any], out_dir: str | Path, *, name: str = "receipt.json") -> Path:
    """Write the machine-readable receipt next to the analysis outputs."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path
