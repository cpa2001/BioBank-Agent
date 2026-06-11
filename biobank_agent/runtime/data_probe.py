"""Data-grounded planning probe.

Before the council drafts a plan, read-only-probe the files the objective actually
references and inject a bounded ``DataContext`` so the planner reasons about REAL
data shape — killing the "CSV named *gwas_results* -> plan a WGS variant-calling
pipeline" misroute at the root (a CSV of GWAS summary stats is tabular data, not
genotypes). Local tabular files get columns/dtypes/rowcount/missingness with
``confidence=high``; a real ``.vcf`` is flagged as genuine genomic data
(``confidence=medium``); paths that don't resolve locally (HPC/dxfuse/Spark/remote)
degrade to ``confidence=unknown`` and NEVER block planning.

Pure read-only, never raises, bounded in size and time.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Absolute/relative path, or a bare filename with a known data extension.
_PATH_RE = re.compile(
    r"((?:[~\w.\-]+)?(?:/[\w.\-]+)+|[\w.\-]+\.(?:csv|tsv|txt|parquet|xlsx?|json|vcf|gz|h5ad|bgen|bed|pgen))",
    re.I,
)
_TABULAR_EXT = {".csv", ".tsv", ".txt"}
_GENOMIC_EXT = {".vcf", ".bgen", ".bed", ".pgen"}
_MAX_FILES = 4
_MAX_COLS = 30


def extract_paths(objective: str) -> list[str]:
    """All distinct file/dir paths referenced in the objective, in order."""
    seen: list[str] = []
    for match in _PATH_RE.finditer(str(objective or "")):
        token = match.group(0)
        if token and token not in seen:
            seen.append(token)
    return seen


def _resolve(token: str, cwd: str | None) -> Path:
    p = Path(token).expanduser()
    if not p.is_absolute() and cwd:
        p = Path(cwd).expanduser() / p
    return p


def _probe_tabular(path: Path, sep: str) -> dict[str, Any]:
    """Best-effort columns/dtypes/rowcount/missingness for a delimited file."""
    info: dict[str, Any] = {"kind": "tabular", "confidence": "high"}
    try:
        import pandas as pd

        head = pd.read_csv(path, sep=sep, nrows=200, low_memory=False)
        info["columns"] = [str(c) for c in head.columns[:_MAX_COLS]]
        info["n_columns"] = int(head.shape[1])
        miss = head.isna().mean().round(3)
        worst = sorted(((str(c), float(v)) for c, v in miss.items()), key=lambda x: x[1], reverse=True)
        info["missingness_top"] = {c: v for c, v in worst[:5] if v > 0}
        n = 0
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for n, _ in enumerate(fh, 0):
                if n > 5_000_000:
                    info["rowcount_note"] = ">5M (scan capped)"
                    break
        info["rowcount_approx"] = max(0, n)
        return info
    except Exception as exc:
        info["confidence"] = "low"
        info["note"] = f"unreadable as tabular ({type(exc).__name__})"
        return info


def _probe_one(token: str, cwd: str | None) -> dict[str, Any]:
    rec: dict[str, Any] = {"ref": token}
    try:
        path = _resolve(token, cwd)
        rec["resolved"] = str(path)
    except Exception:
        rec.update({"exists": False, "confidence": "unknown", "note": "unresolvable path"})
        return rec
    try:
        exists = path.exists()
    except OSError:
        exists = False
    if not exists:
        rec.update({"exists": False, "confidence": "unknown",
                    "note": "does not resolve locally (may be an HPC/cluster/remote path) — not probed"})
        return rec
    rec["exists"] = True
    try:
        if path.is_dir():
            entries = []
            for i, child in enumerate(sorted(path.iterdir())):
                if i >= 12:
                    break
                entries.append(child.name)
            rec.update({"kind": "directory", "confidence": "medium", "entries": entries})
            return rec
        size = path.stat().st_size
        rec["size_bytes"] = size
        suffixes = "".join(path.suffixes).lower()
        ext = path.suffix.lower()
        if ext in _TABULAR_EXT:
            rec.update(_probe_tabular(path, sep="\t" if ext == ".tsv" else ","))
        elif ext == ".parquet":
            rec["kind"] = "tabular"
            rec["confidence"] = "high"
            try:
                import pandas as pd

                head = pd.read_parquet(path) if size < 50_000_000 else None
                if head is not None:
                    rec["columns"] = [str(c) for c in head.columns[:_MAX_COLS]]
                    rec["n_columns"] = int(head.shape[1])
                    rec["rowcount_approx"] = int(head.shape[0])
            except Exception:
                rec["confidence"] = "low"
                rec["note"] = "parquet schema unread"
        elif ext in _GENOMIC_EXT or suffixes.endswith(".vcf.gz"):
            rec.update({"kind": "genomic_variants", "confidence": "medium",
                        "note": "genotype/variant file — genomics workflow is appropriate here"})
        elif ext == ".h5ad":
            rec.update({"kind": "single_cell_h5ad", "confidence": "medium"})
        else:
            rec.update({"kind": ext.lstrip(".") or "file", "confidence": "low"})
    except Exception as exc:
        rec.update({"confidence": "low", "note": f"probe error ({type(exc).__name__})"})
    return rec


def probe_objective(objective: str, *, cwd: str | None = None, max_files: int = _MAX_FILES) -> dict[str, Any]:
    """Probe up to ``max_files`` referenced files. Returns {files: [...], text: <block>}."""
    paths = extract_paths(objective)[:max_files]
    files = [_probe_one(tok, cwd) for tok in paths]
    return {"files": files, "text": format_data_context(files)}


def format_data_context(files: list[dict[str, Any]]) -> str:
    """A bounded, plain-text DataContext block for the planner prompt (<= ~800 tok)."""
    if not files:
        return ""
    lines = ["DataContext (read-only probe of the inputs you referenced — plan around the ACTUAL data shape):"]
    for rec in files:
        ref = rec.get("ref", "?")
        if not rec.get("exists", False):
            lines.append(f"- {ref}: NOT found locally (confidence=unknown). {rec.get('note', '')}")
            continue
        kind = rec.get("kind", "file")
        conf = rec.get("confidence", "low")
        if kind == "tabular":
            cols = rec.get("columns") or []
            shown = ", ".join(cols[:20]) + (" …" if len(cols) > 20 else "")
            rows = rec.get("rowcount_approx")
            rowtxt = rec.get("rowcount_note") or (f"~{rows} rows" if rows is not None else "rows unknown")
            miss = rec.get("missingness_top") or {}
            misstxt = ("; missing: " + ", ".join(f"{c}={v:.0%}" for c, v in miss.items())) if miss else ""
            lines.append(f"- {ref}: TABULAR ({rec.get('n_columns', len(cols))} cols, {rowtxt}; confidence={conf}). "
                         f"columns: {shown}{misstxt}. This is a data TABLE — analyze it as such, not as genotypes.")
        elif kind == "genomic_variants":
            lines.append(f"- {ref}: GENOMIC variant/genotype file (confidence={conf}). A genomics/WGS workflow is appropriate.")
        elif kind == "directory":
            entries = ", ".join(rec.get("entries") or [])
            lines.append(f"- {ref}: DIRECTORY (confidence={conf}). contains: {entries}")
        else:
            lines.append(f"- {ref}: {kind} (confidence={conf}). {rec.get('note', '')}")
    return "\n".join(lines)[:3200]
