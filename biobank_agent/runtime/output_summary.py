"""Aggregate the on-disk outputs a run produced into one user-facing summary.

Each skill returns its own artifact paths, but a run never told the user, in one place, WHERE its
outputs landed — so a completed run could look like it produced nothing. This walks the per-step
result payloads (including nested dicts/lists) and surfaces every path-like value, so a run can close
with an explicit "your outputs are here". It does not rely on a fixed key allowlist: a path returned
under a non-standard key (or buried in a nested structure) is still surfaced rather than dropped.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

# Keys whose value is a path by contract — trusted even when the value is an extension-less directory
# (e.g. a timestamped report dir) that the generic path heuristic would otherwise miss.
_PATH_KEYS = frozenset({
    "report_dir", "report_markdown", "report_html", "report_path", "log_path", "output_path",
    "output_dir", "target_path", "artifact_path", "csv_path", "figure_path", "plot_path",
    "html_path", "pdf_path", "vcf_path", "model_path", "results_path", "saved_to", "path", "dir",
})


def _path_like(value: str) -> bool:
    """A single, space-free token that looks like a filesystem path (absolute, dir-qualified, or extant)."""
    text = value.strip()
    if not text or len(text) > 4096 or any(ws in text for ws in (" ", "\n", "\t")):
        return False
    if text.startswith(("/", "./", "../", "~/")):
        return True
    base = os.path.basename(text.rstrip("/"))
    if "/" in text and "." in base and not base.startswith("."):
        return True
    return os.path.exists(text)


def summarize_output_paths(step_results: Iterable[Any]) -> dict[str, Any]:
    """Collect every output path referenced across a run's step results, deduped and ordered.

    ``step_results`` is any iterable of per-step result dicts (the dicts skills return). Each is walked
    recursively; a string is surfaced if its key is a path-by-contract key OR the value itself looks
    like a path. Returns ``{"count", "paths", "message"}`` — ``message`` is ready to show at run end.
    """
    paths: list[str] = []
    seen: set[str] = set()

    def _add(value: Any, *, trusted: bool) -> None:
        text = str(value or "").strip()
        if not text or text in seen:
            return
        if trusted or _path_like(text):
            seen.add(text)
            paths.append(text)

    def _walk(obj: Any, key_hint: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                _walk(value, str(key).strip().lower())
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _walk(item, key_hint)
        elif isinstance(obj, str):
            _add(obj, trusted=key_hint in _PATH_KEYS)

    for result in step_results or []:
        if isinstance(result, dict):
            _walk(result)

    if paths:
        message = "Outputs from this run were written to:\n" + "\n".join(f"  - {p}" for p in paths)
    else:
        message = "This run did not record any output file paths."
    return {"count": len(paths), "paths": paths, "message": message}
