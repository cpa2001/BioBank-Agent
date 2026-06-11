"""Aggregate the on-disk outputs a run produced into one user-facing summary.

Each skill returns its own artifact paths, but a run never told the user, in one place, WHERE its
outputs landed — so a completed run could look like it produced nothing. This scans the per-step
result payloads for path-like fields and returns a deduped, ordered list, so a run can close with an
explicit "your outputs are here" instead of leaving the user to hunt through logs.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

# Scalar result keys that carry a single path.
_PATH_KEYS = (
    "report_dir", "report_markdown", "report_html", "report_path", "log_path", "output_path",
    "output_dir", "target_path", "artifact_path", "csv_path", "figure_path", "plot_path", "saved_to",
)
# Result keys that carry a list of paths.
_LIST_KEYS = ("figure_artifacts", "reproducibility_artifacts", "artifacts", "output_paths", "outputs")


def _looks_like_path(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    # A real path has a separator or a filename with an extension; reject bare words/ids.
    return ("/" in text) or (os.sep in text) or ("." in os.path.basename(text) and len(text) > 3)


def summarize_output_paths(step_results: Iterable[Any]) -> dict[str, Any]:
    """Collect every output path referenced across a run's step results, deduped and ordered.

    ``step_results`` is any iterable of per-step result dicts (the dicts skills return). Non-dict
    items are skipped. Returns ``{"count", "paths", "message"}`` — ``message`` is ready to show the
    user at run end.
    """
    paths: list[str] = []
    seen: set[str] = set()

    def _add(candidate: Any) -> None:
        text = str(candidate or "").strip()
        if text and text not in seen and _looks_like_path(text):
            seen.add(text)
            paths.append(text)

    for result in step_results or []:
        if not isinstance(result, dict):
            continue
        for key in _PATH_KEYS:
            if result.get(key):
                _add(result[key])
        for key in _LIST_KEYS:
            value = result.get(key)
            if isinstance(value, (list, tuple)):
                for item in value:
                    _add(item)

    if paths:
        message = "Outputs from this run were written to:\n" + "\n".join(f"  - {p}" for p in paths)
    else:
        message = "This run did not record any output file paths."
    return {"count": len(paths), "paths": paths, "message": message}
