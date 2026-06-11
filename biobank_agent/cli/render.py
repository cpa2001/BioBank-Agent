"""Compact, reusable renderers for tool/step result payloads (v3 CLI).

The interactive shell executes plan steps autonomously via ``run_turn`` and used
to surface only a done/failed status — the real output a step produced (column
names, row previews, output file paths, key metrics) was captured in the turn's
``tool_results`` but never shown (issue #7). These helpers turn a tool-result
dict into a bounded Rich renderable so the user can actually see what happened
without flooding the terminal. Pure Rich + json, so they are unit-testable in
isolation from the (large) interactive module.
"""

from __future__ import annotations

import json
from typing import Any

from rich import box
from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

# Result-dict keys that carry an output file path worth surfacing — printing the
# absolute path is the whole point of issue #7 ("where did my results go?").
_PATH_KEYS = (
    "path", "output_path", "out_path", "report_path", "figure", "figures",
    "artifact", "artifacts", "file", "files", "saved_to", "output", "outputs",
)
_COLUMN_KEYS = ("columns", "column_names", "colnames", "fields", "header")
_ROW_KEYS = ("rows", "head", "preview", "sample", "records", "data")
# Captured console output (shell_exec / python_exec put the real result here).
_STDOUT_KEYS = ("stdout", "stdout_tail", "output_text", "logs", "log")
_STDERR_KEYS = ("stderr", "stderr_tail")
# Keys that are control/noise — never worth echoing as a metric.
_SKIP_KEYS = ("status", "ok", "awaiting_user", "error")


def _short(value: str, limit: int = 40) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _tail_block(value: str, *, max_lines: int, max_chars: int) -> str:
    """Last ``max_lines`` lines of ``value``, bounded to ``max_chars``."""
    text = str(value).rstrip("\n")
    lines = text.split("\n")
    clipped = lines[-max_lines:]
    out = "\n".join(clipped)
    if len(out) > max_chars:
        out = "…" + out[-max_chars:]
    if len(lines) > max_lines:
        out = f"… (+{len(lines) - max_lines} earlier line(s))\n" + out
    return out


def _coerce_rows(value: Any) -> list[dict] | None:
    """Normalize a rows-like value into a list of dicts, or None if it isn't one."""
    if isinstance(value, list) and value:
        if all(isinstance(row, dict) for row in value):
            return value
        if all(isinstance(row, (list, tuple)) for row in value):
            return [{f"c{i + 1}": cell for i, cell in enumerate(row)} for row in value]
    return None


def _rows_table(rows: list[dict], *, max_rows: int, max_cols: int) -> Table:
    columns: list[str] = []
    for row in rows:
        for key in row.keys():
            if str(key) not in columns:
                columns.append(str(key))
    columns = columns[:max_cols]
    table = Table(box=box.SIMPLE, show_edge=False, pad_edge=False)
    for col in columns:
        table.add_column(col, overflow="fold", max_width=24)
    for row in rows[:max_rows]:
        table.add_row(*[_short(row.get(col, "")) for col in columns])
    return table


def render_result_payload(
    result: dict[str, Any],
    *,
    max_rows: int = 5,
    max_cols: int = 8,
    max_chars: int = 800,
) -> RenderableType | None:
    """Return a compact Rich renderable previewing ``result``, or None if empty.

    Recognizes (in order) column lists, row/head previews, output file paths and
    short scalar metrics; falls back to a truncated JSON dump. Output is bounded
    by ``max_rows`` / ``max_cols`` / ``max_chars`` so it never floods the screen.
    """
    if not isinstance(result, dict) or not result:
        return None
    pieces: list[RenderableType] = []

    # 1) Column names.
    for key in _COLUMN_KEYS:
        value = result.get(key)
        if isinstance(value, (list, tuple)) and value:
            shown = ", ".join(str(c) for c in list(value)[:30])
            extra = "" if len(value) <= 30 else f" … (+{len(value) - 30} more)"
            pieces.append(Text(f"columns ({len(value)}): {shown}{extra}", style="cyan"))
            break

    # 2) Row / head preview as a small table.
    for key in _ROW_KEYS:
        rows = _coerce_rows(result.get(key))
        if rows:
            pieces.append(Text(f"{key} (showing {min(len(rows), max_rows)}/{len(rows)}):", style="dim"))
            pieces.append(_rows_table(rows, max_rows=max_rows, max_cols=max_cols))
            break

    # 2b) Captured console output — for shell_exec/python_exec this IS the result
    # (e.g. the printed column names + first rows). Show a bounded tail.
    for key in _STDOUT_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            block = _tail_block(value, max_lines=20, max_chars=max_chars)
            pieces.append(Text(f"{key}:", style="dim"))
            pieces.append(Text(block, style="white"))
            break
    for key in _STDERR_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            block = _tail_block(value, max_lines=8, max_chars=400)
            pieces.append(Text(f"{key}:", style="yellow"))
            pieces.append(Text(block, style="yellow"))
            break

    # 3) Output file paths (so the user knows where results landed).
    paths: list[str] = []
    for key in _PATH_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            paths.append(value.strip())
        elif isinstance(value, (list, tuple)):
            paths.extend(str(v).strip() for v in value if str(v).strip())
    paths = list(dict.fromkeys(paths))  # dedupe, preserve order
    if paths:
        ptext = Text("output:", style="green")
        for path in paths[:6]:
            ptext.append(f"\n  {path}", style="green")
        pieces.append(ptext)

    # 4) Short scalar metrics not already shown.
    shown_keys = (
        set(_COLUMN_KEYS) | set(_ROW_KEYS) | set(_PATH_KEYS)
        | set(_STDOUT_KEYS) | set(_STDERR_KEYS) | set(_SKIP_KEYS)
    )
    kv: list[str] = []
    for key, value in result.items():
        if key in shown_keys:
            continue
        if isinstance(value, bool) or isinstance(value, (str, int, float)):
            text = f"{key}={value}"
            if len(text) <= 80:
                kv.append(text)
    if kv:
        pieces.append(Text("  ".join(kv[:10]), style="white"))

    if pieces:
        return Group(*pieces)

    # 5) Fallback: truncated JSON dump.
    try:
        dump = json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        dump = str(result)
    if len(dump) > max_chars:
        dump = dump[:max_chars] + " …(truncated)"
    return Text(dump, style="dim")


# ── Run-tree rendering (M15 /trace) ──────────────────────────────────────────
# Duck-typed over runtime.run_tree.RunNode (name/kind/status/duration/children) so
# this stays a pure Rich helper with no runtime import — same ethos as the payload
# renderer above.

_STATUS_GLYPH = {"ok": "[green]✓[/]", "error": "[red]✗[/]", "running": "[yellow]…[/]"}


def _span_label(node: Any) -> str:
    glyph = _STATUS_GLYPH.get(str(getattr(node, "status", "") or ""), "[dim]·[/]")
    name = str(getattr(node, "name", "") or "?")
    kind = str(getattr(node, "kind", "") or "")
    duration = getattr(node, "duration", None)
    dur = f" [dim]{duration:.2f}s[/]" if isinstance(duration, (int, float)) else ""
    return f"{glyph} {name} [dim]({kind})[/]{dur}"


def render_run_tree(root: Any) -> Tree:
    """Render a run tree (runtime.run_tree.RunNode) as a Rich ``Tree``.

    Each node shows a status glyph, span name, kind and wall-clock duration. Pure and
    bounded by the tree the caller passes in, so it is unit-testable in isolation."""
    tree = Tree(_span_label(root))

    def _attach(parent: Tree, node: Any) -> None:
        branch = parent.add(_span_label(node))
        for child in getattr(node, "children", None) or []:
            _attach(branch, child)

    for child in getattr(root, "children", None) or []:
        _attach(tree, child)
    return tree
