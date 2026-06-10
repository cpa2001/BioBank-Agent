"""Tests for v3 CLI result-payload rendering (issue #7: surface step output)."""

from __future__ import annotations

import types

from rich.console import Console

from biobank_agent.cli.interactive import InteractiveShell
from biobank_agent.cli.render import render_result_payload


def _render_to_text(renderable) -> str:
    console = Console(width=100, record=True, file=None)
    with console.capture() as cap:
        console.print(renderable)
    return cap.get()


def test_render_payload_none_for_empty():
    assert render_result_payload({}) is None
    assert render_result_payload(None) is None  # type: ignore[arg-type]


def test_render_payload_columns_and_rows():
    payload = render_result_payload({
        "status": "ok",
        "columns": ["trait", "beta", "pval"],
        "rows": [
            {"trait": "E11", "beta": 0.12, "pval": 5e-12},
            {"trait": "I10", "beta": -0.03, "pval": 3e-9},
        ],
        "n_rows": 243,
    })
    assert payload is not None
    text = _render_to_text(payload)
    assert "columns (3)" in text
    assert "trait" in text and "beta" in text and "pval" in text
    assert "E11" in text and "I10" in text
    assert "n_rows=243" in text
    # control keys are not echoed as metrics
    assert "status=ok" not in text


def test_render_payload_surfaces_output_paths():
    payload = render_result_payload({
        "status": "ok",
        "report_path": "/proj/X/reports/r.md",
        "auc": 0.81,
    })
    assert payload is not None
    text = _render_to_text(payload)
    assert "/proj/X/reports/r.md" in text
    assert "auc=0.81" in text


def test_render_payload_row_and_col_caps():
    rows = [{"colA": i, "colB": i, "colC": i, "colZ": i} for i in range(20)]
    payload = render_result_payload({"rows": rows}, max_rows=3, max_cols=2)
    text = _render_to_text(payload)
    assert "showing 3/20" in text          # row cap honored
    assert "colA" in text and "colB" in text
    assert "colZ" not in text              # column cap honored (only first 2 shown)


def test_render_payload_surfaces_exec_stdout():
    # shell_exec / python_exec put the real result in stdout — it must be shown,
    # not dropped in favor of returncode/cwd metrics (issue #7 canonical case).
    payload = render_result_payload({
        "status": "success",
        "returncode": 0,
        "cwd": "/proj/X",
        "stdout": "columns: ['trait', 'beta', 'pval']\nE11 0.12 5e-12\nI10 -0.03 3e-9",
        "stderr": "",
    })
    text = _render_to_text(payload)
    assert "stdout:" in text
    assert "trait" in text and "E11" in text
    assert "returncode=0" in text


def test_render_payload_surfaces_stderr_on_error():
    payload = render_result_payload({
        "status": "error", "returncode": 1, "stdout": "",
        "stderr": "FileNotFoundError: results.csv",
    })
    text = _render_to_text(payload)
    assert "stderr:" in text
    assert "FileNotFoundError" in text


def test_render_payload_tail_block_caps_long_stdout():
    big = "\n".join(f"line {i}" for i in range(100))
    payload = render_result_payload({"stdout": big})
    text = _render_to_text(payload)
    assert "line 99" in text          # tail kept
    assert "line 0" not in text       # head dropped
    assert "earlier line" in text     # truncation noted


def test_render_payload_json_fallback_for_opaque():
    payload = render_result_payload({"nested": {"x": [1, 2, 3]}})
    text = _render_to_text(payload)
    assert "nested" in text


def _step(step_id: str):
    return types.SimpleNamespace(id=step_id, title=f"step {step_id}", status="done")


def test_final_answer_text_picks_last_step_with_text():
    plan = types.SimpleNamespace(steps=[_step("s1"), _step("s2"), _step("s3")])
    step_outputs = {
        "s1": {"tool_results": [], "text": "first"},
        "s2": {"tool_results": [], "text": "second"},
        # s3 produced no closing text -> fall back to the last one that did (s2)
        "s3": {"tool_results": [], "text": ""},
    }
    assert InteractiveShell._final_answer_text(plan, step_outputs) == "second"


def test_final_answer_text_empty_when_no_text():
    plan = types.SimpleNamespace(steps=[_step("s1")])
    assert InteractiveShell._final_answer_text(plan, {"s1": {"tool_results": [], "text": ""}}) == ""
    assert InteractiveShell._final_answer_text(plan, {}) == ""
