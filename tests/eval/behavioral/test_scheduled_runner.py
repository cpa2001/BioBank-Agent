"""Tests for scheduled behavioral/evolution quality gates."""

from __future__ import annotations

import json
from pathlib import Path

from biobank_agent.eval.scheduled import run_scheduled_quality_gates


def test_scheduled_quality_gates_write_manifest_and_histories(tmp_path: Path):
    history = tmp_path / "failures.jsonl"
    history.write_text(
        "\n".join(
            json.dumps({
                "skill": "generate_report",
                "args": {"format": "dual"},
                "success": False,
                "error": "report prerequisites missing: no statistical_review",
                "elapsed_s": 0.1,
            })
            for _ in range(3)
        ),
        encoding="utf-8",
    )

    result = run_scheduled_quality_gates(
        output_dir=tmp_path / "scheduled",
        behavioral_output_dir=tmp_path / "behavioral",
        evolution_output_dir=tmp_path / "evolution",
        behavioral_policies=("always",),
        evolution_history=history,
    )

    assert result.status == "NEEDS_REVIEW"
    assert result.behavioral_status == "PASS"
    assert result.evolution_status == "NEEDS_REVIEW"
    assert "repeated tool failures need review" in result.action_required
    latest = json.loads(Path(result.artifacts["latest_json"]).read_text(encoding="utf-8"))
    assert latest["status"] == "NEEDS_REVIEW"
    assert latest["behavioral_trend"]["latest_status"] == "PASS"
    assert latest["behavioral_trend"]["mean_pass_rate"] == 1.0
    assert latest["evolution_trend"]["latest_status"] == "NEEDS_REVIEW"
    assert Path(latest["behavioral_artifacts"]["history_jsonl"]).exists()
    assert Path(latest["evolution_artifacts"]["history_jsonl"]).exists()
    assert "Scheduled Quality Gates" in Path(result.artifacts["run_md"]).read_text(encoding="utf-8")


def test_scheduled_quality_gates_can_fail_on_evolution_patterns(tmp_path: Path):
    history = tmp_path / "failures.jsonl"
    history.write_text(
        "\n".join(
            json.dumps({"skill": "generate_report", "success": False, "error": "same failure"})
            for _ in range(3)
        ),
        encoding="utf-8",
    )

    result = run_scheduled_quality_gates(
        output_dir=tmp_path / "scheduled",
        behavioral_output_dir=tmp_path / "behavioral",
        evolution_output_dir=tmp_path / "evolution",
        behavioral_policies=("always",),
        evolution_history=history,
        fail_on_evolution_patterns=True,
    )

    assert result.status == "FAIL"
