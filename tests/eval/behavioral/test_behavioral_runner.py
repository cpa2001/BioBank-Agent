"""Tests for the behavioral eval pass-rate history runner."""

from __future__ import annotations

import json
from pathlib import Path

from biobank_agent.eval.behavioral import load_behavioral_cases, run_behavioral_eval


CASE_DIR = Path(__file__).parent


def test_behavioral_runner_writes_latest_run_and_history(tmp_path: Path):
    result = run_behavioral_eval(
        case_dir=CASE_DIR,
        output_dir=tmp_path,
        policies=("always", "usually"),
    )

    assert result.status == "PASS"
    assert result.n_passed == result.n_total
    assert result.n_total >= 4

    run_path = Path(result.artifacts["run_json"])
    latest_path = Path(result.artifacts["latest_json"])
    history_path = Path(result.artifacts["history_jsonl"])
    assert run_path.exists()
    assert latest_path.exists()
    assert history_path.exists()

    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    assert latest["status"] == "PASS"
    assert latest["n_passed"] == latest["n_total"]
    assert latest["artifacts"]["run_json"] == str(run_path)
    assert latest["artifacts"]["latest_json"] == str(latest_path)
    assert latest["artifacts"]["history_jsonl"] == str(history_path)
    assert {case["policy"] for case in latest["cases"]} == {"always", "usually"}

    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines()]
    assert history[-1]["status"] == "PASS"
    assert history[-1]["run_path"] == str(run_path)


def test_behavioral_runner_can_filter_policy(tmp_path: Path):
    cases = load_behavioral_cases(CASE_DIR, policies=("always",))
    assert cases
    assert {case["_policy"] for case in cases} == {"always"}

    result = run_behavioral_eval(
        case_dir=CASE_DIR,
        output_dir=tmp_path,
        policies=("always",),
    )

    assert result.status == "PASS"
    assert {case.policy for case in result.cases} == {"always"}
