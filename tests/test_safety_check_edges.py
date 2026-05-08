"""Focused safety-check coverage for privacy and governance branches."""

from types import SimpleNamespace

from biobank_agent.skills.safety_check import safety_check
from biobank_agent.state import AnalysisRecord


def _record(skill, key_results):
    return AnalysisRecord(
        timestamp="2026-05-08T00:00:00",
        skill=skill,
        args={},
        key_results=key_results,
        figure_paths=[],
    )


def _ctx(records):
    return SimpleNamespace(
        state=SimpleNamespace(records=records),
        settings=SimpleNamespace(biobank_name="UK Biobank"),
    )


def test_safety_check_no_records_reports_no_data():
    result = safety_check(ctx=_ctx([]))

    assert result["overall"] == "NO DATA"
    assert result["issues"] == []


def test_last_scope_only_checks_most_recent_record():
    records = [
        _record("predict", {"predictions": [{"eid": "1"}], "n_patients": 1}),
        _record("prevalence", {"n_cases": 100, "n_total": 5000}),
    ]

    result = safety_check(scope="last", ctx=_ctx(records))

    assert result["overall"].startswith("PASS")
    assert result["n_issues"] == 0


def test_session_scope_collects_and_sorts_all_safety_issues():
    records = [
        _record("predict", {"predictions": [{"eid": "1"}, {"eid": "2"}], "n_patients": 2}),
        _record("cohort_summary", {"n_cases": 4, "n_total": 8}),
        _record("biomarker_dist", {"n_patients": 7}),
        _record("cohort_card", {"n_cases": 3, "banks": ["ukb", "hpp"]}),
        _record("world_model_audit", {"safety_status": "FAIL"}),
    ]

    result = safety_check(scope="session", k=5, ctx=_ctx(records))

    assert result["overall"].startswith("BLOCK")
    assert result["n_issues"] == 9
    severities = [issue["severity"] for issue in result["issues"]]
    assert severities == sorted(severities, key={"CRITICAL": 0, "WARNING": 1, "INFO": 2}.get)
    types = {issue["type"] for issue in result["issues"]}
    assert {
        "k_anonymity_violation",
        "min_cell_count",
        "reidentification_risk",
        "irb_reminder",
        "cross_cohort_release_review",
        "blocked_world_model_claim",
    }.issubset(types)
    assert result["k_threshold"] == 5
    assert result["scope"] == "session"


def test_warning_only_safety_result_requires_review():
    records = [_record("biomarker_dist", {"n_patients": 9})]

    result = safety_check(scope="session", ctx=_ctx(records))

    assert result["overall"].startswith("REVIEW")
    assert result["issues"][0]["type"] == "reidentification_risk"


def test_zero_counts_do_not_trigger_small_count_rules():
    records = [
        _record("cohort_summary", {"n_cases": 0, "n_total": 0}),
        _record("cohort_summary", {"n_cases": 0, "n_total": 0}),
        _record("cohort_card", {"n_cases": 0, "banks": ["ukb"]}),
        _record("world_model_audit", {"safety_status": "PASS"}),
    ]

    result = safety_check(scope="session", ctx=_ctx(records))

    assert result["overall"].startswith("PASS")
    assert result["issues"] == []


def test_empty_prediction_list_does_not_trigger_k_anonymity():
    records = [_record("predict", {"predictions": []})]

    result = safety_check(scope="session", ctx=_ctx(records))

    assert result["overall"].startswith("PASS")
    assert result["issues"] == []
