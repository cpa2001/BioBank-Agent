"""Edge coverage for session state, provenance, and summaries."""

import pandas as pd

from biobank_agent.state import AnalysisRecord, Provenance, SessionState, TokenUsage


def _record(i):
    return AnalysisRecord(
        timestamp=f"2026-05-08T00:{i:02d}:00",
        skill=f"skill_{i}",
        args={"x": i},
        key_results={"n": i},
        figure_paths=[],
    )


def test_token_usage_total_and_negative_values_are_clamped():
    usage = TokenUsage(prompt_tokens=2, completion_tokens=3)

    usage.update({"prompt_tokens": -10, "completion_tokens": "4"})

    assert usage.prompt_tokens == 2
    assert usage.completion_tokens == 7
    assert usage.total_tokens == 9


def test_provenance_hash_and_id_are_deterministic():
    first_hash = Provenance.compute_hash({"b": 2, "a": 1})
    second_hash = Provenance.compute_hash({"a": 1, "b": 2})
    first_id = Provenance.make_id("prevalence", {"code": "E11"}, "2026-05-08")
    second_id = Provenance.make_id("prevalence", {"code": "E11"}, "2026-05-08")

    assert first_hash == second_hash
    assert len(first_hash) == 12
    assert first_id == second_id
    assert len(first_id) == 8


def test_context_summary_reports_empty_session():
    assert SessionState().context_summary() == "No analyses performed yet."


def test_context_summary_with_records_only_omits_optional_sections():
    state = SessionState()
    state.add_record(_record(1))

    summary = state.context_summary()

    assert "skill_1(x=1)" in summary
    assert "Active cohorts:" not in summary
    assert "Trained models:" not in summary


def test_context_summary_includes_recent_records_cohorts_and_models():
    state = SessionState()
    state.cohorts = {
        "diabetes": pd.DataFrame({"eid": [1, 2, 3], "label": [1, 0, 1]}),
        "unknown": pd.DataFrame({"eid": [4, 5]}),
    }
    state.models = {"E11:xgboost": object()}
    for i in range(25):
        state.add_record(_record(i))

    summary = state.context_summary()

    assert "Session history (25 steps, showing last 20)" in summary
    assert "skill_4(" not in summary
    assert "skill_5(x=5)" in summary
    assert "skill_24(x=24)" in summary
    assert "diabetes: 3 subjects (2 cases)" in summary
    assert "unknown: 2 subjects (? cases)" in summary
    assert "Trained models: E11:xgboost" in summary
