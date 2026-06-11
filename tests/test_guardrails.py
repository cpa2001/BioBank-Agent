"""Tests for delegation and temporal guardrails."""

from biobank_agent.guardrails import DelegationGuardrails, TemporalRule, TemporalSafetyChecker


def test_history_inheritance_warns_on_multiple_user_turns():
    guardrails = DelegationGuardrails()
    violations = guardrails.check_no_history_inheritance([
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ])

    assert len(violations) == 1
    assert violations[0].rule == "no_history_inheritance"
    assert violations[0].severity == "WARN"
    assert guardrails.check_no_history_inheritance([{"role": "user", "content": "self-contained"}]) == []


def test_vague_spec_requires_concrete_files():
    guardrails = DelegationGuardrails()
    violations = guardrails.check_no_vague_spec(
        "Please figure out the best way to implement something useful for the report quality pipeline."
    )

    assert {v.rule for v in violations} == {"no_vague_spec"}
    assert len(violations) >= 1


def test_specific_spec_passes_vagueness_check():
    guardrails = DelegationGuardrails()
    violations = guardrails.check_no_vague_spec(
        "1. Update biobank_agent/skills/report.py to add references.\n"
        "2. Add tests/test_report_v2.py cases.\n"
        "3. Verify with pytest tests/test_report_v2.py -q."
    )

    assert violations == []


def test_write_overlap_blocks_parallel_workers():
    guardrails = DelegationGuardrails()
    violations = guardrails.check_no_write_overlap([
        {"owned_files": ["biobank_agent/agent.py", "tests/test_agent.py"]},
        {"owned_files": ["biobank_agent/agent.py"]},
    ])

    assert len(violations) == 1
    assert violations[0].severity == "BLOCK"
    assert "biobank_agent/agent.py" in violations[0].description
    assert guardrails.check_no_write_overlap([
        {"owned_files": ["a.py"]},
        {"owned_files": ["b.py"]},
    ]) == []


def test_decision_complete_checks_file_steps_and_verification():
    guardrails = DelegationGuardrails()
    vague = guardrails.check_decision_complete("Fix the agent")
    complete = guardrails.check_decision_complete(
        "1. Edit biobank_agent/agent.py.\n"
        "2. Edit tests/test_agent_multiagent.py.\n"
        "3. Verify with pytest tests/test_agent_multiagent.py -q."
    )

    assert {v.rule for v in vague} == {"decision_complete"}
    assert complete == []


def test_check_all_composes_guardrails():
    violations = DelegationGuardrails().check_all(
        role="worker",
        messages=[{"role": "user"}, {"role": "user"}],
        spec="figure out implementation",
        workers=[{"owned_files": ["a.py"]}, {"owned_files": ["a.py"]}],
    )

    assert {v.rule for v in violations} >= {
        "no_history_inheritance",
        "no_vague_spec",
        "no_write_overlap",
    }
    assert DelegationGuardrails().check_all(role="worker") == []


def test_temporal_checker_warns_missing_required_predecessor():
    checker = TemporalSafetyChecker([
        TemporalRule("safety_before_report", "safety_check", "report", reason="required"),
    ])

    violations = checker.check_plan([{"id": "s1", "skill": "report"}])

    assert len(violations) == 1
    assert "safety_check" in violations[0].description


def test_temporal_checker_warns_wrong_order_and_suggests_reorder():
    checker = TemporalSafetyChecker([
        TemporalRule("prevalence_before_train", "prevalence", "train_model"),
    ])
    plan = [
        {"id": "s1", "skill": "train_model"},
        {"id": "s2", "skill": "prevalence"},
    ]

    violations = checker.check_plan(plan)
    reordered = checker.suggest_reorder(plan)

    assert len(violations) == 1
    assert [s["skill"] for s in reordered] == ["prevalence", "train_model"]


def test_temporal_checker_allows_safe_plan_and_ignores_duplicates():
    checker = TemporalSafetyChecker([
        TemporalRule("prevalence_before_train", "prevalence", "train_model"),
    ])
    plan = [
        {"id": "s1", "skill": "prevalence"},
        {"id": "s2", "skill": "prevalence"},
        {"id": "s3", "skill": "train_model"},
    ]

    assert checker.check_plan(plan) == []


def test_temporal_checker_ignores_rules_without_after_and_noop_reorder():
    checker = TemporalSafetyChecker([
        TemporalRule("a_before_b", "a", "b"),
        TemporalRule("c_before_b", "c", "b"),
    ])
    plan = [{"id": "s1", "skill": "a"}]

    assert checker.check_plan(plan) == []
    assert checker.suggest_reorder(plan) == plan
