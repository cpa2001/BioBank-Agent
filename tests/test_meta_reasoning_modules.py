"""Tests for verdict, reflexion, tool-learning, and critical-thinking modules."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from biobank_agent import verdict
from biobank_agent.reflexion import Correction, ReflectionResult, ReflexionEngine
from biobank_agent.skills import critical_thinking as critical_mod
from biobank_agent.tool_learner import ToolLearner


def test_verdict_summary_report_and_status_rules():
    checks = [
        verdict.Check("schema", command="pytest", output="ok", passed=True),
        verdict.Check("bounds", command="custom", output="bad value", passed=False),
    ]
    issues = [
        verdict.Issue(
            file="skills/example.py",
            line=12,
            severity=verdict.Severity.BLOCKER,
            description="negative count",
            suggested_fix="Clamp counts at zero",
        ),
        verdict.Issue(file="general", severity=verdict.Severity.WARNING, description="weak evidence"),
    ]

    result = verdict.build_verdict(checks, issues, rationale="Audited")

    assert result.status is verdict.VerdictStatus.FAIL
    assert result.n_blockers == 1
    assert result.n_warnings == 1
    assert result.summary() == "**VERDICT: FAIL** — (1/2 checks passed) — 1 blockers — 1 warnings"
    report = result.report()
    assert "## Verdict: FAIL" in report
    assert "Command: `custom`" in report
    assert "[BLOCKER] skills/example.py:12" in report
    assert "Fix: Clamp counts at zero" in report

    assert verdict.determine_verdict([verdict.Check("ok")], []) is verdict.VerdictStatus.PASS


def test_verdict_summary_report_without_optional_sections():
    clean = verdict.VerdictResult(status=verdict.VerdictStatus.PASS)
    assert clean.summary() == "**VERDICT: PASS**"
    assert clean.report() == "## Verdict: PASS\n"

    warning_only = verdict.VerdictResult(
        status=verdict.VerdictStatus.PARTIAL,
        checks=[
            verdict.Check("passed", passed=True),
            verdict.Check("failed_no_output", passed=False),
        ],
        issues=[verdict.Issue(file="general", severity=verdict.Severity.WARNING, description="warn")],
    )

    summary = warning_only.summary()
    report = warning_only.report()

    assert "blockers" not in summary
    assert "1 warnings" in summary
    assert "Command:" not in report
    assert "Output:" not in report
    assert "[WARNING] general" in report


def test_verdict_engine_skill_result_checks_and_formal_fallback(monkeypatch):
    engine = verdict.VerdictEngine(llm=SimpleNamespace())

    class FakeFormalVerifier:
        def verify_skill_output(self, skill_name, result):
            return verdict.VerdictResult(
                status=verdict.VerdictStatus.PASS,
                checks=[verdict.Check("formal_bounds", passed=True)],
                issues=[],
            )

    engine._formal_verifier = FakeFormalVerifier()
    clean = engine.verify_skill_result("summary", {}, {"n_cases": 12, "auc_mean": 0.8})

    assert clean.status is verdict.VerdictStatus.PASS
    assert [c.name for c in clean.checks] == ["formal_bounds", "no_error", "non_empty_result"]

    engine._formal_verifier = SimpleNamespace(
        verify_skill_output=lambda skill_name, result: (_ for _ in ()).throw(RuntimeError("z3 unavailable"))
    )
    errored = engine.verify_skill_result("model", {}, {"error": "failed", "n_cases": -1, "auc_mean": 101})
    not_dict = engine.verify_skill_result("model", {}, ["bad"])
    empty = engine.verify_skill_result("model", {}, {})

    assert errored.status is verdict.VerdictStatus.FAIL
    assert any(i.severity is verdict.Severity.BLOCKER for i in errored.issues)
    assert any(i.severity is verdict.Severity.WARNING for i in errored.issues)
    assert not_dict.status is verdict.VerdictStatus.FAIL
    assert any("instead of dict" in i.description for i in not_dict.issues)
    assert empty.status is verdict.VerdictStatus.FAIL
    assert any(c.name == "non_empty_result" and not c.passed for c in empty.checks)


def test_verdict_engine_lazy_formal_verifier_and_mesh_warning(monkeypatch):
    engine = verdict.VerdictEngine(llm=SimpleNamespace())
    formal = engine.formal_verifier
    assert engine.formal_verifier is formal

    engine._formal_verifier = SimpleNamespace(
        verify_skill_output=lambda skill_name, result: verdict.VerdictResult(
            status=verdict.VerdictStatus.PASS,
            checks=[],
            issues=[],
        )
    )
    engine._mesh = SimpleNamespace(
        verify_result=lambda result, skip_sample_sizes=True: SimpleNamespace(
            checks=[
                SimpleNamespace(
                    passed=False,
                    severity="warning",
                    verifier="url_resolver",
                    claim="Reference resolves",
                    detail="HTTP 404",
                )
            ]
        )
    )

    result = engine.verify_skill_result("report", {}, {"ok": True})

    assert result.status is verdict.VerdictStatus.PASS
    assert result.n_warnings == 1
    assert "[VerifierMesh/url_resolver]" in result.issues[0].description

    engine._mesh = SimpleNamespace(
        verify_result=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("mesh unavailable"))
    )
    skipped = engine.verify_skill_result("report", {}, {"ok": True})
    assert skipped.status is verdict.VerdictStatus.PASS

    engine._mesh = SimpleNamespace(
        verify_result=lambda result, skip_sample_sizes=True: SimpleNamespace(
            checks=[
                SimpleNamespace(
                    passed=False,
                    severity="info",
                    verifier="metadata",
                    claim="Optional note",
                    detail="ignored",
                )
            ]
        )
    )
    informational = engine.verify_skill_result("report", {}, {"ok": True})
    assert informational.status is verdict.VerdictStatus.PASS
    assert informational.issues == []


class FakeMemory:
    def get_error_suggestions(self, error_type, skill_name):
        return ["reduce folds for small cohorts"]


class FakeLLM:
    def __init__(self, text=None, error=None):
        self.text = text
        self.error = error

    def chat(self, messages, max_tokens=512):
        if self.error:
            raise self.error
        return SimpleNamespace(text=self.text)


def test_reflexion_result_retry_rules_and_fast_paths():
    result = ReflectionResult(
        root_cause="small cohort",
        error_category="data",
        retry_recommended=True,
        corrections=[Correction("n_folds", 5, 3, "smaller CV")],
    )
    assert result.corrected_args == {"n_folds": 3}

    engine = ReflexionEngine(FakeLLM("{}"), memory=FakeMemory())
    assert engine.should_retry(ValueError("bad input")) is False
    assert engine.should_retry(RuntimeError("file not found")) is False
    assert engine.should_retry(RuntimeError("temporary backend issue")) is True

    insufficient = engine.reflect(
        "train_model",
        {"n_folds": 5, "controls_ratio": 4},
        RuntimeError("too few cases"),
    )
    timeout = engine.reflect(
        "phewas",
        {"top_n": 20, "sample_size": 1000, "n_folds": 4},
        TimeoutError("timed out"),
    )

    assert insufficient.known_fix_used is True
    assert insufficient.corrected_args == {"n_folds": 4, "controls_ratio": 0}
    assert timeout.corrected_args == {"top_n": 10, "n_folds": 2}
    assert timeout.error_category == "external"

    prompt = engine._build_reflection_prompt(
        "train_model",
        {"n_folds": 5},
        "RuntimeError",
        "backend failed",
        ["Use fewer folds", "Check cohort size"],
        "recent context",
    )
    assert "Previously successful fixes" in prompt
    assert "Use fewer folds" in prompt
    assert "recent context" in prompt


def test_reflexion_llm_parse_code_fence_and_fallbacks():
    llm_text = """```json
{
  "root_cause": "sample size too small",
  "category": "parameter",
  "retry": true,
  "confidence": 0.77,
  "corrections": [
    {"param": "n_folds", "new_value": 3, "reason": "less split pressure"},
    {"param": "not_present", "new_value": 1, "reason": "ignored"}
  ]
}
```"""
    engine = ReflexionEngine(FakeLLM(llm_text), max_reflection_tokens=128)
    parsed = engine.reflect("train_model", {"n_folds": 5}, RuntimeError("backend failed"), context="E11")

    assert parsed.retry_recommended is True
    assert parsed.confidence == pytest.approx(0.77)
    assert parsed.corrected_args == {"n_folds": 3}
    assert parsed.summary == "Reflexion: sample size too small"

    invalid = ReflexionEngine(FakeLLM("not json")).reflect("skill", {"top_n": 9}, RuntimeError("bad"))
    failed_llm = ReflexionEngine(FakeLLM(error=RuntimeError("llm down"))).reflect(
        "skill",
        {"top_n": 9},
        RuntimeError("bad"),
    )

    assert invalid.corrected_args == {"top_n": 4}
    assert failed_llm.corrected_args == {"top_n": 4}
    assert failed_llm.confidence == 0.3


def test_reflexion_branch_variants_without_llm_roundtrip():
    engine = ReflexionEngine(FakeLLM("{}"))

    fenced_without_json = engine._parse_reflection(
        "```text\nnot json\n```",
        {"top_n": 8},
        "RuntimeError",
        "bad input",
    )
    assert fenced_without_json.corrected_args == {"top_n": 4}

    insufficient_controls_ratio_one = engine._fast_path_fix(
        "train_model",
        {"n_folds": 2, "controls_ratio": 1},
        "RuntimeError",
        "insufficient cases",
    )
    assert insufficient_controls_ratio_one.corrected_args == {"controls_ratio": 0}

    insufficient_controls_only = engine._fast_path_fix(
        "train_model",
        {"controls_ratio": 3},
        "RuntimeError",
        "too few cases",
    )
    assert insufficient_controls_only.corrected_args == {"controls_ratio": 0}

    memory_without_integer_args = engine._fast_path_fix(
        "phewas",
        {"top_n": "many", "sample_size": None},
        "MemoryError",
        "out of memory",
    )
    assert memory_without_integer_args is None


def test_tool_learner_records_stats_suggestions_gaps_and_summary():
    learner = ToolLearner()
    learner.record("train_model", {"n_folds": 5, "model_type": "xgb"}, {"ok": True}, elapsed_s=2.0)
    learner.record("train_model", {"n_folds": 3, "model_type": "xgb"}, {"ok": True}, elapsed_s=4.0)
    learner.record("train_model", {"n_folds": 3, "model_type": "lgbm"}, {"ok": True}, elapsed_s=6.0)
    learner.record("train_model", {"n_folds": 10}, {"error": "too few cases"}, elapsed_s=1.0)
    learner.record("phewas", {"field_id": "30740"}, {"error": "missing field"}, elapsed_s=0.2)
    learner.record("phewas", {"field_id": "30750"}, {"error": "missing field"}, elapsed_s=0.3)
    learner.record("phewas", {"field_id": "30870"}, {"error": "timeout"}, elapsed_s=0.4)

    stats = learner.get_stats("train_model")
    suggestions = learner.suggest_params("train_model")
    all_stats = learner.all_stats()

    assert stats.total_calls == 4
    assert stats.successes == 3
    assert stats.failures == 1
    assert stats.success_rate == pytest.approx(0.75)
    assert stats.avg_elapsed_s == pytest.approx(3.25)
    assert {s.param: s.suggested_value for s in suggestions} == {"n_folds": 3, "model_type": "xgb"}
    assert all_stats[0].skill_name == "train_model"
    assert learner.suggest_params("unknown") == []
    assert learner.identify_skill_gap(["Need methylation analysis", "methylation QC failed"]) == (
        "Repeated query terms not served by existing skills: methylation"
    )

    summary = learner.summary()
    assert "train_model: 4 calls, 75% success" in summary
    assert "Low performers: phewas" in summary


def test_tool_learner_history_trims_and_empty_gap_summary():
    learner = ToolLearner()
    assert learner.summary() == ""
    assert learner.identify_skill_gap([]) is None
    assert learner.identify_skill_gap(["one unrelated query"]) is None

    for _ in range(3):
        learner.record("ctx_skill", {"ctx": object(), "mode": "fast"}, {"ok": True}, elapsed_s=0.0)
    learner._stats["ctx_skill"].common_args["empty"] = []
    suggestions = learner.suggest_params("ctx_skill")
    assert {s.param for s in suggestions} == {"mode"}

    unique = ToolLearner()
    unique.record("unique_skill", {"mode": "a"}, {"ok": True}, elapsed_s=0.0)
    unique.record("unique_skill", {"mode": "b"}, {"ok": True}, elapsed_s=0.0)
    unique.record("unique_skill", {"mode": "c"}, {"ok": True}, elapsed_s=0.0)
    assert unique.suggest_params("unique_skill") == []
    assert "Low performers" not in unique.summary()

    for i in range(205):
        learner.record("skill", {"top_n": i}, {"ok": True}, elapsed_s=0.0)

    assert len(learner._history) == 200
    assert learner._history[0]["args"]["top_n"] == 5


def test_critical_thinking_defaults_context_and_biobank_specific_flags():
    default_flags = critical_mod._get_biobank_red_flags()
    assert default_flags[0]["flag"] == "Healthy volunteer bias"
    assert len(default_flags) == 10

    ctx = SimpleNamespace(
        settings=SimpleNamespace(
            biobank_name="UK Biobank",
            biobank_abbreviation="UKB",
            biobank_description="middle-aged UK participants",
            biobank_caveats="European ancestry enrichment",
        )
    )

    framework = critical_mod._build_evaluation_framework(
        "HbA1c predicts diabetes",
        "prospective cohort",
    )
    no_context_framework = critical_mod._build_evaluation_framework(
        "HbA1c predicts diabetes",
        "",
    )
    flags_section = critical_mod._build_red_flags_section(ctx)
    result = critical_mod.critical_thinking(
        "HbA1c predicts diabetes",
        context="prospective cohort",
        ctx=ctx,
    )

    assert "## Additional Context" in framework
    assert "## Additional Context" not in no_context_framework
    assert "Step 7: Proportionate Conclusion" in framework
    assert "UK Biobank participants" in flags_section
    assert result["claim"] == "HbA1c predicts diabetes"
    assert result["n_steps"] == 7
    assert result["n_red_flags"] == 10
    assert result["red_flags_checklist"][0] == {"flag": "Healthy volunteer bias", "severity": "high"}
    assert all(step["status"] == "to_evaluate" for step in result["steps"])
