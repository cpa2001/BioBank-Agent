"""Predefined benchmarks for Biobank Agent evaluation.

Benchmarks:
  - SkillSchemaBenchmark: Verify all skill schemas are valid OpenAI format
  - BiomedQABenchmark: 20 golden biomedical Q&A pairs
  - SkillCallBenchmark: Verify the right skills are called for common queries
"""

from __future__ import annotations

from .harness import Benchmark, TestCase, TestResult


class SkillSchemaBenchmark(Benchmark):
    """Verify all registered skill schemas are valid OpenAI function-calling format."""
    name = "skill_schemas"

    def __init__(self) -> None:
        from ..registry import get_registry
        reg = get_registry()
        self.cases = []
        for skill_info in reg.list_skills():
            self.cases.append(TestCase(
                id=f"schema_{skill_info['name']}",
                query=f"__schema_check__{skill_info['name']}",
                tags=["schema"],
            ))
        self._registry = reg

    def score(self, result: TestResult, case: TestCase) -> float:
        """Check schema validity — not through agent.run(), just schema inspection."""
        skill_name = case.id.replace("schema_", "")
        schemas = self._registry.tool_schemas()
        for schema in schemas:
            if schema.get("function", {}).get("name") == skill_name:
                func = schema.get("function", {})
                if not func.get("name"):
                    result.errors.append("Missing function name")
                    result.passed = False
                    return 0.0
                if not func.get("description"):
                    result.errors.append("Missing function description")
                    result.passed = False
                    return 0.5
                params = func.get("parameters", {})
                if params.get("type") != "object":
                    result.errors.append("Parameters type must be 'object'")
                    result.passed = False
                    return 0.3
                result.passed = True
                return 1.0
        result.errors.append(f"Skill {skill_name} not found in registry")
        result.passed = False
        return 0.0


class BiomedQABenchmark(Benchmark):
    """20 golden biomedical Q&A pairs for answer quality assessment."""
    name = "biomedical_qa"

    def __init__(self) -> None:
        self.cases = [
            TestCase(
                id="qa_icd10_t2dm",
                query="What is the ICD10 code for Type 2 Diabetes?",
                expected_contains=["E11"],
                tags=["knowledge"],
            ),
            TestCase(
                id="qa_auc_meaning",
                query="What does an AUC of 0.85 indicate for a classifier?",
                expected_contains=["discriminat"],
                tags=["knowledge"],
            ),
            TestCase(
                id="qa_bmi_field",
                query="What UK Biobank field ID corresponds to BMI?",
                expected_contains=["21001"],
                tags=["knowledge"],
            ),
            TestCase(
                id="qa_hba1c",
                query="What biomarker is most commonly used to diagnose diabetes?",
                expected_contains=["HbA1c"],
                tags=["knowledge"],
            ),
            TestCase(
                id="qa_hypertension_code",
                query="What ICD10 code represents essential hypertension?",
                expected_contains=["I10"],
                tags=["knowledge"],
            ),
            TestCase(
                id="qa_prevalence_top",
                query="Show me the top 5 most prevalent diseases in the biobank",
                expected_skills=["prevalence"],
                tags=["skill_call"],
            ),
            TestCase(
                id="qa_survival_meaning",
                query="What does a log-rank P < 0.001 mean in survival analysis?",
                expected_contains=["significant"],
                tags=["knowledge"],
            ),
            TestCase(
                id="qa_cv_folds",
                query="Why do we use 5-fold cross-validation instead of a simple train/test split?",
                expected_contains=["overfit"],
                tags=["knowledge"],
            ),
            TestCase(
                id="qa_shap",
                query="What are SHAP values and why are they useful for model interpretation?",
                expected_contains=["feature"],
                tags=["knowledge"],
            ),
            TestCase(
                id="qa_bonferroni",
                query="When should we use Bonferroni correction vs FDR in biomarker studies?",
                expected_contains=["multiple"],
                tags=["knowledge"],
            ),
        ]

    def score(self, result: TestResult, case: TestCase) -> float:
        """Score based on how many expected elements are present."""
        if not result.actual_text:
            return 0.0
        total_checks = len(case.expected_contains) + len(case.expected_skills)
        if total_checks == 0:
            return 1.0 if result.passed else 0.0
        passed_checks = total_checks - len(result.errors)
        return max(0.0, passed_checks / total_checks)


class SkillCallBenchmark(Benchmark):
    """Verify the agent calls the right skills for common analytical queries."""
    name = "skill_calls"

    def __init__(self) -> None:
        self.cases = [
            TestCase(
                id="call_prevalence",
                query="What are the most common diseases in this biobank?",
                expected_skills=["prevalence"],
                tags=["routing"],
            ),
            TestCase(
                id="call_train",
                query="Build a predictive model for Type 2 Diabetes (E11) using XGBoost",
                expected_skills=["train_model"],
                tags=["routing"],
            ),
            TestCase(
                id="call_survival",
                query="Run a survival analysis for heart failure patients",
                expected_skills=["survival"],
                tags=["routing"],
            ),
            TestCase(
                id="call_cohort",
                query="Build a case-control cohort for E11 diabetes",
                expected_skills=["cohort_summary"],
                tags=["routing"],
            ),
            TestCase(
                id="call_report",
                query="Generate a research report summarizing our analyses",
                expected_skills=["generate_report"],
                tags=["routing"],
            ),
        ]

    def score(self, result: TestResult, case: TestCase) -> float:
        """Score based on whether the expected skills were called."""
        if not case.expected_skills:
            return 1.0
        called = set(result.actual_skills)
        expected = set(case.expected_skills)
        if expected.issubset(called):
            return 1.0
        return len(expected & called) / len(expected)
