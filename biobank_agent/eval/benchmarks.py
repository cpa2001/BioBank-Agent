"""Predefined benchmarks for Biobank Agent evaluation.

Benchmarks:
  - SkillSchemaBenchmark: Verify all skill schemas are valid OpenAI format
  - BiomedQABenchmark: 20 golden biomedical Q&A pairs
  - SkillCallBenchmark: Verify the right skills are called for common queries
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from .harness import Benchmark, TestCase, TestResult
from .metrics import report_quality_score


PUBLIC_REPORT_REFERENCES = [
    {
        "title": "Disease prediction with multi-omics and biomarkers empowers case-control genetic discoveries in the UK Biobank",
        "url": "https://www.nature.com/articles/s41588-024-01898-1",
        "doi": "10.1038/s41588-024-01898-1",
        "journal": "Nature Genetics",
    },
    {
        "title": "Plasma proteomic associations with genetics and health in the UK Biobank",
        "url": "https://www.nature.com/articles/s41586-023-06592-6",
        "doi": "10.1038/s41586-023-06592-6",
        "journal": "Nature",
    },
    {
        "title": "Plasma proteomic profiles predict individual future health risk",
        "url": "https://www.nature.com/articles/s41467-023-43575-7",
        "doi": "10.1038/s41467-023-43575-7",
        "journal": "Nature Communications",
    },
]


class SkillSchemaBenchmark(Benchmark):
    """Verify all registered skill schemas are valid OpenAI function-calling format."""
    name = "skill_schemas"

    def __init__(self) -> None:
        from ..registry import autodiscover_skills, discover_custom_skills, get_registry
        try:
            from ..config import get_settings
            settings = get_settings()
            autodiscover_skills()
            discover_custom_skills(settings.custom_skills_dir)
        except Exception:
            autodiscover_skills()
        reg = get_registry()
        self.cases = []
        for skill_info in reg.list_skills():
            self.cases.append(TestCase(
                id=f"schema_{skill_info['name']}",
                query=f"__schema_check__{skill_info['name']}",
                tags=["schema"],
            ))
        self._registry = reg

    def run_case(self, case: TestCase, agent) -> TestResult:
        """Inspect registry schemas directly without calling the LLM agent."""
        return TestResult(
            case_id=case.id,
            passed=True,
            actual_text="schema inspection only",
            metadata={"tags": case.tags, "direct_schema_check": True},
        )

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


class ReportQualityBenchmark(Benchmark):
    """Deterministic report-quality benchmark using public-paper-style evidence.

    This benchmark intentionally avoids live LLM calls and private biobank data. It
    synthesizes analysis records that mirror common UK Biobank publication
    workflows, generates reports, and scores whether the output has enough
    structure, quantitative detail, governance language, and references to support
    a Nature-family experimental/reporting section.
    """

    name = "report_quality"

    def __init__(self) -> None:
        self.cases = [
            TestCase(
                id="paper_diabetes_biomarker_prediction",
                query=(
                    "Generate an IMRaD paper-style report for an E11 biomarker "
                    "prediction workflow, grounded in public UK Biobank biomarker "
                    "and multi-omics papers."
                ),
                expected_contains=[
                    "Abstract",
                    "Methods",
                    "Results",
                    "Discussion",
                    "Reproducibility",
                    "Data Availability",
                    "References",
                    "10.1038/s41588-024-01898-1",
                    "not causal",
                ],
                tags=["report", "paper", "public_literature", "complex"],
            ),
            TestCase(
                id="technical_proteomics_risk_report",
                query=(
                    "Generate a technical report for a UK Biobank proteomic risk "
                    "workflow with figures, governance, statistical checks, and "
                    "public-paper references."
                ),
                expected_contains=[
                    "Key Findings",
                    "Executive Summary",
                    "Methodology Notes",
                    "Reproducibility and Governance",
                    "References",
                    "10.1038/s41586-023-06592-6",
                    "Small-cell outputs",
                ],
                tags=["report", "technical", "public_literature"],
            ),
        ]

    def run_case(self, case: TestCase, agent) -> TestResult:
        """Generate a report from synthetic records and score the artifact."""
        from biobank_agent.skills.report import generate_report

        report_format = "paper" if "paper" in case.tags else "report"
        with tempfile.TemporaryDirectory(prefix=f"biobank_report_eval_{case.id}_") as tmp:
            root = Path(tmp)
            ctx = self._build_ctx(root)
            result = generate_report(
                title=(
                    "Biomarker prediction and proteomic risk signatures in UK Biobank"
                    if report_format == "paper"
                    else "UK Biobank Proteomic Risk Technical Report"
                ),
                format=report_format,
                ctx=ctx,
            )
            md_path = Path(result["markdown"])
            text = md_path.read_text(encoding="utf-8")

            errors = []
            lower = text.lower()
            for token in case.expected_contains:
                if token.lower() not in lower:
                    errors.append(f"Missing expected report content: {token}")

            checks = self._report_checks(text)
            for name, passed in checks.items():
                if not passed:
                    errors.append(f"Report quality check failed: {name}")

            metadata = {
                "tags": case.tags,
                "format": report_format,
                "report_path": str(md_path),
                "quality_checks": checks,
                "quality_score": report_quality_score(text),
                "public_references": [r["doi"] for r in PUBLIC_REPORT_REFERENCES],
                # Direct report generation has no orchestrated claims, so provide
                # an artifact-level evidence proxy for the observability dashboard.
                "claim_count": 1,
                "evidence_count": len(PUBLIC_REPORT_REFERENCES),
                "safety_status": "PASS",
            }

            return TestResult(
                case_id=case.id,
                passed=not errors,
                actual_skills=["generate_report"],
                actual_text=text[:5000],
                errors=errors,
                elapsed_s=0.0,
                metadata=metadata,
            )

    def score(self, result: TestResult, case: TestCase) -> float:
        """Score report output using structural and publication-readiness checks."""
        checks = result.metadata.get("quality_checks", {}) if result.metadata else {}
        if not checks:
            return 0.0
        structural = result.metadata.get("quality_score", {}).get("overall", 0.0)
        checklist = sum(1 for ok in checks.values() if ok) / max(1, len(checks))
        missing_penalty = min(0.35, 0.05 * len(result.errors))
        return max(0.0, min(1.0, 0.4 * structural + 0.6 * checklist - missing_penalty))

    @staticmethod
    def _report_checks(text: str) -> dict[str, bool]:
        lower = text.lower()
        return {
            "has_nature_sections": all(k in lower for k in ("abstract", "methods", "results", "discussion"))
            or all(k in lower for k in ("key findings", "executive summary", "methodology notes")),
            "has_quantitative_statistics": all(k in lower for k in ("auc", "95% ci")) and (
                "p <" in lower or "p =" in lower
            ),
            "has_sample_sizes": "n =" in lower or "n_cases" in lower or "cases" in lower,
            "has_figures_or_captions": "figure" in lower,
            "has_governance": "reproducibility" in lower and "governance" in lower,
            "has_data_access_or_availability": "data availability" in lower or "data access agreement" in lower,
            "has_noncausal_caveat": "not causal" in lower or "precludes causal inference" in lower,
            "has_small_cell_guardrail": "small-cell" in lower or "suppression" in lower,
            "has_public_references": all(ref["doi"].lower() in lower for ref in PUBLIC_REPORT_REFERENCES[:2]),
            "no_placeholders": all(
                bad not in lower
                for bad in ("[citation_needed]", "references to be added", "results pending", "todo")
            ),
        }

    @staticmethod
    def _build_ctx(root: Path):
        report_dir = root / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        fig_path = report_dir / "figure_roc.svg"
        fig_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="240">'
            '<text x="10" y="20">AUC = 0.842</text></svg>',
            encoding="utf-8",
        )

        records = [
            SimpleNamespace(
                timestamp="2026-01-01T00:00:00",
                skill="deep_research",
                args={"topic": "UK Biobank biomarker disease prediction"},
                key_results={
                    "sources": PUBLIC_REPORT_REFERENCES,
                    "n_sources": len(PUBLIC_REPORT_REFERENCES),
                },
                figure_paths=[],
            ),
            SimpleNamespace(
                timestamp="2026-01-01T00:01:00",
                skill="read_paper",
                args={"paper_path_or_doi": PUBLIC_REPORT_REFERENCES[0]["doi"], "focus": "methods"},
                key_results={
                    "title": PUBLIC_REPORT_REFERENCES[0]["title"],
                    "authors": ["Shen X", "Li J", "UK Biobank collaborators"],
                    "doi": PUBLIC_REPORT_REFERENCES[0]["doi"],
                    "url": PUBLIC_REPORT_REFERENCES[0]["url"],
                },
                figure_paths=[],
            ),
            SimpleNamespace(
                timestamp="2026-01-01T00:02:00",
                skill="cohort_summary",
                args={"icd10_code": "E11", "controls_ratio": 4},
                key_results={"n_cases": 12450, "n_controls": 49800, "n_features": 54},
                figure_paths=[],
            ),
            SimpleNamespace(
                timestamp="2026-01-01T00:03:00",
                skill="train_model",
                args={"icd10_code": "E11", "model_type": "xgb", "n_folds": 5},
                key_results={
                    "model_key": "E11_xgb",
                    "mean_auc": 0.842,
                    "auc_95ci": "[0.831, 0.853]",
                    "n_cases": 12450,
                    "n_controls": 49800,
                    "n_features": 54,
                    "fold_aucs": [0.835, 0.846, 0.841, 0.849, 0.839],
                },
                figure_paths=[str(fig_path)],
            ),
            SimpleNamespace(
                timestamp="2026-01-01T00:04:00",
                skill="biomarker_dist",
                args={"field_id": "30750", "field_name": "HbA1c"},
                key_results={"p_value": 2.1e-28, "effect_size": 0.42},
                figure_paths=[],
            ),
            SimpleNamespace(
                timestamp="2026-01-01T00:05:00",
                skill="phewas",
                args={"field_id": "30750", "min_cases": 500},
                key_results={"n_significant": 18, "n_fields_tested": 3213, "correction": "fdr"},
                figure_paths=[],
            ),
            SimpleNamespace(
                timestamp="2026-01-01T00:06:00",
                skill="statistical_review",
                args={"scope": "session"},
                key_results={"overall_assessment": "MINOR -- some issues noted", "n_warnings": 1},
                figure_paths=[],
            ),
            SimpleNamespace(
                timestamp="2026-01-01T00:07:00",
                skill="safety_check",
                args={"scope": "session"},
                key_results={"overall": "PASS", "small_cell_suppression": True},
                figure_paths=[],
            ),
        ]

        cohort = pd.DataFrame(
            {
                "eid": range(100),
                "label": [1] * 20 + [0] * 80,
                "30750-0.0": [65.0] * 20 + [39.0] * 80,
            }
        )
        state = SimpleNamespace(
            records=records,
            figures=[str(fig_path)],
            cohorts={"E11_1:4": cohort},
            models={"E11_xgb": object()},
            model_metadata={
                "E11_xgb": {
                    "model_type": "xgb",
                    "mean_auc": 0.842,
                    "auc_mean": 0.842,
                    "auc_95ci": "[0.831, 0.853]",
                    "n_cases": 12450,
                    "n_features": 54,
                }
            },
            provenances=[SimpleNamespace(provenance_id="prov1")],
        )
        settings = SimpleNamespace(
            biobank_name="UK Biobank",
            biobank_description="a population-scale prospective cohort with linked phenotypes, biomarkers and health outcomes",
            biobank_caveats="healthy volunteer bias, predominantly middle-aged recruitment, and ancestry imbalance",
        )
        return SimpleNamespace(report_dir=report_dir, state=state, settings=settings)


class AgentReportWorkflowBenchmark(Benchmark):
    """Run staged report-generation prompts through an agent and diagnose failure stage.

    Unlike :class:`ReportQualityBenchmark`, this benchmark is designed to exercise
    ``agent.run``. It keeps the stages explicit so a failed live run records where
    the workflow stopped: literature grounding, cohort/model execution, guardrail
    review, or final report generation.
    """

    name = "agent_report_workflow"

    def __init__(self) -> None:
        self.cases = [
            TestCase(
                id="agent_e11_paper_workflow",
                query="Run an E11 prediction workflow and produce an IMRaD paper-style report.",
                expected_skills=[
                    "deep_research",
                    "cohort_summary",
                    "train_model",
                    "statistical_review",
                    "safety_check",
                    "generate_report",
                ],
                expected_contains=[
                    "Abstract",
                    "Methods",
                    "Results",
                    "Discussion",
                    "References",
                    "10.1038/s41588-024-01898-1",
                    "not causal",
                ],
                tags=["agent_workflow", "report", "paper", "complex"],
            ),
            TestCase(
                id="agent_proteomics_technical_workflow",
                query="Run a proteomics risk workflow and produce a technical report.",
                expected_skills=[
                    "deep_research",
                    "read_paper",
                    "cohort_summary",
                    "statistical_review",
                    "safety_check",
                    "generate_report",
                ],
                expected_contains=[
                    "Key Findings",
                    "Executive Summary",
                    "Methodology Notes",
                    "Reproducibility",
                    "Small-cell",
                    "10.1038/s41586-023-06592-6",
                ],
                tags=["agent_workflow", "report", "technical", "complex"],
            ),
        ]
        self._stages = {
            "agent_e11_paper_workflow": [
                ("literature", "Search and summarize public UK Biobank biomarker prediction papers, keeping DOI references."),
                ("cohort_model", "Build an E11 case-control cohort, train an XGBoost biomarker model, and record AUC with 95% CI."),
                ("guardrails", "Run statistical_review and safety_check on the current session before any final conclusions."),
                ("report", "Generate an IMRaD paper report with references, data availability, and non-causal caveats."),
            ],
            "agent_proteomics_technical_workflow": [
                ("literature", "Search/read public UK Biobank proteomics risk papers and keep DOI references."),
                ("cohort_model", "Construct a proteomics risk cohort summary and capture quantitative model evidence."),
                ("guardrails", "Run statistical_review and safety_check, including small-cell suppression checks."),
                ("report", "Generate a technical report with key findings, methods, governance, and references."),
            ],
        }

    def run_case(self, case: TestCase, agent) -> TestResult:
        t0 = time.time()
        if not hasattr(agent, "run"):
            return TestResult(
                case_id=case.id,
                passed=False,
                errors=["Agent unavailable: object has no run(query) method"],
                elapsed_s=0.0,
                metadata={
                    "tags": case.tags,
                    "failure_stage": "agent_unavailable",
                    "stage_results": [],
                    "claim_count": 1,
                    "evidence_count": 0,
                    "safety_status": "FAIL",
                },
            )

        stage_results = []
        failure_stage = None
        errors: list[str] = []
        responses: list[str] = []
        for stage, prompt in self._stages.get(case.id, [("report", case.query)]):
            try:
                response = agent.run(prompt)
                responses.append(str(response))
                stage_results.append({"stage": stage, "status": "PASS"})
            except Exception as e:
                failure_stage = stage
                msg = f"Stage {stage} failed: {type(e).__name__}: {str(e)[:200]}"
                errors.append(msg)
                stage_results.append({"stage": stage, "status": "FAIL", "error": str(e)[:200]})
                break

        actual_skills = self._actual_skills(agent)
        for skill in case.expected_skills:
            if skill not in actual_skills:
                errors.append(f"Expected skill '{skill}' not called")

        report_text, report_path = self._latest_report_text(agent, responses[-1] if responses else "")
        lower = report_text.lower()
        for token in case.expected_contains:
            if token.lower() not in lower:
                errors.append(f"Missing expected report content: {token}")

        checks = ReportQualityBenchmark._report_checks(report_text) if report_text else {}
        for name, passed in checks.items():
            if not passed:
                errors.append(f"Report quality check failed: {name}")

        evidence_count = sum(1 for ref in PUBLIC_REPORT_REFERENCES if ref["doi"].lower() in lower)
        passed = failure_stage is None and not errors
        return TestResult(
            case_id=case.id,
            passed=passed,
            actual_skills=actual_skills[-12:],
            actual_text=report_text[:5000],
            errors=errors,
            elapsed_s=time.time() - t0,
            metadata={
                "tags": case.tags,
                "failure_stage": failure_stage,
                "stage_results": stage_results,
                "report_path": str(report_path) if report_path else "",
                "quality_checks": checks,
                "quality_score": report_quality_score(report_text),
                "claim_count": 1,
                "evidence_count": evidence_count,
                "safety_status": "PASS" if passed else "FAIL",
            },
        )

    def score(self, result: TestResult, case: TestCase) -> float:
        if result.metadata.get("failure_stage"):
            completed = sum(1 for item in result.metadata.get("stage_results", []) if item.get("status") == "PASS")
            return min(0.2, 0.05 * completed)
        checks = result.metadata.get("quality_checks", {}) if result.metadata else {}
        if not checks:
            return 0.0
        checklist = sum(1 for ok in checks.values() if ok) / max(1, len(checks))
        structural = result.metadata.get("quality_score", {}).get("overall", 0.0)
        called = set(result.actual_skills)
        expected = set(case.expected_skills)
        skill_score = len(called & expected) / max(1, len(expected))
        penalty = min(0.4, 0.04 * len(result.errors))
        return max(0.0, min(1.0, 0.35 * checklist + 0.35 * structural + 0.30 * skill_score - penalty))

    @staticmethod
    def _actual_skills(agent) -> list[str]:
        records = getattr(getattr(agent, "state", None), "records", []) or []
        return [getattr(r, "skill", "") for r in records if getattr(r, "skill", "") and getattr(r, "skill", "") != "think"]

    @staticmethod
    def _latest_report_text(agent, fallback: str) -> tuple[str, Path | None]:
        records = getattr(getattr(agent, "state", None), "records", []) or []
        for rec in reversed(records):
            if getattr(rec, "skill", "") != "generate_report":
                continue
            result = getattr(rec, "key_results", {}) or {}
            md_path = result.get("markdown") if isinstance(result, dict) else None
            if md_path:
                path = Path(md_path)
                try:
                    return path.read_text(encoding="utf-8"), path
                except OSError:
                    return fallback, path
        return fallback, None


class ResearchEvalV1(Benchmark):
    """120-case reliability suite for biobank scientific agent evaluation.

    Four categories × 30 each:
    - complexity: complex planning/routing
    - evidence: evidence-chain and provenance prompts
    - statistics: guardrail-sensitive statistical prompts
    - execution: concrete skill-invocation tasks
    """

    name = "research_eval_v1"

    def __init__(self) -> None:
        self.cases = []
        self._build_cases()

    def _build_cases(self) -> None:
        diseases = ["E11", "I10", "I21", "N18", "C34", "J44", "F32", "K76", "G20", "M81"]
        biomarkers = ["HbA1c", "CRP", "LDL", "HDL", "BMI", "ALT", "AST", "Creatinine", "Triglycerides", "eGFR"]

        # Category 1: complex route/decomposition tasks.
        for i in range(30):
            d = diseases[i % len(diseases)]
            b = biomarkers[i % len(biomarkers)]
            self.cases.append(TestCase(
                id=f"complex_{i+1:03d}",
                query=(
                    f"Comprehensive multi-step analysis for {d}: prevalence baseline, "
                    f"train prediction model, check calibration, and critique confounders involving {b}. "
                    "Provide uncertainty and verification plan."
                ),
                expected_contains=["uncertainty"],
                tags=["complex", "routing", "planning"],
            ))

        # Category 2: evidence/provenance tasks.
        for i in range(30):
            d = diseases[i % len(diseases)]
            self.cases.append(TestCase(
                id=f"evidence_{i+1:03d}",
                query=(
                    f"For disease {d}, provide claim-evidence chain with at least two evidence links "
                    "and explain what additional retrieval would falsify your conclusion."
                ),
                expected_contains=["evidence"],
                tags=["evidence", "provenance"],
            ))

        # Category 3: statistical guardrail tasks.
        for i in range(30):
            d = diseases[i % len(diseases)]
            self.cases.append(TestCase(
                id=f"statistics_{i+1:03d}",
                query=(
                    f"Assess {d} cardiovascular risk with explicit multiple-testing considerations, "
                    "95% confidence intervals, and confounder adjustment rationale."
                ),
                expected_contains=["95", "confidence"],
                tags=["statistics", "guardrail"],
            ))

        # Category 4: execution-centric tasks.
        for i in range(30):
            d = diseases[i % len(diseases)]
            self.cases.append(TestCase(
                id=f"execution_{i+1:03d}",
                query=f"Run prevalence and then a predictive modeling workflow for ICD10 {d}.",
                expected_skills=["prevalence"],
                tags=["execution"],
            ))

    def score(self, result: TestResult, case: TestCase) -> float:
        """Composite score: pass/fail plus evidence metadata bonus."""
        base = 1.0 if result.passed else 0.0
        claim_count = int(result.metadata.get("claim_count", 0))
        evidence_count = int(result.metadata.get("evidence_count", 0))
        if claim_count <= 0:
            return base * 0.8
        evidence_ratio = min(1.0, evidence_count / claim_count)
        return max(0.0, 0.7 * base + 0.3 * evidence_ratio)
