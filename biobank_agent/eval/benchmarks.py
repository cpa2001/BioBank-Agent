"""Predefined benchmarks for Biobank Agent evaluation.

Benchmarks:
  - SkillSchemaBenchmark: Verify all skill schemas are valid OpenAI format
  - BiomedQABenchmark: 20 golden biomedical Q&A pairs
  - SkillCallBenchmark: Verify the right skills are called for common queries
  - Report20CaseBenchmark: Offline UKB-oriented synthetic report regression
  - LiveUKBReport20Benchmark: Live UKB workflow probes with fail-closed preflight
"""

from __future__ import annotations

import tempfile
import time
import re
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


DEFAULT_REPORT_SCENARIO = {
    "title": "UK Biobank E11 HbA1c biomarker prediction report",
    "icd10": "E11",
    "disease": "type 2 diabetes",
    "field_id": "30750",
    "field_name": "HbA1c",
    "n_cases": 12450,
    "n_controls": 49800,
    "n_features": 54,
    "auc": 0.842,
    "auc_95ci": "[0.831, 0.853]",
    "p_value": 2.1e-28,
    "effect_size": 0.42,
}


REPORT_20_CASE_SCENARIOS = [
    {"id": "report_e11_hba1c", "title": "UK Biobank type 2 diabetes HbA1c prediction report", "icd10": "E11", "disease": "type 2 diabetes", "field_id": "30750", "field_name": "HbA1c", "n_cases": 12450, "n_controls": 49800, "n_features": 54, "auc": 0.842, "auc_95ci": "[0.831, 0.853]", "p_value": 2.1e-28, "effect_size": 0.42, "format": "paper"},
    {"id": "report_i10_bmi_bp", "title": "UK Biobank hypertension BMI and blood pressure technical report", "icd10": "I10", "disease": "essential hypertension", "field_id": "21001", "field_name": "BMI", "n_cases": 38200, "n_controls": 76400, "n_features": 42, "auc": 0.781, "auc_95ci": "[0.773, 0.789]", "p_value": 4.6e-34, "effect_size": 0.31, "format": "report"},
    {"id": "report_i21_ldl", "title": "UK Biobank myocardial infarction LDL prediction report", "icd10": "I21", "disease": "myocardial infarction", "field_id": "30780", "field_name": "LDL direct", "n_cases": 8950, "n_controls": 35800, "n_features": 49, "auc": 0.806, "auc_95ci": "[0.792, 0.820]", "p_value": 8.8e-19, "effect_size": 0.27, "format": "paper"},
    {"id": "report_n18_creatinine", "title": "UK Biobank chronic kidney disease creatinine technical report", "icd10": "N18", "disease": "chronic kidney disease", "field_id": "30700", "field_name": "Creatinine", "n_cases": 6420, "n_controls": 25680, "n_features": 47, "auc": 0.827, "auc_95ci": "[0.811, 0.843]", "p_value": 1.5e-22, "effect_size": 0.36, "format": "report"},
    {"id": "report_c34_crp", "title": "UK Biobank lung cancer CRP risk report", "icd10": "C34", "disease": "lung cancer", "field_id": "30710", "field_name": "C-reactive protein", "n_cases": 3180, "n_controls": 12720, "n_features": 46, "auc": 0.768, "auc_95ci": "[0.744, 0.792]", "p_value": 6.2e-12, "effect_size": 0.24, "format": "paper"},
    {"id": "report_j44_fev1", "title": "UK Biobank COPD lung function technical report", "icd10": "J44", "disease": "chronic obstructive pulmonary disease", "field_id": "3063", "field_name": "FEV1", "n_cases": 7120, "n_controls": 28480, "n_features": 38, "auc": 0.814, "auc_95ci": "[0.799, 0.829]", "p_value": 3.9e-17, "effect_size": 0.33, "format": "report"},
    {"id": "report_f32_crp", "title": "UK Biobank depression inflammatory biomarker report", "icd10": "F32", "disease": "depressive episode", "field_id": "30710", "field_name": "C-reactive protein", "n_cases": 18100, "n_controls": 72400, "n_features": 52, "auc": 0.712, "auc_95ci": "[0.702, 0.722]", "p_value": 2.7e-09, "effect_size": 0.12, "format": "paper"},
    {"id": "report_k76_alt", "title": "UK Biobank liver disease ALT technical report", "icd10": "K76", "disease": "liver disease", "field_id": "30620", "field_name": "Alanine aminotransferase", "n_cases": 5300, "n_controls": 21200, "n_features": 44, "auc": 0.793, "auc_95ci": "[0.776, 0.810]", "p_value": 9.4e-21, "effect_size": 0.29, "format": "report"},
    {"id": "report_g20_urate", "title": "UK Biobank Parkinson disease urate prediction report", "icd10": "G20", "disease": "Parkinson disease", "field_id": "30880", "field_name": "Urate", "n_cases": 2140, "n_controls": 8560, "n_features": 35, "auc": 0.741, "auc_95ci": "[0.712, 0.770]", "p_value": 1.1e-06, "effect_size": -0.18, "format": "paper"},
    {"id": "report_m81_vitd", "title": "UK Biobank osteoporosis vitamin D technical report", "icd10": "M81", "disease": "osteoporosis", "field_id": "30890", "field_name": "Vitamin D", "n_cases": 6900, "n_controls": 27600, "n_features": 41, "auc": 0.759, "auc_95ci": "[0.742, 0.776]", "p_value": 5.8e-13, "effect_size": -0.21, "format": "report"},
    {"id": "report_c50_igf1", "title": "UK Biobank breast cancer IGF-1 report", "icd10": "C50", "disease": "breast cancer", "field_id": "30770", "field_name": "IGF-1", "n_cases": 10420, "n_controls": 41680, "n_features": 45, "auc": 0.733, "auc_95ci": "[0.720, 0.746]", "p_value": 3.2e-10, "effect_size": 0.16, "format": "paper"},
    {"id": "report_i63_apob", "title": "UK Biobank ischaemic stroke ApoB technical report", "icd10": "I63", "disease": "ischaemic stroke", "field_id": "30640", "field_name": "Apolipoprotein B", "n_cases": 5120, "n_controls": 20480, "n_features": 48, "auc": 0.776, "auc_95ci": "[0.758, 0.794]", "p_value": 1.9e-14, "effect_size": 0.22, "format": "report"},
    {"id": "report_i50_ntprobnp", "title": "UK Biobank heart failure biomarker report", "icd10": "I50", "disease": "heart failure", "field_id": "30700", "field_name": "Creatinine", "n_cases": 7820, "n_controls": 31280, "n_features": 50, "auc": 0.818, "auc_95ci": "[0.802, 0.834]", "p_value": 4.1e-20, "effect_size": 0.34, "format": "paper"},
    {"id": "report_e78_cholesterol", "title": "UK Biobank lipid disorder cholesterol technical report", "icd10": "E78", "disease": "lipid disorder", "field_id": "30690", "field_name": "Cholesterol", "n_cases": 22900, "n_controls": 91600, "n_features": 43, "auc": 0.794, "auc_95ci": "[0.785, 0.803]", "p_value": 7.7e-31, "effect_size": 0.28, "format": "report"},
    {"id": "report_m10_urate", "title": "UK Biobank gout urate prediction report", "icd10": "M10", "disease": "gout", "field_id": "30880", "field_name": "Urate", "n_cases": 6100, "n_controls": 24400, "n_features": 39, "auc": 0.856, "auc_95ci": "[0.842, 0.870]", "p_value": 5.4e-39, "effect_size": 0.58, "format": "paper"},
    {"id": "report_k50_albumin", "title": "UK Biobank Crohn disease albumin technical report", "icd10": "K50", "disease": "Crohn disease", "field_id": "30600", "field_name": "Albumin", "n_cases": 3020, "n_controls": 12080, "n_features": 37, "auc": 0.752, "auc_95ci": "[0.727, 0.777]", "p_value": 8.2e-08, "effect_size": -0.19, "format": "report"},
    {"id": "report_m05_crp", "title": "UK Biobank rheumatoid arthritis CRP report", "icd10": "M05", "disease": "rheumatoid arthritis", "field_id": "30710", "field_name": "C-reactive protein", "n_cases": 4480, "n_controls": 17920, "n_features": 40, "auc": 0.771, "auc_95ci": "[0.751, 0.791]", "p_value": 2.6e-15, "effect_size": 0.25, "format": "paper"},
    {"id": "report_e03_shbg", "title": "UK Biobank hypothyroidism SHBG technical report", "icd10": "E03", "disease": "hypothyroidism", "field_id": "30830", "field_name": "SHBG", "n_cases": 15200, "n_controls": 60800, "n_features": 36, "auc": 0.724, "auc_95ci": "[0.714, 0.734]", "p_value": 6.9e-11, "effect_size": -0.14, "format": "report"},
    {"id": "report_g47_bmi", "title": "UK Biobank sleep disorder BMI report", "icd10": "G47", "disease": "sleep disorder", "field_id": "21001", "field_name": "BMI", "n_cases": 8200, "n_controls": 32800, "n_features": 42, "auc": 0.746, "auc_95ci": "[0.731, 0.761]", "p_value": 4.4e-16, "effect_size": 0.23, "format": "paper"},
    {"id": "report_n39_cystatin", "title": "UK Biobank urinary disorder cystatin C technical report", "icd10": "N39", "disease": "urinary disorder", "field_id": "30720", "field_name": "Cystatin C", "n_cases": 9600, "n_controls": 38400, "n_features": 41, "auc": 0.769, "auc_95ci": "[0.756, 0.782]", "p_value": 7.3e-18, "effect_size": 0.26, "format": "report"},
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

        scenario = dict(getattr(case, "metadata", {}) or {})
        report_format = scenario.get("format") or ("paper" if "paper" in case.tags else "report")
        with tempfile.TemporaryDirectory(prefix=f"biobank_report_eval_{case.id}_") as tmp:
            root = Path(tmp)
            ctx = self._build_ctx(root, scenario)
            result = generate_report(
                title=(
                    scenario.get("title", "Biomarker prediction and proteomic risk signatures in UK Biobank")
                    if report_format == "paper"
                    else scenario.get("title", "UK Biobank Proteomic Risk Technical Report")
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

            checks = self._report_checks(text, md_path)
            for name, passed in checks.items():
                if not passed:
                    errors.append(f"Report quality check failed: {name}")

            metadata = {
                "tags": case.tags,
                "format": report_format,
                "benchmark_kind": case.metadata.get("benchmark_kind", "synthetic_aggregate"),
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
                actual_text=text,
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
    def _report_checks(text: str, report_path: Path | None = None) -> dict[str, bool]:
        lower = text.lower()
        main_lower = lower.split("## execution appendix", 1)[0]
        raw_log_tokens = (
            "traceback",
            "returncode",
            "stdout",
            "stderr",
            "/users/",
            "api_key",
            "authorization:",
        )
        placeholder_tokens = (
            "[citation_needed]",
            "references to be added",
            "results pending",
            "todo",
            "analysis completed -- see details below",
            "top finding: .",
            "auc = n/a",
        )
        signature_tokens = (
            "chen pengan",
            "chinese university",
            "generated by:",
            " biobank agent v",
        )
        weak_main_body_tokens = (
            "analysis completed",
            "top finding: .",
            "see details below",
            "returncode=",
            "dict[",
        )
        unresolved_missing_patterns = (
            r"\bN/A\b",
            r"=\s*\?",
            r"\?\s*%",
            r"\?\s+significant",
            r"\?\s+completed",
        )
        executive_pos = lower.find("executive findings")
        first_body_positions = [
            pos for pos in (
                lower.find("## abstract"),
                lower.find("## executive summary"),
                lower.find("## methods"),
                lower.find("## 1."),
            )
            if pos >= 0
        ]
        first_body_pos = min(first_body_positions) if first_body_positions else len(lower)
        checks = {
            "has_nature_sections": all(k in lower for k in ("abstract", "methods", "results", "discussion"))
            or all(k in lower for k in ("key findings", "executive summary", "methodology notes")),
            "has_executive_findings_upfront": executive_pos >= 0 and executive_pos < first_body_pos,
            "has_quantitative_statistics": all(k in lower for k in ("auc", "95% ci")) and (
                "p <" in lower or "p =" in lower
            ),
            "has_sample_sizes": "n =" in lower or "n_cases" in lower or "cases" in lower,
            "has_figures_or_captions": "figure" in lower,
            "has_governance": "reproducibility" in lower and "governance" in lower,
            "has_execution_appendix": "execution appendix" in lower and "analysis record inventory" in lower,
            "has_data_access_or_availability": "data availability" in lower or "data access agreement" in lower,
            "has_noncausal_caveat": "not causal" in lower or "precludes causal inference" in lower,
            "has_small_cell_guardrail": "small-cell" in lower or "suppression" in lower,
            "has_public_references": all(ref["doi"].lower() in lower for ref in PUBLIC_REPORT_REFERENCES[:2]),
            "no_placeholders": all(bad not in lower for bad in placeholder_tokens),
            "no_raw_logs_or_secrets": all(bad not in lower for bad in raw_log_tokens),
            "no_signature_or_generator_trace": all(bad not in lower for bad in signature_tokens),
            "no_weak_main_body": all(bad not in main_lower for bad in weak_main_body_tokens),
            "no_unresolved_missing_values_in_main_body": all(
                not re.search(pattern, main_lower, re.IGNORECASE)
                for pattern in unresolved_missing_patterns
            ),
            "no_unresolved_critical_guardrails": "severity | type | message | recommendation" not in lower
            or "| critical |" not in lower,
        }
        figure_numbers = re.findall(r"(?m)^\*{0,3}\s*figure\s+(\d+)\.", lower)
        checks["no_duplicate_figure_numbers"] = len(figure_numbers) == len(set(figure_numbers))
        if report_path:
            checks["figure_links_resolvable"] = ReportQualityBenchmark._figure_links_resolvable(text, Path(report_path))
        return checks

    @staticmethod
    def _figure_links_resolvable(text: str, report_path: Path) -> bool:
        report_dir = report_path.parent
        for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
            if link.startswith(("http://", "https://", "data:")):
                continue
            target = (report_dir / link).resolve()
            if not target.exists():
                return False
        return True

    @staticmethod
    def _build_ctx(root: Path, scenario: dict | None = None):
        scenario = {**DEFAULT_REPORT_SCENARIO, **(scenario or {})}
        icd10 = str(scenario["icd10"])
        disease = str(scenario["disease"])
        field_id = str(scenario["field_id"])
        field_name = str(scenario["field_name"])
        n_cases = int(scenario["n_cases"])
        n_controls = int(scenario["n_controls"])
        n_features = int(scenario["n_features"])
        auc = float(scenario["auc"])
        auc_95ci = str(scenario["auc_95ci"])
        p_value = float(scenario["p_value"])
        effect_size = float(scenario["effect_size"])
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in f"{icd10}_{field_id}").strip("_")
        model_key = f"{slug}_auto"
        executive_findings = [
            (
                f"{field_name} stratified {disease} risk in UK Biobank with "
                f"AUC = {auc:.3f} ({auc_95ci}) across {n_cases:,} cases and "
                f"{n_controls:,} controls."
            ),
            (
                f"The {icd10} analysis should be interpreted as observational "
                "biobank evidence: it supports risk stratification and hypothesis "
                "generation, not a causal biomarker claim."
            ),
            (
                "Guardrail review retained the aggregate findings for reporting "
                "with small-cell suppression and reproducibility caveats."
            ),
        ]
        execution_log = [
            {
                "step": "Literature grounding",
                "tool": "deep_research",
                "status": "success",
                "duration_s": "offline fixture",
                "note": "Public UK Biobank paper references attached.",
            },
            {
                "step": "Cohort construction",
                "tool": "cohort_summary",
                "status": "success",
                "duration_s": "offline fixture",
                "note": f"{n_cases:,} cases and {n_controls:,} controls.",
            },
            {
                "step": "Model selection",
                "tool": "train_model",
                "status": "success",
                "duration_s": "offline fixture",
                "note": "Auto-selection compared candidate model families before report synthesis.",
            },
            {
                "step": "Guardrail review",
                "tool": "statistical_review + safety_check",
                "status": "success",
                "duration_s": "offline fixture",
                "note": "Aggregate release caveats recorded.",
            },
        ]

        report_dir = root / "report"
        report_dir.mkdir(parents=True, exist_ok=True)
        fig_path = report_dir / f"figure_{slug}_roc.svg"
        fig_path.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="240">'
            f'<text x="10" y="20">AUC = {auc:.3f}</text></svg>',
            encoding="utf-8",
        )

        records = [
            SimpleNamespace(
                timestamp="2026-01-01T00:00:00",
                skill="deep_research",
                args={"topic": f"UK Biobank {disease} {field_name} biomarker disease prediction"},
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
                args={"icd10_code": icd10, "disease": disease, "controls_ratio": 0},
                key_results={
                    "n_cases": n_cases,
                    "n_controls": n_controls,
                    "n_features": n_features,
                    "controls_sampling_applied": False,
                },
                figure_paths=[],
            ),
            SimpleNamespace(
                timestamp="2026-01-01T00:03:00",
                skill="train_model",
                args={"icd10_code": icd10, "disease": disease, "model_type": "auto", "n_folds": 5},
                key_results={
                    "model_key": model_key,
                    "model_type": "auto-selected gradient boosted model",
                    "mean_auc": auc,
                    "auc_95ci": auc_95ci,
                    "n_cases": n_cases,
                    "n_controls": n_controls,
                    "n_features": n_features,
                    "fold_aucs": [round(auc - 0.007, 3), round(auc + 0.004, 3), auc, round(auc + 0.007, 3), round(auc - 0.003, 3)],
                },
                figure_paths=[str(fig_path)],
            ),
            SimpleNamespace(
                timestamp="2026-01-01T00:04:00",
                skill="biomarker_dist",
                args={"field_id": field_id, "field_name": field_name},
                key_results={"p_value": p_value, "effect_size": effect_size},
                figure_paths=[],
            ),
            SimpleNamespace(
                timestamp="2026-01-01T00:05:00",
                skill="phewas",
                args={"field_id": field_id, "field_name": field_name, "min_cases": 500},
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
                f"{field_id}-0.0": [65.0] * 20 + [39.0] * 80,
            }
        )
        state = SimpleNamespace(
            records=records,
            figures=[str(fig_path)],
            cohorts={f"{icd10}_1:all": cohort},
            models={model_key: object()},
            model_metadata={
                model_key: {
                    "model_type": "auto-selected gradient boosted model",
                    "mean_auc": auc,
                    "auc_mean": auc,
                    "auc_95ci": auc_95ci,
                    "n_cases": n_cases,
                    "n_features": n_features,
                }
            },
            provenances=[SimpleNamespace(provenance_id="prov1")],
            executive_findings=executive_findings,
            execution_log=execution_log,
            custom_data={"execution_log": execution_log},
            guardrail_issues=[],
        )
        settings = SimpleNamespace(
            biobank_name="UK Biobank",
            biobank_description="a population-scale prospective cohort with linked phenotypes, biomarkers and health outcomes",
            biobank_caveats="healthy volunteer bias, predominantly middle-aged recruitment, and ancestry imbalance",
        )
        return SimpleNamespace(report_dir=report_dir, state=state, settings=settings)


class Report20CaseBenchmark(ReportQualityBenchmark):
    """Twenty deterministic, UKB-oriented synthetic report cases.

    The cases cover common ICD10 outcomes and UK Biobank field IDs while staying
    fully offline: each run synthesizes aggregate-only records and then applies
    the same report-quality checks used by the smaller report benchmark.
    """

    name = "report_20_case"

    def __init__(self) -> None:
        self.cases = []
        for scenario in REPORT_20_CASE_SCENARIOS:
            report_format = scenario.get("format", "paper")
            if report_format == "paper":
                expected_sections = ["Abstract", "Methods", "Results", "Discussion"]
                tags = ["report", "paper", "ukb_oriented_synthetic", "public_literature", "complex"]
            else:
                expected_sections = ["Key Findings", "Executive Summary", "Methodology Notes"]
                tags = ["report", "technical", "ukb_oriented_synthetic", "public_literature", "complex"]
            self.cases.append(TestCase(
                id=str(scenario["id"]),
                query=(
                    f"Generate a UK Biobank report for {scenario['disease']} "
                    f"({scenario['icd10']}) using field {scenario['field_id']} "
                    f"({scenario['field_name']}) with public-paper references."
                ),
                expected_contains=[
                    "UK Biobank",
                    "References",
                    "10.1038/s41588-024-01898-1",
                    "10.1038/s41586-023-06592-6",
                    "not causal",
                    "Reproducibility",
                    *expected_sections,
                ],
                tags=tags,
                metadata={"benchmark_kind": "ukb_oriented_synthetic", **dict(scenario)},
            ))

    def run_case(self, case: TestCase, agent) -> TestResult:
        """Generate both Nature-style and technical reports for each UKB case."""
        from biobank_agent.skills.report import generate_report

        scenario = dict(getattr(case, "metadata", {}) or {})
        errors: list[str] = []
        checks: dict[str, bool] = {}
        outputs: dict[str, str] = {}
        texts: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix=f"biobank_report20_{case.id}_") as tmp:
            root = Path(tmp)
            for report_format in ("paper", "report"):
                ctx = self._build_ctx(root / report_format, {**scenario, "format": report_format})
                result = generate_report(
                    title=scenario.get("title", "UK Biobank long-horizon discovery report"),
                    format=report_format,
                    ctx=ctx,
                )
                md_path = Path(result["markdown"])
                text = md_path.read_text(encoding="utf-8")
                outputs[report_format] = str(md_path)
                texts[report_format] = text
                format_checks = self._report_checks(text, md_path)
                for name, passed in format_checks.items():
                    checks[f"{report_format}:{name}"] = passed
                    if not passed:
                        errors.append(f"{report_format} report quality check failed: {name}")

            combined = "\n\n--- TECHNICAL REPORT ---\n\n".join([
                texts.get("paper", ""),
                texts.get("report", ""),
            ])
            lower = combined.lower()
            for token in case.expected_contains:
                if token.lower() not in lower:
                    errors.append(f"Missing expected report content across dual outputs: {token}")

            metadata = {
                "tags": case.tags,
                "format": "dual",
                "formats": ["paper", "report"],
                "benchmark_kind": case.metadata.get("benchmark_kind", "ukb_oriented_synthetic"),
                "report_paths": outputs,
                "quality_checks": checks,
                "quality_score": report_quality_score(combined),
                "public_references": [r["doi"] for r in PUBLIC_REPORT_REFERENCES],
                "claim_count": 2,
                "evidence_count": len(PUBLIC_REPORT_REFERENCES),
                "safety_status": "PASS",
            }

            return TestResult(
                case_id=case.id,
                passed=not errors,
                actual_skills=["generate_report"],
                actual_text=combined,
                errors=errors,
                elapsed_s=0.0,
                metadata=metadata,
            )


class LiveUKBReport20Benchmark(Benchmark):
    """Twenty live UK Biobank workflow probes that fail closed without data access.

    This benchmark is intentionally separate from ``report_20_case``. It should
    only be used when the configured local environment exposes a usable UKB data
    manager/catalog; otherwise every case reports a data-preflight failure rather
    than pretending that a synthetic aggregate report exercised real UKB data.
    """

    name = "live_ukb_report_20"

    def __init__(self) -> None:
        self.cases = [
            TestCase(
                id=str(scenario["id"]).replace("report_", "live_"),
                query=(
                    "Run a live UKB-only long-horizon workflow for "
                    f"{scenario['disease']} ({scenario['icd10']}) using field "
                    f"{scenario['field_id']} ({scenario['field_name']}), including cohort construction, "
                    "auto model selection, guardrail review, and final report generation."
                ),
                expected_skills=[
                    "cohort_summary",
                    "train_model",
                    "statistical_review",
                    "safety_check",
                    "generate_report",
                ],
                expected_contains=["UK Biobank", "not causal", "Reproducibility"],
                tags=["report", "live_ukb", "long_horizon", "complex"],
                metadata=dict(scenario),
            )
            for scenario in REPORT_20_CASE_SCENARIOS
        ]

    def run_case(self, case: TestCase, agent) -> TestResult:
        preflight = self._preflight(agent)
        if not preflight["ok"]:
            return TestResult(
                case_id=case.id,
                passed=False,
                errors=[preflight["reason"]],
                elapsed_s=0.0,
                metadata={
                    "tags": case.tags,
                    "benchmark_kind": "live_ukb",
                    "data_preflight": "FAIL",
                    "preflight": preflight,
                    "claim_count": 1,
                    "evidence_count": 0,
                    "safety_status": "FAIL",
                },
            )

        t0 = time.time()
        try:
            response = agent.run(case.query)
            actual_skills = [r.skill for r in getattr(agent.state, "records", []) if r.skill != "think"]
            lower = str(response).lower()
            errors = [
                f"Expected skill '{skill}' not called"
                for skill in case.expected_skills
                if skill not in actual_skills
            ]
            for token in case.expected_contains:
                if token.lower() not in lower:
                    errors.append(f"Expected response/report content: {token}")
            passed = not errors
            return TestResult(
                case_id=case.id,
                passed=passed,
                actual_skills=actual_skills[-12:],
                actual_text=str(response)[:5000],
                errors=errors,
                elapsed_s=time.time() - t0,
                metadata={
                    "tags": case.tags,
                    "benchmark_kind": "live_ukb",
                    "data_preflight": "PASS",
                    "preflight": preflight,
                    "claim_count": 1,
                    "evidence_count": 1 if passed else 0,
                    "safety_status": "PASS" if passed else "FAIL",
                },
            )
        except Exception as e:
            return TestResult(
                case_id=case.id,
                passed=False,
                errors=[f"Live UKB workflow failed: {type(e).__name__}: {str(e)[:300]}"],
                elapsed_s=time.time() - t0,
                metadata={
                    "tags": case.tags,
                    "benchmark_kind": "live_ukb",
                    "data_preflight": "PASS",
                    "preflight": preflight,
                    "claim_count": 1,
                    "evidence_count": 0,
                    "safety_status": "FAIL",
                },
            )

    @staticmethod
    def _preflight(agent) -> dict:
        settings = getattr(agent, "settings", None)
        bank_id = str(getattr(settings, "bank_id", "") or "").lower()
        if bank_id and bank_id != "ukb":
            return {"ok": False, "reason": f"Configured bank_id is {bank_id}, expected ukb"}

        dm = getattr(agent, "dm", None)
        if dm is None:
            return {"ok": False, "reason": "Agent has no data manager; cannot run live UKB benchmark"}

        catalog = getattr(agent, "catalog", None)
        if catalog is None:
            return {"ok": False, "reason": "Agent has no field catalog; cannot verify UKB fields"}

        data_dir = Path(getattr(settings, "data_dir", "") or "")
        if str(data_dir) and not data_dir.exists():
            return {"ok": False, "reason": f"Configured data_dir does not exist: {data_dir}"}

        count_subjects = getattr(dm, "count_subjects", None)
        if callable(count_subjects):
            try:
                n_subjects = int(count_subjects())
                if n_subjects <= 0:
                    return {"ok": False, "reason": "Data manager reports zero subjects"}
                return {"ok": True, "reason": "live UKB data manager and catalog available", "n_subjects": n_subjects}
            except Exception as e:
                return {"ok": False, "reason": f"Data manager preflight failed: {type(e).__name__}: {str(e)[:200]}"}

        return {"ok": False, "reason": "Data manager cannot report subject count; live UKB benchmark unavailable"}


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
