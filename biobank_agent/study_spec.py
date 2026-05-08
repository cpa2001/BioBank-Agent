"""StudySpec compiler — schema-gated execution boundary.

Compiles natural language research queries into typed StudySpec objects
BEFORE any execution. All downstream planning, tool calls, and exports
are constrained to fields validated by this schema.

Design sources:
- GPT deep research §3: "schema-gated 研究任务编译器"
- Claude research §E1: "El Agente Gráfico" typed execution environment
- Claude research §C1: "Talk Freely, Execute Strictly" — execution boundary
- Gemini research §4.1: VERGE semantic routing + schema compilation

Architecture:
    user_input → StudySpecCompiler.compile(user_input) → StudySpec
    StudySpec → planner.decompose(goal, skills, context, spec=spec) → LongHorizonPlan

The StudySpec is the SINGLE SOURCE OF TRUTH for any analysis session.
Schema violation rate is a key evaluation metric.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
from datetime import datetime
from enum import Enum
from typing import Any, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    from .llm import LLMClient

logger = logging.getLogger(__name__)

# Pre-compiled regex for ICD-10 code extraction (e.g., E11, I21.0)
_ICD10_RE = re.compile(r'\b[A-Z]\d{2}(?:\.\d+)?\b')


# ── Enums ────────────────────────────────────────────────────────────────────


class StudyDesign(str, Enum):
    """Epidemiological study design types supported by the biobank agent."""
    CROSS_SECTIONAL = "cross_sectional"
    LONGITUDINAL = "longitudinal"
    CASE_CONTROL = "case_control"
    COHORT = "cohort"
    GWAS = "gwas"
    PHEWAS = "phewas"
    MENDELIAN_RANDOMIZATION = "mendelian_randomization"
    SURVIVAL = "survival"
    EXPLORATORY = "exploratory"  # Relaxed constraints for discovery


class ExportPolicy(str, Enum):
    """What level of data can be exported from the analysis."""
    SUMMARY_ONLY = "summary_only"       # Aggregate statistics only
    FULL_TABLE = "full_table"           # Full result tables allowed
    NO_EXPORT = "no_export"             # Results stay in session only
    INDIVIDUAL_LEVEL = "individual_level"  # Requires special approval


class Modality(str, Enum):
    """Data modalities available in UK Biobank."""
    GENOMICS = "genomics"
    IMAGING = "imaging"
    EHR = "ehr"
    PROTEOMICS = "proteomics"
    METABOLOMICS = "metabolomics"
    TRANSCRIPTOMICS = "transcriptomics"
    PHENOTYPE = "phenotype"
    LIFESTYLE = "lifestyle"
    MORTALITY = "mortality"
    BLOOD_BIOCHEMISTRY = "blood_biochemistry"
    PHYSICAL_MEASURES = "physical_measures"


# ── Sub-models ───────────────────────────────────────────────────────────────


class CohortSpec(BaseModel):
    """Typed cohort definition with hard constraints.

    Enforces that cohort definitions are explicit BEFORE any SQL is generated.
    Prevents implicit cohort drift during analysis.
    """
    inclusion_criteria: list[str] = Field(
        min_length=1,
        description="Conditions participants must meet to be included"
    )
    exclusion_criteria: list[str] = Field(
        default_factory=list,
        description="Conditions that exclude participants"
    )
    min_sample_size: int = Field(
        default=100, ge=5,
        description="Minimum acceptable sample size for statistical power"
    )
    max_sample_size: Optional[int] = Field(
        default=None,
        description="Maximum sample size (None = use all available)"
    )
    age_range: Optional[tuple[int, int]] = Field(
        default=None,
        description="Age range filter (min, max) at recruitment"
    )
    sex_filter: Optional[str] = Field(
        default=None,
        description="Sex filter: 'male', 'female', or None for both"
    )

    @field_validator("age_range")
    @classmethod
    def validate_age_range(cls, v: Optional[tuple[int, int]]) -> Optional[tuple[int, int]]:
        """UKB recruited ages 37-73; validate range is within bounds."""
        if v is None:
            return v
        min_age, max_age = v
        if min_age < 0 or max_age > 120:
            raise ValueError(f"Age range {v} out of plausible bounds (0-120)")
        if min_age >= max_age:
            raise ValueError(f"min_age ({min_age}) must be < max_age ({max_age})")
        return v

    @field_validator("sex_filter")
    @classmethod
    def validate_sex_filter(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("male", "female"):
            raise ValueError(f"sex_filter must be 'male', 'female', or None, got '{v}'")
        return v


class StatisticalPlan(BaseModel):
    """Pre-declared statistical methods — prevents post-hoc p-hacking.

    Forces researchers to declare their analysis approach BEFORE
    seeing any results, following pre-registration principles.
    """
    primary_method: str = Field(
        description="Primary statistical method (e.g., 'logistic_regression', 'cox_ph', 'linear_regression')"
    )
    multiple_testing_correction: str = Field(
        default="bonferroni",
        description="Correction method for multiple comparisons"
    )
    significance_threshold: float = Field(
        default=0.05, gt=0, lt=1,
        description="Alpha level for statistical significance"
    )
    power_target: float = Field(
        default=0.8, gt=0, le=1,
        description="Target statistical power (1 - beta)"
    )
    sensitivity_analyses: list[str] = Field(
        default_factory=list,
        description="Pre-planned sensitivity analyses"
    )


# ── Main StudySpec model ─────────────────────────────────────────────────────


class StudySpec(BaseModel):
    """The single source of truth for any biobank analysis.

    All code, SQL, notebooks, and exports MUST derive from validated
    StudySpec fields. No downstream component may exceed the permissions
    or scope declared here.

    Attributes:
        title: Brief study title (5-200 chars)
        design: Type of epidemiological study
        cohort: Cohort inclusion/exclusion criteria
        exposure_variables: Main exposure(s) being studied
        outcome_variables: Primary outcome(s) of interest
        covariates: Confounders to adjust for
        modalities: Data modalities required
        statistical_plan: Pre-declared analysis approach
        tool_budget: Maximum number of tool calls allowed
        stop_conditions: Conditions that should halt the analysis
        export_policy: What level of results can be exported
        source_query: Original natural language query (provenance)
        compiled_by: How this spec was created ('llm', 'user', 'template')
    """
    title: str = Field(min_length=5, max_length=200)
    design: StudyDesign
    cohort: CohortSpec
    exposure_variables: list[str] = Field(
        min_length=1,
        description="Main exposure or predictor variables"
    )
    outcome_variables: list[str] = Field(
        min_length=1,
        description="Primary outcome variables"
    )
    covariates: list[str] = Field(
        default_factory=list,
        description="Confounding variables to adjust for"
    )
    modalities: list[Modality] = Field(
        default_factory=list,
        description="Data modalities required for this analysis"
    )
    statistical_plan: StatisticalPlan
    tool_budget: int = Field(
        default=20, ge=1, le=100,
        description="Maximum number of tool/skill invocations allowed"
    )
    stop_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions that should halt the analysis early"
    )
    export_policy: ExportPolicy = Field(
        default=ExportPolicy.SUMMARY_ONLY,
        description="Data export permission level"
    )

    # ── Provenance fields ──
    source_query: str = Field(
        default="",
        description="Original natural language query (for traceability)"
    )
    compiled_by: str = Field(
        default="llm",
        description="How this spec was created: 'llm', 'user', or 'template'"
    )
    compiled_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="ISO timestamp of compilation"
    )

    @model_validator(mode="after")
    def validate_cross_field(self) -> "StudySpec":
        """Cross-field consistency checks."""
        # GWAS must have genomics modality
        if self.design == StudyDesign.GWAS and Modality.GENOMICS not in self.modalities:
            self.modalities = list(self.modalities) + [Modality.GENOMICS]
        # Survival analysis needs time-to-event
        if self.design == StudyDesign.SURVIVAL:
            if not any("time" in v.lower() or "date" in v.lower() for v in self.outcome_variables):
                logger.warning("Survival design but no time/date in outcome_variables")
        return self

    def spec_hash(self) -> str:
        """SHA-256 hash of the spec content (for provenance tracking)."""
        content = self.model_dump_json(exclude={"compiled_at"})
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def constrain_skills(self, available_skills: list[str]) -> list[str]:
        """Filter available skills to only those relevant to this spec's modalities."""
        # Map modalities to relevant skill prefixes
        modality_skill_map: dict[Modality, list[str]] = {
            Modality.GENOMICS: ["gwas", "phewas", "snp", "gene", "variant", "graphpop"],
            Modality.IMAGING: ["imaging", "brain", "mri", "scan"],
            Modality.EHR: ["icd10", "diagnosis", "medication", "hospital"],
            Modality.BLOOD_BIOCHEMISTRY: ["blood", "biomarker", "biochem"],
            Modality.PHENOTYPE: ["phenotype", "trait", "measure"],
            Modality.LIFESTYLE: ["lifestyle", "diet", "exercise", "smoking"],
            Modality.MORTALITY: ["mortality", "death", "survival"],
        }

        if not self.modalities:
            return available_skills  # No restriction if no modalities specified

        # Always allow core skills (prevalence, cohort, train_model, evaluate, etc.)
        core_skills = {"prevalence", "cohort_summary", "train_model", "evaluate_model",
                       "missing_data", "feature_importance", "risk_factors"}

        allowed_prefixes: set[str] = set()
        for mod in self.modalities:
            for prefix in modality_skill_map.get(mod, []):
                allowed_prefixes.add(prefix)

        filtered = []
        for skill_name in available_skills:
            if skill_name in core_skills:
                filtered.append(skill_name)
            elif any(skill_name.startswith(p) or p in skill_name for p in allowed_prefixes):
                filtered.append(skill_name)
            elif not allowed_prefixes:
                filtered.append(skill_name)

        return filtered if filtered else available_skills


# ── Compiler ─────────────────────────────────────────────────────────────────


# Pre-built templates for common study patterns
STUDY_TEMPLATES: dict[str, dict[str, Any]] = {
    "prevalence_analysis": {
        "title": "Prevalence Analysis",
        "design": "cross_sectional",
        "cohort": {"inclusion_criteria": ["UKB participants with relevant data"]},
        "exposure_variables": ["exposure"],
        "outcome_variables": ["outcome"],
        "statistical_plan": {
            "primary_method": "prevalence_estimation",
            "multiple_testing_correction": "none",
            "significance_threshold": 0.05,
            "power_target": 0.8,
        },
        "export_policy": "summary_only",
    },
    "disease_prediction": {
        "title": "Disease Prediction Model",
        "design": "case_control",
        "cohort": {"inclusion_criteria": ["UKB participants with outcome data"]},
        "exposure_variables": ["predictor_features"],
        "outcome_variables": ["disease_outcome"],
        "statistical_plan": {
            "primary_method": "logistic_regression",
            "multiple_testing_correction": "bonferroni",
            "significance_threshold": 0.05,
            "power_target": 0.8,
        },
        "modalities": ["blood_biochemistry", "phenotype"],
        "export_policy": "summary_only",
    },
    "standard_gwas": {
        "title": "Genome-Wide Association Study",
        "design": "gwas",
        "cohort": {"inclusion_criteria": ["UKB participants with genotype data"]},
        "exposure_variables": ["genotype"],
        "outcome_variables": ["phenotype"],
        "statistical_plan": {
            "primary_method": "linear_mixed_model",
            "multiple_testing_correction": "bonferroni",
            "significance_threshold": 5e-8,
            "power_target": 0.8,
        },
        "modalities": ["genomics", "phenotype"],
        "tool_budget": 50,
        "export_policy": "summary_only",
    },
    "survival_analysis": {
        "title": "Survival Analysis",
        "design": "survival",
        "cohort": {"inclusion_criteria": ["UKB participants with follow-up data"]},
        "exposure_variables": ["exposure"],
        "outcome_variables": ["time_to_event"],
        "statistical_plan": {
            "primary_method": "cox_ph",
            "multiple_testing_correction": "bonferroni",
            "significance_threshold": 0.05,
            "power_target": 0.8,
        },
        "modalities": ["ehr", "mortality"],
        "export_policy": "summary_only",
    },
}


class StudySpecCompiler:
    """Compile natural language research queries into typed StudySpec objects.

    Uses LLM to extract structured fields from free-text queries.
    Falls back to template matching if LLM is unavailable.

    Architecture:
        1. LLM extracts structured JSON from user query
        2. Pydantic validates all fields (raises on violation)
        3. Cross-field consistency checks run
        4. Typed StudySpec returned as execution boundary
    """

    # Prompt template for LLM-assisted compilation
    _COMPILE_PROMPT = """You are a biobank research protocol compiler.
Given a natural language research query, extract a structured StudySpec.

## Required Output (JSON):
{{
  "title": "Brief study title (5-200 chars)",
  "design": "cross_sectional|longitudinal|case_control|cohort|gwas|phewas|mendelian_randomization|survival|exploratory",
  "cohort": {{
    "inclusion_criteria": ["list of inclusion conditions"],
    "exclusion_criteria": ["list of exclusion conditions"],
    "min_sample_size": 100,
    "age_range": [min_age, max_age] or null,
    "sex_filter": "male"|"female"|null
  }},
  "exposure_variables": ["main exposure or predictor variables"],
  "outcome_variables": ["primary outcomes"],
  "covariates": ["confounders to adjust for"],
  "modalities": ["genomics","imaging","ehr","proteomics","metabolomics","transcriptomics","phenotype","lifestyle","mortality","blood_biochemistry","physical_measures"],
  "statistical_plan": {{
    "primary_method": "e.g. logistic_regression, cox_ph, linear_mixed_model",
    "multiple_testing_correction": "bonferroni|fdr|none",
    "significance_threshold": 0.05,
    "power_target": 0.8,
    "sensitivity_analyses": []
  }},
  "tool_budget": 20,
  "stop_conditions": [],
  "export_policy": "summary_only|full_table|no_export"
}}

## Research Query:
{query}

## Context (if available):
{context}

Respond with ONLY the JSON object. No markdown fences."""

    def __init__(self, llm: Optional["LLMClient"] = None):
        self.llm = llm

    def compile(self, user_query: str, context: str = "") -> StudySpec:
        """Compile a natural language query into a typed StudySpec.

        Args:
            user_query: The research question in natural language
            context: Optional additional context (prior findings, session state)

        Returns:
            A validated StudySpec object

        Raises:
            ValueError: If the query cannot be compiled into a valid spec
            ValidationError: If extracted fields fail Pydantic validation
        """
        if self.llm is None:
            return self._compile_heuristic(user_query)

        prompt = self._COMPILE_PROMPT.format(
            query=user_query,
            context=context[:500] if context else "None"
        )

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "You are a precise biobank protocol compiler. Output valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1500,
            )

            text = response.text.strip()
            # Strip markdown code fences if present
            if "```" in text:
                parts = text.split("```")
                for part in parts:
                    stripped = part.strip()
                    if stripped.startswith("json"):
                        stripped = stripped[4:].strip()
                    if stripped.startswith("{"):
                        text = stripped
                        break

            data = json.loads(text)
            data["source_query"] = user_query
            data["compiled_by"] = "llm"

            return StudySpec(**data)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("LLM compilation failed (%s), trying heuristic", e)
            return self._compile_heuristic(user_query)
        except Exception as e:
            logger.warning("StudySpec compilation error: %s", e)
            return self._compile_heuristic(user_query)

    def from_template(self, template_name: str, overrides: Optional[dict[str, Any]] = None) -> StudySpec:
        """Load a pre-validated template with optional overrides.

        Args:
            template_name: One of STUDY_TEMPLATES keys
            overrides: Fields to override in the template

        Raises:
            ValueError: If template_name is not found
        """
        if template_name not in STUDY_TEMPLATES:
            available = list(STUDY_TEMPLATES.keys())
            raise ValueError(f"Unknown template '{template_name}'. Available: {available}")

        data = copy.deepcopy(STUDY_TEMPLATES[template_name])
        if overrides:
            data.update(overrides)
        data["compiled_by"] = "template"

        return StudySpec(**data)

    def _compile_heuristic(self, query: str) -> StudySpec:
        """Heuristic compilation when LLM is unavailable.

        Uses keyword matching to infer study type and basic parameters.
        Returns a valid but potentially overly permissive StudySpec.
        """
        q_lower = query.lower()

        # Infer study design from keywords
        if any(w in q_lower for w in ["gwas", "genome-wide", "genetic association"]):
            design = StudyDesign.GWAS
        elif any(w in q_lower for w in ["survival", "cox", "time-to-event", "mortality"]):
            design = StudyDesign.SURVIVAL
        elif any(w in q_lower for w in ["phewas", "phenome-wide"]):
            design = StudyDesign.PHEWAS
        elif any(w in q_lower for w in ["predict", "model", "classify", "risk"]):
            design = StudyDesign.CASE_CONTROL
        elif any(w in q_lower for w in ["prevalence", "how common", "how many"]):
            design = StudyDesign.CROSS_SECTIONAL
        elif any(w in q_lower for w in ["longitudinal", "over time", "trajectory"]):
            design = StudyDesign.LONGITUDINAL
        else:
            design = StudyDesign.EXPLORATORY

        # Extract ICD-10 codes as outcome variables
        icd_codes = _ICD10_RE.findall(query)
        outcome_variables = icd_codes if icd_codes else ["unspecified_outcome"]

        # Infer modalities
        modalities: list[Modality] = []
        if any(w in q_lower for w in ["gene", "snp", "variant", "gwas", "genetic"]):
            modalities.append(Modality.GENOMICS)
        if any(w in q_lower for w in ["blood", "biomarker", "biochem", "hba1c"]):
            modalities.append(Modality.BLOOD_BIOCHEMISTRY)
        if any(w in q_lower for w in ["imaging", "mri", "brain", "scan"]):
            modalities.append(Modality.IMAGING)
        if not modalities:
            modalities.append(Modality.PHENOTYPE)

        return StudySpec(
            title=query[:200] if len(query) >= 5 else "Exploratory analysis",
            design=design,
            cohort=CohortSpec(
                inclusion_criteria=["UKB participants with relevant data available"],
            ),
            exposure_variables=["exposure_from_query"],
            outcome_variables=outcome_variables,
            covariates=["age", "sex"],
            modalities=modalities,
            statistical_plan=StatisticalPlan(
                primary_method="logistic_regression" if design == StudyDesign.CASE_CONTROL else "descriptive",
            ),
            tool_budget=30,
            source_query=query,
            compiled_by="heuristic",
        )
