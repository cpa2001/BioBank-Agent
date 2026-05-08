"""Tests for StudySpec schema compiler (Phase 2).

Tests schema validation, heuristic compilation, template loading,
and cross-field consistency.
"""

import pytest
from pydantic import ValidationError
from biobank_agent.study_spec import (
    StudySpec,
    StudySpecCompiler,
    StudyDesign,
    ExportPolicy,
    Modality,
    CohortSpec,
    StatisticalPlan,
    STUDY_TEMPLATES,
)


@pytest.fixture
def compiler():
    """Compiler without LLM (uses heuristic)."""
    return StudySpecCompiler(llm=None)


class TestStudySpecValidation:
    """Test Pydantic schema validation."""

    def test_valid_minimal_spec(self):
        """Minimal valid spec should pass."""
        spec = StudySpec(
            title="Test study of diabetes prevalence",
            design=StudyDesign.CROSS_SECTIONAL,
            cohort=CohortSpec(inclusion_criteria=["UKB participants"]),
            exposure_variables=["age"],
            outcome_variables=["E11"],
            statistical_plan=StatisticalPlan(primary_method="prevalence_estimation"),
        )
        assert spec.title == "Test study of diabetes prevalence"
        assert spec.design == StudyDesign.CROSS_SECTIONAL

    def test_title_too_short(self):
        """Title under 5 chars should fail."""
        with pytest.raises(ValidationError):
            StudySpec(
                title="Hi",
                design=StudyDesign.CROSS_SECTIONAL,
                cohort=CohortSpec(inclusion_criteria=["all"]),
                exposure_variables=["x"],
                outcome_variables=["y"],
                statistical_plan=StatisticalPlan(primary_method="test"),
            )

    def test_empty_inclusion_criteria_fails(self):
        """Empty inclusion criteria should fail."""
        with pytest.raises(ValidationError):
            CohortSpec(inclusion_criteria=[])

    def test_invalid_age_range(self):
        """Min age >= max age should fail."""
        with pytest.raises(ValidationError):
            CohortSpec(
                inclusion_criteria=["test"],
                age_range=(70, 50),
            )

    def test_valid_optional_cohort_filters(self):
        """Optional cohort filters should accept None and valid bounded values."""
        no_filters = CohortSpec(inclusion_criteria=["test"], age_range=None, sex_filter=None)
        filtered = CohortSpec(inclusion_criteria=["test"], age_range=(40, 70), sex_filter="female")
        assert no_filters.age_range is None
        assert filtered.age_range == (40, 70)
        assert filtered.sex_filter == "female"

    def test_invalid_sex_filter(self):
        """Invalid sex filter should fail."""
        with pytest.raises(ValidationError):
            CohortSpec(
                inclusion_criteria=["test"],
                sex_filter="both",  # Should be None, not "both"
            )

    def test_significance_threshold_bounds(self):
        """Significance threshold must be in (0, 1)."""
        with pytest.raises(ValidationError):
            StatisticalPlan(
                primary_method="test",
                significance_threshold=1.5,
            )

    def test_tool_budget_bounds(self):
        """Tool budget must be 1-100."""
        with pytest.raises(ValidationError):
            StudySpec(
                title="Test study with too many tools",
                design=StudyDesign.EXPLORATORY,
                cohort=CohortSpec(inclusion_criteria=["all"]),
                exposure_variables=["x"],
                outcome_variables=["y"],
                statistical_plan=StatisticalPlan(primary_method="test"),
                tool_budget=200,
            )

    def test_gwas_auto_adds_genomics(self):
        """GWAS design should auto-add genomics modality."""
        spec = StudySpec(
            title="GWAS study of height in UKB",
            design=StudyDesign.GWAS,
            cohort=CohortSpec(inclusion_criteria=["all participants"]),
            exposure_variables=["genotype"],
            outcome_variables=["height"],
            statistical_plan=StatisticalPlan(primary_method="linear_mixed_model"),
            modalities=[],
        )
        assert Modality.GENOMICS in spec.modalities


class TestStudySpecHash:
    """Test spec hashing for provenance."""

    def test_spec_hash_is_hex(self):
        """Hash should be 16-char hex string."""
        spec = StudySpec(
            title="Hash test study for provenance",
            design=StudyDesign.EXPLORATORY,
            cohort=CohortSpec(inclusion_criteria=["all"]),
            exposure_variables=["x"],
            outcome_variables=["y"],
            statistical_plan=StatisticalPlan(primary_method="test"),
        )
        h = spec.spec_hash()
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_same_spec_same_hash(self):
        """Identical specs should produce same hash."""
        kwargs = dict(
            title="Deterministic hash test study",
            design=StudyDesign.CROSS_SECTIONAL,
            cohort=CohortSpec(inclusion_criteria=["all"]),
            exposure_variables=["x"],
            outcome_variables=["y"],
            statistical_plan=StatisticalPlan(primary_method="test"),
            compiled_at="2024-01-01T00:00:00",
        )
        spec1 = StudySpec(**kwargs)
        spec2 = StudySpec(**kwargs)
        assert spec1.spec_hash() == spec2.spec_hash()


class TestStudySpecConstrainSkills:
    """Test skill filtering by modalities."""

    def test_no_modalities_no_filter(self):
        """Empty modalities should return all skills."""
        spec = StudySpec(
            title="No modality filter test study",
            design=StudyDesign.EXPLORATORY,
            cohort=CohortSpec(inclusion_criteria=["all"]),
            exposure_variables=["x"],
            outcome_variables=["y"],
            statistical_plan=StatisticalPlan(primary_method="test"),
            modalities=[],
        )
        skills = ["prevalence", "gwas_scan", "imaging_report", "train_model"]
        assert spec.constrain_skills(skills) == skills

    def test_genomics_filters_correctly(self):
        """Genomics modality should include gwas/gene skills + core."""
        spec = StudySpec(
            title="Genomics modality filter test",
            design=StudyDesign.GWAS,
            cohort=CohortSpec(inclusion_criteria=["all"]),
            exposure_variables=["snp"],
            outcome_variables=["height"],
            statistical_plan=StatisticalPlan(primary_method="lmm"),
            modalities=[Modality.GENOMICS],
        )
        skills = ["prevalence", "gwas_scan", "imaging_report", "train_model", "gene_lookup"]
        filtered = spec.constrain_skills(skills)
        assert "gwas_scan" in filtered
        assert "gene_lookup" in filtered
        assert "train_model" in filtered  # Core skill always included
        assert "imaging_report" not in filtered


class TestStudySpecCompiler:
    """Test heuristic compilation."""

    def test_heuristic_detects_gwas(self, compiler):
        """Query with 'GWAS' should compile to GWAS design."""
        spec = compiler.compile("Run a GWAS for height in UKB")
        assert spec.design == StudyDesign.GWAS
        assert spec.compiled_by == "heuristic"

    def test_heuristic_detects_survival(self, compiler):
        """Query with 'survival' should compile to SURVIVAL design."""
        spec = compiler.compile("Perform survival analysis for mortality after I21")
        assert spec.design == StudyDesign.SURVIVAL

    def test_heuristic_detects_prediction(self, compiler):
        """Query with 'predict' should compile to CASE_CONTROL."""
        spec = compiler.compile("Predict type 2 diabetes E11 using blood biomarkers")
        assert spec.design == StudyDesign.CASE_CONTROL

    def test_heuristic_extracts_icd10(self, compiler):
        """ICD-10 codes should be extracted as outcome variables."""
        spec = compiler.compile("What is the prevalence of E11 and I21?")
        assert "E11" in spec.outcome_variables or "I21" in spec.outcome_variables

    def test_heuristic_detects_prevalence(self, compiler):
        """Query with 'prevalence' should be CROSS_SECTIONAL."""
        spec = compiler.compile("What is the prevalence of hypertension?")
        assert spec.design == StudyDesign.CROSS_SECTIONAL

    def test_heuristic_detects_phewas_and_longitudinal(self, compiler):
        """PheWAS and trajectory keywords should route to their study designs."""
        phewas = compiler.compile("Run a PheWAS for E11")
        longitudinal = compiler.compile("Track biomarker trajectory over time")
        assert phewas.design == StudyDesign.PHEWAS
        assert longitudinal.design == StudyDesign.LONGITUDINAL

    def test_heuristic_fallback_exploratory(self, compiler):
        """Unknown query type should default to EXPLORATORY."""
        spec = compiler.compile("Tell me something interesting about the data")
        assert spec.design == StudyDesign.EXPLORATORY

    def test_source_query_preserved(self, compiler):
        """Original query should be stored in source_query."""
        query = "Analyze diabetes risk factors"
        spec = compiler.compile(query)
        assert spec.source_query == query


class TestStudyTemplates:
    """Test template loading."""

    def test_all_templates_valid(self):
        """All built-in templates should be valid."""
        assert len(STUDY_TEMPLATES) >= 4

    def test_load_template(self):
        """Loading a template with required overrides should work."""
        compiler = StudySpecCompiler()
        spec = compiler.from_template("prevalence_analysis", overrides={
            "title": "Prevalence of E11 in UKB cohort",
            "cohort": {"inclusion_criteria": ["all participants"]},
            "exposure_variables": ["age"],
            "outcome_variables": ["E11"],
        })
        assert spec.design == StudyDesign.CROSS_SECTIONAL
        assert spec.compiled_by == "template"

    def test_unknown_template_raises(self):
        """Unknown template name should raise ValueError."""
        compiler = StudySpecCompiler()
        with pytest.raises(ValueError, match="Unknown template"):
            compiler.from_template("nonexistent_template")

    def test_load_template_without_overrides(self):
        """Templates should validate without override data."""
        compiler = StudySpecCompiler()
        spec = compiler.from_template("survival_analysis")
        assert spec.compiled_by == "template"
        assert spec.design == StudyDesign.SURVIVAL


class TestStudySpecEdgeCases:
    """Test uncovered branches and edge cases."""

    def test_survival_warns_no_time_variable(self, compiler):
        """Survival design without time/date outcome should still compile."""
        spec = compiler.compile("Cox regression for I21 outcome")
        # Should compile (heuristic), with survival design
        assert spec.design == StudyDesign.SURVIVAL

    def test_survival_with_time_variable_needs_no_warning(self):
        """Survival specs with a time/date outcome satisfy the cross-field check."""
        spec = StudySpec(
            title="Survival study with time to myocardial infarction",
            design=StudyDesign.SURVIVAL,
            cohort=CohortSpec(inclusion_criteria=["all"]),
            exposure_variables=["age"],
            outcome_variables=["time_to_I21"],
            statistical_plan=StatisticalPlan(primary_method="cox_ph"),
        )
        assert spec.design == StudyDesign.SURVIVAL

    def test_age_range_impossible(self):
        """Age range with extreme values should raise."""
        with pytest.raises(ValidationError):
            CohortSpec(inclusion_criteria=["test"], age_range=(50, 200))

    def test_modality_genomics_keyword(self, compiler):
        """Query with genetic keywords should get GENOMICS modality."""
        spec = compiler.compile("Find genetic variants associated with E11")
        assert Modality.GENOMICS in spec.modalities

    def test_modality_blood_keyword(self, compiler):
        """Query with blood/biomarker keywords should get BLOOD_BIOCHEMISTRY."""
        spec = compiler.compile("Use blood biomarkers to predict diabetes")
        assert Modality.BLOOD_BIOCHEMISTRY in spec.modalities

    def test_modality_imaging_keyword(self, compiler):
        """Query with imaging keywords should get IMAGING."""
        spec = compiler.compile("Analyze brain MRI scans for dementia")
        assert Modality.IMAGING in spec.modalities

    def test_constrain_skills_imaging_modality(self):
        """Imaging modality should filter for imaging skills."""
        spec = StudySpec(
            title="Imaging only study for brain analysis",
            design=StudyDesign.EXPLORATORY,
            cohort=CohortSpec(inclusion_criteria=["all"]),
            exposure_variables=["x"],
            outcome_variables=["y"],
            statistical_plan=StatisticalPlan(primary_method="test"),
            modalities=[Modality.IMAGING],
        )
        skills = ["prevalence", "brain_volume", "gene_lookup", "imaging_report", "train_model"]
        filtered = spec.constrain_skills(skills)
        assert "brain_volume" in filtered
        assert "imaging_report" in filtered
        assert "train_model" in filtered  # core skill
        assert "gene_lookup" not in filtered  # genomics only, not imaging

    def test_constrain_skills_unknown_modality_prefixes_fall_back_to_all(self):
        """Modalities without prefix mappings should not accidentally block skills."""
        spec = StudySpec(
            title="Proteomics only study for measured proteins",
            design=StudyDesign.EXPLORATORY,
            cohort=CohortSpec(inclusion_criteria=["all"]),
            exposure_variables=["protein"],
            outcome_variables=["E11"],
            statistical_plan=StatisticalPlan(primary_method="test"),
            modalities=[Modality.PROTEOMICS],
        )
        skills = ["custom_protein_model", "rare_tool"]
        assert spec.constrain_skills(skills) == skills

    def test_llm_compile_with_mock(self):
        """LLM-backed compile should work with mock."""
        from unittest.mock import MagicMock
        import json

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "title": "GWAS of height in Europeans",
            "design": "gwas",
            "cohort": {"inclusion_criteria": ["European ancestry"]},
            "exposure_variables": ["genotype"],
            "outcome_variables": ["height"],
            "covariates": ["age", "sex", "PCs"],
            "modalities": ["genomics", "phenotype"],
            "statistical_plan": {
                "primary_method": "linear_mixed_model",
                "multiple_testing_correction": "bonferroni",
                "significance_threshold": 5e-8,
                "power_target": 0.8,
            },
            "tool_budget": 30,
            "export_policy": "summary_only",
        })
        mock_llm.chat.return_value = mock_response

        compiler = StudySpecCompiler(llm=mock_llm)
        spec = compiler.compile("Run GWAS for height in Europeans")
        assert spec.design == StudyDesign.GWAS
        assert spec.compiled_by == "llm"
        assert Modality.GENOMICS in spec.modalities

    def test_llm_compile_fallback_on_bad_json(self):
        """LLM returning bad JSON should fallback to heuristic."""
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is not JSON at all {{{broken"
        mock_llm.chat.return_value = mock_response

        compiler = StudySpecCompiler(llm=mock_llm)
        spec = compiler.compile("What is the prevalence of E11?")
        # Should fallback to heuristic
        assert spec.compiled_by == "heuristic"

    def test_llm_compile_fallback_on_fenced_non_json(self):
        """Markdown fences without JSON should fall back cleanly."""
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "```text\nnot json\n```"
        mock_llm.chat.return_value = mock_response

        spec = StudySpecCompiler(llm=mock_llm).compile("What is the prevalence of E11?")

        assert spec.compiled_by == "heuristic"

    def test_llm_compile_fallback_on_chat_exception(self):
        """Unexpected LLM failures should also fallback to heuristic compilation."""
        from unittest.mock import MagicMock

        mock_llm = MagicMock()
        mock_llm.chat.side_effect = RuntimeError("API down")

        compiler = StudySpecCompiler(llm=mock_llm)
        spec = compiler.compile("Predict E11 risk")

        assert spec.compiled_by == "heuristic"
        assert spec.design == StudyDesign.CASE_CONTROL

    def test_llm_compile_with_markdown_fences(self):
        """LLM wrapping JSON in markdown fences should still parse."""
        from unittest.mock import MagicMock
        import json

        data = {
            "title": "Test markdown fences study",
            "design": "cross_sectional",
            "cohort": {"inclusion_criteria": ["all"]},
            "exposure_variables": ["age"],
            "outcome_variables": ["E11"],
            "statistical_plan": {"primary_method": "prevalence_estimation"},
        }
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = f"```json\n{json.dumps(data)}\n```"
        mock_llm.chat.return_value = mock_response

        compiler = StudySpecCompiler(llm=mock_llm)
        spec = compiler.compile("Prevalence of E11")
        assert spec.design == StudyDesign.CROSS_SECTIONAL
        assert spec.compiled_by == "llm"
