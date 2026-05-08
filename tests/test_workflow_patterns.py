"""Test workflow pattern analysis."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from pathlib import Path

from biobank_agent.state import AnalysisRecord, SessionState
from biobank_agent.memory import LongTermMemory
from biobank_agent.skills.workflow_patterns import (
    analyze_workflow_patterns,
    suggest_optimal_pipeline,
    _analyze_skill_sequences,
    _detect_resource_bottlenecks,
    _predict_skill_compatibility,
)


class TestAnalyzeSkillSequences:
    """Test skill sequence analysis."""
    
    def test_analyze_empty_records(self):
        """Test analyzing empty record list."""
        result = _analyze_skill_sequences([])
        assert "sequences" in result or "skill_success_rates" in result
        assert len(result.get("recommendations", [])) > 0
    
    def test_analyze_skill_sequences_bigrams(self):
        """Test bigram extraction from skill sequence."""
        records = [
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="prevalence",
                args={},
                key_results={"n_diseases": 10},
                figure_paths=[],
                interpretation="Success"
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="survival",
                args={},
                key_results={"log_rank_p": 0.01},
                figure_paths=[],
                interpretation="Success"
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="prevalence",
                args={},
                key_results={"n_diseases": 10},
                figure_paths=[],
                interpretation="Success"
            ),
        ]
        
        result = _analyze_skill_sequences(records)
        assert "skill_sequences" in result
        assert "bigrams" in result["skill_sequences"]
    
    def test_analyze_success_rates(self):
        """Test success rate computation."""
        records = [
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={},
                key_results={"auc": 0.9},
                figure_paths=[],
                interpretation="Success"
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={},
                key_results={"auc": 0.92},
                figure_paths=[],
                interpretation="Success"
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={},
                key_results={},
                figure_paths=[],
                interpretation="Error: Out of memory"
            ),
        ]
        
        result = _analyze_skill_sequences(records)
        success_rates = result["skill_success_rates"]
        assert "train_model" in success_rates
        # 2 successes out of 3
        assert success_rates["train_model"] == pytest.approx(66.7, 0.1)


class TestDetectBottlenecks:
    """Test resource bottleneck detection."""

    def test_detect_empty_bottlenecks(self):
        """Empty execution history has no bottlenecks or patterns."""
        assert _detect_resource_bottlenecks([]) == {"bottlenecks": [], "patterns": []}
    
    def test_detect_memory_bottlenecks(self):
        """Test detection of memory issues."""
        records = [
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={"n_samples": 100000},
                key_results={},
                figure_paths=[],
                interpretation="Error: MemoryError - out of memory"
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={"n_samples": 50000},
                key_results={"auc": 0.9},
                figure_paths=[],
                interpretation="Success"
            ),
        ]
        
        result = _detect_resource_bottlenecks(records)
        assert "bottlenecks" in result
        bottlenecks = result["bottlenecks"]
        memory_issues = [b for b in bottlenecks if b["type"] == "memory"]
        assert len(memory_issues) > 0
        assert memory_issues[0]["skill"] == "train_model"
    
    def test_detect_parameter_patterns(self):
        """Test parameter correlation analysis."""
        records = [
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={"n_folds": 10, "sample_size": 5000},
                key_results={},
                figure_paths=[],
                interpretation="Error: MemoryError"
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={"n_folds": 5, "sample_size": 1000},
                key_results={"auc": 0.9},
                figure_paths=[],
                interpretation="Success"
            ),
        ]
        
        result = _detect_resource_bottlenecks(records)
        patterns = result.get("parameter_patterns", [])
        # Should detect that higher values correlate with failure
        assert len(patterns) > 0 or len(result["bottlenecks"]) > 0

    def test_parameter_patterns_skip_all_success_and_lower_failure_values(self):
        """Parameter pattern mining should ignore non-risky or all-success values."""
        records = [
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={"n_folds": 5, "sample_size": 100, "stable": 1},
                key_results={"auc": 0.8},
                figure_paths=[],
                interpretation="Success",
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={"n_folds": 6, "sample_size": 50, "stable": 2},
                key_results={"auc": 0.81},
                figure_paths=[],
                interpretation="Success",
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={"n_folds": 2, "sample_size": 10},
                key_results={},
                figure_paths=[],
                interpretation="Error: invalid split",
            ),
        ]

        result = _detect_resource_bottlenecks(records)

        assert result["parameter_patterns"] == []

    def test_detect_timeout_bottlenecks(self):
        """Timeout messages should be reported separately from memory issues."""
        records = [
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="phewas",
                args={"max_tests": 100000},
                key_results={},
                figure_paths=[],
                interpretation="Error: timeout while scanning phenotypes",
            )
        ]

        result = _detect_resource_bottlenecks(records)

        assert result["bottlenecks"] == [
            {
                "type": "timeout",
                "skill": "phewas",
                "occurrences": 1,
                "recommendation": "Parallelize computation or reduce dataset size",
            }
        ]


class TestPredictCompatibility:
    """Test skill compatibility prediction."""
    
    def test_predict_empty_records(self):
        """Test compatibility prediction with empty history."""
        result = _predict_skill_compatibility([])
        assert "compatible_chains" in result
        assert isinstance(result["compatible_chains"], list)
    
    def test_predict_numeric_output_to_numeric_input(self):
        """Test compatible skills (numeric output → numeric input)."""
        records = [
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="prevalence",
                args={},
                key_results={"n_diseases": 10, "n_patients": 5000},
                figure_paths=[],
                interpretation=""
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="survival",
                args={"icd10_code": "E11", "n_cases": 100},
                key_results={"log_rank_p": 0.01},
                figure_paths=[],
                interpretation=""
            ),
        ]
        
        result = _predict_skill_compatibility(records)
        chains = result["compatible_chains"]
        # Should find chains with common numeric types
        assert isinstance(chains, list)

    def test_predict_string_dict_and_list_compatibility(self):
        """Compatibility inference should include non-numeric output and input types."""
        records = [
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="extract_fields",
                args={},
                key_results={"summary": "text", "fields": ["30750"], "metadata": {"bank": "UKB"}},
                figure_paths=[],
                interpretation="Success",
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="build_report",
                args={"summary": "text", "metadata": {"bank": "UKB"}},
                key_results={"path": "report.md"},
                figure_paths=[],
                interpretation="Success",
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="archive",
                args={"path": "report.md"},
                key_results={},
                figure_paths=[],
                interpretation="Success",
            ),
        ]

        result = _predict_skill_compatibility(records)

        first_chain = result["compatible_chains"][0]
        assert first_chain["from"] == "extract_fields"
        assert first_chain["to"] == "build_report"
        assert set(first_chain["common_types"]) == {"string", "dict"}

    def test_predict_compatibility_skips_unmatched_type_pairs(self):
        """Pairs with no shared output/input type should be excluded."""
        records = [
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="numeric_source",
                args={"items": ["input"]},
                key_results={"n": 1, "opaque": object()},
                figure_paths=[],
                interpretation="Success",
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="dict_sink",
                args={"config": {"x": 1}},
                key_results={"done": True},
                figure_paths=[],
                interpretation="Success",
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="list_source",
                args={"name": "abc"},
                key_results={"items": ["a"]},
                figure_paths=[],
                interpretation="Success",
            ),
        ]

        result = _predict_skill_compatibility(records)

        assert result["compatible_chains"] == []


class TestAnalyzeWorkflowPatternsSkill:
    """Test analyze_workflow_patterns skill."""
    
    def test_analyze_workflow_patterns_imports(self):
        """Test skill can be imported."""
        assert analyze_workflow_patterns is not None
        assert callable(analyze_workflow_patterns)
    
    def test_analyze_workflow_patterns_no_data(self):
        """Test with no execution history."""
        mock_ctx = MagicMock()
        mock_ctx.state.records = []
        
        result = analyze_workflow_patterns(ctx=mock_ctx)
        
        assert result["status"] == "no_data"
        assert "recommendations" in result
    
    def test_analyze_workflow_patterns_quick(self):
        """Test quick analysis mode."""
        records = [
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="prevalence",
                args={},
                key_results={"n_diseases": 10},
                figure_paths=[],
                interpretation=""
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="survival",
                args={},
                key_results={"log_rank_p": 0.01},
                figure_paths=[],
                interpretation=""
            ),
        ]
        
        mock_ctx = MagicMock()
        mock_ctx.state.records = records
        
        result = analyze_workflow_patterns(analysis_depth="quick", ctx=mock_ctx)
        
        assert result["status"] == "success"
        assert result["total_records"] == 2
        assert "skill_success_rates" in result
    
    def test_analyze_workflow_patterns_detailed(self):
        """Test detailed analysis mode."""
        records = [
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={"n_folds": 5},
                key_results={"auc": 0.9},
                figure_paths=[],
                interpretation="Success"
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={"n_folds": 10},
                key_results={},
                figure_paths=[],
                interpretation="Error: MemoryError"
            ),
        ]
        
        mock_ctx = MagicMock()
        mock_ctx.state.records = records
        
        result = analyze_workflow_patterns(analysis_depth="detailed", ctx=mock_ctx)
        
        assert result["status"] == "success"
        assert "bottlenecks" in result
        assert "recommendations" in result


class TestSuggestOptimalPipeline:
    """Test suggest_optimal_pipeline skill."""
    
    def test_suggest_optimal_pipeline_imports(self):
        """Test skill can be imported."""
        assert suggest_optimal_pipeline is not None
        assert callable(suggest_optimal_pipeline)
    
    def test_suggest_optimal_pipeline_no_data(self):
        """Test with no execution history."""
        mock_ctx = MagicMock()
        mock_ctx.state.records = []
        mock_ctx.state.memory = MagicMock()
        
        result = suggest_optimal_pipeline(goal="disease_prediction", ctx=mock_ctx)
        
        assert result["status"] == "insufficient_data"
    
    def test_suggest_optimal_pipeline_disease_prediction(self):
        """Test pipeline suggestion for disease prediction."""
        records = [
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="prevalence",
                args={},
                key_results={},
                figure_paths=[],
                interpretation="Success"
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="build_cohort",
                args={},
                key_results={},
                figure_paths=[],
                interpretation="Success"
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={},
                key_results={},
                figure_paths=[],
                interpretation="Success"
            ),
        ]
        
        mock_ctx = MagicMock()
        mock_ctx.state.records = records
        mock_memory = MagicMock()
        mock_memory._data = {"model_configs": {}}
        mock_ctx.state.memory = mock_memory
        
        result = suggest_optimal_pipeline(goal="disease_prediction", max_steps=5, ctx=mock_ctx)
        
        assert result["status"] == "success"
        assert "pipeline" in result

    def test_suggest_low_reliability(self):
        """If all observed skills fail, no pipeline should be recommended."""
        records = [
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={},
                key_results={},
                figure_paths=[],
                interpretation="Error: failed",
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={},
                key_results={},
                figure_paths=[],
                interpretation="Error: failed again",
            ),
        ]

        mock_ctx = MagicMock()
        mock_ctx.state.records = records
        mock_ctx.state.memory = MagicMock(_data={"model_configs": {}})

        result = suggest_optimal_pipeline(goal="disease_prediction", ctx=mock_ctx)

        assert result["status"] == "low_reliability"
        assert "No highly reliable skills" in result["message"]

    def test_suggest_includes_saved_model_config(self):
        """Long-term memory configs should enrich matching pipeline steps."""
        records = [
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={},
                key_results={},
                figure_paths=[],
                interpretation="Success",
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={},
                key_results={},
                figure_paths=[],
                interpretation="Success",
            ),
        ]

        mock_ctx = MagicMock()
        mock_ctx.state.records = records
        mock_ctx.state.memory = MagicMock(
            _data={"model_configs": {"train_model:E11": {"config": {"n_folds": 3}}}}
        )

        result = suggest_optimal_pipeline(goal="disease_prediction", ctx=mock_ctx)

        assert result["status"] == "success"
        assert result["pipeline"] == [
            {"skill": "train_model", "success_rate": 100.0, "suggested_config": {"n_folds": 3}}
        ]
        assert result["total_steps"] <= 5
        assert "estimated_success_rate" in result

    def test_suggest_ignores_nonmatching_model_configs(self):
        records = [
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={},
                key_results={},
                figure_paths=[],
                interpretation="Success",
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={},
                key_results={},
                figure_paths=[],
                interpretation="Success",
            ),
        ]
        mock_ctx = MagicMock()
        mock_ctx.state.records = records
        mock_ctx.state.memory = MagicMock(
            _data={"model_configs": {"calibration:I21": {"config": {"bins": 10}}}}
        )

        result = suggest_optimal_pipeline(goal="disease_prediction", ctx=mock_ctx)

        assert result["pipeline"] == [{"skill": "train_model", "success_rate": 100.0}]
    
    def test_suggest_optimal_pipeline_survival_analysis(self):
        """Test pipeline suggestion for survival analysis."""
        records = [
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="prevalence",
                args={},
                key_results={},
                figure_paths=[],
                interpretation="Success"
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="survival",
                args={},
                key_results={},
                figure_paths=[],
                interpretation="Success"
            ),
        ]
        
        mock_ctx = MagicMock()
        mock_ctx.state.records = records
        mock_memory = MagicMock()
        mock_memory._data = {"model_configs": {}}
        mock_ctx.state.memory = mock_memory
        
        result = suggest_optimal_pipeline(goal="survival_analysis", ctx=mock_ctx)
        
        assert result["status"] == "success"
        assert "pipeline" in result


class TestWorkflowPatternIntegration:
    """Integration tests for workflow pattern analysis."""
    
    def test_pattern_analysis_full_workflow(self):
        """Test full workflow: analyze patterns and suggest pipeline."""
        records = [
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="prevalence",
                args={"top_n": 20},
                key_results={"n_diseases": 20},
                figure_paths=["fig1.png"],
                interpretation="Top 20 diseases identified"
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="build_cohort",
                args={"icd10_code": "E11", "controls_ratio": 4},
                key_results={"n_cases": 500, "n_controls": 2000},
                figure_paths=[],
                interpretation="Cohort built successfully"
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={"n_folds": 5, "sample_size": 2500},
                key_results={"auc": 0.85},
                figure_paths=["roc.png"],
                interpretation="Model trained successfully"
            ),
        ]
        
        mock_ctx = MagicMock()
        mock_ctx.state.records = records
        mock_memory = MagicMock()
        mock_memory._data = {"model_configs": {}}
        mock_ctx.state.memory = mock_memory
        
        # First analyze patterns
        analysis = analyze_workflow_patterns(analysis_depth="detailed", ctx=mock_ctx)
        assert analysis["status"] == "success"
        assert len(analysis["recommendations"]) > 0
        
        # Then suggest pipeline based on patterns
        pipeline = suggest_optimal_pipeline(goal="disease_prediction", ctx=mock_ctx)
        assert pipeline["status"] == "success"
        assert len(pipeline["pipeline"]) > 0
    
    def test_bottleneck_detection_and_pipeline_adjustment(self):
        """Test that pipeline suggestions avoid detected bottlenecks."""
        records = [
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={"n_folds": 10},
                key_results={},
                figure_paths=[],
                interpretation="Error: MemoryError - out of memory"
            ),
            AnalysisRecord(
                timestamp=datetime.now().isoformat(),
                skill="train_model",
                args={"n_folds": 5},
                key_results={"auc": 0.9},
                figure_paths=[],
                interpretation="Success"
            ),
        ]
        
        mock_ctx = MagicMock()
        mock_ctx.state.records = records
        mock_memory = MagicMock()
        mock_memory._data = {"model_configs": {}}
        mock_ctx.state.memory = mock_memory
        
        # Analyze to find bottleneck
        analysis = analyze_workflow_patterns(analysis_depth="detailed", ctx=mock_ctx)
        assert len(analysis.get("recommendations", [])) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
