"""Test pipeline replay and macro recording skills."""

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime


class TestReplayPipeline:
    """Test replay_pipeline skill execution."""
    
    def test_replay_pipeline_imports(self):
        """Test that replay_pipeline skill can be imported."""
        from biobank_agent.skills.replay_pipeline import replay_pipeline
        assert callable(replay_pipeline)
        assert hasattr(replay_pipeline, '_skill_name')
        assert replay_pipeline._skill_name == 'replay_pipeline'

    def test_replay_pipeline_schema(self):
        """Test replay_pipeline has correct parameters."""
        from biobank_agent.skills.replay_pipeline import replay_pipeline
        schema = replay_pipeline._skill_schema
        
        assert schema['type'] == 'function'
        func_def = schema['function']
        assert func_def['name'] == 'replay_pipeline'
        assert 'pipeline_name' in func_def['parameters']['properties']
        assert 'overrides' in func_def['parameters']['properties']
        assert 'pipeline_name' in func_def['parameters']['required']

    def test_replay_pipeline_with_mock_context(self):
        """Test replay_pipeline execution with mocked context."""
        from biobank_agent.skills.replay_pipeline import replay_pipeline
        
        # Create mock context
        ctx = MagicMock()
        ctx.state.memory.get_pipeline.return_value = [
            {"skill": "prevalence", "args": {"top_n": 10}},
            {"skill": "min_sample", "args": {"icd10_code": "E11"}},
        ]
        ctx.registry.execute.side_effect = [
            {"result": "prevalence_data"},
            {"result": "sample_data"},
        ]
        ctx.state.records = []
        
        # Call with no overrides
        result = replay_pipeline("test_pipeline", ctx=ctx)
        
        assert result["pipeline"] == "test_pipeline"
        assert result["steps_executed"] == 2
        assert result["steps_succeeded"] == 2
        assert len(result["results"]) == 2

    def test_replay_pipeline_with_parameter_override(self):
        """Test that parameter overrides are applied."""
        from biobank_agent.skills.replay_pipeline import replay_pipeline
        
        ctx = MagicMock()
        ctx.state.memory.get_pipeline.return_value = [
            {"skill": "prevalence", "args": {"top_n": 10}},
        ]
        ctx.registry.execute.side_effect = [{"result": "data"}]
        ctx.state.records = []
        
        # Replay with overrides
        overrides_json = json.dumps({"top_n": 50})
        result = replay_pipeline("test_pipeline", overrides_json, ctx=ctx)
        
        # Verify execute was called with merged args
        call_args = ctx.registry.execute.call_args
        executed_args = call_args[0][1]  # second positional arg
        assert executed_args["top_n"] == 50  # override applied

    def test_replay_pipeline_not_found(self):
        """Test handling of non-existent pipeline."""
        from biobank_agent.skills.replay_pipeline import replay_pipeline
        
        ctx = MagicMock()
        ctx.state.memory.get_pipeline.return_value = None
        ctx.state.memory.list_pipelines.return_value = ["existing_pipeline"]
        
        result = replay_pipeline("nonexistent", ctx=ctx)
        
        assert "error" in result
        assert "not found" in result["error"]
        assert "existing_pipeline" in result["available_pipelines"]

    def test_replay_pipeline_invalid_json_override(self):
        """Test handling of invalid JSON in overrides."""
        from biobank_agent.skills.replay_pipeline import replay_pipeline
        
        ctx = MagicMock()
        ctx.state.memory.get_pipeline.return_value = [
            {"skill": "prevalence", "args": {"top_n": 10}},
        ]
        
        result = replay_pipeline("test", "invalid json!", ctx=ctx)
        
        assert "error" in result
        assert "JSON" in result["error"]

    def test_replay_pipeline_execution_error_handling(self):
        """Test handling of errors during skill execution."""
        from biobank_agent.skills.replay_pipeline import replay_pipeline
        
        ctx = MagicMock()
        ctx.state.memory.get_pipeline.return_value = [
            {"skill": "prevalence", "args": {"top_n": 10}},
            {"skill": "broken_skill", "args": {}},
        ]
        ctx.registry.execute.side_effect = [
            {"result": "ok"},
            Exception("Skill failed"),
        ]
        ctx.state.records = []
        
        result = replay_pipeline("test", ctx=ctx)
        
        assert result["steps_executed"] == 2
        assert result["steps_succeeded"] == 1
        assert result["steps_failed"] == 1
        assert result["results"][1]["status"] == "error"


class TestRecordMacro:
    """Test record_macro skill."""
    
    def test_record_macro_imports(self):
        """Test that record_macro skill can be imported."""
        from biobank_agent.skills.record_macro import record_macro
        assert callable(record_macro)
        assert hasattr(record_macro, '_skill_name')
        assert record_macro._skill_name == 'record_macro'

    def test_record_macro_schema(self):
        """Test record_macro has correct parameters."""
        from biobank_agent.skills.record_macro import record_macro
        schema = record_macro._skill_schema
        
        assert schema['type'] == 'function'
        func_def = schema['function']
        assert func_def['name'] == 'record_macro'
        assert 'name' in func_def['parameters']['properties']
        assert 'from_record_idx' in func_def['parameters']['properties']
        assert 'name' in func_def['parameters']['required']

    def test_record_macro_execution(self):
        """Test record_macro saves pipeline correctly."""
        from biobank_agent.skills.record_macro import record_macro
        
        ctx = MagicMock()
        ctx.state.records = [
            {"skill": "prevalence", "args": {"top_n": 20}},
            {"skill": "min_sample", "args": {"icd10_code": "E11"}},
        ]
        ctx.state.memory = MagicMock()
        
        result = record_macro("my_macro", ctx=ctx)
        
        assert result["saved"] == "my_macro"
        assert result["steps"] == 2
        # Verify save_pipeline was called
        ctx.state.memory.save_pipeline.assert_called_once()

    def test_record_macro_filters_think_skill(self):
        """Test that 'think' skills are filtered out."""
        from biobank_agent.skills.record_macro import record_macro
        
        ctx = MagicMock()
        ctx.state.records = [
            {"skill": "prevalence", "args": {"top_n": 20}},
            {"skill": "think", "args": {"query": "reasoning"}},
            {"skill": "min_sample", "args": {"icd10_code": "E11"}},
        ]
        ctx.state.memory = MagicMock()
        
        result = record_macro("my_macro", ctx=ctx)
        
        # Should record only 2 steps (think filtered out)
        assert result["steps"] == 2
        saved_steps = ctx.state.memory.save_pipeline.call_args[0][1]
        assert len(saved_steps) == 2

    def test_record_macro_from_record_idx(self):
        """Test filtering records by index."""
        from biobank_agent.skills.record_macro import record_macro
        
        ctx = MagicMock()
        ctx.state.records = [
            {"skill": "prevalence", "args": {"top_n": 20}},
            {"skill": "min_sample", "args": {"icd10_code": "E11"}},
            {"skill": "survival", "args": {"icd10_code": "E11"}},
        ]
        ctx.state.memory = MagicMock()
        
        # Record only from index 1 onward
        result = record_macro("recent_analysis", from_record_idx=1, ctx=ctx)
        
        assert result["steps"] == 2
        saved_steps = ctx.state.memory.save_pipeline.call_args[0][1]
        assert saved_steps[0]["skill"] == "min_sample"

    def test_record_macro_no_records(self):
        """Test handling when there are no records."""
        from biobank_agent.skills.record_macro import record_macro
        
        ctx = MagicMock()
        ctx.state.records = []
        
        result = record_macro("empty_macro", ctx=ctx)
        
        assert "error" in result
        assert "No records" in result["error"]


class TestListPipelines:
    """Test list_pipelines skill."""
    
    def test_list_pipelines_imports(self):
        """Test that list_pipelines skill can be imported."""
        from biobank_agent.skills.list_pipelines import list_pipelines
        assert callable(list_pipelines)
        assert hasattr(list_pipelines, '_skill_name')
        assert list_pipelines._skill_name == 'list_pipelines'

    def test_list_pipelines_execution(self):
        """Test list_pipelines returns pipeline names."""
        from biobank_agent.skills.list_pipelines import list_pipelines
        
        ctx = MagicMock()
        ctx.state.memory.list_pipelines.return_value = [
            "pipeline1", "pipeline2", "pipeline3"
        ]
        
        result = list_pipelines(ctx=ctx)
        
        assert result["count"] == 3
        assert "pipeline1" in result["pipelines"]
        assert "pipeline2" in result["pipelines"]

    def test_list_pipelines_with_details(self):
        """Test list_pipelines shows details when requested."""
        from biobank_agent.skills.list_pipelines import list_pipelines
        
        ctx = MagicMock()
        ctx.state.memory.list_pipelines.return_value = ["pipeline1"]
        ctx.state.memory.get_pipeline.return_value = [
            {"skill": "prevalence", "args": {"top_n": 20}},
            {"skill": "min_sample", "args": {"icd10_code": "E11"}},
        ]
        
        result = list_pipelines(show_details="true", ctx=ctx)
        
        assert "details" in result
        assert "pipeline1" in result["details"]
        assert result["details"]["pipeline1"]["steps"] == 2

    def test_list_pipelines_empty(self):
        """Test handling when no pipelines exist."""
        from biobank_agent.skills.list_pipelines import list_pipelines
        
        ctx = MagicMock()
        ctx.state.memory.list_pipelines.return_value = []
        
        result = list_pipelines(ctx=ctx)
        
        assert result["count"] == 0
        assert len(result["pipelines"]) == 0
        assert "No saved pipelines" in result["message"]


class TestPipelineIntegration:
    """Integration tests for the pipeline system."""
    
    def test_record_and_replay_workflow(self):
        """Test the complete record -> replay workflow."""
        from biobank_agent.skills.record_macro import record_macro
        from biobank_agent.skills.replay_pipeline import replay_pipeline
        
        # Step 1: Create context with session records
        ctx = MagicMock()
        ctx.state.records = [
            {"skill": "prevalence", "args": {"top_n": 10}},
            {"skill": "min_sample", "args": {"icd10_code": "E11"}},
        ]
        
        # Step 2: Mock memory
        stored_pipeline = {}
        def mock_save(name, steps):
            stored_pipeline[name] = steps
        
        def mock_get(name):
            return stored_pipeline.get(name)
        
        ctx.state.memory.save_pipeline.side_effect = mock_save
        ctx.state.memory.get_pipeline.side_effect = mock_get
        
        # Step 3: Record macro
        record_result = record_macro("workflow", ctx=ctx)
        assert record_result["saved"] == "workflow"
        assert record_result["steps"] == 2
        
        # Step 4: Replay (create new context for replay)
        replay_ctx = MagicMock()
        replay_ctx.state.memory.get_pipeline.side_effect = mock_get
        replay_ctx.registry.execute.side_effect = [
            {"result": "data1"},
            {"result": "data2"},
        ]
        replay_ctx.state.records = []
        
        replay_result = replay_pipeline("workflow", ctx=replay_ctx)
        assert replay_result["pipeline"] == "workflow"
        assert replay_result["steps_executed"] == 2
        assert replay_result["steps_succeeded"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
