"""Test error tracking and suggestion engine."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import json

from biobank_agent.memory import LongTermMemory
from biobank_agent.skills.track_error import track_error, list_errors
from biobank_agent.skills.error_suggestions import _suggest_fixes_for_error, suggest_error_fix


class TestLongTermMemoryErrorCatalog:
    """Test error tracking in LongTermMemory."""
    
    def test_record_error_basic(self):
        """Test recording a basic error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = LongTermMemory(Path(tmpdir))
            
            memory.record_error(
                error_type="ValueError",
                error_message="Invalid parameter",
                skill_name="test_skill",
                suggested_fix="Check parameter format"
            )
            
            assert "errors" in memory._data
            errors = memory._data["errors"]
            assert "ValueError:test_skill" in errors
            assert errors["ValueError:test_skill"]["count"] == 1
    
    def test_record_error_increments_count(self):
        """Test error count increments on repeated errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = LongTermMemory(Path(tmpdir))
            
            for _ in range(3):
                memory.record_error(
                    error_type="MemoryError",
                    error_message="Out of memory",
                    skill_name="big_task",
                )
            
            errors = memory._data["errors"]
            assert errors["MemoryError:big_task"]["count"] == 3
    
    def test_record_error_keeps_last_messages(self):
        """Test that only last 5 messages are kept."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = LongTermMemory(Path(tmpdir))
            
            for i in range(7):
                memory.record_error(
                    error_type="RuntimeError",
                    error_message=f"Error {i}",
                    skill_name="test",
                )
            
            errors = memory._data["errors"]
            messages = errors["RuntimeError:test"]["messages"]
            assert len(messages) <= 5
            assert "Error 2" in messages  # Kept from middle
            assert "Error 6" in messages  # Kept from end
    
    def test_record_error_with_context(self):
        """Test recording error with context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = LongTermMemory(Path(tmpdir))
            
            context = {"n_samples": 10000, "icd10_code": "E11"}
            memory.record_error(
                error_type="MemoryError",
                error_message="Out of memory",
                skill_name="train_model",
                context=context,
            )
            
            errors = memory._data["errors"]
            contexts = errors["MemoryError:train_model"]["contexts"]
            assert len(contexts) == 1
            assert contexts[0]["n_samples"] == 10000
    
    def test_get_error_suggestions(self):
        """Test retrieving suggestions for an error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = LongTermMemory(Path(tmpdir))
            
            memory.record_error(
                error_type="ValueError",
                error_message="Bad value",
                skill_name="skill1",
                suggested_fix="Use valid ICD10 code",
            )
            memory.record_error(
                error_type="ValueError",
                error_message="Bad value",
                skill_name="skill1",
                suggested_fix="Check input format",
            )
            
            suggestions = memory.get_error_suggestions("ValueError", "skill1")
            assert len(suggestions) == 2
            assert "Use valid ICD10 code" in suggestions
    
    def test_most_common_errors(self):
        """Test getting most common errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = LongTermMemory(Path(tmpdir))
            
            # Record errors
            for _ in range(5):
                memory.record_error("MemoryError", "OOM", "skill1")
            for _ in range(3):
                memory.record_error("ValueError", "Bad val", "skill2")
            for _ in range(1):
                memory.record_error("TimeoutError", "Timeout", "skill3")
            
            top_errors = memory.most_common_errors(10)
            assert len(top_errors) == 3
            assert top_errors[0]["count"] == 5
            assert top_errors[0]["error_type"] == "MemoryError"


class TestTrackErrorSkill:
    """Test track_error skill."""
    
    def test_track_error_imports(self):
        """Test track_error skill can be imported."""
        assert track_error is not None
        assert callable(track_error)
    
    def test_track_error_execution(self):
        """Test executing track_error skill."""
        mock_ctx = MagicMock()
        mock_memory = MagicMock()
        mock_ctx.state.memory = mock_memory
        
        result = track_error(
            error_type="ValueError",
            error_message="Invalid parameter",
            skill_name="test_skill",
            suggested_fix="Check format",
            ctx=mock_ctx,
        )
        
        assert result["status"] == "tracked"
        assert result["error_type"] == "ValueError"
        assert result["skill"] == "test_skill"
        mock_memory.record_error.assert_called_once()
    
    def test_track_error_returns_suggestions(self):
        """Test track_error returns known suggestions."""
        mock_ctx = MagicMock()
        mock_memory = MagicMock()
        mock_memory.get_error_suggestions.return_value = ["Try again", "Use different params"]
        mock_memory._data = {"errors": {"ValueError:skill1": {"count": 2}}}
        mock_ctx.state.memory = mock_memory
        
        result = track_error(
            error_type="ValueError",
            error_message="Bad value",
            skill_name="skill1",
            ctx=mock_ctx,
        )
        
        assert len(result["known_fixes"]) == 2
        assert "Try again" in result["known_fixes"]

    def test_track_error_with_explicit_context_no_catalog_or_fixes(self):
        """Explicit context should pass through when no prior error catalog exists."""
        mock_ctx = MagicMock()
        mock_memory = MagicMock()
        mock_memory.get_error_suggestions.return_value = []
        mock_memory._data = {}
        mock_ctx.state.memory = mock_memory

        result = track_error(
            error_type="RuntimeError",
            error_message="offline",
            skill_name="web_search",
            suggested_fix="",
            context={"query": "diabetes"},
            ctx=mock_ctx,
        )

        assert result["known_fixes"] == []
        assert result["occurrence_count"] == 0
        mock_memory.record_error.assert_called_once_with(
            error_type="RuntimeError",
            error_message="offline",
            skill_name="web_search",
            suggested_fix=None,
            context={"query": "diabetes"},
        )
    
    def test_list_errors_execution(self):
        """Test list_errors skill execution."""
        mock_ctx = MagicMock()
        mock_memory = MagicMock()
        mock_memory.most_common_errors.return_value = [
            {
                "error_type": "MemoryError",
                "skill": "train_model",
                "count": 5,
                "last_seen": "2024-01-15T10:30:00",
                "suggested_fixes": ["Reduce sample size"],
            }
        ]
        mock_ctx.state.memory = mock_memory
        
        result = list_errors(top_n=10, ctx=mock_ctx)
        
        assert result["status"] == "success"
        assert len(result["top_errors"]) == 1
        assert result["top_errors"][0]["occurrences"] == 5
    
    def test_list_errors_empty(self):
        """Test list_errors with no errors."""
        mock_ctx = MagicMock()
        mock_memory = MagicMock()
        mock_memory.most_common_errors.return_value = []
        mock_ctx.state.memory = mock_memory
        
        result = list_errors(ctx=mock_ctx)
        
        assert result["status"] == "no_errors"


class TestSuggestErrorFixSkill:
    """Test suggest_error_fix skill."""
    
    def test_suggest_error_fix_imports(self):
        """Test suggest_error_fix can be imported."""
        assert suggest_error_fix is not None
        assert callable(suggest_error_fix)
    
    def test_suggest_memory_error(self):
        """Test suggestions for MemoryError."""
        mock_ctx = MagicMock()
        mock_memory = MagicMock()
        mock_memory.get_error_suggestions.return_value = []
        mock_ctx.state.memory = mock_memory
        
        result = suggest_error_fix(
            error_type="MemoryError",
            error_message="Out of memory",
            skill_name="train_model",
            current_parameters={"n_folds": 5, "sample_size": 1000},
            ctx=mock_ctx,
        )
        
        assert result["status"] == "success"
        # Check that memory-related suggestions exist (case-insensitive)
        suggestions_lower = " ".join(s["suggestion"].lower() for s in result["suggestions"])
        assert "memory" in suggestions_lower or "reduce" in suggestions_lower or "sample" in suggestions_lower
        assert result["retry_recommended"] is True
    
    def test_suggest_key_error(self):
        """Test suggestions for KeyError."""
        mock_ctx = MagicMock()
        mock_memory = MagicMock()
        mock_memory.get_error_suggestions.return_value = []
        mock_ctx.state.memory = mock_memory
        
        result = suggest_error_fix(
            error_type="KeyError",
            error_message="Field 'unknown_field' not found",
            skill_name="load_data",
            ctx=mock_ctx,
        )
        
        assert result["status"] == "success"
        assert result["retry_recommended"] is False
        # Check for relevant suggestions (verify/check/field)
        suggestions_lower = " ".join(s["suggestion"].lower() for s in result["suggestions"])
        assert "verify" in suggestions_lower or "check" in suggestions_lower or "field" in suggestions_lower
    
    def test_suggest_parameter_mutations_for_memory(self):
        """Test parameter mutation suggestions for memory errors."""
        mock_ctx = MagicMock()
        mock_memory = MagicMock()
        mock_memory.get_error_suggestions.return_value = []
        mock_ctx.state.memory = mock_memory
        
        result = suggest_error_fix(
            error_type="MemoryError",
            error_message="Out of memory",
            skill_name="train_model",
            current_parameters={
                "n_folds": 5,
                "sample_size": 1000,
                "n_repeats": 3,
                "top_n": 50,
            },
            ctx=mock_ctx,
        )
        
        mutations = result["suggested_parameter_mutations"]
        assert mutations["n_folds"] == 4  # Reduced by 1
        assert mutations["sample_size"] == 500  # Halved
        assert mutations["n_repeats"] == 2  # Reduced by 1
        assert mutations["top_n"] == 10  # Capped at 10
    
    def test_suggest_combines_known_and_generic(self):
        """Test suggestions combine known fixes and generic ones."""
        mock_ctx = MagicMock()
        mock_memory = MagicMock()
        mock_memory.get_error_suggestions.return_value = ["Previously worked fix"]
        mock_ctx.state.memory = mock_memory
        
        result = suggest_error_fix(
            error_type="RuntimeError",
            error_message="Temporary issue",
            skill_name="analyze",
            ctx=mock_ctx,
        )
        
        suggestions = result["suggestions"]
        # First should be from history
        assert suggestions[0]["source"] == "from_history"
        assert suggestions[0]["suggestion"] == "Previously worked fix"
        # Rest should be generic
        assert any(s["source"] == "generic" for s in suggestions)
    
    def test_suggest_no_duplicates(self):
        """Test suggestions don't have duplicates."""
        mock_ctx = MagicMock()
        mock_memory = MagicMock()
        mock_memory.get_error_suggestions.return_value = ["Fix A"]
        mock_ctx.state.memory = mock_memory
        
        result = suggest_error_fix(
            error_type="ValueError",
            error_message="Invalid input",
            skill_name="skill1",
            ctx=mock_ctx,
        )
        
        suggestions = result["suggestions"]
        suggestion_texts = [s["suggestion"] for s in suggestions]
        assert len(suggestion_texts) == len(set(suggestion_texts))

    def test_suggestion_categories_cover_remaining_error_types(self):
        cases = [
            ("TimeoutError", "timed out", "timeout duration"),
            ("AttributeError", "missing attr", "expected method"),
            ("OSError", "file missing", "file/directory"),
            ("TypeError", "bad type", "parameter types"),
            ("ImportError", "import failed", "Install missing package"),
            ("MysteryError", "unexpected", "Review error message"),
        ]

        for error_type, message, expected in cases:
            suggestions = _suggest_fixes_for_error(error_type, message, "skill")
            assert any(expected in suggestion for suggestion in suggestions)

    def test_suggest_memory_mutation_floor_values_and_empty_history_items(self):
        mock_ctx = MagicMock()
        mock_memory = MagicMock()
        mock_memory.get_error_suggestions.return_value = ["", "Reduce sample size (pass lower value to sample_size parameter)"]
        mock_ctx.state.memory = mock_memory

        result = suggest_error_fix(
            error_type="RuntimeError",
            error_message="memory pressure",
            skill_name="train_model",
            current_parameters={"sample_size": 5, "n_repeats": 1, "top_n": 4},
            ctx=mock_ctx,
        )

        assert result["suggested_parameter_mutations"] == {
            "sample_size": 10,
            "n_repeats": 1,
            "top_n": 4,
        }
        suggestions = result["suggestions"]
        assert suggestions[0]["source"] == "from_history"
        assert all(s["suggestion"] for s in suggestions)

        n_folds_only = suggest_error_fix(
            error_type="MemoryError",
            error_message="memory pressure",
            skill_name="train_model",
            current_parameters={"n_folds": 2},
            ctx=mock_ctx,
        )
        assert n_folds_only["suggested_parameter_mutations"] == {"n_folds": 2}


class TestErrorTrackingIntegration:
    """Integration tests for error tracking system."""
    
    def test_track_and_suggest_workflow(self):
        """Test workflow: track error, then get suggestions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = LongTermMemory(Path(tmpdir))
            
            # Simulate tracking an error
            memory.record_error(
                error_type="MemoryError",
                error_message="Out of memory",
                skill_name="train_model",
                suggested_fix="Reduce sample size to 5000",
            )
            
            # Get suggestions for same error
            suggestions = memory.get_error_suggestions("MemoryError", "train_model")
            assert "Reduce sample size to 5000" in suggestions
    
    def test_error_persistence(self):
        """Test errors persist across memory loads."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_dir = Path(tmpdir)
            
            # Record error in first instance
            memory1 = LongTermMemory(memory_dir)
            memory1.record_error(
                error_type="ValueError",
                error_message="Bad input",
                skill_name="skill1",
                suggested_fix="Validate input",
            )
            # Force save
            memory1._save()
            del memory1
            
            # Load in second instance
            memory2 = LongTermMemory(memory_dir)
            errors = memory2._data.get("errors", {})
            assert "ValueError:skill1" in errors
            assert errors["ValueError:skill1"]["count"] == 1
    
    def test_error_catalog_growth(self):
        """Test error catalog grows with different error types."""
        with tempfile.TemporaryDirectory() as tmpdir:
            memory = LongTermMemory(Path(tmpdir))
            
            error_configs = [
                ("ValueError", "Bad value", "skill1", "Validate"),
                ("MemoryError", "OOM", "skill2", "Reduce size"),
                ("TimeoutError", "Timeout", "skill3", "Increase timeout"),
                ("KeyError", "Not found", "skill4", "Check field"),
            ]
            
            for etype, emsg, skill, fix in error_configs:
                memory.record_error(etype, emsg, skill, fix)
            
            all_errors = memory.most_common_errors(10)
            assert len(all_errors) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
