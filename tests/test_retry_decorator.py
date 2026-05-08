"""Test auto-retry decorator with parameter mutation."""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock, call

from biobank_agent.skills.retry import (
    retry_on_error,
    _modify_parameters_for_retry,
    should_retry_on_error,
)


class TestRetryDecorator:
    """Test the retry_on_error decorator."""
    
    def test_decorator_imports(self):
        """Test retry decorator can be imported."""
        assert retry_on_error is not None
        assert callable(retry_on_error)

    def test_successful_execution_no_retry(self):
        """Test function succeeds on first attempt."""
        call_count = [0]
        
        @retry_on_error(max_retries=3)
        def successful_func(x, *, ctx=None):
            call_count[0] += 1
            return {"result": x * 2}
        
        result = successful_func(5, ctx=None)
        
        assert result == {"result": 10}
        assert call_count[0] == 1  # Only called once

    def test_failure_then_success_on_retry(self):
        """Test function fails then succeeds on retry."""
        call_count = [0]
        
        @retry_on_error(max_retries=3, initial_delay=0.01)
        def flaky_func(x, n_folds=5, *, ctx=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("First attempt fails")
            return {"result": x * 2}
        
        result = flaky_func(5, n_folds=5, ctx=None)
        
        assert result == {"result": 10}
        assert call_count[0] == 2  # Called twice

    def test_exhausts_retries_returns_error(self):
        """Test function fails all retries and returns error dict."""
        
        @retry_on_error(max_retries=3, initial_delay=0.01)
        def always_fails(x, *, ctx=None):
            raise RuntimeError("Always fails")
        
        result = always_fails(5, ctx=None)
        
        assert "error" in result
        assert "Failed after 3 retries" in result["error"]
        assert result["last_attempt"] == 3

    def test_parameter_mutation_n_folds(self):
        """Test n_folds is reduced on retry."""
        kwargs = {"n_folds": 5}
        _modify_parameters_for_retry(kwargs, 1)
        assert kwargs["n_folds"] == 4
        
        _modify_parameters_for_retry(kwargs, 2)
        assert kwargs["n_folds"] == 3

    def test_parameter_mutation_n_folds_minimum(self):
        """Test n_folds doesn't go below minimum."""
        kwargs = {"n_folds": 2}
        _modify_parameters_for_retry(kwargs, 1)
        assert kwargs["n_folds"] == 2  # Stays at minimum

    def test_parameter_mutation_n_repeats(self):
        """Test n_repeats is reduced on retry."""
        kwargs = {"n_repeats": 5}
        _modify_parameters_for_retry(kwargs, 1)
        assert kwargs["n_repeats"] == 4

    def test_parameter_mutation_sample_size(self):
        """Test sample_size is halved on retry."""
        kwargs = {"sample_size": 1000}
        _modify_parameters_for_retry(kwargs, 1)
        assert kwargs["sample_size"] == 500
        
        _modify_parameters_for_retry(kwargs, 2)
        assert kwargs["sample_size"] == 250

    def test_parameter_mutation_top_n(self):
        """Test top_n is capped at 10."""
        kwargs = {"top_n": 50}
        _modify_parameters_for_retry(kwargs, 1)
        assert kwargs["top_n"] == 10
        
        # Already at cap, stays at 10
        _modify_parameters_for_retry(kwargs, 2)
        assert kwargs["top_n"] == 10

    def test_parameter_mutation_multiple_params(self):
        """Test multiple parameters are mutated together."""
        kwargs = {
            "n_folds": 5,
            "sample_size": 1000,
            "top_n": 50,
            "max_depth": 10,
        }
        
        _modify_parameters_for_retry(kwargs, 1)
        
        assert kwargs["n_folds"] == 4
        assert kwargs["sample_size"] == 500
        assert kwargs["top_n"] == 10
        assert kwargs["max_depth"] == 9

    def test_parameter_mutation_case_threshold_and_estimators(self):
        """Case thresholds and estimator counts should be reduced with floors."""
        kwargs = {"n_cases_min": 200, "n_estimators": 200}
        _modify_parameters_for_retry(kwargs, 1)
        assert kwargs["n_cases_min"] == 180
        assert kwargs["n_estimators"] == 100

        floor_kwargs = {"n_cases_min": 40, "n_estimators": 5}
        _modify_parameters_for_retry(floor_kwargs, 1)
        assert floor_kwargs["n_cases_min"] == 50
        assert floor_kwargs["n_estimators"] == 10

    def test_backoff_delay(self):
        """Test exponential backoff is applied."""
        call_times = []
        
        @retry_on_error(max_retries=3, backoff_factor=2.0, initial_delay=0.05)
        def flaky_func(*, ctx=None):
            call_times.append(time.time())
            if len(call_times) < 3:
                raise RuntimeError("Fail")
            return {"ok": True}
        
        result = flaky_func(ctx=None)
        
        assert result == {"ok": True}
        assert len(call_times) == 3
        
        # Check delays are roughly exponential
        delay1 = call_times[1] - call_times[0]
        delay2 = call_times[2] - call_times[1]
        
        # delay1 should be ~0.05s, delay2 should be ~0.1s
        assert delay1 >= 0.04  # Allow small tolerance
        assert delay2 >= 0.08

    def test_preserves_function_metadata(self):
        """Test decorator preserves function name and docstring."""
        @retry_on_error()
        def my_skill(x, *, ctx=None):
            """My skill docstring."""
            return x * 2
        
        assert my_skill.__name__ == "my_skill"
        assert "My skill docstring" in my_skill.__doc__

    def test_error_dict_contains_original_error(self):
        """Test error dict includes original error message."""
        
        @retry_on_error(max_retries=1, initial_delay=0.01)
        def fails_with_message(*, ctx=None):
            raise ValueError("Specific error message")
        
        result = fails_with_message(ctx=None)
        
        assert "original_error" in result
        assert "Specific error message" in result["original_error"]

    def test_zero_retries_returns_unexpected_state_fallback(self):
        """A zero retry budget should hit the defensive fallback path."""
        @retry_on_error(max_retries=0)
        def never_called(*, ctx=None):
            raise AssertionError("should not execute")

        result = never_called(ctx=None)

        assert result["error"] == "Unexpected state: exhausted retries"
        assert result["skill"] == "never_called"


class TestShouldRetryOnError:
    """Test should_retry_on_error function."""
    
    def test_retryable_memory_error(self):
        """Test MemoryError is retryable."""
        error = MemoryError("Out of memory")
        assert should_retry_on_error(error) is True

    def test_retryable_timeout_error(self):
        """Test TimeoutError is retryable."""
        error = TimeoutError("Connection timeout")
        assert should_retry_on_error(error) is True

    def test_retryable_runtime_error(self):
        """Test RuntimeError is retryable."""
        error = RuntimeError("Temporary resource issue")
        assert should_retry_on_error(error) is True

    def test_permanent_value_error(self):
        """Test ValueError is not retryable."""
        error = ValueError("Invalid parameter")
        assert should_retry_on_error(error) is False

    def test_permanent_key_error(self):
        """Test KeyError is not retryable."""
        error = KeyError("Field not found")
        assert should_retry_on_error(error) is False

    def test_permanent_attribute_error(self):
        """Test AttributeError is not retryable."""
        error = AttributeError("No such attribute")
        assert should_retry_on_error(error) is False

    def test_permanent_type_error_by_type(self):
        """Permanent exception types should be rejected even without keywords."""
        assert should_retry_on_error(TypeError("wrong shape")) is False

    def test_error_message_keyword_retryable(self):
        """Test error message keywords indicate retryable."""
        error = RuntimeError("Please try again later")
        assert should_retry_on_error(error) is True

    def test_error_message_keyword_retryable_for_unknown_type(self):
        """Retryable keywords should apply to otherwise unknown exception types."""
        class CustomError(Exception):
            pass

        assert should_retry_on_error(CustomError("worker busy")) is True

    def test_error_message_keyword_permanent(self):
        """Test error message keywords indicate permanent."""
        error = RuntimeError("Field not found in data")
        assert should_retry_on_error(error) is False

    def test_unknown_error_defaults_to_retryable(self):
        """Test unknown errors default to retryable."""
        
        class CustomError(Exception):
            pass
        
        error = CustomError("Custom error")
        assert should_retry_on_error(error) is True


class TestRetryIntegration:
    """Integration tests for retry decorator."""
    
    def test_retry_with_parameter_mutation_reduces_workload(self):
        """Test that retries use progressively simpler parameters."""
        execution_params = []
        
        @retry_on_error(max_retries=3, initial_delay=0.01)
        def skill_with_params(n_folds=5, top_n=50, *, ctx=None):
            execution_params.append({"n_folds": n_folds, "top_n": top_n})
            if len(execution_params) < 3:
                raise RuntimeError("Fail")
            return {"ok": True}
        
        result = skill_with_params(n_folds=5, top_n=50, ctx=None)
        
        assert result == {"ok": True}
        assert len(execution_params) == 3
        
        # First attempt: original params
        assert execution_params[0] == {"n_folds": 5, "top_n": 50}
        # Second attempt: reduced params
        assert execution_params[1]["n_folds"] == 4
        assert execution_params[1]["top_n"] == 10
        # Third attempt: further reduced
        assert execution_params[2]["n_folds"] == 3

    def test_context_preserved_across_retries(self):
        """Test context is preserved across retries."""
        contexts = []
        
        @retry_on_error(max_retries=2, initial_delay=0.01)
        def skill_with_ctx(*, ctx=None):
            contexts.append(ctx)
            if len(contexts) < 2:
                raise RuntimeError("Fail")
            return {"ok": True}
        
        mock_ctx = MagicMock()
        result = skill_with_ctx(ctx=mock_ctx)
        
        assert result == {"ok": True}
        assert len(contexts) == 2
        assert all(ctx is mock_ctx for ctx in contexts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
