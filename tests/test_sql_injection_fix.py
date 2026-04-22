"""Test SQL injection vulnerability fixes in data layer and skills."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import inspect


class TestSQLInjectionFix:
    """Test that SQL injection vulnerabilities are fixed."""
    
    def test_query_method_has_params_argument(self):
        """Test that query() method supports parameter binding."""
        from biobank_agent.data.loader import DataManager
        from biobank_agent.config import get_settings
        
        settings = get_settings()
        dm = DataManager(settings)
        
        # Verify method signature has params argument
        sig = inspect.signature(dm.query)
        assert 'params' in sig.parameters, "query() should have 'params' parameter"
    
    def test_prevalence_uses_parameterized_query(self):
        """Test that prevalence skill uses parameterized queries."""
        from biobank_agent.skills.prevalence import prevalence
        
        # Get the source code
        source = inspect.getsource(prevalence)
        
        # Verify it doesn't use f-string interpolation with chapter_filter
        assert "f\" AND diag_icd10 LIKE" not in source, \
            "Should not use f-string with chapter_filter in SQL"
        
        # Verify it uses ? placeholder for parameter binding
        assert "LIKE ?" in source, \
            "Should use ? placeholder for parameter binding"

        # Verify it builds params list
        assert "params = []" in source or "params =" in source, "Should initialize params list"

    def test_survival_uses_parameterized_query(self):
        """Test that survival skill uses parameterized queries."""
        from biobank_agent.skills.survival import survival
        
        source = inspect.getsource(survival)
        
        # Verify it doesn't use f-string with user input directly in SQL
        assert "LIKE '{icd" not in source, \
            "Should not use f-string interpolation for diagnosis code in SQL"

        # Verify it uses ? placeholder and dm.query with params
        assert "LIKE ?" in source, \
            "Should use ? placeholder for parameter binding"
        
        # Verify no uniform random distribution for follow-up times
        assert "np.random.uniform(1, 16)" not in source, \
            "Should not use artificial uniform distribution for follow-up times"
        
        # Verify realistic follow-up time handling
        assert "beta" in source or "follow_up_days" in source or "date" in source.lower(), \
            "Should use realistic date-based or beta-distributed follow-up times"

    def test_cohort_uses_parameterized_queries(self):
        """Test that cohort builder uses parameterized queries."""
        from biobank_agent.data.cohort import build_cohort

        source = inspect.getsource(build_cohort)

        # Verify no f-string interpolation of user input in WHERE clauses
        assert "LIKE '{icd" not in source, \
            "Should not use f-string interpolation for diagnosis code"

        # Verify ? placeholders are used
        count_placeholders = source.count("LIKE ?")
        assert count_placeholders >= 2, \
            "cohort.py should have at least 2 parameterized LIKE queries (diagnoses + deaths)"

        # Verify parameters are passed as list
        assert "icd_code}%\"]" in source or "icd_code}%']" in source, \
            "Should pass parameters as list to execute()"


class TestParameterization:
    """Test that SQL parameterization prevents injection."""
    
    def test_malicious_input_treated_as_literal(self):
        """Test that malicious input is treated as literal data."""
        from biobank_agent.data.loader import DataManager
        from biobank_agent.config import get_settings
        
        settings = get_settings()
        dm = DataManager(settings)
        
        # Mock DuckDB connection
        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_conn.execute.return_value = mock_result
        mock_result.df.return_value = Mock()
        dm.conn = mock_conn
        
        # Attempt SQL injection
        malicious = "E11'; DROP TABLE diagnoses; --"
        sql = "SELECT * FROM diagnoses WHERE diag_icd10 LIKE ?"
        
        dm.query(sql, [f"{malicious}%"])
        
        # Verify: SQL is clean, malicious string is in params
        call_args = mock_conn.execute.call_args
        executed_sql = call_args[0][0]
        executed_params = call_args[0][1]
        
        # SQL should have only ? placeholder, no injection
        assert "DROP TABLE" not in executed_sql, "Malicious SQL not in query string"
        # Malicious input should be in parameters only
        assert f"{malicious}%" in executed_params, "Malicious input in params, not SQL"

    def test_get_diagnoses_uses_parameterized_query(self):
        """Test that get_diagnoses uses parameterized query."""
        from biobank_agent.data.loader import DataManager
        from biobank_agent.config import get_settings
        
        settings = get_settings()
        dm = DataManager(settings)
        
        # Mock query method to inspect call signature
        original_query = dm.query
        calls = []
        
        def mock_query(sql, params=None):
            calls.append((sql, params))
            return Mock()
        
        dm.query = mock_query
        
        # Call get_diagnoses with a parameter
        try:
            dm.get_diagnoses("E11")
        except Exception:
            pass  # We only care about the call signature
        
        # Verify query was called with params
        assert len(calls) > 0, "get_diagnoses should call query()"
        sql, params = calls[0]
        assert params is not None, "Params should not be None for parameterized queries"
        assert isinstance(params, (list, tuple)), "Params should be list or tuple"


class TestFollowUpTimeRealism:
    """Test that follow-up times in survival analysis are realistic."""
    
    def test_survival_does_not_use_artificial_uniform_distribution(self):
        """Verify that survival.py doesn't use np.random.uniform for follow-up."""
        from biobank_agent.skills.survival import survival
        
        source = inspect.getsource(survival)
        
        # Check the artificial distribution is removed
        assert "np.random.uniform(1, 16)" not in source, \
            "Should not use artificial uniform(1, 16) distribution"
        
        # Verify realistic alternatives are used
        has_beta = "np.random.beta" in source
        has_date_computation = "follow_up_days" in source or "death_date" in source
        has_fallback = "assessment_date" in source
        
        assert has_beta or has_date_computation or has_fallback, \
            "Should use realistic follow-up time: beta distribution, date computation, or fallback"


class TestSkillImports:
    """Test that all modified skills can be imported successfully."""
    
    def test_prevalence_imports(self):
        """Test prevalence skill can be imported."""
        try:
            from biobank_agent.skills.prevalence import prevalence
            assert callable(prevalence), "prevalence should be callable"
        except ImportError as e:
            pytest.skip(f"Dependencies not available: {e}")

    def test_survival_imports(self):
        """Test survival skill can be imported."""
        try:
            from biobank_agent.skills.survival import survival
            assert callable(survival), "survival should be callable"
        except ImportError as e:
            pytest.skip(f"Dependencies not available: {e}")

    def test_cohort_builder_imports(self):
        """Test cohort builder can be imported."""
        try:
            from biobank_agent.data.cohort import build_cohort
            assert callable(build_cohort), "build_cohort should be callable"
        except ImportError as e:
            pytest.skip(f"Dependencies not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
