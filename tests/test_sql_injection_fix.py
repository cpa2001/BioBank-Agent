"""Test SQL injection vulnerability fixes in data layer and skills."""

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestSQLInjectionFix:
    """Test that SQL injection vulnerabilities are fixed."""
    
    def test_query_method_has_params_argument(self):
        """Test that query() method supports parameter binding."""
        from biobank_agent.data.loader import DataManager
        from biobank_agent.config import get_settings
        
        settings = get_settings()
        dm = DataManager(settings)
        
        # Verify method signature has params argument
        import inspect
        sig = inspect.signature(dm.query)
        assert 'params' in sig.parameters, "query() should have 'params' parameter"
    
    def test_prevalence_uses_parameterized_query(self):
        """Test that prevalence skill uses parameterized queries."""
        from biobank_agent.skills.prevalence import prevalence
        import inspect
        
        # Get the source code
        source = inspect.getsource(prevalence)
        
        # Verify it doesn't use f-string interpolation with chapter_filter
        assert "f\" AND diag_icd10 LIKE" not in source, \
            "Should not use f-string with chapter_filter in SQL"
        
        # Verify it uses ? placeholder
        assert "diag_icd10 LIKE ?" in source, \
            "Should use ? placeholder for parameter binding"
        
        # Verify it builds params list
        assert "params = []" in source, "Should initialize params list"
        assert "params.append" in source, "Should append to params list"


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
