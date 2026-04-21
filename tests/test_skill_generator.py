"""Test AST-based skill code generation and validation."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock


class TestSkillGenerator:
    """Test SkillGenerator class."""
    
    def test_generator_imports(self):
        """Test SkillGenerator can be imported."""
        from biobank_agent.skills.generator import SkillGenerator
        assert SkillGenerator is not None

    def test_validate_code_syntax_error(self):
        """Test validation catches syntax errors."""
        from biobank_agent.skills.generator import SkillGenerator
        
        bad_code = "if True\n  pass"
        is_valid, msg = SkillGenerator.validate_code(bad_code)
        
        assert not is_valid
        assert "Syntax error" in msg

    def test_validate_code_forbidden_import(self):
        """Test validation catches forbidden imports."""
        from biobank_agent.skills.generator import SkillGenerator
        
        bad_code = "import os\nprint('hello')"
        is_valid, msg = SkillGenerator.validate_code(bad_code)
        
        assert not is_valid
        assert "Forbidden import" in msg
        assert "os" in msg

    def test_validate_code_forbidden_call(self):
        """Test validation catches forbidden calls."""
        from biobank_agent.skills.generator import SkillGenerator
        
        bad_code = "code = compile('1+1', '<string>', 'eval')"
        is_valid, msg = SkillGenerator.validate_code(bad_code)
        
        assert not is_valid
        assert "Forbidden" in msg
        assert "compile" in msg

    def test_validate_code_eval_forbidden(self):
        """Test validation blocks eval()."""
        from biobank_agent.skills.generator import SkillGenerator
        
        bad_code = "result = eval('x + 1')"
        is_valid, msg = SkillGenerator.validate_code(bad_code)
        
        assert not is_valid
        assert "eval" in msg

    def test_validate_code_exec_forbidden(self):
        """Test validation blocks exec()."""
        from biobank_agent.skills.generator import SkillGenerator
        
        bad_code = "exec('print(1)')"
        is_valid, msg = SkillGenerator.validate_code(bad_code)
        
        assert not is_valid
        assert "exec" in msg

    def test_validate_code_open_forbidden(self):
        """Test validation blocks open()."""
        from biobank_agent.skills.generator import SkillGenerator
        
        bad_code = "f = open('/etc/passwd')"
        is_valid, msg = SkillGenerator.validate_code(bad_code)
        
        assert not is_valid
        assert "open" in msg

    def test_validate_code_private_attribute_forbidden(self):
        """Test validation blocks private attribute access."""
        from biobank_agent.skills.generator import SkillGenerator
        
        bad_code = "x = obj._private_attr"
        is_valid, msg = SkillGenerator.validate_code(bad_code)
        
        assert not is_valid
        assert "Private attribute" in msg

    def test_validate_code_allowed_import_numpy(self):
        """Test validation allows numpy import."""
        from biobank_agent.skills.generator import SkillGenerator
        
        good_code = "import numpy as np\nx = np.array([1, 2, 3])"
        is_valid, msg = SkillGenerator.validate_code(good_code)
        
        assert is_valid
        assert msg == "OK"

    def test_validate_code_allowed_import_pandas(self):
        """Test validation allows pandas import."""
        from biobank_agent.skills.generator import SkillGenerator
        
        good_code = "import pandas as pd\ndf = pd.DataFrame()"
        is_valid, msg = SkillGenerator.validate_code(good_code)
        
        assert is_valid

    def test_validate_code_allowed_import_scipy(self):
        """Test validation allows scipy import."""
        from biobank_agent.skills.generator import SkillGenerator
        
        good_code = "from scipy import stats\nresult = stats.norm.pdf(0)"
        is_valid, msg = SkillGenerator.validate_code(good_code)
        
        assert is_valid

    def test_validate_code_allowed_sklearn(self):
        """Test validation allows sklearn import."""
        from biobank_agent.skills.generator import SkillGenerator
        
        good_code = "from sklearn.model_selection import cross_val_score"
        is_valid, msg = SkillGenerator.validate_code(good_code)
        
        assert is_valid

    def test_validate_code_allowed_builtins(self):
        """Test validation allows safe builtins."""
        from biobank_agent.skills.generator import SkillGenerator
        
        good_code = """
x = [1, 2, 3]
y = len(x)
z = sum(x)
w = max(x)
"""
        is_valid, msg = SkillGenerator.validate_code(good_code)
        
        assert is_valid

    def test_get_validation_errors_multiple(self):
        """Test get_validation_errors returns all errors."""
        from biobank_agent.skills.generator import SkillGenerator
        
        bad_code = """
import os
import sys
x = obj._private
exec('code')
"""
        errors = SkillGenerator.get_validation_errors(bad_code)
        
        assert len(errors) > 0
        assert any("os" in e for e in errors)
        assert any("_private" in e for e in errors)
        assert any("exec" in e for e in errors)

    def test_template_generation(self):
        """Test skill template generation."""
        from biobank_agent.skills.generator import SkillGenerator
        
        params = {
            "top_n": {
                "type": "integer",
                "description": "Number of items",
                "default": 10,
            },
            "icd10_code": {
                "type": "string",
                "description": "Disease code",
            },
        }
        
        code = "return {'result': 'ok'}"
        
        template = SkillGenerator.template(
            name="test_skill",
            description="Test skill",
            parameters=params,
            code_body=code,
        )
        
        assert "@skill" in template
        assert 'name="test_skill"' in template
        assert 'description="Test skill"' in template
        assert '"top_n"' in template
        assert '"integer"' in template
        assert '"icd10_code"' in template
        assert '"string"' in template
        assert "def test_skill(top_n, icd10_code, *, ctx=None)" in template

    def test_template_with_defaults(self):
        """Test template handles default values."""
        from biobank_agent.skills.generator import SkillGenerator
        
        params = {
            "n": {"type": "integer", "description": "Count", "default": 42},
            "mode": {"type": "string", "description": "Mode", "default": "strict"},
        }
        
        template = SkillGenerator.template(
            name="test_skill",
            description="Test",
            parameters=params,
            code_body="pass",
        )
        
        assert '"default": 42' in template
        assert '"default": "strict"' in template

    def test_template_code_indentation(self):
        """Test code body is properly indented."""
        from biobank_agent.skills.generator import SkillGenerator
        
        code_body = "x = 1\ny = 2\nreturn {'result': x + y}"
        
        template = SkillGenerator.template(
            name="test_skill",
            description="Test",
            parameters={},
            code_body=code_body,
        )
        
        # Code should be indented with 4 spaces in the function body
        assert "    x = 1" in template
        assert "    y = 2" in template
        assert "    return {'result': x + y}" in template


class TestCreateSkill:
    """Test create_skill skill."""
    
    def test_create_skill_imports(self):
        """Test create_skill can be imported."""
        from biobank_agent.skills.create_skill import create_skill
        assert callable(create_skill)
        assert create_skill._skill_name == "create_skill"

    def test_create_skill_schema(self):
        """Test create_skill has correct parameters."""
        from biobank_agent.skills.create_skill import create_skill
        schema = create_skill._skill_schema
        
        func_def = schema['function']
        assert 'name' in func_def['parameters']['properties']
        assert 'description' in func_def['parameters']['properties']
        assert 'parameters' in func_def['parameters']['properties']
        assert 'code_body' in func_def['parameters']['properties']

    def test_create_skill_invalid_name(self):
        """Test create_skill rejects invalid names."""
        from biobank_agent.skills.create_skill import create_skill
        
        ctx = MagicMock()
        ctx.settings.reports_dir = Path(tempfile.gettempdir())
        
        result = create_skill(
            name="123-invalid",
            description="Test",
            parameters="{}",
            code_body="pass",
            ctx=ctx,
        )
        
        assert result["status"] == "validation_failed"
        assert "Invalid skill name" in result["error"]

    def test_create_skill_invalid_json_parameters(self):
        """Test create_skill rejects invalid parameter JSON."""
        from biobank_agent.skills.create_skill import create_skill
        
        ctx = MagicMock()
        
        result = create_skill(
            name="test_skill",
            description="Test",
            parameters="not valid json",
            code_body="pass",
            ctx=ctx,
        )
        
        assert result["status"] == "validation_failed"
        assert "JSON" in result["error"]

    def test_create_skill_code_validation(self):
        """Test create_skill validates code."""
        from biobank_agent.skills.create_skill import create_skill
        
        ctx = MagicMock()
        
        result = create_skill(
            name="test_skill",
            description="Test",
            parameters="{}",
            code_body="exec('malicious')",
            ctx=ctx,
        )
        
        assert result["status"] == "validation_failed"
        assert "Code validation failed" in result["error"]
        assert len(result["validation_errors"]) > 0

    def test_create_skill_success(self):
        """Test create_skill generates valid skill."""
        from biobank_agent.skills.create_skill import create_skill
        
        ctx = MagicMock()
        temp_dir = Path(tempfile.gettempdir())
        ctx.settings.reports_dir = temp_dir
        
        params = json.dumps({
            "top_n": {
                "type": "integer",
                "description": "Count",
                "default": 10,
            }
        })
        
        result = create_skill(
            name="my_analysis",
            description="My analysis skill",
            parameters=params,
            code_body="return {'status': 'success'}",
            ctx=ctx,
        )
        
        assert result["status"] == "success"
        assert "generated_skill_path" in result
        assert "code" in result
        assert "@skill" in result["code"]
        assert "my_analysis" in result["code"]
        
        # Verify file was created
        skill_path = Path(result["generated_skill_path"])
        assert skill_path.exists()
        assert "@skill" in skill_path.read_text()


class TestSkillGeneratorSecurity:
    """Test security features of SkillGenerator."""
    
    def test_blocks_lambda_eval_trick(self):
        """Test blocking of lambda-based code execution."""
        from biobank_agent.skills.generator import SkillGenerator
        
        # Lambda used to call eval indirectly
        code = "f = lambda x: eval(x); result = f('1+1')"
        errors = SkillGenerator.get_validation_errors(code)
        
        # Should catch the eval call
        assert any("eval" in e for e in errors)

    def test_blocks_getattr_trick(self):
        """Test private attributes are blocked."""
        from biobank_agent.skills.generator import SkillGenerator
        
        code = "x = getattr(obj, '__class__')"
        errors = SkillGenerator.get_validation_errors(code)
        
        # Should catch __class__ access
        assert len(errors) > 0

    def test_blocks_import_builtin(self):
        """Test __import__ is blocked."""
        from biobank_agent.skills.generator import SkillGenerator
        
        code = "os = __import__('os')"
        is_valid, msg = SkillGenerator.validate_code(code)
        
        assert not is_valid
        assert "__import__" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
