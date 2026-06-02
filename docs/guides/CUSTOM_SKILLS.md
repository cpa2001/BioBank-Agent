# Quick Start: Creating Custom Skills for Biobank Agent

## 1. Basic Skill Template

```python
"""Your skill module description."""

from biobank_agent.registry import skill

@skill(
    name="my_analysis",
    description="One-line description of what your skill does",
    parameters={
        "param1": {
            "type": "string",
            "description": "Description of param1"
        },
        "param2": {
            "type": "integer",
            "description": "Description of param2",
            "default": 10,  # Optional parameters need a default
        },
    },
    required=["param1"],  # List params without defaults
)
def my_analysis(param1: str, param2: int = 10, *, ctx=None) -> dict:
    """Implementation of your skill.
    
    Parameters
    ----------
    param1 : str
        First parameter
    param2 : int, optional
        Second parameter (default: 10)
    ctx : SkillContext
        Agent context (injected automatically)
    
    Returns
    -------
    dict
        Result dictionary
    """
    # Extract tools from context
    dm = ctx.dm                    # Query data
    settings = ctx.settings       # Access config
    state = ctx.state             # Session state
    memory = ctx.memory           # Long-term memory
    report_dir = ctx.report_dir   # Save outputs here
    catalog = ctx.catalog         # Field metadata
    
    # Your implementation
    result = {
        "status": "success",
        "message": f"Processed {param1}",
        "data": {...}
    }
    
    return result
```

## 2. Installation Steps

### Step 1: Create Your Skill File
```bash
# Create custom_skills directory if it doesn't exist
mkdir -p ./custom_skills

# Create your skill
cat > ./custom_skills/my_analysis.py << 'SKILL'
from biobank_agent.registry import skill

@skill(
    name="my_analysis",
    description="My custom analysis",
    parameters={
        "code": {
            "type": "string",
            "description": "ICD10 code to analyze"
        },
    },
)
def my_analysis(code: str, *, ctx=None) -> dict:
    dm = ctx.dm
    result = dm.query(f"SELECT * FROM diagnoses WHERE diag_icd10 LIKE '{code}%'")
    return {"count": len(result), "data": result.to_dict("records")}
SKILL
```

### Step 2: Verify Setup in .env
```bash
# Check or add to .env
echo "CUSTOM_SKILLS_DIR=./custom_skills" >> .env
```

### Step 3: Restart Agent
```bash
# Your custom skill will be loaded automatically at startup
python -m biobank_agent.cli
```

### Step 4: Test the Skill
```
/skills
# Should see "my_analysis" in the list

# Use it
analyze ICD10 code E11 using my_analysis with code="E11"
```

## 3. Advanced: Using Data & Memory

```python
from biobank_agent.registry import skill
from pathlib import Path
import pandas as pd

@skill(
    name="advanced_analysis",
    description="Advanced analysis with data and memory",
    parameters={
        "disease_code": {
            "type": "string",
            "description": "ICD10 disease code",
        },
    },
)
def advanced_analysis(disease_code: str, *, ctx=None) -> dict:
    dm = ctx.dm
    state = ctx.state
    memory = ctx.memory
    report_dir = ctx.report_dir
    settings = ctx.settings
    
    # Query data
    sql = f"""
        SELECT {settings.subject_id_col}, {settings.diagnoses_code_col}
        FROM diagnoses
        WHERE {settings.diagnoses_code_col} LIKE '{disease_code}%'
    """
    df = dm.query(sql)
    
    # Record to long-term memory
    finding = f"Found {len(df)} cases of {disease_code}"
    memory.domain.append_finding(f"Disease: {disease_code}", finding)
    
    # Save outputs
    output_path = report_dir / f"{disease_code}_analysis.csv"
    df.to_csv(output_path, index=False)
    
    # Record figures for report
    state.figures.append(str(output_path))
    
    return {
        "count": len(df),
        "disease_code": disease_code,
        "output_file": str(output_path),
    }
```

## 4. Testing Locally

```python
# tests/test_my_skill.py
import pytest
from biobank_agent.registry import get_registry, autodiscover_skills, discover_custom_skills
from pathlib import Path

autodiscover_skills()
discover_custom_skills(Path("./custom_skills"))

def test_my_analysis(synthetic_ctx):
    reg = get_registry()
    result = reg.execute("my_analysis", {"code": "E11"}, ctx=synthetic_ctx)
    assert "count" in result
    assert result["count"] >= 0
```

## 5. Parameter Types

**Supported types in parameters dict:**
- `"string"` - Text parameter
- `"integer"` - Whole number
- `"number"` - Float/decimal
- `"boolean"` - True/False
- `"array"` - List (limited support)

**Parameter options:**
```python
parameters={
    "name": {
        "type": "string",
        "description": "Human-readable description",
        "default": "optional_default_value",  # Makes param optional
    },
}
```

## 6. Common Mistakes

❌ **Don't forget the `ctx` parameter:**
```python
# WRONG
def my_skill(param1: str) -> dict:
    return {"result": param1}

# RIGHT
def my_skill(param1: str, *, ctx=None) -> dict:
    return {"result": param1}
```

❌ **Don't modify the decorator parameter definitions:**
```python
# WRONG - Parameter object structure is special
parameters={"field": {"type": "string"}}

# RIGHT
parameters={
    "field": {
        "type": "string",
        "description": "...",
    }
}
```

❌ **Don't forget to return a dict:**
```python
# WRONG - Returns string
def my_skill(x: str, *, ctx=None) -> dict:
    return x

# RIGHT
def my_skill(x: str, *, ctx=None) -> dict:
    return {"result": x}
```

## 7. Access Patterns in Context

```python
# Data queries
df = ctx.dm.query("SELECT * FROM biomarkers LIMIT 10")
total_subjects = ctx.dm.count_subjects()

# Configuration
biobank_name = ctx.settings.biobank_name
data_dir = ctx.settings.data_dir
subject_id_col = ctx.settings.subject_id_col

# Session state
current_records = ctx.state.records
context_summary = ctx.state.context_summary()

# Memory operations
memory = ctx.memory
memory.domain.append_finding("Category", "Finding text")
memory.user.upsert_preference("key", "value")

# Field catalog
field_info = ctx.catalog.fields.get("30600-0.0")

# Output directory
output_path = ctx.report_dir / "my_output.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text("...")

# Record figures for report
ctx.state.figures.append(str(output_path))
```

## 8. File Organization

```
biobank_agent/
|-- config.py              # custom_skills_dir defined here (line 87)
|-- registry.py            # @skill decorator and discovery functions
|-- agent.py               # calls autodiscover_skills() and discover_custom_skills()
|-- skills/                # built-in skills (currently 105 registered tools)
|   |-- __init__.py
|   |-- prevalence.py
|   |-- think.py
|   `-- ...
|
./custom_skills/           # YOUR CUSTOM SKILLS GO HERE
    |-- my_analysis.py
    |-- special_query.py
    `-- ...
```

## 9. Testing with CLI

```bash
# List all skills (including your custom ones)
/skills

# Get help on a skill
/help my_analysis

# View session status (includes skill count)
/status

# Use the skill in a query
analyze disease E11 using my_analysis
```

## 10. Troubleshooting

**Q: My skill doesn't appear in /skills**
- Check if file is in `./custom_skills/`
- Check if filename doesn't start with `_`
- Look for errors in logs: `python -m biobank_agent.cli 2>&1 | grep -i "custom"`
- Restart the agent

**Q: Skill shows up but fails when called**
- Check parameter names match function signature
- Make sure `ctx` parameter is keyword-only (after `*`)
- Verify all required parameters are provided
- Check logs for detailed error message

**Q: How do I reload my skill without restarting?**
- Currently not supported, but you can:
  1. Edit the skill file
  2. Exit and restart the agent
  3. Or use `registry.reload_skill("my_analysis")` programmatically

---

**For detailed architecture documentation, see:** [`../architecture/SKILLS.md`](../architecture/SKILLS.md)
