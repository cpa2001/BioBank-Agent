# Biobank Agent: Custom Skills Loading & Invocation Analysis

## Executive Summary

The biobank_agent project uses a **decorator-based skill registry** with support for both built-in skills and custom skills. Skills are:
1. Discovered via Python module introspection and loaded at agent initialization
2. Registered through the `@skill` decorator
3. Executed through a central registry during the ReAct agent loop
4. Custom skills are loaded from a configurable directory

---

## 1. SKILL DISCOVERY & LOADING MECHANISM

### 1.1 Entry Point: Agent Initialization

**File:** `biobank_agent/agent.py` (lines 116-119)

```python
# Skills
autodiscover_skills()
discover_custom_skills(settings.custom_skills_dir)
self.registry = get_registry()
logger.info("Loaded %d skills", len(self.registry))
```

**Initialization sequence:**
1. `autodiscover_skills()` - Loads all built-in skills from `biobank_agent.skills` package
2. `discover_custom_skills(settings.custom_skills_dir)` - Loads custom skills from configured directory
3. `get_registry()` - Returns the global `SkillRegistry` singleton

---

### 1.2 Built-in Skill Discovery: `autodiscover_skills()`

**File:** `biobank_agent/registry.py` (lines 184-201)

```python
def autodiscover_skills(package_path: str = "biobank_agent.skills") -> None:
    """Import all modules in the skills package to trigger @skill decorators."""
    try:
        pkg = importlib.import_module(package_path)
    except ImportError:
        logger.warning("Could not import %s", package_path)
        return

    pkg_dir = Path(pkg.__file__).parent
    for finder, module_name, is_pkg in pkgutil.iter_modules([str(pkg_dir)]):
        if module_name.startswith("_"):
            continue
        full_name = f"{package_path}.{module_name}"
        try:
            importlib.import_module(full_name)
            logger.debug("Loaded skill module: %s", full_name)
        except Exception as e:
            logger.warning("Failed to load skill %s: %s", full_name, e)
```

**How it works:**
1. Uses `pkgutil.iter_modules()` to discover all `.py` modules in `biobank_agent/skills/`
2. Skips modules starting with `_` (private modules)
3. Dynamically imports each module with `importlib.import_module()`
4. As each module is imported, the `@skill` decorators automatically register skills in the global registry
5. **58 skills** are currently available in the project

**Current Skills:** 58 total, including data analysis, modelling, genetic target
interpretation, literature, reporting, guardrail, memory, project documentation,
and external review tools.

---

### 1.3 Custom Skill Discovery: `discover_custom_skills()`

**File:** `biobank_agent/registry.py` (lines 204-233)

```python
def discover_custom_skills(custom_dir: Path) -> int:
    """Discover and load skills from a custom directory.

    Scans ``custom_dir`` for ``.py`` files containing @skill-decorated functions.
    Returns the number of newly loaded skills.
    """
    import sys

    if not custom_dir.exists():
        return 0

    before = len(_registry)
    for py_file in sorted(custom_dir.glob("*.py")):
        if py_file.name.startswith("_"):
            continue
        module_name = f"custom_skills.{py_file.stem}"
        spec = importlib.util.spec_from_file_location(module_name, py_file)
        if spec and spec.loader:
            try:
                mod = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = mod
                spec.loader.exec_module(mod)
                logger.info("Loaded custom skill: %s", py_file.name)
            except Exception as e:
                logger.warning("Failed to load custom skill %s: %s", py_file.name, e)

    loaded = len(_registry) - before
    if loaded > 0:
        logger.info("Loaded %d custom skill(s) from %s", loaded, custom_dir)
    return loaded
```

**How it works:**
1. Checks if custom directory exists (default: `./custom_skills/`)
2. Scans all `.py` files (excluding those starting with `_`)
3. Uses `importlib.util.spec_from_file_location()` to dynamically load each file
4. Registers the module in `sys.modules` with namespace `custom_skills.<filename>`
5. Executes the module, triggering any `@skill` decorators
6. Returns count of newly loaded skills

**Configuration:** Defined in config.py line 87
```python
custom_skills_dir: Path = Path("./custom_skills")
```

---

## 2. SKILL INTERFACE & BASE CLASS (Decorator Pattern)

### 2.1 The `@skill` Decorator

**File:** `biobank_agent/registry.py` (lines 137-179)

```python
def skill(
    name: str,
    description: str,
    parameters: dict[str, dict],
    required: list[str] | None = None,
):
    """Decorator to register a function as an agent skill (tool)."""
    # Build OpenAI function-calling schema
    props = {}
    for pname, pdef in parameters.items():
        props[pname] = {k: v for k, v in pdef.items()}

    if required is None:
        required = [p for p in parameters if "default" not in parameters[p]]

    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }

    def decorator(func: Callable) -> Callable:
        func._skill_name = name
        func._skill_schema = schema
        _registry.register(name, func, schema)
        return func

    return decorator
```

### 2.2 Skill Function Signature

**Required Convention:**
```python
def skill_function(param1: str, param2: int = 10, *, ctx=None) -> dict:
    """Skill implementation.
    
    Args:
        param1: Positional/keyword argument
        param2: Optional parameter with default
        ctx: Special keyword-only context parameter
        
    Returns:
        dict: Result dictionary
    """
    # Access context
    dm = ctx.dm                 # DataManager
    settings = ctx.settings    # Settings
    state = ctx.state          # SessionState
    memory = ctx.memory        # LongTermMemory
    report_dir = ctx.report_dir  # Path for outputs
    catalog = ctx.catalog      # FieldCatalog
    
    # Implementation
    result = {...}
    return result
```

### 2.3 Example: Prevalence Skill

**File:** `biobank_agent/skills/prevalence.py` (lines 1-76)

```python
@skill(
    name="prevalence",
    description="Calculate disease prevalence in the biobank cohort...",
    parameters={
        "top_n": {
            "type": "integer",
            "description": "Number of top diseases to show (default 20)",
            "default": 20,
        },
        "chapter_filter": {
            "type": "string",
            "description": "Optional ICD10 chapter letter to filter (e.g. 'E' for endocrine)",
            "default": "",
        },
    },
    required=[],
)
def prevalence(top_n: int = 20, chapter_filter: str = "", *, ctx=None) -> dict:
    dm = ctx.dm
    diag_col = ctx.settings.diagnoses_code_col
    id_col = ctx.settings.subject_id_col
    
    # Query data
    sql = f"""
        SELECT LEFT({diag_col}, 3) AS code, COUNT(DISTINCT {id_col}) AS n_patients
        FROM diagnoses
        WHERE {diag_col} IS NOT NULL AND {diag_col} != ''
    """
    # ... implementation ...
    
    return {
        "total_subjects": total,
        "top_diseases": [...],
        "figure": str(paths[0]),
    }
```

### 2.4 Context Object Injected at Runtime

**File:** `biobank_agent/agent.py` (lines 600-615)

```python
def _build_ctx(self, report_dir: Optional[Path] = None):
    """Build the context object passed to skills."""
    class SkillContext:
        pass

    ctx = SkillContext()
    ctx.dm = self.dm                    # DataManager
    ctx.catalog = self.catalog          # FieldCatalog
    ctx.state = self.state              # SessionState
    ctx.state.memory = self.memory      # Expose memory to skills
    ctx.settings = self.settings        # Settings
    ctx.memory = self.memory            # LongTermMemory
    ctx.report_dir = report_dir or self.settings.reports_dir
    return ctx
```

**Context provides access to:**
- `ctx.dm` - Data queries, subject counts, column access
- `ctx.catalog` - Field definitions and metadata
- `ctx.state` - Session state, figures list, analysis records
- `ctx.settings` - Configuration (biobank name, paths, etc.)
- `ctx.memory` - Long-term memory, error tracking, domain knowledge
- `ctx.report_dir` - Directory for saving outputs

---

## 3. SKILL REGISTRY & INVOCATION

### 3.1 The SkillRegistry Class

**File:** `biobank_agent/registry.py` (lines 29-124)

```python
class SkillRegistry:
    """Central registry of all available skills (tools)."""

    def __init__(self) -> None:
        self._schemas: dict[str, dict] = {}          # name → OpenAI tool schema
        self._callables: dict[str, Callable] = {}    # name → function (lazy)
        self._module_paths: dict[str, str] = {}      # name → "biobank_agent.skills.xyz"
        self._descriptions: dict[str, str] = {}      # name → description

    # Registration
    def register(self, name: str, func: Callable, schema: dict) -> None:
        """Eagerly register a skill."""
        self._schemas[name] = schema
        self._callables[name] = func
        self._descriptions[name] = schema["function"]["description"]

    def register_lazy(self, name: str, module_path: str, schema: dict) -> None:
        """Register schema only — implementation loaded on first call."""
        self._schemas[name] = schema
        self._module_paths[name] = module_path
        self._descriptions[name] = schema["function"]["description"]

    # Execution
    def execute(self, name: str, args: dict, ctx: Any = None) -> Any:
        """Execute a skill by name, injecting ctx if the function accepts it."""
        if name not in self._callables:
            if name in self._module_paths:
                # Lazy-load the module and find the skill function
                mod = importlib.import_module(self._module_paths[name])
                func = getattr(mod, name, None)
                if func is None:
                    # Search for any decorated function with matching _skill_name
                    for attr_name in dir(mod):
                        attr = getattr(mod, attr_name)
                        if callable(attr) and getattr(attr, "_skill_name", None) == name:
                            func = attr
                            break
                if func is None:
                    raise ValueError(f"Skill '{name}' not found in {self._module_paths[name]}")
                self._callables[name] = func
            else:
                raise ValueError(f"Unknown skill: {name}")

        func = self._callables[name]
        if ctx is not None:
            args["ctx"] = ctx
        return func(**args)

    # Schema access
    def tool_schemas(self) -> list[dict]:
        """Return OpenAI-format tool schemas for all registered skills."""
        return list(self._schemas.values())

    def list_skills(self) -> list[dict]:
        """Return skill name + description pairs."""
        return [
            {"name": n, "description": self._descriptions.get(n, "")}
            for n in self._schemas
        ]

    def __len__(self) -> int:
        return len(self._schemas)
```

**Key features:**
- **Lazy loading:** Skills can be registered with module path but not loaded until execution
- **Dual storage:** Schemas for LLM, callables for execution
- **Context injection:** Automatically adds `ctx` parameter if called with context
- **Dynamic lookup:** Searches for functions with matching `_skill_name` attribute

### 3.2 Global Registry Singleton

**File:** `biobank_agent/registry.py` (lines 126-132)

```python
# Global registry instance
_registry = SkillRegistry()

def get_registry() -> SkillRegistry:
    return _registry
```

---

## 4. SKILL INVOCATION FLOW IN THE AGENT LOOP

### 4.1 Tool Call Handling in agent.run()

**File:** `biobank_agent/agent.py` (lines 280-490)

**Flow:**
```
1. User query → agent.run(query)
   ↓
2. LLM generates tool_calls with function names and arguments
   ↓
3. For each tool_call:
   a. Build skill context: ctx = self._build_ctx(report_dir)
   b. Execute: result = self.registry.execute(tc.name, tc.args, ctx=ctx)
   c. Catch errors, record metrics, track provenance
   ↓
4. Append tool result to message history
   ↓
5. Loop until LLM produces text-only response (no tool_calls)
```

**Specific execution code (lines 339-345):**
```python
for tc in response.tool_calls:
    logger.info("Tool call: %s(%s)", tc.name, tc.args)
    t0 = time.time()
    figs_before = len(self.state.figures)
    ctx = self._build_ctx(_report_dir)
    try:
        result = self.registry.execute(tc.name, tc.args, ctx=ctx)
        result_str = json.dumps(result, default=str, ensure_ascii=False)
```

### 4.2 Tool Schema Generation for LLM

**File:** `biobank_agent/agent.py` (lines 304-311)

```python
if self.settings.multi_model_enabled and len(self.orchestrator.model_pool) > 1:
    response = self.orchestrator.route(
        query=user_query,
        messages=all_messages,
        tools=self.registry.tool_schemas() or None,  # ← Schemas for LLM
        records=self.state.records,
        force_strategy=force_strategy,
    )
else:
    response = self.llm.chat(
        messages=all_messages,
        tools=self.registry.tool_schemas() or None,  # ← Schemas for LLM
    )
```

The registry provides OpenAI-format tool schemas that the LLM uses to generate tool_calls.

---

## 5. CONFIG.PY CONNECTION

### 5.1 custom_skills_dir Configuration

**File:** `biobank_agent/config.py` (line 87)

```python
# ── Custom Skills ────────────────────────────────────────
custom_skills_dir: Path = Path("./custom_skills")
```

**Flow:**
1. Defined in `Settings` class with default value `Path("./custom_skills")`
2. Read from `.env` file via pydantic-settings
3. Passed to `discover_custom_skills()` during agent initialization
4. Used to dynamically load `.py` files containing `@skill`-decorated functions

**Example .env override:**
```bash
CUSTOM_SKILLS_DIR="/path/to/my/skills"
```

---

## 6. SLASH COMMANDS & CLI INTEGRATION

### 6.1 /skills Command

**File:** `biobank_agent/cli.py` (lines 484-614)

```python
elif cmd == "/skills":
    _show_skills(agent)

def _show_skills(agent: Agent) -> None:
    table = Table(title="Available Skills", show_lines=False, padding=(0, 1))
    table.add_column("Skill", style="cyan", no_wrap=True)
    table.add_column("Description", style="dim")
    for s in agent.registry.list_skills():
        table.add_row(s["name"], s["description"][:80])
    console.print(table)
```

**What it does:**
- Lists all registered skills with descriptions
- Uses `agent.registry.list_skills()` to fetch available skills
- Displays in a formatted table

### 6.2 /status Command

**File:** `biobank_agent/cli.py` (lines 618-643)

Shows:
```
Skills: <count from len(agent.registry)>
```

---

## 7. DYNAMIC SKILL CREATION

### 7.1 create_skill() Tool

**File:** `biobank_agent/skills/create_skill.py` (lines 18-100+)

The agent can dynamically create new skills:

```python
@skill(
    name="create_skill",
    description="Create a new analysis skill from code with AST validation...",
    parameters={
        "name": {"type": "string", "description": "Skill name"},
        "description": {"type": "string", "description": "One-line description"},
        "parameters": {"type": "string", "description": "JSON dict of parameters"},
        "code_body": {"type": "string", "description": "Python code implementing the skill"},
    },
)
def create_skill(name: str, description: str, parameters: str, code_body: str, *, ctx=None) -> dict:
    # Validates skill name, parameters JSON, code AST
    # Generates skill file to reports/generated_skills/
    # Returns generated code for manual review
    # User must move to biobank_agent/skills/ and restart to activate
```

**Workflow:**
1. LLM calls `create_skill()` with code
2. AST validation checks for:
   - Allowed imports (pandas, numpy, scipy, etc.)
   - Forbidden calls (exec, eval, etc.)
3. Generated skill saved to `reports/generated_skills/`
4. User reviews and manually moves to `biobank_agent/skills/`
5. Restart agent to load (via `autodiscover_skills()`)

---

## 8. ERROR HANDLING & RECOVERY

### 8.1 Error Tracking

**File:** `biobank_agent/agent.py` (lines 347-365)

```python
except Exception as e:
    logger.error("Tool %s failed: %s", tc.name, e)
    
    # Track error in long-term memory
    try:
        self.memory.record_error(
            error_type=type(e).__name__,
            error_message=str(e),
            skill_name=tc.name,
            context={"args": {k: str(v)[:100] for k, v in tc.args.items()}},
        )
    except Exception:
        pass
    
    # Get suggestions from error history
    suggestions = self.memory.get_error_suggestions(type(e).__name__, tc.name)
```

### 8.2 Reflexion-Based Retry

**File:** `biobank_agent/agent.py` (lines 367-393)

```python
if self.reflexion and self.reflexion.should_retry(e, tc.name):
    reflection = self.reflexion.reflect(
        skill_name=tc.name,
        args=tc.args,
        error=e,
        context=self.state.context_summary()[:500],
    )
    if reflection.retry_recommended and reflection.corrections:
        retry_args = {**tc.args, **reflection.corrected_args}
        result = self.registry.execute(tc.name, retry_args, ctx=ctx)
```

---

## 9. TESTING & VERIFICATION

### 9.1 Skill Integration Tests

**File:** `tests/test_skill_integration.py`

```python
from biobank_agent.registry import get_registry, autodiscover_skills

autodiscover_skills()

class TestPrevalenceSkill:
    def test_runs_without_error(self, synthetic_ctx):
        reg = get_registry()
        result = reg.execute("prevalence", {"top_n": 5}, ctx=synthetic_ctx)
        assert isinstance(result, dict)
        assert "error" not in result

class TestSkillRegistration:
    def test_all_skills_registered(self):
        reg = get_registry()
        assert len(reg) >= 43, f"Expected 43+ skills, got {len(reg)}"

    def test_all_schemas_have_name(self):
        reg = get_registry()
        for schema in reg.tool_schemas():
            func = schema.get("function", {})
            assert func.get("name")
```

---

## 10. SUMMARY: LOADING & INVOCATION ARCHITECTURE

### Discovery Pipeline
```
┌─────────────────────────────────────────────────────┐
│ Agent.__init__() [agent.py:116-119]                 │
├─────────────────────────────────────────────────────┤
│ 1. autodiscover_skills()                            │
│    └─ Scans biobank_agent/skills/*.py               │
│       └─ Each module @skill decorator registers     │
│                                                      │
│ 2. discover_custom_skills(settings.custom_skills_dir)
│    └─ Scans ./custom_skills/*.py (configurable)    │
│       └─ Each custom skill @skill decorator registers
│                                                      │
│ 3. get_registry()                                   │
│    └─ Returns global _registry singleton            │
└─────────────────────────────────────────────────────┘
```

### Invocation Pipeline
```
┌──────────────────────────────────┐
│ User Query                        │
├──────────────────────────────────┤
│ LLM receives tool_schemas()      │
│ ↓                                │
│ LLM generates tool_calls         │
│ ↓                                │
│ For each tool_call:              │
│ ├─ Build context                 │
│ ├─ registry.execute(name, args)  │
│ ├─ Inject ctx parameter          │
│ ├─ Execute function              │
│ └─ Return result                 │
│ ↓                                │
│ Append tool results to history   │
│ ↓                                │
│ Loop until no tool_calls         │
└──────────────────────────────────┘
```

### Registry State
```
SkillRegistry._schemas    → name → OpenAI tool schema (for LLM)
SkillRegistry._callables  → name → Function object (for execution)
SkillRegistry._module_paths → name → Module path (for lazy loading)
SkillRegistry._descriptions → name → Skill description
```

---

## 11. KEY FILES & LINE NUMBERS REFERENCE

| File | Purpose | Key Lines |
|------|---------|-----------|
| `biobank_agent/config.py` | Configuration, custom_skills_dir default | 87 |
| `biobank_agent/registry.py` | Core registry, decorator, discovery | 29-233 |
| `biobank_agent/agent.py` | Agent initialization, skill invocation | 116-119, 304-345 |
| `biobank_agent/skills/__init__.py` | Skill package initialization | 1-7 |
| `biobank_agent/skills/prevalence.py` | Example skill (prevalence calculation) | 1-76 |
| `biobank_agent/skills/think.py` | Example skill (reasoning tool) | 1-66 |
| `biobank_agent/skills/create_skill.py` | Dynamic skill creation from code | 18-100+ |
| `biobank_agent/cli.py` | CLI /skills command | 484-614 |
| `tests/test_skill_integration.py` | Skill testing framework | 1-170 |
