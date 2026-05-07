# Biobank Agent: Skills System - Documentation Index

## Overview

The biobank_agent project uses a **decorator-based skill registry system** for extensible command execution. This index provides navigation to all documentation about skills architecture, discovery, invocation, and custom skill creation.

---

## 📚 Documentation Files

### 1. **[architecture/SKILL_ARCHITECTURE.md](architecture/SKILL_ARCHITECTURE.md)** (654 lines)
**Comprehensive technical reference** - Read this first for deep understanding

- **Section 1:** Skill Discovery & Loading Mechanism
  - Entry point: Agent initialization (agent.py:116-119)
  - Built-in skill auto-discovery (registry.py:184-201)
  - Custom skill discovery (registry.py:204-233)
  - How 44 built-in skills are loaded

- **Section 2:** Skill Interface & Base Class
  - @skill decorator mechanism (registry.py:137-179)
  - Skill function signature conventions
  - Example: prevalence.py skill
  - Context object injection at runtime

- **Section 3:** Skill Registry & Invocation
  - SkillRegistry class architecture (registry.py:29-124)
  - Registration vs. lazy loading
  - Global singleton pattern
  - Execution flow with context injection

- **Section 4:** Agent Loop Invocation
  - Tool call handling in agent.run() (agent.py:280-490)
  - Tool schema generation for LLM
  - ReAct loop with skill execution

- **Section 5:** Configuration Connection
  - custom_skills_dir config (config.py:87)
  - How config flows to discovery
  - .env override examples

- **Section 6:** CLI Integration
  - /skills command (cli.py:484-614)
  - /status command
  - Listing available skills

- **Section 7:** Dynamic Skill Creation
  - create_skill() tool for runtime generation
  - Workflow: generate → review → activate

- **Section 8:** Error Handling & Recovery
  - Error tracking in memory
  - Reflexion-based retry mechanism

- **Section 9:** Testing & Verification
  - Skill integration test patterns
  - test_skill_integration.py reference

- **Section 10-11:** Reference tables
  - Key files with line numbers
  - Loading & invocation architecture summary

---

### 2. **[CUSTOM_SKILLS_GUIDE.md](CUSTOM_SKILLS_GUIDE.md)** (326 lines)
**Quick start guide** - Start here if you want to create a custom skill

- **Section 1:** Basic Skill Template
  - Complete minimal example
  - Parameter definition format
  - Function signature conventions

- **Section 2:** Installation Steps (4-step process)
  - Create skill file in ./custom_skills/
  - Configure .env (optional)
  - Restart agent
  - Test via CLI

- **Section 3:** Advanced Examples
  - Using data manager (ctx.dm)
  - Interacting with memory (ctx.memory)
  - Saving outputs to report_dir
  - Recording figures for reports

- **Section 4:** Local Testing
  - Unit test template
  - Using autodiscover_skills()
  - Using discover_custom_skills()

- **Section 5:** Parameter Types
  - Supported JSON types
  - Default values
  - Required vs optional

- **Section 6:** Common Mistakes
  - Missing ctx parameter
  - Incorrect decorator format
  - Wrong return type

- **Section 7:** Context Access Patterns
  - Data queries via ctx.dm
  - Configuration via ctx.settings
  - Session state via ctx.state
  - Memory operations via ctx.memory
  - Output directory via ctx.report_dir

- **Section 8:** File Organization
  - Where skills go
  - Directory structure

- **Section 9-10:** Testing & Troubleshooting
  - CLI testing commands
  - Common problems and solutions

---

### 3. **[architecture/ARCHITECTURE_DIAGRAMS.md](architecture/ARCHITECTURE_DIAGRAMS.md)** (717 lines)
**Visual reference** - Read sections matching your task

- **Diagram 1:** Skill Discovery Flow at Initialization
  - Complete flow: autodiscover → decorator → register
  - Both built-in and custom skill paths
  - Singleton pattern

- **Diagram 2:** Skill Invocation During agent.run()
  - Message building with tool schemas
  - Tool call generation by LLM
  - Execution with context injection
  - Result processing and feedback loop

- **Diagram 3:** Registry Architecture
  - Internal data structures (_schemas, _callables, _module_paths, _descriptions)
  - Storage format for each
  - Key methods (register, execute, tool_schemas, list_skills)

- **Diagram 4:** @skill Decorator Flow
  - Parameter extraction
  - Schema building
  - Registration
  - Function preservation

- **Diagram 5:** Custom Skills Discovery Process
  - Directory scanning
  - File filtering (skip _*)
  - Dynamic module loading
  - Namespace creation

- **Diagram 6:** Context Injection at Execution
  - Function lookup
  - Context attachment as keyword argument
  - Available context properties
  - Return to agent loop

- **Diagram 7:** Complete Message Flow Through Agent Loop
  - Round-by-round progression
  - Message history accumulation
  - Tool execution
  - Final response

- **Diagram 8:** Config to Runtime Flow
  - config.py definition
  - .env override
  - Agent initialization
  - Registry population

---

## 🎯 Quick Navigation by Task

### "I want to understand the architecture"
1. Read: `architecture/SKILL_ARCHITECTURE.md` sections 1-4
2. Reference: `architecture/ARCHITECTURE_DIAGRAMS.md` Diagrams 1, 2, 3

### "I want to create a custom skill"
1. Read: `CUSTOM_SKILLS_GUIDE.md` sections 1-3
2. Copy: Section 1 template
3. Follow: Section 2 installation steps
4. Test: Section 9 CLI commands

### "I want to debug skill loading"
1. Reference: `architecture/ARCHITECTURE_DIAGRAMS.md` Diagrams 1, 5
2. Read: `architecture/SKILL_ARCHITECTURE.md` sections 1.2, 1.3
3. Check: registry.py lines 184-233

### "I want to debug skill execution"
1. Reference: `architecture/ARCHITECTURE_DIAGRAMS.md` Diagrams 2, 6, 7
2. Read: `architecture/SKILL_ARCHITECTURE.md` section 4
3. Check: agent.py lines 339-345

### "I want to see an example skill"
1. Read: `CUSTOM_SKILLS_GUIDE.md` section 1
2. Look at: `architecture/SKILL_ARCHITECTURE.md` section 2.3 (prevalence.py)
3. Browse: biobank_agent/skills/ directory

### "I want to understand context"
1. Read: `architecture/SKILL_ARCHITECTURE.md` section 2.4
2. Reference: `CUSTOM_SKILLS_GUIDE.md` section 7
3. See: agent.py lines 600-615

### "I want to extend the registry"
1. Read: `architecture/SKILL_ARCHITECTURE.md` section 3.1
2. Study: registry.py lines 29-124

---

## 🔍 Key Concepts at a Glance

### Core Pattern: Decorator-Based Registration
```python
@skill(name="x", description="y", parameters={...})
def x(param1, *, ctx=None) -> dict:
    # ctx injected automatically
    return {...}
```

### Discovery: Two Paths
1. **Built-in:** `autodiscover_skills()` scans `biobank_agent/skills/*.py`
2. **Custom:** `discover_custom_skills()` scans `./custom_skills/*.py` (configurable)

### Registry: Singleton Storage
- `_schemas` → LLM tool definitions
- `_callables` → Function objects
- `_module_paths` → Lazy loading paths
- `_descriptions` → Metadata

### Execution: Inject Context
```python
registry.execute(name, args, ctx=ctx)
→ automatically injects ctx as keyword argument
→ calls func(**args, ctx=ctx)
```

### Configuration
- Default: `custom_skills_dir = Path("./custom_skills")` (config.py:87)
- Override: `CUSTOM_SKILLS_DIR=/path/to/skills` in .env

---

## 📊 Key Statistics

| Metric | Value |
|--------|-------|
| Built-in Skills | 44 |
| Max Tool Rounds | 30 (configurable) |
| Total Documentation Lines | 1,926 |
| Skill Definition Files | 44 in biobank_agent/skills/ |
| Current Custom Skills | 0 (ready for extension) |
| Supported Parameter Types | 5 (string, integer, number, boolean, array) |

---

## 🔗 Key Files Referenced

| File | Purpose | Key Lines |
|------|---------|-----------|
| **biobank_agent/config.py** | Configuration | 87 |
| **biobank_agent/registry.py** | Core registry & decorator | 29-233 |
| **biobank_agent/agent.py** | Agent init & execution | 116-119, 280-490, 600-615 |
| **biobank_agent/skills/__init__.py** | Skill package | 1-7 |
| **biobank_agent/skills/prevalence.py** | Example skill | 1-76 |
| **biobank_agent/cli.py** | CLI /skills command | 484-614 |
| **tests/test_skill_integration.py** | Skill tests | 1-170 |

---

## 🚀 Getting Started Checklist

### To Create Your First Custom Skill:
- [ ] Read `CUSTOM_SKILLS_GUIDE.md` section 1
- [ ] Create `./custom_skills/my_skill.py`
- [ ] Copy template from section 1
- [ ] Implement your skill function
- [ ] Test: `/skills` to see it listed
- [ ] Test: Call the skill from agent
- [ ] Reference: Section 7 for context access patterns

### To Understand Architecture:
- [ ] Read `architecture/SKILL_ARCHITECTURE.md` section 1 (discovery)
- [ ] Read `architecture/SKILL_ARCHITECTURE.md` section 3 (registry)
- [ ] Study `architecture/ARCHITECTURE_DIAGRAMS.md` Diagram 1
- [ ] Study `architecture/ARCHITECTURE_DIAGRAMS.md` Diagram 2
- [ ] Browse biobank_agent/registry.py

### To Debug Issues:
- [ ] Check logs for "custom_skills" errors
- [ ] Verify file in ./custom_skills/ (not _*)
- [ ] Verify @skill decorator syntax
- [ ] Verify ctx parameter (keyword-only, after *)
- [ ] Verify return type is dict
- [ ] See `CUSTOM_SKILLS_GUIDE.md` section 10 for troubleshooting

---

## 📝 Document Maintenance

This documentation was generated on **April 23, 2026** by analyzing:
- biobank_agent/config.py
- biobank_agent/registry.py
- biobank_agent/agent.py
- biobank_agent/skills/ (44 modules)
- biobank_agent/cli.py
- tests/test_skill_integration.py

The documentation is accurate as of the last code review but may need updates if:
- New discovery mechanisms are added
- Skill registration changes
- Context object structure changes
- CLI commands are modified

---

## 🤝 Contributing Custom Skills

When creating skills for this project:

1. **Follow the template** in `CUSTOM_SKILLS_GUIDE.md` section 1
2. **Place in** `./custom_skills/` directory
3. **Test locally** before committing
4. **Document** your skill's parameters
5. **Follow naming** conventions (snake_case)
6. **Return dict** with clear keys
7. **Use context** for data access (ctx.dm, ctx.state, etc.)
8. **Handle errors** gracefully

---

**For questions or updates to this documentation, refer to the source files listed above.**
