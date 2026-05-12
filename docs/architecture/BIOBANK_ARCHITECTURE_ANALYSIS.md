# BioBank Agent: Comprehensive Architecture & Extensibility Analysis

**Project**: Biobank Agent v3.0.0-rc1  
**Author**: CHEN Pengan (CUHK, SAIS)  
**Repository**: https://github.com/cpa2001/BioBank-Agent  
**Analysis Date**: 2026-05-11

---

## Executive Summary

**BioBank Agent** is a production-grade autonomous scientific discovery system for population-scale biobank research, combining ReAct agent orchestration with 64 registered skills for hypothesis-driven discovery through data analysis, predictive modeling, literature integration, and publication-quality reporting.

**Key Infrastructure Findings:**
- ✅ **Highly extensible**: Decorator-based skill registry with hot-reload, custom skill discovery, multi-model orchestration
- ✅ **Multi-biobank capable**: Bank adapter pattern supports UKB, FinnGen (CKB), HPP with config overrides
- ✅ **Production deployment ready**: GitHub Actions CI/CD, eval harness, telemetry layer, MCP integration
- ✅ **Portable architecture**: Platform support (macOS ARM, Linux x86+GPU), Python 3.10+, OpenAI-compatible LLM APIs
- ⚠️ **In v3 transition**: RC1 with 5 blocked criteria (mostly credential-dependent); v3.0 target adds MCP STDIO, remote CI, RAP path
- 📦 **Complex dependency graph**: 51 core dependencies + 7 optional groups (GPU, research, literature, etc.)

---

## 1. Project Metadata & Packaging

### pyproject.toml Configuration

**Version & Build:**
```toml
[project]
name = "biobank-agent"
version = "3.0.0-rc1"
description = "Autonomous scientific discovery agent for population-scale biobank research"
requires-python = ">=3.10"
```

**Entry Point:**
```toml
[project.scripts]
biobank = "biobank_agent.cli:main"
```

### Dependencies (51 core + 7 optional)

**Core Stack:**
- **Data/ML**: duckdb, pyarrow, pandas, numpy, scikit-learn, xgboost, lightgbm
- **Visualization**: matplotlib, seaborn (Nature/ICML-style SVG+PDF)
- **Agent**: openai (1.50+), rich, prompt-toolkit
- **Analysis**: lifelines (survival), scipy, networkx (pathway analysis)
- **Config**: pydantic, pydantic-settings, python-dotenv
- **Web & PDF**: pymupdf, httpx, html2text, beautifulsoup4, ddgs (DuckDuckGo), pyyaml

**Optional Dependency Groups:**
- `gpu` — CatBoost, SHAP, UMAP (hardware acceleration)
- `verify` — Z3 SMT solver (formal constraint verification)
- `graphpop` — Neo4j (population genetics graph queries)
- `research` — inferactively-pymdp (active learning)
- `literature` — paper-qa (citation-first literature search)
- `structured` — instructor (Pydantic-enforced LLM outputs)
- `tui` — textual (terminal UI)
- `tracing` — OpenTelemetry (observability)
- `enrichment` — gseapy (gene set enrichment)
- `single-cell` — cellxgene-census (optional, Python <3.13)

**Installation:**
```bash
pip install -e ".[all]"          # All optional features
pip install -e ".[gpu,literature,tracing]"  # Custom subset
```

---

## 2. CLI Structure & Commands

### Entry Point: `biobank_agent/cli_legacy.py`

The CLI package is a compatibility wrapper (v3 decomposition in progress) that routes through `cli_legacy.py` into a modern `cli/commands/` submodule hierarchy.

**CLI Architecture:**
```
biobank_agent/
├── cli/                          # v3 modular decomposition (in progress)
│   ├── __init__.py              # Compatibility wrapper → cli_legacy
│   ├── __main__.py              # Entry for python -m biobank_agent.cli
│   ├── commands/                # Slash command handlers
│   │   ├── base.py              # CommandContext, RegisteredCommand, @action decorator
│   │   ├── mcp.py               # /mcp-list, /mcp-start, /mcp-health, /mcp-call, /mcp-stop
│   │   ├── plan.py              # /plan <task>
│   │   ├── memory.py            # /memory
│   │   ├── session.py           # /clear, /cost, /export, /status, /history
│   │   ├── registry.py          # /skills, /models
│   │   └── reproducibility.py   # /record, /pipelines, /errors
│   ├── tui/                     # Terminal UI (Textual-based, v3 scaffold)
│   │   ├── main.py              # App entry
│   │   ├── main_screen.py       # Multi-panel layout (input, results, tools, disclosure)
│   │   ├── tool_status.py       # Live tool execution view
│   │   ├── progress_panel.py    # Task progress tracker
│   │   └── ...                  # Other UI components
│   └── live/
│       └── streaming_renderer.py # Event-to-terminal renderer
│
└── cli_legacy.py                # Current v2.4 CLI implementation (compatibility mode)
```

### 22 Slash Commands (Comprehensive)

| Category | Command | Purpose |
|----------|---------|---------|
| **Help & Discovery** | `/help` | Show all commands |
| | `/skills` | List 64 registered analysis skills |
| | `/models-available` | Fetch supported models from LLM relay |
| | `/external-agents` | Check Codex/Claude Code CLI availability |
| **Analysis** | `/plan <task>` | Enter structured planning mode (INTAKE → ALIGNMENT → EXECUTION → DONE) |
| | `/think <query>` | Invoke internal reasoning tool |
| | `/model <name>` | Switch active LLM model |
| **Session** | `/clear` | Reset conversation state |
| | `/compact` | Compress history (structured extraction) |
| | `/status` | Full session status (cost, cohorts, models, memory) |
| | `/cost` | Token usage + estimated USD |
| | `/history` | Analysis transcript replay |
| **Output & Export** | `/export <fmt>` | Session export (JSON/Markdown) |
| | `/figures` | List generated SVG/PDF figures |
| | `/cohorts` | Active case/control cohort summaries |
| | `/models` | Trained model AUC/performance |
| **Pipelines & Memory** | `/record <name>` | Save workflow as replayable pipeline |
| | `/pipelines` | List saved pipelines |
| | `/replay <name>` | Re-run saved pipeline |
| | `/memory` | 8-tier memory summary (short/mid/long/error/domain/user/episodic/action-graph) |
| | `/errors` | Error catalog + suggested fixes |
| **External Review** | `/codex-plan <task>` | Ask Codex for read-only plan |
| | `/codex-check [focus]` | Ask Codex to review execution |
| | `/claude-plan <task>` | Ask Claude Code for read-only plan |
| | `/claude-check [focus]` | Ask Claude Code to review execution |
| **MCP Integration** | `/mcp-list` | List configured MCP servers & loaded tools |
| | `/mcp-start` | Start STDIO MCP servers, register remote tools |
| | `/mcp-health [--repair]` | Probe MCP servers, optionally reconnect |
| | `/mcp-call <tool> [json]` | Direct MCP tool invocation (audit/debug) |
| | `/mcp-stop` | Stop MCP clients, unregister tools |

### Command Registration Pattern

Commands use a decorator-based registration system:

```python
# From biobank_agent/cli/commands/base.py
@action("skill_schemas")  # Exposes a skill_schemas action
def _registered_command(ctx: CommandContext, arg: str):
    ctx.console.print("[bold]Available Skills[/]")
    for skill in ctx.agent.registry.list_skills():
        ctx.console.print(f"  {skill['name']}: {skill['description']}")

RegisteredCommand(
    "/skills",                              # slash command name
    "/skills",                              # help syntax
    "List available analysis skills",       # help text
    _registered_command                     # handler function
)
```

---

## 3. Configuration System (Pydantic-based)

### `biobank_agent/config.py` - `Settings` Class

**Core Design:**
- Pydantic v2 BaseSettings with `.env` file support
- `AliasChoices` for backward compatibility (legacy env var names)
- Biobank-identity overridable (UKB defaults, extensible to CKB/FinnGen/HPP)

**Key Settings:**

```python
# LLM Configuration
llm_base_url: str = "https://openrouter.ai/api/v1"  # OpenAI-compatible endpoint
llm_api_key: str = ""
llm_model: str = "deepseek/deepseek-v4-pro"

# Data Paths (backward-compatible with legacy names)
data_dir: Path = Path("./milton_data")    # UKB_PARQUET_DIR (alias)
raw_dir: Path = Path("./UKB")             # UKB_RAW_DIR (alias)
full_ukb_feature_store_dir: str = ""      # Full UKB parquet rebuild path

# Biobank Identity (overridable for FinnGen, CKB, HPP)
bank_id: str = "ukb"
biobank_name: str = "UK Biobank"
biobank_abbreviation: str = "UKB"
biobank_description: str = "..."
biobank_caveats: str = "healthy volunteer cohort with known selection biases"

# Column/Schema Identity
subject_id_col: str = "eid"             # UKB: eid; CKB: study_id; HPP: participant_id
diagnoses_code_col: str = "diag_icd10"  # ICD-10 for UKB/CKB, ICD-9 for HPP
deaths_code_col: str = "cause_icd10"

# Data Governance
data_deidentified: bool = True
data_use_full_dataset_default: bool = True
max_train_rows_default: int = 0  # 0 = no cap
disclosure_control_mode: str = "internal"  # internal | external | strict

# Web Search
search_provider: str = "duckduckgo"  # or brave, serper
search_api_key: str = ""

# Plan Mode
plan_repair_budget_per_step: int = 3
plan_repair_budget_total: int = 8
plan_external_council_enabled: bool = True
plan_external_council_policy: str = "requested"  # requested | always | never
plan_external_council_timeout_s: int = 180
plan_external_council_agents: str = "codex,claude,gemini"

# Custom Skills
custom_skills_dir: Path = Path("./custom_skills")

# Agent Limits
max_tool_rounds: int = 30
context_window: int = 180_000

# Telemetry (privacy-preserving, local-only JSONL)
telemetry_enabled: bool = True
otel_enabled: bool = False
otel_exporter: str = "none"  # none | console | otlp
```

### `.env.example` Template

```bash
# === LLM API ===
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=your-api-key-here
LLM_MODEL=deepseek/deepseek-v4-pro

# === Multi-Agent Routing (Optional) ===
MULTI_MODEL_ENABLED=true
AUTO_DISCOVER_MODELS=true
PREFERRED_MULTI_MODELS=deepseek/deepseek-v4-pro,moonshotai/kimi-k2.6,z-ai/glm-5.1

# === Data Paths ===
DATA_DIR=./data
RAW_DIR=./raw

# === Biobank Identity (Override for non-UKB) ===
# BIOBANK_NAME=UK Biobank
# BIOBANK_ABBREVIATION=UKB
# SUBJECT_ID_COL=eid
# DIAGNOSES_CODE_COL=diag_icd10

# === Web Search ===
SEARCH_PROVIDER=duckduckgo

# === Optional Paths ===
REPORTS_DIR=./reports
MEMORY_DIR=~/.biobank_agent
PLANS_DIR=./plans
CUSTOM_SKILLS_DIR=./custom_skills
```

---

## 4. Skill Registry & Extensibility

### Decorator-Based Skill Registration

**File:** `biobank_agent/registry.py`

```python
from biobank_agent.registry import skill

@skill(
    name="prevalence",
    description="Calculate disease prevalence for an ICD10 code",
    parameters={
        "icd10_code": {"type": "string", "description": "ICD10 code prefix"},
        "sample_size": {"type": "integer", "description": "Max records to analyze", "default": 0},
    },
    required=["icd10_code"]
)
def prevalence(icd10_code: str, sample_size: int = 0, *, ctx) -> dict:
    """Skill implementation."""
    return {
        "icd10_code": icd10_code,
        "prevalence": 0.15,
        "n": 75300,
    }
```

### SkillRegistry Class Architecture

**Core Methods:**

```python
class SkillRegistry:
    def register(self, name: str, func: Callable, schema: dict) -> None:
        """Eagerly register a skill (used by @skill decorator)."""
        
    def register_lazy(self, name: str, module_path: str, schema: dict) -> None:
        """Register schema only — implementation loaded on first call."""
        
    def execute(self, name: str, args: dict | None, ctx: Any = None) -> Any:
        """Execute a skill by name, injecting ctx if the function accepts it."""
        
    def reload_skill(self, name: str) -> None:
        """Hot-reload a skill: reimport its module and re-register."""
        
    def unregister(self, name: str) -> None:
        """Remove a skill by name."""
        
    def tool_schemas(self) -> list[dict]:
        """Return OpenAI-format tool schemas for all registered skills."""
        
    def list_skills(self) -> list[dict]:
        """Return skill name + description pairs."""
```

### 64 Registered Skills (Auto-Discovered)

**Location:** `biobank_agent/skills/*.py`

#### Analysis Skills (17)
```
prevalence — disease prevalence calculation
cohort — case/control cohort construction
cohort_summary — cohort demographics
biomarker_dist — biomarker distribution
correlation — feature correlation matrix
train_model — XGBoost/LightGBM/CatBoost model training
evaluate_model — cross-validated performance metrics
feature_importance — SHAP/permutation importance
calibration — probability calibration curves
survival — Kaplan-Meier survival analysis
phewas — phenome-wide association study
comorbidity — comorbidity network analysis
min_sample — statistical power calculations
data_query — arbitrary SQL-like queries (DuckDB)
bank_data_probe — check data availability
bank_data_readiness — audit biobank readiness
embedding — document/sequence embeddings
```

#### Discovery Skills (7)
```
predict — predictive model inference
discover — automated biomarker discovery pipeline
gwas_proxy — GWAS-style summary statistics
genetic_target_hypothesis — rare-variant burden ranking
target_annotation_context — Open Targets, UniProt, GTEx, ClinicalTrials
target_enrichment — local GMT gene set enrichment
smart_plot — Nature/ICML/NEJM/Lancet style plotting
```

#### Research Skills (7)
```
web_search — DuckDuckGo/Brave/Serper web search
web_fetch — fetch & parse web content (HTML→Markdown)
read_pdf — extract text/images from PDFs (PyMuPDF)
fetch_paper — arXiv/bioRxiv paper download
read_paper — full-text paper parsing
deep_research — multi-turn literature synthesis
nature_writer — Nature-style manuscript sections
```

#### Ideation Skills (2)
```
brainstorm — hypothesis generation
critical_thinking — structured critique
```

#### Self-Evolution Skills (9)
```
create_skill — generate new skills (Instructor-backed)
record_macro — save workflows as pipelines
replay_pipeline — re-run saved pipelines
track_error — log failure pattern
error_suggestions — suggest error fixes
suggest_error_fix — automated remediation
environment_repair — fix environment issues
project_doc — read project documentation
external_agents — Codex/Claude Code status & calls
```

#### Report/Governance Skills (15+)
```
report — generate technical/Nature/paper formats
statistical_review — verify statistical validity
safety_check — detect causal language, disclosure issues
paper_replication_compare — compare with published findings
phenotype_harmonize — cross-biobank code harmonization
world_model_audit — verify consistency of world model
trajectory_tokenize — token accounting for long traces
```

### Custom Skills Discovery

**Workflow:**

1. **Directory Scan**: `biobank_agent/registry.py:discover_custom_skills()`
   ```python
   custom_dir = settings.custom_skills_dir  # ./custom_skills
   # Finds all *.py files with @skill decorators
   ```

2. **Hot-Reload on Load**:
   ```bash
   ./custom_skills/
   ├── my_skill.py          # @skill(name="my_analysis", ...)
   ├── another_tool.py      # @skill(name="batch_process", ...)
   └── utilities.py         # Non-decorated helper (ignored)
   ```

3. **In-Session Activation**:
   - Skill loads automatically when agent starts
   - Or manually reloaded: `registry.reload_skill("my_analysis")`

4. **Context Injection**:
   ```python
   @skill(name="my_analysis", parameters={...})
   def my_analysis(input_field: str, *, ctx):
       # ctx has access to:
       #  - ctx.agent — main agent instance
       #  - ctx.llm — LLM client
       #  - ctx.data — DataManager (parquet loader)
       #  - ctx.catalog — FieldCatalog (field metadata)
       #  - ctx.config — Settings (config)
       return {"result": ...}
   ```

---

## 5. Multi-Agent & Orchestration Patterns

### Multi-Model Orchestrator

**File:** `biobank_agent/orchestrator.py`

**Architecture:**
- **Route by Complexity**: `classify_complexity()` — SIMPLE, MEDIUM, HARD
- **Model Pool**: User-configured list of frontier models (default: Deepseek, Kimi, GLM)
- **Debate Mechanism**: Red/blue teams with anonymous voting (MAS-v2 style)
- **Subagent Roles**:
  - `EXPLORER` — read-only search (file:line references)
  - `WORKER` — one-shot execution (narrow validation test)
  - `VERIFIER` — adversarial review (PASS/FAIL/PARTIAL verdict)

**Multi-Agent Config** (`.env`):
```bash
MULTI_MODEL_ENABLED=true
AUTO_DISCOVER_MODELS=true
PREFERRED_MULTI_MODELS=deepseek/deepseek-v4-pro,moonshotai/kimi-k2.6,z-ai/glm-5.1
```

### Complexity Classification

```python
def classify_complexity(query: str) -> Strategy:
    """Classify task as SIMPLE, MEDIUM, or HARD."""
    # Heuristics:
    # - Keywords: "plan", "review", "complex" → HARD
    # - Biomarker discovery, paper replication → MEDIUM
    # - Single-tool queries → SIMPLE
```

### ReAct Agent Loop

**File:** `biobank_agent/agent.py`

**Loop Structure:**
```
User Query
    ↓
[Check Session State]
    ↓
[Classify Complexity]
    ↓
[Route to MultiModelOrchestrator if HARD]
    ↓
[LLM + Tool Calling Loop]
    ├─ Generate response + tool calls
    ├─ Execute skills (registry.execute)
    ├─ Verify outputs (VerdictEngine)
    ├─ Check guardrails (DelegationGuardrails, NLI causal check)
    └─ Loop up to max_tool_rounds
    ↓
[Reflexion Recovery if Failure]
    ↓
[Generate Final Report]
    ↓
[Disclosure/Safety Filtering]
    ↓
User Receives Output
```

### Reflexion Engine

**File:** `biobank_agent/reflexion.py`

**Capabilities:**
- **Failure Detection**: Track failed skill calls
- **Root Cause Analysis**: Suggest parameter mutations
  - Out of memory (OOM) → swap to "lite" algorithm
  - Cohort too small → insert prerequisite cohort call
  - Numeric arg overflow → halve argument
- **Automatic Retry**: Up to `plan_repair_budget_per_step` attempts

---

## 6. MCP (Model Context Protocol) Integration

### MCP Architecture

**File:** `biobank_agent/mcp/`

**Components:**
- `graphpop_client.py` — Neo4j graph queries (21 tools for population genomics)
- `/mcp-list` — discover configured servers
- `/mcp-start` — boot STDIO servers
- `/mcp-health` — probe server health
- `/mcp-call` — direct tool invocation
- `/mcp-stop` — shutdown servers

### MCP Configuration (Planned for v3.0)

**Status (RC1):**
- ✅ STDIO JSON-RPC transport implemented & tested
- ⚠️ HTTP/SSE transport protocol-implemented, awaiting production server (v3.1)
- ⚠️ Real MCP compatibility matrix (criterion 14) — targeted for rc2

**Future Roadmap:**
- GitHub MCP server (filesystem + workflow automation)
- Slack MCP server (data sharing)
- Custom institutional MCP servers

---

## 7. Bank Adapter Pattern (Multi-Biobank Support)

### Bank Configuration System

**Location:** `biobank_agent/banks/`

**Supported Banks:**

| Bank | Config | Subject ID | Diagnoses | Status |
|------|--------|------------|-----------|--------|
| **UKB** | `ukb.yaml` | `eid` | `diag_icd10` | ✅ Full support |
| **CKB** | `ckb.yaml` | `study_id` | Native → ICD-10 | ⚠️ Config ready, needs data |
| **HPP** | `hpp.yaml` | `participant_id` | ICD-9 → ICD-10 | ⚠️ Config ready, credential-gated |
| **RAP** | (dynamic) | (dialect-aware) | SparkSQL | ⚠️ v3.1 milestone |

### Bank Adapter Interface

```python
# biobank_agent/banks/adapter.py
class BankAdapter:
    def __init__(self, bank_id: str, config: Path):
        self.bank_id = bank_id
        self.subject_id_col = config.subject_id_col
        self.diagnoses_code_col = config.diagnoses_code_col
        
    def build_cohort(self, icd_prefix: str, case_control: bool = True):
        """Construct case/control cohort using bank-specific SQL."""
        
    def query_biomarkers(self, field_ids: list[str]):
        """Fetch biomarker data (Parquet, CSV, SparkSQL)."""
        
    def harmonize_codes(self, codes: list[str], from_coding: str, to_coding: str):
        """Cross-biobank code translation (ICD-9 → ICD-10, etc.)."""
```

### Configuration Overrides in `.env`

```bash
# Override biobank identity
BANK_ID=ckb
BIOBANK_NAME=China Kadoorie Biobank
BIOBANK_ABBREVIATION=CKB
SUBJECT_ID_COL=study_id
DIAGNOSES_CODE_COL=diag_icd10_native

# Point to CKB data
DATA_DIR=/path/to/ckb_parquet
RAW_DIR=/path/to/ckb_raw_csv
```

---

## 8. Testing Infrastructure

### Test Suite

**Total Tests**: 1,148 passing (as of v3.0-rc1)  
**Test Count**: 94 test files across 2 main directories

**Structure:**
```
tests/
├── core/                    # Core runtime tests (50+ files)
│   ├── test_bank_data_path.py     — multi-bank adapter
│   ├── test_compaction.py         — context window compression
│   ├── test_events.py             — event bus & telemetry
│   ├── test_m3_domain.py          — biobank domain logic
│   ├── test_m4_evolution.py       — reflexion & tool learning
│   └── ...
├── eval/                    # Behavioral evaluation suites
│   ├── behavioral/
│   │   ├── always_passes.yaml     — regression suite (critical)
│   │   └── usually_passes.yaml    — soft acceptance gates
│   └── scheduled/                 — GitHub Actions weekly runs
└── [individual test_*.py files]   — 40+ skill/unit tests
```

### Test Markers

```python
@pytest.mark.integration    # Requires real LLM API (skipped locally)
@pytest.mark.usually        # Soft behavioral evaluations (manual/scheduled)
```

### Eval Suites

```bash
biobank eval --suite skill_schemas --mode baseline
    # Verify all 64 skill schemas are valid

biobank eval --suite report_20_case --mode baseline
    # 20 deterministic synthetic report cases (fast regression)

biobank eval --suite live_ukb_report_20 --mode baseline
    # 20 live UK Biobank workflow probes (requires local UKB data)

biobank eval --suite agent_report_workflow --mode baseline
    # Literature grounding, cohort execution, guardrails, report generation

biobank eval --suite v3_completion --run-dir <run>
    # Maps full v3 objective to evidence (17 criteria, 12 passing, 5 blocked)
```

### GitHub Actions CI/CD

**File:** `.github/workflows/biobank-scheduled-eval.yml`

**Schedule:**
- **Trigger**: Monday 3am UTC (weekly)
- **Manual override**: GitHub Actions UI
- **Python**: 3.11
- **Artifacts**: `reports/eval/scheduled/`

**Workflow:**
```yaml
- Checkout repo
- Setup Python 3.11
- Install package with dev dependencies
- Run scheduled quality gates:
    biobank_agent.eval.scheduled
    --behavioral-policy all
    --output-dir reports/eval/scheduled
- Upload eval artifacts
```

**Test Count**: 1,148 passed, 11 skipped

---

## 9. Deployment & Infrastructure

### Platform Support

| Platform | Support | Notes |
|----------|---------|-------|
| **macOS ARM (Apple Silicon)** | ✅ Full | CPU mode, tested |
| **Linux x86_64** | ✅ Full | CPU mode, tested |
| **Linux + NVIDIA GPU** | ✅ Full | CUDA optional, auto-fallback |

**GPU-Dependent Features:**
- SHAP deep explainer (auto-fallback to CPU)
- UMAP with RAPIDS (auto-fallback to sklearn)

### Dependency Management

**Build System**: setuptools + wheel

```bash
python -m pip install -e ".[all]"              # Full install
python -m pip install -e ".[gpu,literature]"   # Custom subset
```

### Data Persistence

**Directory Structure:**
```
./data/                     # Parquet files (fast path)
./raw/                      # Raw CSV fallback (optional)
./milton_data/              # UKB Milton subset (legacy alias)
./reports/                  # Generated analysis outputs (gitignored)
./plans/                    # Agent plan execution traces (gitignored)
./custom_skills/            # Custom skill plugins (gitignored)
~/.biobank_agent/           # Long-term memory & telemetry (configurable)
```

### Telemetry & Observability

**Local Telemetry** (Privacy-Preserving):
```python
telemetry_enabled: bool = True
telemetry_jsonl: str = ""  # JSONL file path

# Records: event type, skill, status, timing only
# Never records: query text, tool args, outputs, identifiers
```

**OpenTelemetry (Optional)**:
```python
otel_enabled: bool = False
otel_exporter: str = "none"  # none | console | otlp
otel_endpoint: str = ""       # e.g. http://localhost:4318/v1/traces
otel_policy: str = "local"    # local | production
```

---

## 10. External Agent Integration (Codex & Claude Code)

### Plugin Architecture

**Codex Plugin:**
- **Location**: `plugins/biobank-agent/`
- **Marketplace**: `.agents/plugins/marketplace.json`

**Claude Code Plugin:**
- **Location**: `plugins/biobank-agent-claude/`
- **Marketplace**: `.claude-plugin/marketplace.json`

### Skills for External Agents

**Built-in Skills:**
- `external_agent_status` — check Codex/Claude CLI availability
- `codex_plan` — ask Codex for read-only planning
- `codex_check_execution` — ask Codex to review execution
- `claude_plan` — ask Claude Code for planning
- `claude_check_execution` — ask Claude Code to review execution

**Default Behavior:**
- Codex is **default** review gate for eval loops
- Claude Code is **opt-in** secondary reviewer
- Both run in **read-only mode** by default (mocked in tests)

**Setup (Claude Code):**
```bash
claude auth status
claude auth login
```

**Verify Bridge from BioBank Agent:**
```
/external-agents
/codex-plan Draft a guarded report workflow
/codex-check Check report guardrails
/claude-plan Draft a guarded report workflow
/claude-check Check report guardrails
```

---

## 11. Data Access & Query Layer

### Data Stack

**File:** `biobank_agent/data/`

```
data/
├── loader.py              # Unified query layer (Parquet + CSV fallback)
├── catalog.py             # Field catalog (11,821 fields, 410 categories)
├── cohort.py              # Case/control cohort builder
├── features.py            # Biomarker group definitions
└── parquet_builder.py     # Batch CSV → Parquet rebuild
```

**Query Backend:**
- **Primary**: DuckDB (in-process SQL engine)
- **Fallback**: Pandas (if CSV only)
- **Optional**: PySpark (RAP path, v3.1)

**Cohort Builder:**
```python
def build_cohort(
    icd_prefix: str,
    sex: str = "both",
    age_range: tuple[int, int] = (37, 73),
    case_control: bool = True,
) -> CohortRecord:
    """Build case/control cohort for diagnosis code."""
    # Returns: case_n, control_n, feature names, cohort IDs
```

### Field Catalog

**Size**: 11,821 fields (UKB), 410 categories

**Categories**:
- Biomarkers (haematological, metabolic, anthropometric, renal, etc.)
- Lifestyle (alcohol, smoking, diet, exercise)
- Demographics (age, sex, ethnicity, education)
- Medical history (diagnoses, medications, procedures)

**Access**:
```python
catalog = FieldCatalog()
field_info = catalog.get_field(20002)  # e.g., ICD-10 diagnoses
field_info.category      # "Diagnoses"
field_info.description   # "Non-cancer illness code, main"
```

---

## 12. Verification & Guardrails

### Multi-Layer Verification Pipeline

**File:** `biobank_agent/verdict.py`

**Layers:**

| Layer | Engine | Purpose |
|-------|--------|---------|
| **Formal** | Z3 SMT solver | Constraint satisfaction (optional) |
| **Numeric** | `NumericRangeChecker` | UKB bounds (502,411 max, age 37–73) |
| **URL/DOI** | `URLDOIResolver` | Reference accessibility check |
| **Entailment** | LLM NLI | Claim ↔ evidence consistency |
| **Verdict** | `VerdictEngine` | Final PASS / FAIL / PARTIAL |

### Guardrails

**File:** `biobank_agent/guardrails.py`

**Safety Checks:**
- **Disclosure Control**: Block undisclosed cell counts (<5)
- **Causal Language Filter**: Flag unqualified causal claims
- **NLI Entailment**: Verify claim-evidence consistency
- **Reference Validation**: Check DOI/URL accessibility

**Guardrail Example:**
```python
if _unqualified_causal_pattern.search(line):
    if not _causal_qualifier_pattern.search(line):
        # Flag: unqualified causal claim
        # Inject: "This is an association, not proven causation"
```

---

## 13. Memory System (8-Tier Architecture)

**File:** `biobank_agent/memory.py`

| Tier | Scope | Persistence | TTL |
|------|-------|-------------|-----|
| **Short-term** | Current conversation | In-memory | Session |
| **Mid-term** | Analysis records (exact metrics) | Session state | Session |
| **Long-term** | Model configs, pipelines | `memory.json` | Permanent |
| **Error Catalog** | Error patterns + fixes | `memory.json` | Permanent |
| **Domain** | Biomedical findings | `domain.md` | Permanent |
| **User Preferences** | Researcher settings | `user.md` | Permanent |
| **Episodic** | Cross-session recall (BM25) | `sessions.db` | Permanent |
| **Action Graph** | Claim ↔ Evidence knowledge graph | `action_graph.db` | Permanent |

### Evidence Lattice

**File:** `biobank_agent/evidence.py`

```python
class EvidenceNode:
    """Claim-level provenance tracking."""
    claim_id: str
    text: str
    evidence: list[EvidenceItem]  # supporting facts
    confidence: float  # 0.0–1.0
    tags: list[str]    # ["genetic", "statistical", "literature"]
```

---

## 14. Plan Mode & Structured Execution

**File:** `biobank_agent/planner.py` + `biobank_agent/plan_executor.py`

### Plan Workflow

```
INTAKE
├─ Parse user goal
├─ Disambiguate domain (biobank analysis, literature, etc.)
└─ Check data readiness

ALIGNMENT
├─ Compile natural language → StudySpec (Pydantic schema)
├─ Verify skill dependencies + execution order
├─ Check resource constraints (max_tool_rounds, context_window)
└─ Get user approval (if --interactive)

EXECUTION
├─ Execute steps sequentially (with inter-step context)
├─ Reflexion recovery on failure (up to plan_repair_budget_per_step)
├─ Record execution trace + skill outputs
└─ Publish progress to event bus

DONE
├─ Compile final report (technical or IMRaD)
├─ Validate report artifacts (dual format, figures)
└─ Return reproducible session snapshot
```

### Plan Configuration

```python
plan_repair_budget_per_step: int = 3
plan_repair_budget_total: int = 8
plan_external_council_enabled: bool = True
plan_external_council_policy: str = "requested"  # requested | always | never
plan_external_council_timeout_s: int = 180
plan_external_council_agents: str = "codex,claude,gemini"
plan_clarification_enabled: bool = True
plan_clarification_policy: str = "critical_only"
```

---

## 15. Version History & Evolution (CHANGELOG)

### Major Milestones

**v3.0.0-rc1** (2026-05-12)
- v3 Foundation Runtime (async event-stream, Textual TUI scaffold)
- Plan/Report Hard Gates (schema-valid skills, failed-skill pause)
- MCP Manager (STDIO JSON-RPC, tool registry, reconnect logic)
- Bank Adapter Main Path (HPP/CKB code paths, synthetic fixtures)
- Paper Replication Scaffold (StudySpec → plan → dual report)
- v3 Completion Audit (13 of 17 criteria pass, 5 `BLOCKED_EXTERNAL`)

**v2.4.0** (2026-05-11)
- MCP Manager with STDIO support
- Bank Adapter foundations
- Paper Replication framework
- OpenTelemetry tracing bridge
- Verification pipeline (Z3, NLI, URL/DOI)

**v2.3.0** (2026-05-08)
- Target Annotation Context skill (Open Targets, UniProt, GTEx)
- Target Enrichment skill (GMT, GSEApy)

**v2.2.0** (2026-05-08)
- Genetic Target Hypothesis skill (rare-variant burden ranking)
- Therapeutic direction + pathway convergence scoring

**v2.1.0** (2026-05-08)
- Schema-Gated Execution (StudySpec compiler)
- Verifier Mesh (multi-strategy verification)
- Evidence Lattice (claim provenance)
- Formal Verification (Z3 SMT)

**v2.0.0** (2026-04-22)
- Scientific Discovery Pipeline (discover, predict, gwas_proxy)
- Literature Research (web_search, read_pdf, read_paper, deep_research)
- Scientific Writing (nature_writer)
- Plan Mode (INTAKE → ALIGNMENT → EXECUTION → DONE)
- Report Dual Format (technical + IMRaD paper)
- 28 skills → 58 skills

**v1.0.0** (2026-04-21)
- Initial release: 28 skills, DuckDB data layer, ReAct loop
- Nature-style plotting, 4-tier memory, 164 tests

---

## 16. Portability & Extensibility Summary

### Highly Portable ✅

**Across Biobanks:**
- Bank adapter pattern supports UKB (primary), CKB (ready), HPP (ready), FinnGen (planning)
- Column names, diagnosis codes, subject IDs overridable via `.env`
- Multi-cohort workflows via bank routing

**Across Platforms:**
- Python 3.10+
- macOS ARM, Linux x86_64 (both CPU & GPU)
- GPU features auto-fallback to CPU (no hard dependencies)

**Across LLM Providers:**
- OpenAI-compatible API (openai-sdk 1.50+)
- Model switching via `/model <name>` or config
- Multi-model orchestration support (Deepseek, Kimi, GLM, etc.)

### Highly Extensible ✅

**Skills:**
- Decorator-based registration (`@skill`)
- Hot-reload on load (`custom_skills/` auto-discovery)
- Context injection (`ctx.agent`, `ctx.llm`, `ctx.data`)
- 64 existing skills as templates

**Orchestration:**
- Multi-agent routing (`MultiModelOrchestrator`)
- Reflexion recovery with parameter mutation
- Custom evaluation suites (pytest markers)

**Data Access:**
- Pluggable backends (Parquet, CSV, PySpark/RAP)
- Bank-agnostic query layer
- Field catalog extensible

**MCP Integration:**
- STDIO JSON-RPC implementation ready
- GraphPop (21 graph tools) included
- HTTP/SSE transport planned (v3.1)

**Reports:**
- Dual format output (technical, IMRaD)
- SVG+PDF figures with style presets
- Filterable for disclosure/safety

---

## 17. Known Limitations & v3 Roadmap

### v3.0-rc1 Blocked Criteria (5 of 17)

| Criterion | Status | Target | Notes |
|-----------|--------|--------|-------|
| 13a | PASS | ✅ Local UKB readiness artifact required |
| 13b | BLOCKED | v3.1 | HPP/CKB/RAP credential-gated |
| 14 | BLOCKED | rc2 | Real MCP compatibility (filesystem + GitHub STDIO) |
| 15 | BLOCKED | rc2 | Remote CI scheduled eval (`gh workflow run`) |
| 16 | BLOCKED | rc2 | Credentialed high-risk PR path (owned repo) |

### Future Work (v3.1–3.2)

- **HTTP/SSE MCP Transport**: Production server + cloud deployment
- **RAP Path**: SparkSQL adapter for UK Biobank secure analytics platform
- **Paper Replication**: 1091-disease MILTON AUC distribution verification
- **Codex Bridge Hardening**: DAAO-inspired task routing refinement
- **CLI Package Decomposition** (`cli_legacy.py` → modular `cli/` package)

---

## 18. Repository Structure (Comprehensive)

```
UKB_agent/
├── README.md                    # Main project readme
├── CHANGELOG.md                 # Version history
├── CLAUDE.md                    # Agent rules (root cleanliness, testing)
├── LICENSE                      # MIT
├── pyproject.toml              # Build config, dependencies, entry points
├── .env / .env.example         # Configuration templates
├── .gitignore                  # Excludes reports/, plans/, custom_skills/
│
├── biobank_agent/              # Main Python package
│   ├── __init__.py
│   ├── agent.py                # ReAct loop orchestration
│   ├── config.py               # Pydantic settings (overridable for other banks)
│   ├── registry.py             # Decorator-based skill registry
│   ├── llm.py                  # OpenAI-compatible LLM client (retry logic)
│   ├── state.py                # Session state + token tracking
│   ├── memory.py               # 8-tier persistent memory system
│   ├── orchestrator.py         # Multi-model orchestration + debate
│   ├── complexity.py           # Task complexity classification
│   ├── reflexion.py            # Failure recovery engine
│   ├── verdict.py              # Multi-layer verification pipeline
│   ├── verifier_mesh.py        # URL/DOI, numeric bounds, NLI checks
│   ├── guardrails.py           # Safety checks (disclosure, causal language)
│   ├── evidence.py             # Claim-evidence lattice
│   ├── study_spec.py           # Schema-gated execution (Pydantic)
│   ├── planner.py              # Plan mode (INTAKE → ALIGNMENT → EXECUTION)
│   ├── plan_executor.py        # Plan step executor + reflexion
│   ├── tool_learner.py         # Error pattern mining + skill generation
│   ├── retrieval.py            # Agentic RAG (BM25, embedding)
│   ├── external_agents.py      # Codex/Claude Code CLI bridges
│   ├── reproducibility.py      # SHA-256 checkpoints, replay
│   ├── world_model.py          # World state consistency checks
│   ├── difficulty.py           # DAAO-inspired task routing
│   ├── constants.py            # UKB domain constants (single source of truth)
│   ├── validators.py           # UKB-specific validation (field ranges, ICD-10)
│   ├── disclosure.py           # Disclosure control (cell count rounding, PII stripping)
│   │
│   ├── cli/                    # CLI v3 modular package (v2 compat wrapper)
│   │   ├── __init__.py         # Routes → cli_legacy (compatibility)
│   │   ├── __main__.py
│   │   ├── commands/
│   │   │   ├── base.py         # CommandContext, @action decorator, RegisteredCommand
│   │   │   ├── mcp.py          # /mcp-* commands
│   │   │   ├── plan.py         # /plan command
│   │   │   ├── memory.py       # /memory command
│   │   │   ├── session.py      # /clear, /cost, /export, /status, /history
│   │   │   ├── registry.py     # /skills, /models
│   │   │   └── reproducibility.py  # /record, /pipelines, /errors
│   │   ├── tui/                # Terminal UI (Textual v3 scaffold)
│   │   │   ├── main.py
│   │   │   ├── main_screen.py
│   │   │   ├── tool_status.py
│   │   │   ├── progress_panel.py
│   │   │   └── ...
│   │   └── live/
│   │       └── streaming_renderer.py
│   ├── cli_legacy.py           # Current v2.4 CLI (compatibility mode)
│   │
│   ├── data/                   # Data access layer
│   │   ├── loader.py           # Unified query (Parquet + CSV)
│   │   ├── catalog.py          # Field catalog (11,821 fields, 410 categories)
│   │   ├── cohort.py           # Case/control builder
│   │   ├── features.py         # Biomarker groups
│   │   └── parquet_builder.py  # CSV → Parquet batch rebuild
│   │
│   ├── banks/                  # Multi-biobank adapter layer
│   │   ├── __init__.py
│   │   ├── adapter.py          # BankAdapter interface
│   │   ├── registry.py         # Bank registration
│   │   └── configs/
│   │       ├── ukb.yaml        # UK Biobank (primary)
│   │       ├── ckb.yaml        # China Kadoorie Biobank
│   │       └── hpp.yaml        # Health Professionals Participant
│   │
│   ├── skills/                 # 64 registered skills (auto-discovered)
│   │   ├── __init__.py
│   │   ├── prevalence.py       # Disease prevalence
│   │   ├── cohort.py           # Cohort building
│   │   ├── train_model.py      # XGBoost/LightGBM/CatBoost
│   │   ├── genetic_target_hypothesis.py  # Rare-variant ranking
│   │   ├── web_search.py       # DuckDuckGo/Brave/Serper
│   │   ├── read_paper.py       # Full-text paper parsing
│   │   ├── report.py           # Report generation
│   │   ├── external_agents.py  # Codex/Claude CLI
│   │   ├── create_skill.py     # Auto skill generation
│   │   ├── record_macro.py     # Pipeline recording
│   │   └── ... (58 total)
│   │
│   ├── mcp/                    # MCP integration layer
│   │   ├── __init__.py
│   │   └── graphpop_client.py  # Neo4j graph queries (21 tools)
│   │
│   ├── core/                   # v3 foundation runtime
│   │   ├── __init__.py
│   │   ├── events.py           # Event bus + PII scrubbing
│   │   ├── runtime.py          # AsyncAgent event-stream runtime
│   │   ├── telemetry.py        # Privacy-preserving JSONL logging
│   │   ├── compaction.py       # Context window compression
│   │   ├── memory/             # Tiered memory backends
│   │   ├── evolution/          # Reflexion + tool learning
│   │   ├── planning/           # Plan executor + phase tracking
│   │   ├── safety/             # NLI, disclosure, causal checks
│   │   ├── orchestration/      # Multi-model routing
│   │   ├── llm/                # LLM client abstractions
│   │   └── tools/              # Tool execution harness
│   │
│   ├── utils/                  # Shared utilities
│   │   ├── plotting.py         # Nature/ICML SVG+PDF (300 DPI)
│   │   ├── report_templates.py # Section templates, CSS, LaTeX
│   │   ├── stats.py            # Mann-Whitney, chi², FDR, log-rank
│   │   ├── icd10.py            # ICD-10 code → name lookup
│   │   └── ...
│   │
│   ├── domain/                 # Domain-specific modules
│   ├── interfaces/             # Plugin interfaces
│   ├── extensions/             # Extension points
│   ├── eval/                   # Evaluation harness
│   │   ├── scheduled.py        # Scheduled eval entry point
│   │   ├── v3_completion.py    # v3 audit (17 criteria)
│   │   └── behavioral/         # Behavioral test suites
│   └── sdk/                    # SDK for programmatic use
│
├── tests/                      # Test suite (1,148 passing)
│   ├── core/                   # Runtime tests (50+ files)
│   │   ├── test_bank_data_path.py
│   │   ├── test_compaction.py
│   │   ├── test_events.py
│   │   ├── test_m3_domain.py
│   │   ├── test_m4_evolution.py
│   │   └── ...
│   ├── eval/                   # Evaluation suites
│   │   ├── behavioral/
│   │   │   ├── always_passes.yaml
│   │   │   └── usually_passes.yaml
│   │   └── scheduled/
│   └── test_*.py               # 40+ skill/unit tests
│
├── docs/                       # Documentation (tracked in git)
│   ├── README.md              # Docs index
│   ├── architecture/          # Architecture & design docs
│   │   ├── V3.md             # v3 scope & plan
│   │   ├── V3_RELEASE_NOTES.md
│   │   ├── COMPATIBILITY.md  # Tier 1/2 support matrix
│   │   └── ...
│   ├── data/                 # UK Biobank data reference
│   ├── guides/               # User & skill guides
│   ├── examples/             # Example workflows (SDK, MCP, etc.)
│   ├── deep_research/        # Research notes
│   └── related_works/        # Related paper citations
│
├── plugins/                   # Repo-local plugin bundles
│   ├── biobank-agent/        # Codex plugin
│   │   └── .codex-plugin/
│   │       └── plugin.json
│   └── biobank-agent-claude/ # Claude Code plugin
│       └── .claude-plugin/
│           └── plugin.json
│
├── .agents/                   # Agent marketplace metadata
│   └── plugins/
│       └── marketplace.json
│
├── .claude-plugin/            # Claude Code plugin registry
│   └── marketplace.json
│
├── .github/
│   └── workflows/
│       └── biobank-scheduled-eval.yml  # Weekly CI/CD
│
├── custom_skills/            # User-defined skill plugins (gitignored)
│   └── [auto-discovered on load]
│
├── reports/                   # Generated outputs (gitignored)
│   ├── eval/                 # Evaluation artifacts
│   ├── checkpoints/          # Plan checkpoints
│   ├── external_agents/      # External review results
│   └── [timestamps]/         # Session reports
│
├── plans/                     # Agent plans (gitignored)
│   └── [plan files with provenance]
│
└── papers/                    # Research papers / fixtures
    └── [paper PDFs & metadata]
```

---

## Conclusion

**BioBank Agent** is a **production-ready, highly extensible scientific discovery system** with:

- ✅ **64 registered skills** with decorator-based registration & hot-reload
- ✅ **Multi-biobank support** (UKB primary, CKB/HPP/FinnGen-ready)
- ✅ **Multi-model orchestration** with complexity-based routing & debate
- ✅ **Rich CLI** with 22 slash commands + v3 Textual TUI scaffold
- ✅ **Comprehensive testing** (1,148 passing, scheduled eval suite)
- ✅ **MCP integration** (STDIO ready, HTTP/SSE planned)
- ✅ **Deployment-ready** (GitHub Actions CI/CD, telemetry, reproducibility)
- ✅ **Portable** (Python 3.10+, macOS/Linux, CPU/GPU auto-fallback)

**Key Extensibility Patterns:**
1. **Skills**: `@skill` decorator, hot-reload, context injection
2. **Banks**: Bank adapter interface + config overrides
3. **Data**: Pluggable backends (Parquet, CSV, PySpark)
4. **Models**: Multi-model pool with complexity routing
5. **MCP**: STDIO JSON-RPC, custom server support
6. **Safety**: Guardrails, disclosure control, NLI verification

The architecture is **modular**, **well-documented**, and **ready for institutional deployment** across multiple biobanks with minimal configuration changes.
