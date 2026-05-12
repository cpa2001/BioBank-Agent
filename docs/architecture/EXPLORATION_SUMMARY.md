# BioBank Agent: Executive Analysis Summary

## Quick Overview

**BioBank Agent** is a production-grade v3.0-rc1 autonomous scientific discovery system combining:
- **64 registered skills** (decorator-based registry, hot-reload)
- **Multi-model orchestration** (complexity routing, red/blue debate)
- **ReAct agent loop** with native tool_use (OpenAI-compatible)
- **Multi-biobank support** (UKB primary, CKB/HPP/FinnGen-ready via bank adapters)
- **Comprehensive CLI** (22 slash commands, v3 Textual TUI scaffold)
- **Production infrastructure** (GitHub Actions CI/CD, telemetry, reproducibility)

---

## Key Findings

### ✅ Highly Extensible

1. **Skills** — Decorator-based system with hot-reload
   ```python
   @skill(name="my_analysis", parameters={...})
   def my_analysis(input_field: str, *, ctx):
       # ctx.agent, ctx.llm, ctx.data accessible
   ```
   - Auto-discovery from `./custom_skills/`
   - 64 existing skills as templates
   - Context injection of agent, LLM, data, catalog

2. **Bank Adapters** — Configure for any biobank
   - UKB (primary), CKB (ready), HPP (ready), FinnGen (planning)
   - Override via `.env`: `BANK_ID`, `SUBJECT_ID_COL`, `DIAGNOSES_CODE_COL`
   - Automatic code harmonization (ICD-9 → ICD-10)

3. **Multi-Agent Routing** — Auto-upgrade complex tasks
   - Complexity classification (SIMPLE, MEDIUM, HARD)
   - Model pool: Deepseek, Kimi, GLM (configurable)
   - Red/blue debate with anonymous voting

4. **MCP Integration** — Extensible external tool layer
   - STDIO JSON-RPC ready (GraphPop: 21 population-genomics tools)
   - HTTP/SSE transport planned (v3.1)
   - Custom MCP servers supported

5. **Data Layer** — Pluggable backends
   - Primary: DuckDB (in-process SQL)
   - Fallback: Pandas/CSV
   - Optional: PySpark/SparkSQL (RAP path, v3.1)

### ✅ Production Deployment Ready

- **CI/CD**: GitHub Actions weekly scheduled eval
- **Testing**: 1,148 passing tests (94 files), pytest + eval harness
- **Telemetry**: Privacy-preserving local JSONL + optional OpenTelemetry
- **Guardrails**: Disclosure control, causal language filtering, NLI entailment
- **Verification**: Multi-layer (formal Z3, numeric bounds, URL/DOI, entailment)

### ✅ Portable Architecture

- **Platforms**: macOS ARM (native), Linux x86_64 (native), Linux + NVIDIA GPU (optional)
- **Python**: 3.10+ only
- **LLM**: Any OpenAI-compatible API (relay-friendly)
- **Installation**: `pip install -e ".[all]"` or custom dependency subset

### ⚠️ In v3 Transition

- **Status**: RC1 (13 of 17 completion criteria pass)
- **Blocked**: 5 criteria mostly credential-dependent (HPP/CKB/RAP data access)
- **Timeline**: v3.0 target adds MCP STDIO, remote CI, RAP path; rc2 targeted for May 2026

---

## Architecture Highlights

### 1. CLI Structure (22 Commands)

| Category | Commands |
|----------|----------|
| Help | `/help`, `/skills`, `/models-available`, `/external-agents` |
| Analysis | `/plan`, `/think`, `/model`, `/compact` |
| Session | `/clear`, `/cost`, `/status`, `/history` |
| Output | `/export`, `/figures`, `/cohorts`, `/models` |
| Pipelines | `/record`, `/pipelines`, `/replay`, `/memory`, `/errors` |
| Review | `/codex-plan`, `/codex-check`, `/claude-plan`, `/claude-check` |
| MCP | `/mcp-list`, `/mcp-start`, `/mcp-health`, `/mcp-call`, `/mcp-stop` |

### 2. Skill Categories (64 Total)

- **Analysis** (17): prevalence, cohort, train_model, survival, phewas, etc.
- **Discovery** (7): predict, discover, gwas_proxy, genetic_target_hypothesis, etc.
- **Research** (7): web_search, read_pdf, fetch_paper, deep_research, etc.
- **Ideation** (2): brainstorm, critical_thinking
- **Self-Evolution** (9): create_skill, record_macro, replay_pipeline, etc.
- **Report/Governance** (15+): report, statistical_review, safety_check, etc.

### 3. Configuration Layers

```
pyproject.toml (dependencies, entry point)
    ↓
.env (runtime config, overrides)
    ↓
Settings (Pydantic v2, backward-compatible aliases)
    ↓
Bank adapters (UKB/CKB/HPP/FinnGen YAML configs)
```

**Key settings:**
- LLM: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`
- Data: `DATA_DIR`, `RAW_DIR`, `FULL_UKB_FEATURE_STORE_DIR`
- Biobank: `BANK_ID`, `BIOBANK_NAME`, `SUBJECT_ID_COL`, `DIAGNOSES_CODE_COL`
- Safety: `disclosure_control_mode`, `plan_external_council_enabled`

### 4. Multi-Agent Orchestration

```
User Query → Classify Complexity → Route if HARD
    ↓
[HARD] → MultiModelOrchestrator
    ├─ Model pool: Deepseek, Kimi, GLM
    ├─ Red/blue debate with voting
    ├─ Evidence linking (claims ↔ evidence)
    └─ Safety checks before finalization
    
[SIMPLE/MEDIUM] → ReAct Agent Loop
    ├─ LLM + tool calling
    ├─ Skill execution (registry.execute)
    ├─ Verification (verdict engine)
    ├─ Reflexion recovery on failure
    └─ Report generation
```

### 5. Memory System (8 Tiers)

| Tier | Persistence | Use |
|------|-------------|-----|
| Short-term | In-memory | Current conversation |
| Mid-term | Session state | Analysis records |
| Long-term | `memory.json` | Model configs, pipelines |
| Error catalog | `memory.json` | Error patterns + fixes |
| Domain | `domain.md` | Biomedical findings |
| User prefs | `user.md` | Researcher settings |
| Episodic | `sessions.db` | Cross-session BM25 recall |
| Action graph | `action_graph.db` | Claim ↔ Evidence lattice |

### 6. Verification Pipeline

```
Skill Output → Formal (Z3) → Numeric (bounds) → URL/DOI
    → NLI (entailment) → Verdict (PASS/FAIL/PARTIAL)
```

---

## Dependency Landscape

**Core** (51 packages):
- Data: duckdb, pyarrow, pandas, numpy
- ML: scikit-learn, xgboost, lightgbm
- Viz: matplotlib, seaborn
- LLM: openai, rich, prompt-toolkit
- Web: httpx, html2text, beautifulsoup4, ddgs
- Analysis: lifelines, scipy, networkx

**Optional Groups** (7):
- `gpu` — CatBoost, SHAP, UMAP
- `verify` — Z3 SMT solver
- `graphpop` — Neo4j
- `research` — inferactively-pymdp
- `literature` — paper-qa
- `structured` — instructor
- `tracing` — OpenTelemetry

---

## Deployment Patterns

### GitHub Actions CI/CD
```yaml
# .github/workflows/biobank-scheduled-eval.yml
Trigger: Monday 3am UTC (weekly) or manual
Steps:
  - Checkout repo
  - Setup Python 3.11
  - Install package + dev deps
  - Run behavioral + evolution gates
  - Upload eval artifacts
```

### Data Directories
```
./data/              # Parquet files (fast path)
./raw/               # CSV fallback
./reports/           # Session outputs (gitignored)
./plans/             # Plan traces (gitignored)
./custom_skills/     # User plugins (gitignored)
~/.biobank_agent/    # Memory + telemetry (configurable)
```

### Testing
- **1,148 tests passing** (94 files)
- **Markers**: `@pytest.mark.integration`, `@pytest.mark.usually`
- **Eval suites**: skill_schemas, report_20_case, live_ukb_report_20, agent_report_workflow, v3_completion

---

## Extension Points (How to Customize)

### 1. Add a New Skill
```bash
# biobank_agent/skills/my_skill.py
from biobank_agent.registry import skill

@skill(name="my_analysis", ...)
def my_analysis(param: str, *, ctx):
    return {"result": ...}

# Or in ./custom_skills/my_skill.py (auto-discovered on load)
```

### 2. Support a New Biobank
```bash
# biobank_agent/banks/configs/mybank.yaml
bank_id: mybank
subject_id_col: participant_id
diagnoses_code_col: diag_icd10

# .env
BANK_ID=mybank
DATA_DIR=/path/to/mybank_data
```

### 3. Integrate an MCP Server
```bash
# Configure in .env or settings
PREFERRED_MULTI_MODELS=...
# Use /mcp-list, /mcp-start, /mcp-call
```

### 4. Add Custom Report Format
```bash
# biobank_agent/utils/report_templates.py
# Extend ReportGenerator.generate() with new format
```

---

## Portability Score

| Dimension | Rating | Evidence |
|-----------|--------|----------|
| **Biobank Portability** | ⭐⭐⭐⭐⭐ | Bank adapter + config overrides (UKB/CKB/HPP/FinnGen) |
| **Platform Portability** | ⭐⭐⭐⭐⭐ | macOS ARM, Linux x86, GPU auto-fallback |
| **LLM Portability** | ⭐⭐⭐⭐⭐ | OpenAI-compatible + multi-model orchestration |
| **Skill Extensibility** | ⭐⭐⭐⭐⭐ | Decorator-based hot-reload + context injection |
| **Data Extensibility** | ⭐⭐⭐⭐ | Pluggable backends (Parquet/CSV/PySpark) |
| **Safety/Governance** | ⭐⭐⭐⭐⭐ | Disclosure control, causal filtering, NLI verification |
| **Deployment Readiness** | ⭐⭐⭐⭐ | CI/CD, telemetry, testing (v3.0 adds MCP STDIO) |

---

## Full Documentation

A comprehensive **1,259-line analysis** has been saved to:
```
docs/architecture/BIOBANK_ARCHITECTURE_ANALYSIS.md
```

This includes:
- Detailed pyproject.toml breakdown
- Complete CLI command reference
- Pydantic Settings structure
- Skill registry implementation details
- Bank adapter patterns
- Multi-agent orchestration flow
- MCP integration roadmap
- Testing infrastructure
- Deployment patterns
- Known limitations & v3 roadmap
- Full repository structure visualization

---

## Conclusion

**BioBank Agent is a highly portable, production-ready, and deeply extensible autonomous discovery system** suitable for:
- ✅ Multi-institutional deployment (UKB, CKB, HPP, FinnGen)
- ✅ Custom skill development (64 existing templates)
- ✅ LLM provider flexibility (OpenAI-compatible APIs)
- ✅ Data backend flexibility (DuckDB, Parquet, CSV, PySpark)
- ✅ Multi-model orchestration (Deepseek, Kimi, GLM, etc.)
- ✅ External tool integration (MCP STDIO ready)

**Key architectural strengths:**
1. Modular decorator-based skill system
2. Configuration-driven biobank routing
3. Multi-layer verification pipeline
4. Comprehensive memory & reproducibility
5. Production-grade testing & telemetry

---

