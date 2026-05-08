<div align="center">

# Biobank Agent

**Autonomous scientific discovery agent for population-scale biobank research**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-1074%20passed-brightgreen.svg)](#testing)

*Natural language interface to large-scale biobank cohorts — from hypothesis to publication-quality report.*

</div>

---

## Overview

Biobank Agent is an LLM-powered scientific discovery system designed for population-scale biobank data analysis. It combines a **ReAct agent loop** with **55 registered analysis, documentation, and review skills** to enable end-to-end research workflows: cohort construction, biomarker discovery, predictive modelling, survival analysis, literature review, cross-agent review, and publication-quality reporting.

**Key capabilities:**
- **Hypothesis-driven discovery** — automated pipelines from cohort building through feature importance to PheWAS
- **Predictive modelling** — XGBoost/LightGBM/CatBoost with cross-validation, calibration, and SHAP explanations
- **Literature integration** — search papers, read PDFs, cross-reference findings with biobank data
- **Publication-quality output** — Nature/ICML-style SVG+PDF figures, dual-format reports (technical & IMRaD)
- **Self-evolution** — learns from errors, records analysis pipelines, generates new skills at runtime
- **Plan mode** — structured multi-step planning with stage gates for complex analyses
- **Native multi-agent routing** — auto-detects complex tasks and coordinates multiple frontier models
- **Project documentation access** — a read-only `project_doc` skill exposes README, data reference, architecture, guide, and plugin Markdown to the agent
- **Local Codex/Claude bridge** — optional `@skill` tools call this workstation's Codex and Claude Code CLIs for independent planning and execution review

Currently validated on **UK Biobank** (502K participants, 4,971 phenotype fields, 6.9M diagnosis records). Architecture supports extension to FinnGen, China Kadoorie Biobank, and other population cohorts.

## Quick Start

```bash
# Install
git clone https://github.com/cpa2001/BioBank-Agent.git
cd BioBank-Agent
pip install -e ".[all]"

# Configure
cp .env.example .env
# Edit .env: set LLM_API_KEY and DATA_DIR

# Run
biobank
```

## Usage Examples

```
biobank> What are the top 20 most common diseases?
biobank> Discover disease-specific biomarkers for Type 2 Diabetes
biobank> Train an XGBoost model to predict E11 and show feature importance
biobank> Show Kaplan-Meier survival curves for acute MI (I21)
biobank> Search for recent CKD biomarker studies and summarize findings
biobank> Read this paper and compare with our cohort data
biobank> /plan Comprehensive cardiovascular risk analysis
biobank> Generate a Nature-quality report for all analyses
```

## Architecture

```
biobank_agent/
├── agent.py                 # ReAct loop with native tool_use
├── planner.py               # Plan mode (INTAKE → ALIGNMENT → EXECUTION → DONE)
├── llm.py                   # OpenAI-compatible client with retry + backoff
├── registry.py              # @skill decorator, auto-discovery, hot-reload
├── config.py                # Pydantic settings
├── state.py                 # Session state + token tracking
├── memory.py                # 8-tier persistent memory
│
├── study_spec.py            # Schema-gated execution (query → typed StudySpec)
├── verdict.py               # Verification engine (PASS/FAIL/PARTIAL)
├── verifier_mesh.py         # URL/DOI, numeric bounds, NLI entailment
├── verification.py          # Z3 SMT formal constraint checking
├── evidence.py              # Claim-evidence lattice with confidence
├── constants.py             # UKB domain constants (single source of truth)
│
├── data/                    # Data access layer (DuckDB + Parquet)
│   ├── loader.py            # Unified query layer (parquet + CSV fallback)
│   ├── catalog.py           # Field catalogue (11,821 fields, 410 categories)
│   ├── cohort.py            # Case/control cohort builder
│   ├── features.py          # Biomarker group definitions
│   └── parquet_builder.py   # Batch CSV → Parquet rebuild
│
├── skills/                  # 55 registered skills (auto-discovered)
│   ├── Analysis (17)        # prevalence, cohort, biomarker_dist, correlation,
│   │                        # train_model, evaluate_model, feature_importance,
│   │                        # calibration, survival, phewas, comorbidity, ...
│   ├── Discovery (4)        # predict, discover, gwas_proxy, smart_plot
│   ├── Research (7)         # web_search, web_fetch, read_pdf, fetch_paper,
│   │                        # read_paper, deep_research, nature_writer
│   ├── Ideation (2)         # brainstorm, critical_thinking
│   └── Self-Evolution (9)   # create_skill, record_macro, replay_pipeline,
│                            # track_error, suggest_error_fix, ...
│
└── utils/                   # Shared utilities
    ├── plotting.py          # Nature/ICML-style SVG+PDF (300 DPI, Okabe-Ito)
    ├── report_templates.py  # Section templates, CSS, LaTeX preamble
    ├── stats.py             # Mann-Whitney, chi², FDR, log-rank
    └── icd10.py             # ICD-10 code → name lookup
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/skills` | List available analysis skills |
| `/plan <task>` | Enter structured plan mode |
| `/compact` | Compress conversation history |
| `/clear` | Reset session state |
| `/cost` | Token usage and estimated cost |
| `/model <name>` | Switch LLM model |
| `/models-available` | Fetch relay-supported model IDs |
| `/figures` | List generated figures |
| `/cohorts` | Active cohorts summary |
| `/models` | Trained models with AUC |
| `/export <fmt>` | Export session (JSON/Markdown) |
| `/history` | Analysis history |
| `/record <name>` | Save session as pipeline |
| `/pipelines` | List saved pipelines |
| `/errors` | Error catalog |
| `/memory` | Long-term memory summary |
| `/status` | Full session status |
| `/external-agents` | Check local Codex/Claude Code availability |
| `/codex-plan <task>` | Ask Codex for a read-only plan |
| `/codex-check [focus]` | Ask Codex to review execution/code |
| `/claude-plan <task>` | Ask Claude Code for a read-only plan |
| `/claude-check [focus]` | Ask Claude Code to review execution/code |

## Local Agent Plugins

Biobank Agent includes repo-local plugin bundles:

- Codex: `plugins/biobank-agent` with marketplace metadata in `.agents/plugins/marketplace.json`
- Claude Code: `plugins/biobank-agent-claude` with marketplace metadata in `.claude-plugin/marketplace.json`

Inside Biobank Agent, the corresponding skills are `external_agent_status`, `codex_plan`, `codex_check_execution`, `claude_plan`, and `claude_check_execution`. These default to read-only planning/review modes and are covered by mocked tests, so the suite does not spend real model quota.

Before using Claude Code from inside Biobank Agent, authenticate Claude Code on
the workstation:

```bash
claude auth status
claude auth login
```

After login, verify the bridge from the Biobank Agent shell:

```text
/external-agents
/claude-plan Draft a guarded report workflow
/claude-check Check report guardrails
```

`external_agent_status` reports Claude as unavailable when `claude auth status`
returns `loggedIn=false`, even if the `claude` binary is installed.

## Configuration

```bash
# .env
LLM_BASE_URL=https://api.openai.com       # or any OpenAI-compatible endpoint
LLM_API_KEY=your-key
LLM_MODEL=gpt-4o                           # or claude-sonnet-4-6, etc.

DATA_DIR=./data                             # Parquet files
RAW_DIR=./raw                               # Raw CSV fallback (optional)

# Optional
SEARCH_PROVIDER=duckduckgo                  # or brave, serper
SEARCH_API_KEY=                             # required for brave/serper
MULTI_MODEL_ENABLED=true                    # auto route hard tasks to multi-agent
AUTO_DISCOVER_MODELS=true                   # query /v1/models from relay
PREFERRED_MULTI_MODELS=gpt-5.4,gemini-3.1-pro-preview
```

## Memory System

| Tier | Scope | Persistence |
|------|-------|-------------|
| Short-term | Current conversation | In-memory |
| Mid-term | Analysis records with exact metrics | Session state |
| Long-term | Model configs, pipelines, field usage | `memory.json` |
| Error catalog | Error patterns + suggested fixes | `memory.json` |
| Domain | Accumulated biomedical findings | `domain.md` |
| User preferences | Researcher settings | `user.md` |
| Episodic | Cross-session BM25-ranked recall | `sessions.db` |
| Action graph | Claim ↔ Evidence knowledge graph | `action_graph.db` |

## Verification Pipeline

All skill outputs pass through a multi-layer verification stack:

| Layer | Engine | Purpose |
|-------|--------|---------|
| Formal | Z3 SMT solver | Constraint satisfaction (optional) |
| Numeric | `NumericRangeChecker` | UKB bounds (502,411 max, age 37–73) |
| URL/DOI | `URLDOIResolver` | Reference accessibility check |
| Entailment | LLM NLI | Claim ↔ evidence consistency |
| Verdict | `VerdictEngine` | Final PASS / FAIL / PARTIAL |

## Testing

```bash
pytest tests/ -v                          # All tests
pytest tests/ -v -m "not integration"     # Skip API-dependent tests
biobank eval --suite skill_schemas --mode baseline
biobank eval --suite report_quality --mode baseline
biobank eval --suite agent_report_workflow --mode baseline
```

## Platform Support

| Platform | Status |
|----------|--------|
| macOS ARM (Apple Silicon) | Full support, CPU |
| Linux x86 + NVIDIA GPU | Full support, CUDA optional |

GPU-dependent features (SHAP deep explainer, UMAP with RAPIDS) auto-fallback to CPU.

## Citation

If you use Biobank Agent in your research, please cite:

```bibtex
@software{biobank_agent,
  title   = {Biobank Agent: Autonomous Scientific Discovery for Population-Scale Biobank Research},
  author  = {Chen, Pengan},
  year    = {2026},
  url     = {https://github.com/cpa2001/BioBank-Agent}
}
```

## License

[MIT](LICENSE)

## Author

AIH Group, Department of Computer Science and Engineering (CSE), The Chinese University of Hong Kong (CUHK)
Shanghai Academy of AI for Science (SAIS)

**CHEN Pengan** · chenpengan@link.cuhk.edu.hk
