<div align="center">

# Biobank Agent

**Autonomous scientific discovery agent for population-scale biobank research**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-265%20passed-brightgreen.svg)](#testing)

*Natural language interface to large-scale biobank cohorts — from hypothesis to publication-quality report.*

</div>

---

## Overview

Biobank Agent is an LLM-powered scientific discovery system designed for population-scale biobank data analysis. It combines a **ReAct agent loop** with **39 specialized analysis skills** to enable end-to-end research workflows: cohort construction, biomarker discovery, predictive modelling, survival analysis, literature review, and publication-quality reporting.

**Key capabilities:**
- **Hypothesis-driven discovery** — automated pipelines from cohort building through feature importance to PheWAS
- **Predictive modelling** — XGBoost/LightGBM/CatBoost with cross-validation, calibration, and SHAP explanations
- **Literature integration** — search papers, read PDFs, cross-reference findings with biobank data
- **Publication-quality output** — Nature/ICML-style SVG+PDF figures, dual-format reports (technical & IMRaD)
- **Self-evolution** — learns from errors, records analysis pipelines, generates new skills at runtime
- **Plan mode** — structured multi-step planning with stage gates for complex analyses
- **Native multi-agent routing** — auto-detects complex tasks and coordinates multiple frontier models

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
├── memory.py                # 4-tier persistent memory
│
├── data/                    # Data access layer (DuckDB + Parquet)
│   ├── loader.py            # Unified query layer (parquet + CSV fallback)
│   ├── catalog.py           # Field catalogue (11,821 fields, 410 categories)
│   ├── cohort.py            # Case/control cohort builder
│   ├── features.py          # Biomarker group definitions
│   └── parquet_builder.py   # Batch CSV → Parquet rebuild
│
├── skills/                  # 39 analysis skills (auto-discovered)
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
| Long-term | Model configs, pipelines, field usage | `~/.biobank_agent/memory.json` |
| Error catalog | Error patterns + suggested fixes | `~/.biobank_agent/memory.json` |

## Testing

```bash
pytest tests/ -v                          # All tests
pytest tests/ -v -m "not integration"     # Skip API-dependent tests
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

**CHEN Pengan** · AIH Group, The Chinese University of Hong Kong

chenpengan@link.cuhk.edu.hk
