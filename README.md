# Biobank Agent

Autonomous scientific discovery agent for UK Biobank phenotype analysis. Natural language interface to 502,370 subjects, 4,971 biomarker fields, and 6.9M diagnosis records.

## Quick Start

```bash
# Install
cd /path/to/UKB_agent
pip install -e ".[all]"

# Configure
cp .env.example .env
# Edit .env with your API key and data paths

# Run
biobank
```

## Usage

```
biobank> What are the top 20 most common diseases in UK Biobank?
biobank> Discover disease-specific biomarkers for Type 2 Diabetes
biobank> Train an XGBoost model to predict E11 and show feature importance
biobank> Read this paper and compare findings with our UK Biobank data
biobank> Search for latest Chronic Kidney Disease biomarker studies
biobank> Generate a Nature Methods-quality report for all analyses
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/skills` | List all available analysis tools |
| `/status` | Session state, platform info, memory summary |
| `/cost` | Show token usage and estimated cost |
| `/compact` | Compress conversation history |
| `/clear` | Reset session state |
| `/plan <task>` | Enter plan mode for complex tasks |
| `/plan-approve` | Approve plan and begin execution |
| `/plan-exit` | Exit plan mode |
| `/model <name>` | Switch LLM model at runtime |
| `/history` | Show analysis history |
| `/figures` | List all generated figures |
| `/cohorts` | List active cohorts |
| `/models` | List trained models with AUC |
| `/record <name>` | Save session as a replayable pipeline |
| `/pipelines` | List saved pipelines |
| `/export <fmt>` | Export session as JSON/Markdown |
| `/errors` | Show error catalog |
| `/memory` | Show long-term memory summary |

### Subcommands

```bash
biobank rebuild-parquet                        # Rebuild parquet from category CSVs
biobank rebuild-parquet --categories=Genomics  # Rebuild specific category only
biobank --model=gpt-5.4                        # Override model at runtime
```

## Architecture

```
biobank_agent/
├── cli.py                  # Rich terminal UI with REPL + 18 slash commands
├── agent.py                # ReAct loop (native tool_use, auto-retry, error tracking)
├── planner.py              # Plan mode engine (INTAKE → ALIGNMENT → EXECUTION → DONE)
├── llm.py                  # OpenAI-compatible client (retry + backoff)
├── registry.py             # Skill auto-discovery via @skill decorator + hot-reload
├── config.py               # Pydantic settings from .env
├── state.py                # Session state (cohorts, models, figures, token tracking)
├── memory.py               # 4-tier persistent memory (configs, pipelines, fields, errors)
│
├── data/
│   ├── loader.py           # DuckDB unified data layer (parquet + CSV fallback)
│   ├── catalog.py          # UKB field catalogue (11,821 fields, 410 categories)
│   ├── cohort.py           # Case/control cohort builder (parameterized SQL)
│   ├── features.py         # Biomarker group definitions
│   └── parquet_builder.py  # Batch CSV→parquet rebuild
│
├── skills/                 # 39 analysis skills (auto-discovered)
│   ├── prevalence.py       # Disease prevalence charts
│   ├── train_model.py      # XGBoost/LightGBM/CatBoost + CV
│   ├── predict.py          # Patient-level disease risk prediction
│   ├── discovery.py        # Automated scientific discovery pipeline
│   ├── gwas_proxy.py       # Phenotype-wide association study
│   ├── survival.py         # Kaplan-Meier + log-rank
│   ├── web_search.py       # DuckDuckGo/Brave/Serper web search
│   ├── web_fetch.py        # URL content fetching + HTML→text
│   ├── read_pdf.py         # PDF text + table extraction (PyMuPDF)
│   ├── fetch_paper.py      # Paper acquisition by DOI/URL/title
│   ├── read_paper.py       # Deep critical paper analysis
│   ├── nature_writer.py    # Nature-quality manuscript writing
│   ├── brainstorm.py       # AI4Science research ideation
│   ├── critical_thinking.py # Scientific claims evaluation
│   ├── deep_research.py    # Multi-source literature research
│   ├── smart_plot.py       # Publication-quality figure generation
│   ├── create_skill.py     # AST-gated skill generation
│   └── ...                 # 22 more skills
│
├── utils/
│   ├── plotting.py         # Nature/ICML-style matplotlib (SVG+PDF, 300 dpi, Okabe-Ito)
│   ├── report_templates.py # Report section templates, CSS, LaTeX preamble
│   ├── stats.py            # Mann-Whitney, chi-squared, FDR, log-rank
│   ├── icd10.py            # ICD10 code → name lookup, chapter grouping
│   └── platform.py         # GPU/CPU detection, architecture info
│
├── interfaces/             # Extension stubs (Foundation Models, Multimodal)
└── plans/                  # Plan mode storage
```

## Skills (39)

### Analysis (17)

| Skill | Description |
|-------|-------------|
| `think` | Internal reasoning trace |
| `field_search` | Search UKB field catalogue by keyword |
| `prevalence` | Disease prevalence bar charts with patient counts |
| `cohort_summary` | Demographics, age/sex distribution for a disease cohort |
| `biomarker_dist` | Cases vs controls violin plots + Mann-Whitney U |
| `correlation` | Clustered correlation heatmaps with dendrograms |
| `missing_data` | Missing data patterns and MCAR/MAR analysis |
| `train_model` | XGBoost / LightGBM / CatBoost with 5-fold CV |
| `evaluate_model` | ROC + PR curves with 95% CI |
| `feature_importance` | SHAP beeswarm / tree-based importance |
| `calibration` | Reliability diagram + ECE / MCE / Brier score |
| `survival` | Kaplan-Meier curves + log-rank test |
| `phewas` | PheWAS Manhattan plot with FDR correction |
| `comorbidity` | Co-occurrence network + odds ratios |
| `embedding` | t-SNE / UMAP patient scatter plots |
| `min_sample` | Sample size sensitivity curves |
| `generate_report` | Markdown + HTML analysis report (technical or paper draft) |

### Scientific Discovery (4)

| Skill | Description |
|-------|-------------|
| `predict` | Patient-level disease risk prediction from trained models |
| `discover` | Automated discovery pipeline (cohort → model → features → PheWAS) |
| `gwas_proxy` | Phenotype-wide association study (GWAS proxy) |
| `smart_plot` | Publication-quality figures with style selection (Nature/ICML) |

### Research & Literature (7)

| Skill | Description |
|-------|-------------|
| `web_search` | Web search via DuckDuckGo/Brave/Serper |
| `web_fetch` | Fetch + parse web pages to text |
| `read_pdf` | PDF text and table extraction |
| `fetch_paper` | Download papers by DOI/URL/title |
| `read_paper` | Deep critical paper analysis |
| `nature_writer` | Nature-quality manuscript section writing |
| `deep_research` | Multi-source research with citations |

### Ideation & Evaluation (2)

| Skill | Description |
|-------|-------------|
| `brainstorm` | AI4Science research ideation with evidence grounding |
| `critical_thinking` | Scientific claims evaluation (7-step protocol) |

### Self-Evolution (9)

| Skill | Description |
|-------|-------------|
| `record_macro` | Record session skill calls into a named pipeline |
| `replay_pipeline` | Replay a saved pipeline with parameter overrides |
| `list_pipelines` | List all saved pipelines |
| `create_skill` | Generate new skills from code (AST safety-gated) |
| `track_error` | Record errors with context to long-term memory |
| `list_errors` | Show most common errors across sessions |
| `suggest_error_fix` | Intelligent error recovery suggestions |
| `analyze_workflow_patterns` | Skill sequence analysis + bottleneck detection |
| `suggest_optimal_pipeline` | Recommend skill sequences for a given task |

## Data

### Parquet (Fast Path)

| Dataset | Rows | Fields | Size |
|---------|------|--------|------|
| `ukb.parquet/` (biomarkers) | 502,370 | 484 field IDs (2,031 cols) | 363 MB |
| `categories/` (6 category parquets) | 502,370 | 4,487 field IDs (10,741 cols) | 1.55 GB |
| `hesin_diag.parquet/` (diagnoses) | 6,946,795 | 4 cols | 37 MB |
| `death_cause.parquet/` (deaths) | 112,917 | 5 cols | 576 KB |

**Total: 4,971 unique field IDs accessible via DuckDB.**

## Configuration

```bash
# .env
LLM_BASE_URL=http://your-api-endpoint
LLM_API_KEY=your-key
LLM_MODEL=claude-sonnet-4-6

UKB_PARQUET_DIR=./milton_data
UKB_RAW_DIR=./UKB

# Optional
SEARCH_PROVIDER=duckduckgo  # or 'brave', 'serper'
SEARCH_API_KEY=             # required for brave/serper
```

## Memory System

| Tier | Scope | Storage |
|------|-------|---------|
| Short-term | Current session messages | In-memory |
| Mid-term | Analysis records (exact numbers) | Session state |
| Long-term | Best model configs, saved pipelines, field usage | `~/.biobank_agent/memory.json` |
| Error catalog | Error patterns + suggested fixes | `~/.biobank_agent/memory.json` |

## Platform Support

- **macOS ARM** (Apple Silicon): Full support, CPU-only
- **Linux x86 + NVIDIA GPU**: Full support, optional CUDA acceleration
- GPU-dependent features (SHAP deep explainer, UMAP with RAPIDS) auto-fallback to CPU

## Testing

```bash
pytest tests/ -v                          # Run all 164 tests
pytest tests/ -v -m "not integration"     # Skip API-dependent tests
```

## License

MIT

## Author

CHEN Pengan, chenpengan@link.cuhk.edu.hk
