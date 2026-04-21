# Biobank Agent (`bb`)

A CLI AI agent for UK Biobank phenotype analysis. Natural language interface to 502,370 subjects, 4,971 biomarker fields, and 6.9M diagnosis records.

## Quick Start

```bash
# Install
cd /path/to/UKB_agent
pip install -e ".[all]"

# Configure
cp .env.example .env
# Edit .env with your API key and data paths

# Run
bb
```

## Usage

```
bb> What are the top 20 most common diseases in UK Biobank?
bb> Train an XGBoost model to predict Type 2 Diabetes from biomarkers
bb> Show Kaplan-Meier survival curves for acute MI (I21)
bb> Run a complete analysis of Chronic Kidney Disease and generate a report
```

### CLI Commands

| Command | Description |
|---------|-------------|
| `/skills` | List all available analysis tools |
| `/status` | Session state, platform info, memory summary |
| `/history` | Show analysis history |
| `/record <name>` | Save current session as a replayable pipeline |
| `/pipelines` | List saved pipelines |
| `/errors` | Show error catalog from long-term memory |
| `/memory` | Show long-term memory summary |

### Subcommands

```bash
bb rebuild-parquet                        # Rebuild parquet from category CSVs
bb rebuild-parquet --categories=Genomics  # Rebuild specific category only
bb --model=gpt-5.4                        # Override model at runtime
```

## Architecture

```
biobank_agent/
├── cli.py                  # Rich terminal UI with REPL
├── agent.py                # ReAct loop (native tool_use, auto-retry, error tracking)
├── llm.py                  # OpenAI-compatible client (retry + backoff)
├── registry.py             # Skill auto-discovery via @skill decorator
├── config.py               # Pydantic settings from .env
├── state.py                # Session state (cohorts, models, figures, records)
├── memory.py               # 4-tier persistent memory (configs, pipelines, fields, errors)
│
├── data/
│   ├── loader.py           # DuckDB unified data layer (parquet + CSV fallback)
│   ├── catalog.py          # UKB field catalogue (11,821 fields, 410 categories)
│   ├── cohort.py           # Case/control cohort builder (parameterized SQL)
│   ├── features.py         # Biomarker group definitions
│   └── parquet_builder.py  # Batch CSV→parquet rebuild
│
├── skills/                 # 26 analysis skills (auto-discovered)
│   ├── prevalence.py       # Disease prevalence charts
│   ├── train_model.py      # XGBoost/LightGBM/CatBoost + CV
│   ├── survival.py         # Kaplan-Meier + log-rank
│   ├── replay_pipeline.py  # Pipeline macro replay
│   ├── create_skill.py     # AST-gated skill generation
│   └── ...
│
├── utils/
│   ├── plotting.py         # Nature-style matplotlib (300 dpi, Arial, Okabe-Ito)
│   ├── stats.py            # Mann-Whitney, chi², FDR correction, log-rank
│   ├── icd10.py            # ICD10 code → name lookup, chapter grouping
│   └── platform.py         # GPU/CPU detection, architecture info
│
└── interfaces/             # Extension stubs (Foundation Models, Multimodal)
    ├── fm_embedding.py     # Abstract: Evo2, ESM-2, BrainLM integration
    ├── multimodal.py       # Abstract: tabular + embedding fusion
    └── evolution.py        # Abstract: self-evolution hooks
```

## Skills

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
| `generate_report` | Markdown + HTML analysis report |

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

### Raw CSV (Comprehensive Fallback)

| File | Columns | Size |
|------|---------|------|
| `ukb672073_Biological_Samples.csv` | 1,775 | 3.5 GB |
| `ukb672073_Health_Related_Outcomes.csv` | 4,896 | 7.3 GB |
| `ukb672073_Online_Follow_up.csv` | 5,467 | 8.3 GB |
| `ukb672073_Genomics.csv` | 194 | 1.0 GB |
| `ukb672073_Additional_Exposures.csv` | 279 | 679 MB |
| `ukb672073_Population_Characteristics.csv` | 34 | 78 MB |

### Metadata

- `field.txt`: 11,821 field definitions
- `category.txt`: 410 categories
- `esimpint.txt` / `ehierint.txt`: encoding lookups

## Configuration

```bash
# .env
LLM_BASE_URL=http://your-api-endpoint
LLM_API_KEY=your-key
LLM_MODEL=claude-sonnet-4-6       # or gpt-5.4, gemini-3.1-pro-preview, etc.
UKB_PARQUET_DIR=./milton_data
UKB_RAW_DIR=./UKB
```

Switch model at runtime: `bb --model=gpt-5.4`

## Memory System

| Tier | Scope | Storage |
|------|-------|---------|
| Short-term | Current session messages | In-memory |
| Mid-term | Analysis records (exact numbers) | Session state |
| Long-term | Best model configs, saved pipelines, field usage | `~/.biobank_agent/memory.json` |
| Error catalog | Error patterns + suggested fixes | `~/.biobank_agent/memory.json` |

## Extension Points

- **Foundation Models**: `interfaces/fm_embedding.py` — plug in Evo2, ESM-2, BrainLM
- **Multimodal Fusion**: `interfaces/multimodal.py` — combine tabular + FM embeddings
- **Self-Evolution**: `interfaces/evolution.py` — pipeline macros, skill generation
- **Custom Skills**: Drop `.py` files in `custom_skills/` with `@skill` decorator

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

CHEN Pengan · chenpengan@link.cuhk.edu.hk
