# Biobank Agent (`bb`)

CLI AI agent for UK Biobank phenotype analysis. Natural language interface to 502K subjects, 8,868 fields, and 6.9M diagnosis records.

## Quick Start

```bash
# Install
cd /path/to/UKB_agent
pip install -e ".[all]"

# Configure
cp .env.example .env
# Edit .env: set UKB_PARQUET_DIR, UKB_RAW_DIR, LLM_API_KEY

# Run
bb
```

## Usage

```
bb> What are the top 20 most common diseases in UK Biobank?
bb> Train an XGBoost model to predict Type 2 Diabetes from biomarkers
bb> Run a complete analysis of Chronic Kidney Disease and generate a report
bb> /skills          # list available analysis tools
bb> /status          # show session state
bb> /history         # show analysis history
```

## Architecture

```
biobank_agent/
├── cli.py              # Rich terminal UI
├── agent.py            # ReAct loop (native tool_use)
├── llm.py              # OpenAI-compatible client (Claude/GPT/Gemini)
├── registry.py         # Skill auto-discovery
├── config.py           # Pydantic settings from .env
├── state.py            # Session state
├── memory.py           # 3-tier persistent memory
├── data/
│   ├── loader.py       # DuckDB on parquet + CSV fallback
│   ├── catalog.py      # UKB field catalogue (11,821 fields)
│   ├── cohort.py       # Case/control builder
│   ├── features.py     # Biomarker groups
│   └── parquet_builder.py  # On-demand CSV→parquet
├── skills/             # 17 analysis skills (auto-discovered)
├── utils/              # Nature-style plots, stats, ICD10
└── interfaces/         # FM embedding, multimodal, evolution stubs
```

## Skills (17)

| Skill | Description |
|-------|-------------|
| `think` | Internal reasoning |
| `field_search` | Search UKB field catalogue |
| `prevalence` | Disease prevalence bar charts |
| `cohort_summary` | Demographics + age/sex distribution |
| `biomarker_dist` | Cases vs controls violin plots |
| `correlation` | Clustered correlation heatmaps |
| `missing_data` | Missing data analysis |
| `train_model` | XGBoost/LightGBM/CatBoost + 5-fold CV |
| `evaluate_model` | ROC + PR curves with metrics |
| `feature_importance` | SHAP / tree importance |
| `calibration` | Reliability diagram + ECE/MCE |
| `survival` | Kaplan-Meier + log-rank test |
| `phewas` | PheWAS Manhattan plot + FDR |
| `comorbidity` | Co-occurrence + odds ratios |
| `embedding` | t-SNE/UMAP scatter plots |
| `min_sample` | Sample size sensitivity curves |
| `generate_report` | Markdown + HTML report |

## Data

- **Parquet (fast):** 502K subjects × 2,031 columns (363 MB)
- **CSV (comprehensive):** 502K subjects × 30,799 columns (54 GB)
- **Diagnoses:** 6.9M ICD10 records
- **Deaths:** 113K cause-of-death records
- **Catalogue:** 11,821 field definitions, 410 categories

## Configuration (.env)

```
LLM_BASE_URL=http://api.shubiaobiao.cn
LLM_API_KEY=your-key
LLM_MODEL=claude-sonnet-4-6
UKB_PARQUET_DIR=/path/to/milton_data
UKB_RAW_DIR=/path/to/UKB
```

## Models Supported

`claude-opus-4-5-20251101`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`, `gemini-3.1-pro-preview`, `gpt-5.4`, `gpt-5.4-pro`

Switch at runtime: `bb --model=gpt-5.4`

## Extension Points

- **Foundation Models:** `interfaces/fm_embedding.py` — plug in Evo2, ESM-2, BrainLM
- **Multimodal Fusion:** `interfaces/multimodal.py` — combine tabular + FM embeddings
- **Self-Evolution:** `interfaces/evolution.py` — pipeline macros, skill generation
- **Custom Skills:** Drop `.py` files in `custom_skills/` with `@skill` decorator

## Author

CHEN Pengan · chenpengan@link.cuhk.edu.hk
