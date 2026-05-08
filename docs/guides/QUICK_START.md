# Quick Start Guide

## Installation

```bash
# Clone and install
git clone <repo-url>
cd UKB_agent
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env: set LLM_API_KEY, DATA_DIR, RAW_DIR
```

## Configuration

Required environment variables in `.env`:
```ini
LLM_API_KEY=<your-api-key>          # OpenRouter or OpenAI-compatible key
LLM_BASE_URL=https://openrouter.ai/api/v1
DATA_DIR=/path/to/processed/data    # Parquet files
RAW_DIR=/path/to/raw/data           # CSV fallback
```

## Launch

```bash
# Interactive CLI
biobank

# Or via module
python -m biobank_agent.cli
```

## Example Queries

```
biobank> What is the prevalence of Type 2 Diabetes (E11)?
biobank> Discover metabolomics biomarkers for hypertension
biobank> Train an XGBoost model to predict myocardial infarction (I21)
biobank> Show Kaplan-Meier survival curves for I21 by sex
biobank> /plan Comprehensive cardiovascular risk factor analysis
```

## CLI Commands

| Command | Purpose |
|---------|---------|
| `/help` | Show all commands |
| `/skills` | List registered skills (58) |
| `/plan <goal>` | Enter plan mode (DAG decomposition) |
| `/show_plan` | Display current plan |
| `/execute_plan` | Execute planned steps |
| `/evidence <id>` | Show evidence for a claim |
| `/memory` | Inspect memory tiers |
| `/models` | Available models |
| `/cost` | Token usage summary |

## Testing

```bash
# Full test suite
pytest tests/ -v

# Quick smoke test
pytest tests/ -q --tb=short

# With coverage
pytest tests/ --cov=biobank_agent
```

## Key Features

- **58 registered skills** — prevalence, GWAS proxy, genetic target hypotheses, target annotation, target enrichment, survival, predictive modeling, deep research, project docs, and external review
- **8-tier memory** — cross-session learning, error catalog, episodic recall
- **Multi-model orchestration** — single/ensemble/debate/supervisor strategies
- **Schema-gated execution** — natural language → typed StudySpec before running
- **Evidence lattice** — claim-level provenance with confidence tracking
- **Verification pipeline** — formal (Z3), numeric bounds, URL/DOI, NLI entailment

## Architecture

```
User Query → StudySpec Compilation → Agent.run() (ReAct loop)
    → LLM decides tool calls → SkillRegistry executes
    → Reflexion on errors → Verdict (PASS/FAIL/PARTIAL)
    → Progressive Disclosure → Response
```

For full architecture details, see [architecture/OVERVIEW.md](../architecture/OVERVIEW.md).

## Troubleshooting

**Model unavailable**: Check API key and base URL in `.env`  
**No data found**: Verify `DATA_DIR` points to Parquet files  
**Import errors**: Run `pip install -e ".[all]"` for optional dependencies
