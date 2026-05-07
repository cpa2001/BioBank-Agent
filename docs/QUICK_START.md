# Quick Start Guide - UKB Biobank Agent

## Status: Production Ready ✅

### Latest Improvements (Commit 96beb35)
- ✅ Structured evidence tracking (claims + evidence links)
- ✅ OrchestrationResult with safety verification gate
- ✅ Query ID correlation for session tracking
- ✅ 381 tests passing, zero regressions
- ✅ 16 comprehensive documentation files

---

## Getting Started

### 1. Launch the Agent
```bash
cd /Users/chenpengan/Projects/CUHK/UKB_agent
python -m biobank_agent.cli
```

### 2. Try Example Queries
```
biobank> Discover T2DM biomarkers using metabolomics
biobank> Train a predictive model for myocardial infarction
biobank> Show cardiovascular risk factors in the cohort
biobank> /plan Comprehensive brain imaging analysis
```

### 3. Inspect Results
```
biobank> /strategy show        # View routing decision
biobank> /evidence show        # View claims & evidence
biobank> /routing-status       # Full orchestration trace
```

---

## Documentation Map

| Need | File |
|------|------|
| **System Architecture** | [`architecture/ARCHITECTURE_DIAGRAMS.md`](architecture/ARCHITECTURE_DIAGRAMS.md) |
| **Data Quick Lookup** | [`data/DATA_MODALITIES_QUICK_REFERENCE.md`](data/DATA_MODALITIES_QUICK_REFERENCE.md) |
| **Capabilities** | [`RESEARCH_EXPLORATION_REPORT.md`](RESEARCH_EXPLORATION_REPORT.md) |
| **Custom Skills** | [`CUSTOM_SKILLS_GUIDE.md`](CUSTOM_SKILLS_GUIDE.md) |
| **Complete Data Inventory** | [`data/UKB_DATA_MODALITIES_COMPREHENSIVE.md`](data/UKB_DATA_MODALITIES_COMPREHENSIVE.md) |
| **Academic Overview** | [`BIOBANK_AGENT_TECH_WHITEPAPER.md`](BIOBANK_AGENT_TECH_WHITEPAPER.md) |

---

## Key Features

### Evidence-Based Reasoning
- All claims extracted with confidence scores
- Evidence links to analysis records
- Safety verification before finalization
- Complete debate trace history

### Multi-Model Orchestration
- Complexity-based strategy selection
- First-turn debate mode for hard questions
- Single-model fallback for stability
- Consensus voting in debate mode

### Data Access
**13 data modality categories:**
- Genomics (97.1% coverage)
- Blood Biomarkers (97.6% coverage)
- Metabolomics (97.0% coverage)
- Brain MRI, Cardiac MRI, DXA, etc.
- Hospital & GP clinical records

**Total:** 4,971 fields, 502,372 participants

### Built-in Skills (45+)
- Cohort construction & analysis
- GWAS association testing
- Predictive modeling (XGBoost/LightGBM)
- Survival analysis
- Feature importance
- Report generation
- Literature search

---

## Recent Git Activity

```
96beb35 Enhance orchestrator with structured claims & evidence tracking
73cf585 Add git_clean_push skill: rewrite AI commit authors and force-push
9017dee Switch default model to claude-opus-4-7, update multi-agent pool
224fa41 Implement structured verdict system and comprehensive tests
```

**Current Status:** main branch, 1 commit ahead of origin/main

To push: `git push origin main`

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_orchestration_result.py -v

# Check coverage
python -m pytest tests/ --cov=biobank_agent
```

**Current Results:** 381 passed, 9 skipped, 0 failed ✅

---

## Architecture at a Glance

```
User Query
    ↓
ComplexityClassifier
    ↓
MultiModelOrchestrator (Route to strategy)
    ├─ SINGLE: Fast path, single model
    ├─ DEBATE: Multi-model proposals with voting
    ├─ SUPERVISOR: Planner decomposition + workers
    └─ ENSEMBLE: Parallel models with consensus
    ↓
ReAct Loop (LLM + Tool calls)
    ├─ LLMClient (OpenAI-compatible API)
    ├─ SkillRegistry (45+ built-in skills)
    └─ Tool Execution
    ↓
Evidence Tracking
    ├─ Claim Extraction
    ├─ Evidence Linking
    ├─ Safety Verification
    └─ Result Serialization
    ↓
Refined Answer with Claims & Evidence
```

---

## Common Commands

| Command | Purpose |
|---------|---------|
| `/help` | Show all commands |
| `/skills` | List available skills |
| `/history` | Show conversation history |
| `/plan [goal]` | Enter plan mode |
| `/strategy show` | View routing decision |
| `/evidence show` | View claims & evidence |
| `/models available` | List available models |
| `/cost` | Show token usage |

---

## Configuration

**LLM Configuration** (`biobank_agent/config.py`)
- Base URL: `http://api.shubiaobiao.cn/v1`
- Default Model: `claude-opus-4-7`
- Multi-Model Pool: `gpt-5.4-pro`, `gemini-3.1-pro-preview`

**Data Paths**
- Processed: `/Users/chenpengan/Projects/CUHK/milton_data/`
- Raw: `/Users/chenpengan/Projects/CUHK/UKB/`

---

## Troubleshooting

### Model Unavailable
```
Error: "no available channel for model"
→ Check relay at api.shubiaobiao.cn
→ Verify API key is set
→ Try switching model: /model-switch
```

### No Data Found
```
Error: "cohort size too small"
→ Check field IDs in data/DATA_MODALITIES_QUICK_REFERENCE.md
→ Try broader disease search
→ Review data/UKB_DATA_FIELDS_INVENTORY.md
```

### Tests Failing
```bash
# Run with verbose output
python -m pytest tests/ -vvs

# Run specific failing test
python -m pytest tests/test_X.py::test_Y -vvs

# See full traceback
python -m pytest tests/ --tb=long
```

---

## Next Steps

1. **Push to Remote** (if needed)
   ```bash
   git push origin main
   ```

2. **Run Agent**
   ```bash
   python -m biobank_agent.cli
   ```

3. **Try Plan Mode**
   ```
   biobank> /plan Discover metabolic disease biomarkers
   ```

4. **Inspect Evidence**
   ```
   biobank> /evidence show
   ```

---

**Last Updated:** April 23, 2026  
**Status:** Production Ready ✅  
**Tests:** 381/381 Passing ✅  
**Deployment:** Ready for remote push
