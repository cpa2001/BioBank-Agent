# BioBank Agent: Quick Reference Guide

## 📊 System Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER INTERACTION LAYER                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  CLI (22 slash commands)  +  Textual TUI (v3 scaffold)  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Complexity Classifier (SIMPLE/MEDIUM/HARD)            │   │
│  │         ↓                                               │   │
│  │  ┌──────────────────┐      ┌──────────────────┐        │   │
│  │  │  SIMPLE/MEDIUM   │      │     HARD (≥2)    │        │   │
│  │  │  ReAct Loop      │  OR  │ MultiModelOrch.  │        │   │
│  │  │  (Single Model)  │      │ (Red/Blue Debate)│        │   │
│  │  └──────────────────┘      └──────────────────┘        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    SKILL EXECUTION LAYER                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  SkillRegistry (64 registered skills)                   │   │
│  │  ├─ Analysis (17)      ├─ Discovery (7)                │   │
│  │  ├─ Research (7)       ├─ Ideation (2)                 │   │
│  │  ├─ Self-Evolution (9) └─ Report/Governance (15+)      │   │
│  │                                                         │   │
│  │  + Hot-reload from ./custom_skills/ (user plugins)     │   │
│  └─────────────────────────────────────────────────────────┘   │
│              ↙                    ↓                    ↘         │
│         ┌────────────┐    ┌────────────┐    ┌────────────┐     │
│         │ Web Search │    │Data Analysis   │ Report Gen │      │
│         │ & Paper ML │    │ & Modeling     │ & Figures  │      │
│         └────────────┘    └────────────┘    └────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    VERIFICATION LAYER                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Multi-Layer Pipeline                                  │   │
│  │  Formal (Z3) → Numeric (bounds) → URL/DOI → NLI       │   │
│  │  ↓ VerdictEngine → PASS/FAIL/PARTIAL                  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   STORAGE & MEMORY LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Short-term   │  │ Long-term    │  │ Action Graph │          │
│  │ (in-memory)  │  │ (memory.json)│  │ (claim ↔ ev) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Sessions DB  │  │ Domain (MD)  │  │ Error Catalog│          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA ACCESS LAYER                             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Bank Adapters (UKB / CKB / HPP / FinnGen)             │   │
│  │  ↓                                                      │   │
│  │  DuckDB (primary) ← Parquet ← CSV (fallback)           │   │
│  │  + Optional: PySpark/SparkSQL (RAP, v3.1)              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Components Quick-Access

### Configuration
```python
# pyproject.toml
name = "biobank-agent"
version = "3.0.0-rc1"
entry_point = "biobank = biobank_agent.cli:main"
requires-python = ">=3.10"
```

```bash
# .env
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=your-key
LLM_MODEL=deepseek/deepseek-v4-pro
DATA_DIR=./data
BANK_ID=ukb
MULTI_MODEL_ENABLED=true
```

### Skills Registry
```python
# Define a skill in biobank_agent/skills/my_skill.py
from biobank_agent.registry import skill

@skill(
    name="my_analysis",
    description="Analyze something cool",
    parameters={
        "input_field": {"type": "string", "description": "Input"}
    }
)
def my_analysis(input_field: str, *, ctx):
    ctx.llm    # LLM client
    ctx.agent  # Agent instance
    ctx.data   # DataManager
    return {"result": "..."}
```

### CLI Commands (Quick Ref)
```bash
biobank                      # Start interactive CLI
/help                        # Show all commands
/skills                      # List 64 skills
/plan "Research CVD"        # Enter plan mode
/cost                        # Token usage
/memory                      # 8-tier memory summary
/codex-plan "Review this"    # Ask Codex
/mcp-list                    # Show MCP servers
```

### Bank Configuration
```yaml
# biobank_agent/banks/configs/mybank.yaml
bank_id: mybank
subject_id_col: participant_id
diagnoses_code_col: diag_icd10
```

```bash
# .env override
BANK_ID=mybank
BIOBANK_NAME=My Biobank
DATA_DIR=/path/to/data
```

---

## 📈 Statistics at a Glance

| Metric | Value |
|--------|-------|
| **Skills Registered** | 64 |
| **CLI Commands** | 22 |
| **Test Files** | 94 |
| **Tests Passing** | 1,148 ✅ |
| **Core Dependencies** | 51 |
| **Optional Groups** | 7 |
| **Memory Tiers** | 8 |
| **Verification Layers** | 5 |
| **Supported Banks** | UKB (✅), CKB (ready), HPP (ready), FinnGen (planning) |
| **Platforms** | macOS ARM, Linux x86, Linux+GPU |
| **v3 Criteria Met** | 13/17 (5 credential-blocked) |

---

## 🔧 Extension Checklist

### To Add a New Skill
- [ ] Create `biobank_agent/skills/my_skill.py` (or `./custom_skills/my_skill.py`)
- [ ] Use `@skill(name=..., parameters={...})` decorator
- [ ] Accept `ctx` kwarg for agent/LLM/data access
- [ ] Return dict or structured output
- [ ] Skill auto-discovered on load

### To Support a New Biobank
- [ ] Create `biobank_agent/banks/configs/mybank.yaml`
- [ ] Set `subject_id_col`, `diagnoses_code_col`, column patterns
- [ ] Point `.env` to data: `DATA_DIR`, `RAW_DIR`
- [ ] Override: `BANK_ID`, `BIOBANK_NAME`, `SUBJECT_ID_COL`
- [ ] Test with `/bank-data-readiness` skill

### To Integrate MCP Server
- [ ] Configure MCP server address/type
- [ ] Run `/mcp-start` to boot servers
- [ ] Use `/mcp-list` to see loaded tools
- [ ] Call tools via `/mcp-call <server__tool>`

### To Add Custom Report Format
- [ ] Extend `biobank_agent/utils/report_templates.py`
- [ ] Add format template in `ReportGenerator`
- [ ] Update CLI with new format option
- [ ] Test with `/report --format=myformat`

---

## 📚 Documentation Map

| Document | Purpose |
|----------|---------|
| **README.md** | Main project overview |
| **BIOBANK_ARCHITECTURE_ANALYSIS.md** | Deep-dive (1,259 lines) |
| **EXPLORATION_SUMMARY.md** | This executive summary |
| **QUICK_REFERENCE.md** | Quick lookup (you are here) |
| **docs/architecture/V3.md** | v3 scope & roadmap |
| **docs/guides/** | User guides, skill guides |
| **docs/examples/** | SDK, MCP, workflow examples |
| **CHANGELOG.md** | Version history |

---

## 🚀 Deployment Checklist

### Local Development
- [ ] `git clone https://github.com/cpa2001/BioBank-Agent.git`
- [ ] `pip install -e ".[all]"`
- [ ] `cp .env.example .env` and edit
- [ ] `biobank` to start CLI

### Production Deployment
- [ ] Configure `.env` for your institution
- [ ] Point `DATA_DIR` to parquet files
- [ ] Set `LLM_BASE_URL` and credentials
- [ ] Run test eval: `biobank eval --suite skill_schemas`
- [ ] Set up GitHub Actions secrets for CI/CD
- [ ] Monitor telemetry via `~/.biobank_agent/telemetry.jsonl`

### Containerization (Future)
- [ ] v3.0 adds Dockerfile support
- [ ] Docker image with all optional deps
- [ ] Kubernetes manifests planned

---

## ❓ FAQ Quick Links

**Q: How do I add custom analysis logic?**  
A: Create a skill in `./custom_skills/my_analysis.py` with `@skill` decorator. It auto-loads on startup.

**Q: Can I use a different LLM provider?**  
A: Yes! Any OpenAI-compatible API works. Set `LLM_BASE_URL` in `.env`.

**Q: How does multi-model orchestration work?**  
A: Complex tasks (classified as HARD) route to `MultiModelOrchestrator` with red/blue debate & voting.

**Q: What biobanks are supported?**  
A: UKB (primary), CKB/HPP (configs ready, need credentials), FinnGen (planning). Use bank adapters.

**Q: How do I verify results are correct?**  
A: Multi-layer verification: formal (Z3), numeric bounds, URL/DOI, NLI entailment. See VerdictEngine.

**Q: Is it GPU-required?**  
A: No. GPU features (SHAP, UMAP) auto-fallback to CPU. Optional: `pip install -e ".[gpu]"`

**Q: How are disclosure controls handled?**  
A: Cell count rounding (nearest 5), PII stripping, causal language filtering, NLI verification.

---

## 🎓 Learning Path

1. **Start here**: README.md (project overview)
2. **Understand architecture**: BIOBANK_ARCHITECTURE_ANALYSIS.md
3. **Run first query**: `biobank` → `/plan "What are top diseases?"`
4. **Add custom skill**: Create `./custom_skills/my_skill.py`
5. **Extend to new biobank**: Add `banks/configs/mybank.yaml`, override `.env`
6. **Deploy multi-model**: Set `MULTI_MODEL_ENABLED=true` + model list
7. **Integrate MCP**: `/mcp-start` + `/mcp-call <tool>`
8. **Generate reports**: Use `/report --format=paper` or `/report --format=technical`

---

## 📞 Support & Contribution

- **Repository**: https://github.com/cpa2001/BioBank-Agent
- **Author**: CHEN Pengan (CUHK, SAIS)
- **Email**: chenpengan@link.cuhk.edu.hk
- **License**: MIT
- **Status**: v3.0-rc1 (production-ready for UKB, others in progress)

---

**Last Updated**: 2026-05-11  
**Analysis Depth**: Comprehensive (1,259 lines) + Executive (this page)
