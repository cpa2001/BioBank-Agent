# Technical Reference

Quick-reference tables for all core subsystems. For narrative explanation, see [architecture/OVERVIEW.md](../architecture/OVERVIEW.md).

---

## 1. Memory System (8-Tier)

| Tier | Name | Storage | Retention | Key Methods |
|------|------|---------|-----------|-------------|
| 1 | Short-term | RAM list | Session | `.messages` |
| 2 | Mid-term | RAM dataclass | Session | `.records` (AnalysisRecord) |
| 3 | Long-term | JSON file | Persistent | `save_pipeline()`, `remember_model_config()` |
| 4 | Error catalog | JSON nested | Persistent | `record_error()`, `get_error_suggestions()` |
| 5 | Domain | Markdown | Persistent | `append_finding()`, `summary()` |
| 6 | User prefs | Markdown | Persistent | `upsert_preference()` |
| 7 | Episodic | SQLite FTS5 | Persistent | `index_turn()`, `search()` |
| 8 | Action graph | SQLite | Persistent | `upsert_node()`, `link_nodes()`, `explain_claim()` |

---

## 2. Context Management

| Component | Default | Purpose |
|-----------|---------|---------|
| Context window | 180,000 tokens | Max tokens per LLM request |
| Max tool rounds | 30 | Prevent infinite loops |
| State injection | `context_summary()` | Last 20 steps + cohorts + models |
| Domain memory | 2000 char max | Accumulated findings |
| Complexity threshold | 0.7 | Multi-model trigger |

---

## 3. Configuration

| Source | Key Settings |
|--------|-------------|
| `.env` | `LLM_API_KEY`, `DATA_DIR`, `RAW_DIR`, `MULTI_MODEL_ENABLED` |
| `config.py` | Typed Pydantic settings with backward-compatible aliases |
| `pyproject.toml` | Dependencies: openai, duckdb, pandas, scikit-learn, xgboost |

**Multi-model options:**
```ini
MULTI_MODEL_ENABLED=true
AUTO_DISCOVER_MODELS=true
PREFERRED_MULTI_MODELS=deepseek/deepseek-v4-pro,moonshotai/kimi-k2.6
DEBATE_ROUNDS=2
```

---

## 4. Data Access

| Data Type | Format | Access |
|-----------|--------|--------|
| Biomarkers | Parquet | DuckDB VIEW, 4,971 fields |
| Diagnoses | Parquet | eid + ICD10 + date |
| Deaths | Parquet | eid + cause_icd10 |
| Categories | Parquet (split) | Per category_id |
| Field catalog | TSV | 11,821 entries |
| Encoding map | TSV | ICD10 ↔ text |

**Access layer:** `DataManager` class → DuckDB zero-copy VIEWs over Parquet

---

## 5. Error Handling

| Component | Max Retries | Strategy |
|-----------|-------------|----------|
| LLM client | 3 | Exponential backoff (2.0×) |
| Skill executor | 3 | Parameter mutation (reduce folds, sample_size) |
| Error catalog | ∞ | Record pattern + fix, track frequency |

**Parameter mutations per retry:**
- `n_folds`: max(2, n−1)
- `sample_size`: × 0.75
- `top_n`: max(5, × 0.5)
- `n_jobs`: → 1

---

## 6. Verification Layers

| Layer | Engine | Output |
|-------|--------|--------|
| Formal | Z3 SMT solver | Constraint satisfaction |
| Numeric | `NumericRangeChecker` | UKB bounds (502,411 max, age 37–73) |
| URL/DOI | `URLDOIResolver` | Reference accessibility |
| Entailment | LLM NLI | Claim ↔ evidence consistency |
| Verdict | `VerdictEngine` | PASS / FAIL / PARTIAL |
| Guardrails | `DelegationGuardrails` | Anti-pattern detection |

**Verdict severity:** BLOCKER → WARNING → SUGGESTION

---

## 7. Report and Eval Gates

| Component | Behavior |
|-----------|----------|
| `generate_report(format="technical")` | Human-facing key findings first; methods, execution logs, and diagnostics in appendices |
| `generate_report(format="nature")` | IMRaD report with abstract, methods, results, discussion, references, and data availability |
| Executive finding filter | Removes raw logs, local paths, reviewer chatter, placeholders, and unsupported causal claims from lead findings |
| Plan execution log | `/plan` steps execute through the agent recorder and are visible to generated report appendices |
| `report_20_case` | 20 UKB-oriented synthetic aggregate cases for fast report regression |
| `live_ukb_report_20` | 20 live UKB probes; fails closed when real UKB data access is unavailable |
| Review loop | Codex/GPT-5.5 xhigh is the default reviewer; Claude Code is optional via `--include-claude` |

---

## 8. Selected Skills Inventory

The current registry discovers 105 skills. This section lists representative
skills by workflow area rather than the full registry.

### Cohort, Data, and Descriptive Analysis
field_search, prevalence, cohort_summary, cohort_card, phenotype_harmonize,
trajectory_tokenize, min_sample, missing_data, biomarker_dist, correlation,
comorbidity, phewas

### Modelling and Discovery
train_model, evaluate_model, feature_importance, calibration, predict,
survival, gwas_proxy, genetic_target_hypothesis, target_annotation_context,
target_enrichment, discover, embedding, smart_plot

### Literature, Documentation, and Writing
web_search, web_fetch, read_pdf, read_paper, fetch_paper, deep_research,
literature_qa, project_doc, brainstorm, critical_thinking, nature_writer

### Reporting, Safety, and Evidence Governance
generate_report, statistical_review, safety_check, hypothesis, world_model_audit

### Memory, Workflow, and Self-Evolution
think, create_skill, suggest_error_fix, track_error, list_errors,
recall_session, record_macro, replay_pipeline, list_pipelines,
analyze_workflow_patterns, suggest_optimal_pipeline

### External Review and Release
external_agent_status, codex_plan, claude_plan, codex_check_execution,
claude_check_execution, git_clean_push

---

## 9. CLI Commands

```
/plan <goal>      — Draft a structured plan (DAG decomposition)
/graph            — Display the current plan / action graph
/plan-approve     — Approve and execute the active plan
/skills           — List registered skills (alias: /tools)
/evidence <id>    — Show evidence for claim
/memory           — Inspect memory tiers
/models           — Show available models
/routing-status   — Show orchestration status
```

---

## 10. Design Principles

| Principle | Implementation |
|-----------|---------------|
| Skill-first | `@skill` decorator, hot-reloadable, 58 composable units |
| Memory-driven | 8-tier persistent system, cross-session learning |
| Verified outputs | Reflexion → Verdict → Guardrails → VerifierMesh |
| Schema-gated | `StudySpec` typed boundary before execution |
| Progressive disclosure | Token-aware injection, 4-layer output |
| Zero-copy data | DuckDB VIEWs over Parquet |
| Graceful degradation | Optional deps with fallbacks (Z3, PaperQA2, Instructor) |
| Biobank-portable | Config-driven column names (UKB/FinnGen/CKB compatible) |
