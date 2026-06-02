# Architecture Overview

## System Summary

**Type**: Single-agent with optional multi-model consultation  
**Pattern**: ReAct loop + Reflexion error correction  
**Framework**: Custom orchestrator (not LangGraph/CrewAI/AutoGen)  
**Tools**: OpenAI-compatible function calling (native, not text parsing)  
**State**: Centralized `SessionState` injected into all skills via context  

---

## Core Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         User Query                                      │
└───────────────┬────────────────────────────────────────────────────────┘
                │
┌───────────────▼────────────────────────────────────────────────────────┐
│ Agent.run()  [agent.py]                                                │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1. Build system message (inject memory, cohort context)         │   │
│  │ 2. Call LLM with tool schemas (105 registered skills)           │   │
│  │ 3. Execute tool calls via SkillRegistry                         │   │
│  │ 4. Record results → AnalysisRecord                              │   │
│  │ 5. Loop until final text response (max 30 rounds)               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Guards: Reflexion ↔ Verdict ↔ Guardrails ↔ VerifierMesh              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Module Map

### Core Orchestration
| Module | Purpose | Key Class |
|--------|---------|-----------|
| `agent.py` | Main ReAct loop | `Agent` |
| `orchestrator.py` | Multi-model routing (single/ensemble/debate/supervisor) | `MultiModelOrchestrator` |
| `planner.py` | Long-horizon DAG decomposition | `LongHorizonPlanner` |
| `complexity.py` | Task complexity scoring | `classify_complexity()` |
| `reflexion.py` | Structured self-correction | `ReflexionEngine` |
| `reasoning.py` | Explicit reasoning traces | `ReasoningEngine` |

### Verification & Safety
| Module | Purpose | Key Class |
|--------|---------|-----------|
| `verdict.py` | PASS/FAIL/PARTIAL output verification | `VerdictEngine` |
| `guardrails.py` | Anti-pattern enforcement, delegation safety | `DelegationGuardrails` |
| `verifier_mesh.py` | URL/DOI, numeric bounds, NLI entailment | `VerifierMesh` |
| `verification.py` | Z3 SMT formal constraint checking | `FormalVerifier` |
| `validators.py` | UKB-specific domain validators | `UKBValidator` |
| `disclosure.py` | Progressive 4-layer result presentation | `ProgressiveDisclosure` |

### Memory & State
| Module | Purpose | Key Class |
|--------|---------|-----------|
| `memory.py` | 8-tier persistent memory system | `LongTermMemory` |
| `state.py` | Session runtime state (cohorts, models, records) | `SessionState` |
| `evidence.py` | Claim-evidence lattice with confidence | `EvidenceLattice` |

### LLM & Registry
| Module | Purpose | Key Class |
|--------|---------|-----------|
| `llm.py` | OpenAI-compatible client (OpenRouter) | `LLMClient` |
| `registry.py` | Skill registry with `@skill` decorator | `SkillRegistry` |
| `study_spec.py` | Schema-gated execution (query → typed spec) | `StudySpecCompiler` |
| `structured.py` | Optional Instructor-based typed outputs | `extract_structured()` |

### Data Layer
| Module | Purpose |
|--------|---------|
| `data/loader.py` | DuckDB + parquet data manager |
| `data/catalog.py` | Field metadata (11,821 UKB fields) |
| `data/cohort.py` | Cohort construction with criteria |
| `data/features.py` | Biomarker definitions |
| `data/parquet_builder.py` | CSV → Parquet conversion |

---

## Execution Flow

```
User Query
    │
    ├─▶ StudySpecCompiler.compile()  →  StudySpec (schema-gated boundary)
    │
    ├─▶ Agent.run() loop:
    │       │
    │       ├─ LLM decides tool calls
    │       ├─ SkillRegistry.execute(name, args, ctx)
    │       ├─ Result → AnalysisRecord → SessionState
    │       ├─ On error → ReflexionEngine → retry with mutations
    │       └─ Loop until final answer or max_rounds
    │
    ├─▶ VerdictEngine.judge(result)
    │       ├─ Formal checks (Z3 constraints)
    │       ├─ VerifierMesh (numeric bounds, URL/DOI, entailment)
    │       └─ Returns PASS / FAIL / PARTIAL
    │
    └─▶ ProgressiveDisclosure → formatted output
```

---

## Skill System

### Registration
```python
from biobank_agent.registry import skill

@skill(name="prevalence", description="...", parameters={...})
def prevalence(top_n: int = 20, *, ctx) -> dict:
    result = ctx.dm.query(...)
    return {"total": ..., "diseases": [...]}
```

### Discovery Flow
1. `autodiscover_skills()` imports all `biobank_agent.skills.*` modules
2. Each `@skill` decorator registers function + OpenAI schema
3. Optional: `discover_custom_skills()` loads from `./custom_skills/`
4. `registry.tool_schemas()` → sent to LLM as available tools

### Context Injection
Every skill receives a `ctx` object with:
- `ctx.dm` — DataManager (DuckDB connection)
- `ctx.state` — SessionState (cohorts, models, records)
- `ctx.memory` — LongTermMemory (8 tiers)
- `ctx.settings` — Configuration
- `ctx.catalog` — Field metadata
- `ctx.report_dir` — Output directory

---

## Multi-Model Orchestration

| Strategy | When | How |
|----------|------|-----|
| SINGLE | Complexity < 0.7 | Direct LLM call |
| ENSEMBLE | Moderate complexity | Parallel answers → consensus |
| DEBATE | High complexity / disagreement | Propose → critique → anonymous vote → adjudicate |
| SUPERVISOR | Long-horizon tasks | Planner decomposes → specialists execute → merge |

**Trigger**: `complexity_threshold` in config (default 0.7)  
**Rollback**: Gate failure → try ensemble → try single → return best available

---

## 8-Tier Memory System

| Tier | Name | Storage | Retention |
|------|------|---------|-----------|
| 1 | Short-term | RAM | Session only |
| 2 | Mid-term | RAM (AnalysisRecord) | Session only |
| 3 | Long-term | JSON file | Persistent |
| 4 | Error catalog | JSON nested | Persistent |
| 5 | Domain | Markdown | Persistent |
| 6 | User preferences | Markdown | Persistent |
| 7 | Episodic | SQLite FTS5 | Persistent |
| 8 | Action graph | SQLite | Persistent |

---

## Verification Pipeline

```
Skill Result
    │
    ├─▶ FormalVerifier (Z3)         → constraint satisfaction
    ├─▶ VerifierMesh                → numeric bounds + URL/DOI + NLI
    ├─▶ VerdictEngine               → PASS / FAIL / PARTIAL
    ├─▶ Guardrails                  → anti-patterns, data leakage
    └─▶ EvidenceLattice             → claim confidence tracking
```

---

## Key Design Principles

| Principle | Implementation |
|-----------|---------------|
| Skill-first | 58 self-contained skills + `@skill` decorator, hot-reloadable |
| Memory-driven | 8-tier persistent memory, cross-session learning |
| Verified outputs | Multi-layer: Reflexion → Verdict → Guardrails → VerifierMesh |
| Schema-gated | Natural language → StudySpec before execution |
| Progressive disclosure | Token-aware context injection, 4-layer output |
| Zero-copy data | DuckDB VIEWs over Parquet (columnar scan) |
| Graceful degradation | Optional deps (Z3, PaperQA2, Instructor) with fallbacks |
| Biobank-portable | All column names configurable, supports UKB/FinnGen/CKB |
