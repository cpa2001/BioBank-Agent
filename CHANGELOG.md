# Changelog

All notable changes to Biobank Agent are documented here.

## [2.3.0] — 2026-05-08

### Added
- **Target Annotation Context Skill**: `target_annotation_context` adds biobank-scoped translational context from Open Targets, UniProt, GTEx, ClinicalTrials.gov, and optional CELLxGENE snapshots
- **Target Enrichment Skill**: `target_enrichment` runs local GMT over-representation analysis, with optional GSEApy support when installed
- **External Context Cache**: Source-level JSON caching, cache-only mode, and source-specific failure isolation keep annotation calls reproducible and non-blocking
- **Agent Governance Integration**: `generate_report`, `statistical_review`, and `safety_check` now distinguish annotation/enrichment context from genetic or causal evidence
- **Action Graph Provenance**: Annotation sources, target context, enrichment terms, and overlap genes are recorded as typed graph evidence
- 12 new tests (1098 total tests passing)

### Changed
- External annotation and enrichment are now first-class biobank target interpretation workflows; they do not alter rare-variant or GWAS-derived target rankings

## [2.2.0] — 2026-05-08

### Added
- **Genetic Target Hypothesis Skill**: `genetic_target_hypothesis` ranks therapeutic target hypotheses from GeneBass-like rare-variant burden summary statistics
- **Genetics-First Target Prioritization**: Composite scoring with loss-of-function therapeutic direction, pLoF/missense concordance, pathway convergence, Bonferroni and Benjamini-Yekutieli tiers
- **Target Hypothesis Artifacts**: Companion Markdown and CSV outputs for ranked gene cards, validation caveats, and next-step triage
- **Agent Governance Integration**: `generate_report`, `statistical_review`, and `safety_check` now understand genetic target hypothesis records
- **Target Hypothesis Hardening**: labelled burden tables now block phenotype mismatches, unlabelled rows require explicit `prefiltered=true`, and multiple-testing scope is reported
- **Burden Evidence Provenance**: Action Graph now records row-level rare-variant burden evidence linked to each target hypothesis
- 12 new tests (1086 total tests passing)

### Changed
- Documentation and plugin guidance now expose rare-variant burden target prioritization as a first-class workflow

## [2.1.0] — 2026-05-08

### Added
- **Schema-Gated Execution**: `StudySpecCompiler` compiles natural language queries into typed `StudySpec` before execution (Pydantic v2)
- **Verifier Mesh**: Multi-strategy verification — URL/DOI resolution, numeric range checking (UKB bounds), NLI claim-evidence entailment
- **Evidence Lattice**: Claim-level provenance tracking with `EvidenceNode`, `ClaimRecord`, confidence computation (supports/refutes weighting)
- **Formal Verification**: Z3 SMT solver integration for constraint satisfaction checking (optional dependency)
- **Constants Module**: Single source of truth for UKB domain constants (502,411 participants, age 37–73, etc.)
- **Difficulty-Aware Routing**: DAAO-inspired learned routing for task complexity
- **Reproducibility Harness**: NeuroClaw-inspired SHA-256 checkpoint verification
- **Progressive Disclosure**: 4-layer result presentation (headline → summary → detail → raw)
- **Structured Outputs**: Optional Instructor integration for Pydantic-enforced LLM outputs
- **Literature QA Skill**: PaperQA2-backed citation-first literature search (optional dependency)
- **External Agents**: Safe subprocess runner for `codex` and `claude` CLI tools
- **Project Documentation Skill**: `project_doc` lists, searches, and reads curated repository Markdown for agent-visible data, guide, architecture, and plugin docs
- **GraphPop MCP Client**: 21-tool registry for population-genomics graph queries (optional)
- **Validators Module**: UKB-specific domain validation (field ranges, ICD-10 format, cohort bounds)
- **Temporal Safety Rules**: LTL-inspired precedence checks in planner decomposition
- 49 new test files (1074 total tests passing)

### Changed
- Planner accepts optional `StudySpec` for skill constraint and tool budget enforcement
- Verdict engine integrates Verifier Mesh as Phase 0.5 after formal checks
- Memory system extended with evidence recording methods
- 8-tier memory → action graph now links to Evidence Lattice
- Documentation consolidated: 23 files → 13 files across 6 logical directories

### Fixed
- Constant mismatch (502,536 vs 502,411) resolved via authoritative `constants.py`
- Non-atomic file writes in Evidence Lattice (now uses tmp + rename)
- Duplicate verification issues from overlapping numeric checks (partitioned responsibility)
- LLM `model_name` AttributeError (uses `getattr(llm, "model", "gpt-4")`)
- Template shallow-copy mutation (now uses `copy.deepcopy()`)
- asyncio deprecated `get_event_loop()` replaced with `get_running_loop()` idiom

## [2.0.0] — 2026-04-22

### Added
- **Scientific Discovery Pipeline**: `discover`, `predict`, `gwas_proxy` skills for automated biomarker discovery
- **Literature Research**: `web_search`, `web_fetch`, `read_pdf`, `fetch_paper`, `read_paper`, `deep_research` skills
- **Scientific Writing**: `nature_writer` for Nature-quality manuscript sections
- **Research Ideation**: `brainstorm`, `critical_thinking` skills for hypothesis generation
- **Plan Mode**: Structured planning workflow (INTAKE → ALIGNMENT → EXECUTION → DONE)
- **Smart Plotting**: `smart_plot` with Nature/ICML/NEJM/Lancet style presets
- **CLI Commands**: 20 slash commands (`/plan`, `/compact`, `/clear`, `/cost`, `/export`, `/help`, etc.)
- **Token Tracking**: Cumulative prompt/completion token counting
- **Hot-Reload Skills**: `custom_skills/` auto-discovery, in-session skill activation
- **Report Dual Format**: Technical report (`format='report'`) and IMRaD paper draft (`format='paper'`)
- **SVG/PDF Figures**: Publication-quality vector output as default
- **ICML Figure Style**: `icml_figure()` for ML conference formatting
- **Panel Labels**: `add_panel_labels()` for multi-panel figures
- New dependencies: pymupdf, httpx, html2text, beautifulsoup4, duckduckgo-search

### Changed
- Renamed CLI entry point to `biobank` (legacy alias removed)
- Report system rewritten with interpretive text and Key Findings
- Plotting defaults to SVG+PDF instead of PNG+PDF
- `create_skill` now auto-activates via `custom_skills/` hot-reload
- Enhanced `registry.py` with `reload_skill()`, `unregister()`, `discover_custom_skills()`

### Fixed
- Correlation skill now uses `save_figure()` instead of manual `savefig()`

## [1.0.0] — 2026-04-21

### Added
- Initial release: 28 skills, DuckDB data layer, ReAct agent loop
- Parquet rebuild from 484 to 4,971 field IDs
- LLM client with retry logic and exponential backoff
- Self-evolution: error catalog, pipeline recording, skill generation
- Nature-style plotting (300 DPI, Okabe-Ito palette)
- 4-tier memory system
- 164 unit tests
