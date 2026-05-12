# Changelog

All notable changes to Biobank Agent are documented here.

## [3.0.0-rc1] — 2026-05-12

> Release candidate cut from the v3-foundation tree. Local 12 of 17 v3-completion
> criteria pass; 5 remain `BLOCKED_EXTERNAL` because they require institutional
> credentials or remote-CI evidence (see `docs/architecture/V3_RELEASE_NOTES.md`).
> The final `v3.0` tag will only be cut when remaining `BLOCKED_EXTERNAL`
> criteria close per the release-tag roadmap in the plan file.

### Updated after UKB full-data audit
- The local UKB readiness path now includes the configured `settings.data_dir`
  fallback, so a real UKB-only `bank_data_readiness` artifact can close
  criterion 13a. With that artifact present, the audit should report 13 of 17
  criteria passing and 4 still `BLOCKED_EXTERNAL` (`13b`, `14`, `15`, `16`).
- Added full-UKB raw CSV inventory and selected-field materialization support:
  `ukb_data_inventory`, `ukb_field_resolve`, `ukb_materialize_fields`, and
  `biobank build-ukb-full-parquet`. These expose `/Users/chenpengan/Projects/CUHK/UKB`
  without replacing the existing Milton parquet subset.

### Added
- **v3-completion audit criterion 13 split**: the previous `credentialed_hpp_ckb_rap_data` criterion has been split into:
  - `credentialed_ukb_data` (criterion 13a) — UKB readiness must `PASS` for the final `v3.0` tag. Run `biobank-agent skill bank_data_readiness --banks ukb` against the configured local UKB data dir, then point `--hpp-ckb-rap-readiness` at the produced JSON.
  - `credentialed_hpp_ckb_rap_data` (criterion 13b) — HPP/CKB/RAP readiness; explicitly scoped to the v3.1 milestone so v3.0 can ship without indefinitely waiting on institutional data access. The audit still reports it as `BLOCKED_EXTERNAL` until evidence arrives.
- **Worker content-quality audit hardening** (`biobank_agent/eval/v3_completion.py:_check_live_run` + new `_worker_has_content`): the live-benchmark audit no longer accepts empty `worker-NN/` directories as evidence. Every existing worker dir must demonstrate ≥10-line transcript or declared `report_dirs` whose `report*.md` exists and is non-empty; missing per-worker audit `report_dirs` is treated as legitimate "no-report query" only when the per-worker audit explicitly recorded `status=PASS`.
- **Compatibility matrix** (`docs/architecture/COMPATIBILITY.md`): Tier 1 / Tier 2 / Not supported across Python, OS, Textual, openai SDK, and MCP transports.
- **v3.0 release notes** (`docs/architecture/V3_RELEASE_NOTES.md`).

### Known limitations (will close in rc2 or v3.1)
- 5 of 17 v3-completion criteria are `BLOCKED_EXTERNAL` at rc1: `credentialed_ukb_data` (13a, requires local UKB readiness artifact), `credentialed_hpp_ckb_rap_data` (13b, **v3.1 milestone**), `real_mcp_compatibility_matrix` (14, targeted for rc2 via filesystem + github MCP STDIO), `remote_ci_scheduled_eval` (15, targeted for rc2 via `gh workflow run`), `credentialed_high_risk_pr_path` (16, targeted for rc2 on an owned repo).
- MCP HTTP/SSE transport is **protocol-implemented and locally tested** but no production HTTP/SSE server is part of v3.0 compatibility evidence. Treat HTTP/SSE as a v3.1 target.
- Paper-replication is wired end-to-end (read_paper → study-spec extract → planner → dual report → `paper_replication_compare` acceptance gates) with the MILTON DOI `10.1038/s41588-024-01898-1` fixture, but this verifies the **tool chain**, not a scientific reproduction of MILTON's 1091-disease AUC distribution. Numeric/figure diffs and additional paper fixtures are v3.2 work.
- `biobank_agent/cli_legacy.py` is the compatibility entry until `cli/`-package decomposition completes in v3.2.

## [2.4.0] — 2026-05-11

### Added
- **v3 Foundation Runtime**: Async event-stream runtime, streaming renderer, and Textual TUI scaffold with plan/tool/disclosure panels.
- **Plan/Report Hard Gates**: Plan approval now requires schema-valid skill calls; failed skills, blocked dependencies, and missing `generate_report(format="dual")` artifacts pause or fail execution.
- **MCP Manager**: STDIO and HTTP/SSE JSON-RPC MCP discovery with `/mcp-list`, `/mcp-start`, `/mcp-health`, `/mcp-call`, `/mcp-stop`, startup retry/backoff, call-failure reconnect, status/error reporting, safe tool-name normalization, and legacy registry bridging.
- **Bank Adapter Main Path**: HPP ICD9/ICD10 and CKB native diagnosis columns are wired into cohort/model code-prefix paths for synthetic cross-bank fixtures.
- **Paper Replication Scaffold**: Review-only paper replication StudySpec/plan artifacts plus table/figure target linking in generated reports.
- **Self-Evolution Gate**: LOW generated-skill proposals require allow-listed targets and ALWAYS_PASSES eval success before application.
- **Strict Live Artifact Audit**: `biobank eval --suite live_artifacts --run-dir <run> --enforce-gate` validates human-style benchmark artifacts for dual reports, MILTON paper acceptance gates, and trajectory token/world-model claim boundaries.
- **v3 Completion Audit**: `biobank eval --suite v3_completion --run-dir <run>` maps the full v3 objective to concrete evidence and reports externally blocked credentialed items separately from local failures.
- **OpenTelemetry Bridge**: Optional event-to-span tracing with low-cardinality metadata only, configurable `none`, `console`, and OTLP/HTTP exporters, a Jaeger runbook, and code-level `otel_policy=production` collector validation; raw queries, arguments, results, per-turn ids, timestamps, and participant identifiers are not recorded.
- **Examples and Verification Docs**: `docs/architecture/V3.md` and `docs/examples/` define current scope, manual plan testing, SDK streaming, MCP smoke testing, and paper-replication fixture workflows.

### Changed
- Data-analysis skills default to full eligible data instead of fixed small samples where the local de-identified dataset can support full execution.
- `web_search` uses `ddgs` first and suppresses the legacy `duckduckgo_search` rename warning.
- Generated code remains review-only by default under `reports/generated_skills/`.

### Fixed
- Long `/plan` runs can no longer display success solely from step accounting when final report files are missing.
- MCP server failures now record actionable status instead of silently looking unavailable.
- MCP STDIO calls now have hard timeouts to avoid CLI hangs on malformed servers.

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
