# Changelog

All notable changes to Biobank Agent are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0-rc1] - 2026-06-11

The v3.1 release candidate adds a capability-acquisition layer on top of the
v3 trustworthy-instrument foundation: a hierarchical skill tree, omics-aware
methodology gates, demand-driven paper→skill synthesis, knowledge-only
external skill ingestion, an online-eval run-tree, an alternative planning
pipeline, and a substantial round of agent-autonomy and interactive-CLI
robustness fixes. Each behavior-changing addition is behind a default-OFF
flag, so the legacy v3.0 plan/report flows remain byte-identical when the
flags are off. Driven by end-to-end testing of the live `biobank` CLI against
a configured LLM, every fix in this release is root-caused from an observed
failure and verified.

### Added
- Hierarchical skill tree in `skills/manifest.json` (22 nodes, every skill
  placed once) with `navigate_tree` / `classify_skill` helpers, a
  registry subtree filter, a `navigate_skill_tree` native tool, and a
  tree-aware executor prompt behind `skill_tree_enabled`; output is
  byte-identical when no tree is present
  (`biobank_agent/skills/skill_tree.py`).
- Method-contract reviewer extended with single-cell / spatial sin checks
  (`pseudoreplicated_de`, `velocity_without_splicing`,
  `batch_confounded_clustering`, `spatial_enrichment_no_null`,
  `coloc_unharmonized_alleles`) plus `check_artifact` for AnnData-style
  postcondition validation, wired into the completion gate behind
  `methodology_gate_enabled` (default OFF)
  (`biobank_agent/runtime/methodology.py`).
- Live `synthesize_skill_from_paper` skill — gated paper→contract→review-branch
  pipeline that pre-checks the methodology gate, applies through the
  transactional review-branch worktree, and classifies the result into the
  skill tree; OFF by default behind `skill_synthesis_enabled`
  (`biobank_agent/runtime/skill_from_paper.py`,
  `biobank_agent/runtime/self_evolve.py`).
- Online-eval run-tree folded into `audit_session()`, plus a `/trace`
  slash command and renderer that produces non-empty trees on real sessions
  (`biobank_agent/runtime/run_tree.py`,
  `biobank_agent/runtime/run_eval.py`,
  `biobank_agent/cli/commands/runtime.py`).
- Alternative planning pipeline behind `adversarial_council_enabled`
  (default OFF). When enabled, planning can route through a proposer/critic
  pipeline with bounded per-model timeouts and an A/B harness that compares
  it against the default planner; behavior is unchanged when the flag is OFF
  (`biobank_agent/runtime/adversarial_council.py`,
  `biobank_agent/runtime/council_ab.py`).
- Knowledge-only external skill ingestion: parses each upstream `SKILL.md`
  and registers it as a deferred, trust-`external` knowledge skill — third
  party code is never written to disk or executed. The repo-URL validator
  requires an immutable commit SHA. OFF by default behind
  `external_skill_ingestion_enabled`
  (`biobank_agent/runtime/skill_ingest.py`,
  `biobank_agent/skills/ingest_skills.py`).
- Load-time skill-execution safety gate: `validate_load_safety` is enforced
  at custom-skill discovery, blocking module-scope `shell_exec`, call-of-call
  patterns, and read-sink obfuscations. `manifest.json` gained a `trust` map
  so `trust='external'` skills are excluded from curator auto-promotion
  (`biobank_agent/registry.py`, `biobank_agent/skills/manifest.py`).
- Autonomous patch generation that closes the self-evolution loop — the
  planner LLM converts an `EvolutionProposal` plus evidence into a concrete
  `{target_path, diff, test_commands}` and routes it through the existing
  transactional apply loop, so `learn → generate → test-gate → apply` is a
  real end-to-end pipeline. The generated patch is untrusted; safety is
  enforced by the isolated worktree, authoritative changed-path validation,
  the allow-list, the test gate, and rollback
  (`biobank_agent/runtime/patch_generation.py`).
- Transactional, test-gated self-evolution: a proposed patch is applied in
  an isolated git worktree, its tests run there, and it is committed only
  if they pass — with automatic rollback on failure. Hardened against
  path-traversal (every actually-changed file is re-validated from
  `git status`, not the diff text), arbitrary test-command execution
  (test gates must be `pytest`/`ruff`/`mypy`/... runners; no bare
  `python -c`, no shell metacharacters, secret-stripped env), and
  dirty-tree merges. Edits to `biobank_agent/core`, `biobank_agent/runtime`,
  or `tests/` are never auto-merged
  (`biobank_agent/runtime/self_evolve.py`).
- Real shell capability for native tools: `shell`/`test_runner` run through
  `bash -c` so pipes, redirects, globs, and `&&`/`||` work
  (`biobank_agent/core/tools/native.py`).
- Multi-model planning council — `/plan` now runs a parallel
  clarification → N candidate drafts (with distinct provider roles) →
  parallel critics → merge → schema validation → review pipeline over the
  runtime `ProviderRouter`. The dashboard surfaces per-model live status
  and per-round revisions; surviving drafts debate over bounded rounds and
  drop low-confidence or homogeneous plans
  (`biobank_agent/runtime/council.py`,
  `biobank_agent/runtime/planner.py`).
- Unified `/research` mode that reuses the council core for a multi-source
  cited research brief: sub-query decomposition → parallel `deep_research`
  retrieval → synthesis with citation sanitization → verifier-mesh claim
  checks (`biobank_agent/runtime/researcher.py`).
- Completion gates — `/verify` assesses plan completion with
  report-contract enforcement, an LLM goal-acceptance judge that never
  auto-accepts on unparseable/unavailable output, completion warnings, and
  bounded LLM-proposed repair through the existing repair recorder
  (`biobank_agent/runtime/completion.py`).
- Multi-model live dashboard with one live row per model, a compact phase
  pipeline, recent-milestones log, height budget against the terminal,
  markup-escaped model-derived text, and per-subagent terminal events on
  council timeout so live timers stop cleanly
  (`biobank_agent/progress.py`).
- Live token streaming into the dashboard via a
  `ProviderRequest.stream_cb` runtime channel (never recorded as events,
  never logged). Streamed `usage` is captured via
  `stream_options.include_usage` with a creation- and iteration-time
  fallback for providers that reject it; orphaned post-timeout deltas are
  dropped through a per-stage cancel plus the global `cancel_event`
  (`biobank_agent/llm.py`, `biobank_agent/progress.py`).
- Doc-consistency test suite that pins README skill/command counts and the
  `pyproject.toml`↔package version (`tests/test_doc_consistency.py`).

### Changed
- `RuntimeConfig.from_settings` now mirrors `methodology_gate_enabled` and
  `adversarial_council_enabled` so flags configured via `.env` actually
  reach the runtime (both were previously unreachable)
  (`biobank_agent/runtime/types.py`).
- `gate_test_source()` is self-contained (no `biobank_agent` import) so
  paper-synthesis and external-skill-ingestion gate tests pass inside the
  secret-stripped, isolated apply worktree
  (`biobank_agent/runtime/self_evolve.py`).
- Single interaction mode: `biobank` (optionally `biobank "<task>"`) is the
  only way to start the agent. The former `--tui` and `--legacy-repl`
  dispatch modes were removed.
- The plan-time dashboard wall is trimmed (preamble panels and a redundant
  third steps panel are dropped) and the Workflow Diagram is shown once
  rather than at draft, approval, and view.
- Data-analysis skills default to full eligible data instead of fixed small
  samples where the local de-identified dataset can support full execution.

### Fixed
- Executor now carries an identity + workspace-sandbox + tool-use-discipline
  system prompt and compact prior-turn history into every turn, so the
  agent no longer rewrites the same file in a loop on speculative tool
  calls (`biobank_agent/runtime/engine.py`).
- Multi-step tool use is no longer broken by malformed assistant
  `tool_calls`: the wire format is normalised at the API boundary in
  `LLMClient.sanitize_messages`, so providers see the OpenAI-compatible
  `{id,type,function:{name,arguments}}` shape and tool results are no
  longer orphaned (`biobank_agent/llm.py`).
- A turn is no longer marked FAILED when any tool errored — turn status is
  now convergence-based (COMPLETED iff the model reached a final answer
  and the provider didn't fail).
- `run_turn` catches provider failures and records an ERROR instead of
  letting the exception escape.
- Runaway tool loops are now bounded by a consecutive no-progress guard
  (soft nudge at 3, hard stop at 6) that does not penalise legitimate
  distinct progress.
- Shell and `test_runner` nonzero exits are surfaced as real errors so the
  loop guard / audit / completion feedback see real failures.
- Workspace-denial messages are now actionable (they tell the model to use
  a path inside the workspace) so the agent adapts instead of retrying a
  rejected path.
- `/plan` no longer times out on slow reasoning models — council timeout
  raised to 360 seconds; `/plan` and `/research` now complete cleanly.
- Constraint verification merges the formal Z3 consistency proof and
  Minimal Correction Subset on top of the descriptive bounds checks, so
  callers always get specific, actionable diagnostics whether or not the
  Z3 solver is installed (`biobank_agent/verification.py`).
- Project-doc search now uses BM25 length-normalised relevance with
  full-coverage and exact-phrase bonuses, replacing raw term-frequency
  scoring that buried short focused guides under long architecture dumps
  (`biobank_agent/skills/project_doc.py`).
- Curated-doc title redaction is now structural at the source in
  `_title_for`, so a heading that contained a credential term can no
  longer leak through the `title` field of list/search/read
  (`biobank_agent/skills/project_doc.py`).
- Interactive menu arrow keys are usable again — `_read_single_key` reads
  raw bytes via `os.read`, parses CSI/SS3 sequences precisely, drains
  bracketed-paste payloads, uses `tty.setcbreak`, confirms the highlighted
  row on Enter, and redraws a single in-place Rich `Live` surface
  (`biobank_agent/cli/interactive.py`).
- `/plan` no longer crashes the CLI with an uncaught `KeyboardInterrupt`
  on menu open — the menu `Live` no longer redirects stdout/stderr, and
  `run()` contains a `KeyboardInterrupt` raised inside command handling so
  a Ctrl-C during a command prints `Cancelled` and returns to the prompt
  (`biobank_agent/cli/interactive.py`).
- `/plan` no longer crashes the CLI with a Rich `MarkupError` — every
  dynamic-style renderable goes through a `_styled` helper that only emits
  a tag when the style is non-empty, status icon/style maps cover the
  `error`/`cancelled` cases, the daemon refresh wraps `live.update` in
  try/except, and every markup `console.print` that interpolates
  exception/user/LLM text now escapes it with `_rich_escape`
  (`biobank_agent/progress.py`, `biobank_agent/cli/interactive.py`).
- Clarification questions are no longer silently dropped for many models —
  the parser accepts the question list under several keys (or a bare
  top-level list), each option as a string or a dict (label under
  `label`/`text`/`value`/`name`/...), and the question text under
  `question`/`text`/`prompt`/`title` (`biobank_agent/runtime/planner.py`).
- Clarification menus accept free-text answers and inline editing — every
  menu gains a "Type a custom answer" row that opens a free-text prompt
  inline inside the panel; Tab on any option opens the same editor
  pre-filled with that row's label. Implementation is opt-in via an
  `edit_handler` so non-text-capable menus are unaffected
  (`biobank_agent/cli/interactive.py`).
- Approving a plan now executes it autonomously with a live progress view.
  Approval restores the user's real permission profile and runs every
  step through `run_turn` in dependency order under a single live
  `PlanRunDashboard` (or a line-logger on a non-TTY)
  (`biobank_agent/cli/interactive.py`).
- Clarification menu windowing keeps the selected row on screen on short
  terminals, the last option remains visible at the boundary, and CJK
  glyphs no longer cause Rich to undercount lines and leave stale
  highlights (`biobank_agent/cli/interactive.py`).

### Removed
- The external CLI agent bridge and its `/external-agents`,
  `/codex-plan`, `/codex-check`, `/claude-plan`, `/claude-check`,
  `/gemini-plan`, and `/gemini-check` slash commands. Equivalent
  cross-tool integration is now provided through the MCP manager
  (`/mcp-list`, `/mcp-start`, `/mcp-health`, `/mcp-call`, `/mcp-stop`).
- A contributor-only git-history helper skill that automated a clean
  force-push workflow; removed in favour of running git directly.

### Security
- Secrets are no longer leakable through the file/shell tools: redaction
  masks secret VALUES (`KEY=...` → `KEY=[REDACTED]`, plus `sk-` / `ghp_` /
  bearer tokens), and `file_read` refuses `.env`, `.pem`, `.key`,
  `id_rsa`, and `credentials` files; `test_runner` runs with a
  secret-stripped env (`biobank_agent/core/tools/native.py`).
- The self-evolution collect-only ban is hardened across every channel
  so an untrusted generated patch can no longer auto-merge a
  body-skipping (collect-only) test. Specifically: pytest gate commands
  get `-o addopts=` appended (last-wins, neutralising the
  `pytest.ini`/`pyproject.toml`/command-line addopts channels);
  `_safe_env()` strips `PYTEST_ADDOPTS`/`PYTEST_PLUGINS`; any patch that
  creates or edits a pytest config or hook file (`pytest.ini`,
  `tox.ini`, `setup.cfg`, `pyproject.toml`, `conftest.py`, ...) is never
  auto-merged. Regression tests reproduce each attack
  (`biobank_agent/runtime/self_evolve.py`).

## [3.0.0-rc1] - 2026-05-12

Release candidate cut from the v3-foundation tree. Local 12 of 17
v3-completion criteria pass; 5 remain `BLOCKED_EXTERNAL` because they
require institutional credentials or remote-CI evidence (see
`docs/architecture/V3_RELEASE_NOTES.md`). The final `v3.0` tag will only
be cut when remaining `BLOCKED_EXTERNAL` criteria close.

### Added
- v3-completion audit criterion 13 split: `credentialed_ukb_data` (13a;
  must `PASS` for the final `v3.0` tag) and `credentialed_hpp_ckb_rap_data`
  (13b; explicitly scoped to v3.1 so v3.0 can ship without waiting on
  institutional data access).
- Worker content-quality audit hardening so the live-benchmark audit no
  longer accepts empty `worker-NN/` directories as evidence; every
  existing worker dir must demonstrate >=10-line transcript or declared
  `report_dirs` whose `report*.md` exists and is non-empty
  (`biobank_agent/eval/v3_completion.py`).
- Compatibility matrix (`docs/architecture/COMPATIBILITY.md`): Tier 1 /
  Tier 2 / Not supported across Python, OS, Textual, openai SDK, and MCP
  transports.
- v3.0 release notes (`docs/architecture/V3_RELEASE_NOTES.md`).

### Changed
- The local UKB readiness path includes the configured
  `settings.data_dir` fallback so a real UKB-only `bank_data_readiness`
  artifact can close criterion 13a.
- Added full-UKB raw CSV inventory and selected-field materialization
  support: `ukb_data_inventory`, `ukb_field_resolve`,
  `ukb_materialize_fields`, and `biobank build-ukb-full-parquet`. These
  expose the local UKB raw data without replacing the existing Milton
  parquet subset.

### Known limitations
- 5 of 17 v3-completion criteria are `BLOCKED_EXTERNAL` at rc1:
  `credentialed_ukb_data` (13a, requires local UKB readiness artifact),
  `credentialed_hpp_ckb_rap_data` (13b, v3.1 milestone),
  `real_mcp_compatibility_matrix` (14, targeted for rc2 via filesystem
  and GitHub MCP STDIO), `remote_ci_scheduled_eval` (15, targeted for
  rc2 via `gh workflow run`), `credentialed_high_risk_pr_path` (16,
  targeted for rc2 on an owned repo).
- MCP HTTP/SSE transport is protocol-implemented and locally tested but
  no production HTTP/SSE server is part of v3.0 compatibility evidence.
  Treat HTTP/SSE as a v3.1 target.
- Paper replication is wired end-to-end (read_paper → study-spec extract
  → planner → dual report → `paper_replication_compare` acceptance
  gates) with the MILTON DOI `10.1038/s41588-024-01898-1` fixture, but
  this verifies the tool chain, not a scientific reproduction of MILTON's
  1091-disease AUC distribution. Numeric/figure diffs and additional
  paper fixtures are v3.2 work.
- `biobank_agent/cli_legacy.py` is the compatibility entry until the
  `cli/`-package decomposition completes in v3.2.

## [2.4.0] - 2026-05-11

### Added
- v3 Foundation Runtime: async event-stream runtime, streaming renderer,
  and Textual TUI scaffold with plan/tool/disclosure panels.
- Plan/Report Hard Gates: plan approval requires schema-valid skill
  calls; failed skills, blocked dependencies, and missing
  `generate_report(format="dual")` artifacts pause or fail execution.
- MCP Manager: STDIO and HTTP/SSE JSON-RPC discovery with `/mcp-list`,
  `/mcp-start`, `/mcp-health`, `/mcp-call`, `/mcp-stop`, startup
  retry/backoff, call-failure reconnect, status/error reporting, safe
  tool-name normalisation, and legacy registry bridging.
- Bank adapter main path: HPP ICD9/ICD10 and CKB native diagnosis columns
  are wired into cohort/model code-prefix paths for synthetic cross-bank
  fixtures.
- Paper replication scaffold: review-only paper replication
  StudySpec/plan artifacts plus table/figure target linking in generated
  reports.
- Self-evolution gate: LOW generated-skill proposals require allow-listed
  targets and ALWAYS_PASSES eval success before application.
- Strict live artifact audit:
  `biobank eval --suite live_artifacts --run-dir <run> --enforce-gate`
  validates human-style benchmark artifacts for dual reports, MILTON
  paper acceptance gates, and trajectory token / world-model claim
  boundaries.
- v3 completion audit:
  `biobank eval --suite v3_completion --run-dir <run>` maps the full v3
  objective to concrete evidence and reports externally blocked
  credentialed items separately from local failures.
- OpenTelemetry bridge: optional event-to-span tracing with
  low-cardinality metadata, configurable `none`, `console`, and
  OTLP/HTTP exporters, a Jaeger runbook, and a code-level
  `otel_policy=production` collector validator. Raw queries, arguments,
  results, per-turn ids, timestamps, and participant identifiers are not
  recorded.
- Examples and verification docs: `docs/architecture/V3.md` and
  `docs/examples/` define current scope, manual plan testing, SDK
  streaming, MCP smoke testing, and paper-replication fixture workflows.

### Changed
- Data-analysis skills default to full eligible data instead of fixed
  small samples where the local de-identified dataset can support full
  execution.
- `web_search` uses `ddgs` first and suppresses the legacy
  `duckduckgo_search` rename warning.
- Generated code remains review-only by default under
  `reports/generated_skills/`.

### Fixed
- Long `/plan` runs can no longer display success solely from step
  accounting when final report files are missing.
- MCP server failures now record actionable status instead of silently
  looking unavailable.
- MCP STDIO calls now have hard timeouts to avoid CLI hangs on
  malformed servers.

## [2.3.0] - 2026-05-08

### Added
- Target annotation context skill: `target_annotation_context` adds
  biobank-scoped translational context from Open Targets, UniProt, GTEx,
  ClinicalTrials.gov, and optional CELLxGENE snapshots.
- Target enrichment skill: `target_enrichment` runs local GMT
  over-representation analysis, with optional GSEApy support when
  installed.
- External context cache: source-level JSON caching, cache-only mode,
  and source-specific failure isolation keep annotation calls
  reproducible and non-blocking.
- Agent governance integration: `generate_report`,
  `statistical_review`, and `safety_check` now distinguish
  annotation/enrichment context from genetic or causal evidence.
- Action graph provenance: annotation sources, target context,
  enrichment terms, and overlap genes are recorded as typed graph
  evidence.
- 12 new tests (1098 total tests passing).

### Changed
- External annotation and enrichment are now first-class biobank
  target-interpretation workflows; they do not alter rare-variant or
  GWAS-derived target rankings.

## [2.2.0] - 2026-05-08

### Added
- Genetic target hypothesis skill: `genetic_target_hypothesis` ranks
  therapeutic target hypotheses from GeneBass-like rare-variant burden
  summary statistics.
- Genetics-first target prioritisation: composite scoring with
  loss-of-function therapeutic direction, pLoF/missense concordance,
  pathway convergence, Bonferroni and Benjamini-Yekutieli tiers.
- Target hypothesis artifacts: companion Markdown and CSV outputs for
  ranked gene cards, validation caveats, and next-step triage.
- Agent governance integration: `generate_report`,
  `statistical_review`, and `safety_check` now understand genetic
  target hypothesis records.
- Target hypothesis hardening: labelled burden tables block phenotype
  mismatches, unlabelled rows require explicit `prefiltered=true`, and
  multiple-testing scope is reported.
- Burden evidence provenance: Action Graph records row-level
  rare-variant burden evidence linked to each target hypothesis.
- 12 new tests (1086 total tests passing).

### Changed
- Documentation and plugin guidance expose rare-variant burden target
  prioritisation as a first-class workflow.

## [2.1.0] - 2026-05-08

### Added
- Schema-gated execution: `StudySpecCompiler` compiles natural-language
  queries into a typed `StudySpec` before execution (Pydantic v2).
- Verifier mesh: multi-strategy verification — URL/DOI resolution,
  numeric range checking (UKB bounds), and NLI claim-evidence
  entailment.
- Evidence lattice: claim-level provenance tracking with
  `EvidenceNode`, `ClaimRecord`, and confidence computation
  (supports/refutes weighting).
- Formal verification: Z3 SMT solver integration for constraint
  satisfaction checking (optional dependency).
- Constants module: single source of truth for UKB domain constants
  (502,411 participants, age 37–73, etc.).
- Difficulty-aware routing: DAAO-inspired learned routing for task
  complexity.
- Reproducibility harness: NeuroClaw-inspired SHA-256 checkpoint
  verification.
- Progressive disclosure: 4-layer result presentation (headline →
  summary → detail → raw).
- Structured outputs: optional Instructor integration for
  Pydantic-enforced LLM outputs.
- Literature QA skill: PaperQA2-backed citation-first literature search
  (optional dependency).
- Project documentation skill: `project_doc` lists, searches, and reads
  curated repository Markdown for agent-visible data, guide,
  architecture, and plugin docs.
- GraphPop MCP client: 21-tool registry for population-genomics graph
  queries (optional).
- Validators module: UKB-specific domain validation (field ranges,
  ICD-10 format, cohort bounds).
- Temporal safety rules: LTL-inspired precedence checks in planner
  decomposition.
- 49 new test files (1074 total tests passing).

### Changed
- Planner accepts an optional `StudySpec` for skill constraint and tool
  budget enforcement.
- Verdict engine integrates the verifier mesh as Phase 0.5 after the
  formal checks.
- Memory system extended with evidence recording methods.
- 8-tier memory → action graph now links to the evidence lattice.
- Documentation consolidated: 23 files → 13 files across 6 logical
  directories.

### Fixed
- Constant mismatch (502,536 vs 502,411) resolved via authoritative
  `constants.py`.
- Non-atomic file writes in the evidence lattice (now uses tmp +
  rename).
- Duplicate verification issues from overlapping numeric checks
  (partitioned responsibility).
- LLM `model_name` AttributeError (uses
  `getattr(llm, "model", "gpt-4")`).
- Template shallow-copy mutation (now uses `copy.deepcopy()`).
- Deprecated `asyncio.get_event_loop()` replaced with
  `get_running_loop()` idiom.

## [2.0.0] - 2026-04-22

### Added
- Scientific discovery pipeline: `discover`, `predict`, `gwas_proxy`
  skills for automated biomarker discovery.
- Literature research: `web_search`, `web_fetch`, `read_pdf`,
  `fetch_paper`, `read_paper`, `deep_research` skills.
- Scientific writing: `nature_writer` for Nature-quality manuscript
  sections.
- Research ideation: `brainstorm`, `critical_thinking` skills for
  hypothesis generation.
- Plan mode: structured planning workflow (INTAKE → ALIGNMENT →
  EXECUTION → DONE).
- Smart plotting: `smart_plot` with Nature/ICML/NEJM/Lancet style
  presets.
- CLI commands: 20 slash commands (`/plan`, `/compact`, `/clear`,
  `/cost`, `/export`, `/help`, etc.).
- Token tracking: cumulative prompt/completion token counting.
- Hot-reload skills: `custom_skills/` auto-discovery, in-session skill
  activation.
- Report dual format: technical report (`format='report'`) and IMRaD
  paper draft (`format='paper'`).
- SVG/PDF figures: publication-quality vector output as default.
- ICML figure style: `icml_figure()` for ML conference formatting.
- Panel labels: `add_panel_labels()` for multi-panel figures.
- New dependencies: pymupdf, httpx, html2text, beautifulsoup4,
  duckduckgo-search.

### Changed
- Renamed CLI entry point to `biobank` (legacy alias removed).
- Report system rewritten with interpretive text and Key Findings.
- Plotting defaults to SVG+PDF instead of PNG+PDF.
- `create_skill` now auto-activates via `custom_skills/` hot-reload.
- Enhanced `registry.py` with `reload_skill()`, `unregister()`,
  `discover_custom_skills()`.

### Fixed
- Correlation skill now uses `save_figure()` instead of manual
  `savefig()`.

## [1.0.0] - 2026-04-21

### Added
- Initial release: 28 skills, DuckDB data layer, ReAct agent loop.
- Parquet rebuild from 484 to 4,971 field IDs.
- LLM client with retry logic and exponential backoff.
- Self-evolution: error catalog, pipeline recording, skill generation.
- Nature-style plotting (300 DPI, Okabe-Ito palette).
- 4-tier memory system.
- 164 unit tests.
