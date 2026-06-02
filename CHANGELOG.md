# Changelog

All notable changes to Biobank Agent are documented here.

## [Unreleased] — Agent autonomy, robustness & self-evolution

Driven by real-environment end-to-end testing of the live `biobank` CLI against
the configured LLM (each fix root-caused from an observed failure, then verified;
every change Codex-reviewed).

### Fixed (agent intelligence & robustness)
- **The executor ran with NO system prompt and NO memory.** `run_turn` sent the
  model only the raw user message, so it flailed — rewriting the same file 6×
  without ever running it, and speculatively tool-calling on garbage input. It now
  sends an identity + workspace-sandbox + tool-use-discipline system prompt
  ("write→run→observe→fix; never repeat a succeeded call; finish with a concrete
  answer") plus compact prior-turn history (`biobank_agent/runtime/engine.py`).
- **Multi-step tool use was broken** by malformed assistant `tool_calls`: the
  runtime emitted the internal `{id,name,args}` shape instead of the OpenAI
  `{id,type,function:{name,arguments}}` wire format, so the relay dropped them and
  orphaned the tool results — the model never saw its prior actions. Normalised at
  the API boundary in `LLMClient.sanitize_messages` (`biobank_agent/llm.py`). This
  is what makes autonomous write→run→fix actually work.
- **A turn was marked FAILED if ANY tool errored**, even after the model recovered.
  Turn status is now convergence-based (COMPLETED iff the model reached a final
  answer and the provider didn't fail).
- **`run_turn` didn't catch provider failures** → now records an ERROR and fails
  the turn gracefully instead of letting the exception escape.
- **Runaway tool loops** (observed 33 calls until `max_tool_rounds=30` burned out)
  → consecutive no-progress guard (nudge at 3, hard-stop at 6) that does not kill
  legitimate distinct progress.
- **Shell/test nonzero exits were silently "ok"** (no `error` field) → now surfaced
  so the loop guard / audit / completion feedback see real failures.
- **Workspace-denial messages are now actionable** (tell the model to use a path
  inside the workspace) so it adapts instead of retrying a rejected path.
- **`/plan` timed out** on slow reasoning models (240s) → council timeout raised to
  360s; `/plan` and `/research` now complete cleanly.
- **Constraint verification silently degraded when `z3-solver` was installed**: the
  Z3 path returned only one opaque verdict ("claims are mutually contradictory")
  and lost the human-readable, per-constraint diagnostics ("n_excluded=… exceeds …",
  "n_cases is negative") that the no-solver fallback produced. `verify_cohort_claims`
  now **always** runs the descriptive bounds checks and *merges* the formal Z3
  consistency proof + Minimal Correction Subset on top — so callers get specific,
  actionable issues whether or not z3 is present (`biobank_agent/verification.py`).
- **Project-doc search ranked sprawling docs over focused ones**: `_score` was raw
  term-frequency over the whole document, so a long architecture dump that merely
  mentioned the query terms buried the one-page guide a user actually wanted. Replaced
  with BM25 length-normalised relevance + full-coverage and exact-phrase bonuses
  (`biobank_agent/skills/project_doc.py`).
- **Curated-doc titles weren't redacted** (only snippets/content were), so a document
  with a credential-bearing heading could leak it through the `title` field of
  list/search/read. Redaction is now structural at the source in `_title_for`
  (Codex-flagged).

### Added
- **Autonomous patch generation closes the self-evolution loop**
  (`biobank_agent/runtime/patch_generation.py`, wired into `/evolve --generate`
  and `/evolve --generate --apply`). `learn_from_session` only ever produced
  *review-only* proposals (no concrete patch), so `--apply` always found nothing
  to apply. The new generator asks the planner LLM to turn a proposal + its
  evidence into a concrete `{target_path, diff, test_commands}`, then feeds it to
  the existing transactional apply loop — making the full **learn → generate →
  test-gate → apply** loop real. The generated patch is **untrusted**; safety is
  enforced entirely by `apply_patch_transactionally` (isolated worktree,
  authoritative changed-path validation, allow-list, test gate, rollback). Design
  was adversarially reviewed across four risk lenses (trust-boundary,
  test-laundering, prompt-injection, API) before implementation; the resulting
  guards: a shared balanced-brace JSON parser (robust to braces/quotes inside a
  diff, to truncation, and to LLM preamble noise), evidence compacted + size-capped
  before it enters the prompt (prompt-injection surface), a syntactic-only path
  pre-check **plus validation of every path a unified diff touches** (the apply
  loop's `.resolve()` on the actually-changed files remains the authoritative
  symlink/escape gate), a test-laundering guard (≥1 real `pytest` run required; all
  three collect-only spellings — `--collect-only`/`--collectonly`/`--co`, incl.
  `=`-glued — banned in one shared place so the generation and apply gates can't
  diverge), and idempotent generation (already-patched proposals are skipped). A
  post-implementation adversarial pass (Codex + a 3-lens review) hardened these and
  closed a `--collectonly` bypass. **Disclosed limitation:** LLM-authored test code runs
  with network access in the disposable worktree before the merge decision, and
  patches are not verified to *exercise* their diff — review test relevance
  before trusting an auto-merge.
- **Real shell capability**: the native `shell`/`test_runner` tools run through a
  real shell (`bash -c`) so pipes, redirects, globs, and `&&`/`||` work
  (`biobank_agent/core/tools/native.py`).
- **Transactional, test-gated self-evolution** (`biobank_agent/runtime/self_evolve.py`,
  wired into `/evolve --apply`): a proposed patch (diff + target_path +
  test_commands) is applied in an **isolated git worktree**, its tests run there,
  and it is committed **only if they pass** — with automatic rollback (zero
  residue) on failure. Hardened against path-traversal (every actually-changed file
  is re-validated from `git status`, not the diff text), arbitrary test-command
  execution (test gates must be `pytest`/`ruff`/`mypy`/… runners, no bare
  `python -c`, no shell metacharacters, secret-stripped env), and dirty-tree merges.
  Edits to `biobank_agent/core`, `biobank_agent/runtime`, or `tests/` are **never
  auto-merged** — they go to a review branch.

### Security / containment
- **Secrets are no longer leakable** through the file/shell tools: redaction masks
  secret VALUES (`KEY=…` → `KEY=[REDACTED]`, plus `sk-`/`ghp_`/bearer tokens), and
  `file_read` refuses `.env`/`.pem`/`.key`/`id_rsa`/`credentials`; `test_runner`
  runs with a secret-stripped env (`biobank_agent/core/tools/native.py`).
- **Self-evolution collect-only ban hardened across every channel** so an untrusted
  generated patch can no longer auto-merge a body-skipping (collect-only) test
  (`biobank_agent/runtime/self_evolve.py`). An adversarial verification pass reproduced
  two laundering paths the literal-flag ban missed: (1) **shell expansion** —
  `pytest x ${OPTS:---collect-only}` passes a token check but `bash -c` expands it into
  `--collect-only`; now `$` is a banned test-command metacharacter. (2) **pytest addopts**
  — a `pytest.ini`/`pyproject.toml` `addopts = --collect-only`, an attacker
  `-o addopts=--collect-only` token, or a `PYTEST_ADDOPTS` env var all skip test bodies
  while exiting 0. Now: pytest gate commands get `-o addopts=` appended (last-wins,
  neutralizing the file/command channels — verified), `_safe_env()` strips
  `PYTEST_ADDOPTS`/`PYTEST_PLUGINS` (the env channel `-o` cannot override), and any patch
  that creates/edits a pytest config or hook file (`pytest.ini`, `tox.ini`, `setup.cfg`,
  `pyproject.toml`, `conftest.py`, …) is **never auto-merged** — it is verified onto a
  review branch for human inspection (a `conftest.py` can subvert collection in ways
  `-o addopts=` cannot stop). Regression tests reproduce each attack (red before, green
  after).

## [Unreleased] — Interactive CLI council overhaul

### Fixed
- **Interactive menu arrow keys were unusable** (`biobank_agent/cli/interactive.py`).
  In the Plan Review / clarification selectors, ↑/↓ "repeated the display" instead
  of moving the selection. Root cause: `_read_single_key` read keys with buffered
  `sys.stdin.read(1)` + `select` on the file descriptor, so a 3-byte arrow burst
  (`\x1b[A`) was split into three separate non-matching keys (one keypress →
  ~3 spurious re-renders, no movement). It now reads raw bytes via `os.read`,
  parses CSI/SS3 escape sequences precisely, drains bracketed-paste payloads, uses
  `tty.setcbreak`, confirms the **highlighted** row on Enter (previously a separate
  `checked` index), and redraws a single in-place Rich `Live` surface (no more
  stacked menus or double-render).
- **`/plan` produced an instant, generic 4-step template** (`context/design/execute/verify`)
  for any objective. The live shell was wired to the static `RuntimeEngine.build_plan`
  template, bypassing real planning. `/plan` now runs the multi-agent council planner.
- **`--model foo` (space form)** leaked the model value into the seeded interactive
  task; the value token is now consumed correctly.
- **Clarification menus were still broken on a real terminal** (arrows didn't move,
  menus stacked, prompts "flew past" with `0 clarification(s) resolved`). The earlier
  fix tested the selector in isolation and missed its interaction with the live
  `PlanRunDashboard`. Real causes, now fixed: (1) the dashboard `Live` and the
  selector `Live` competed on one console — the planning dashboard now **lazily
  starts only after clarification finishes** (the selector is the sole `Live`, so
  each question blocks until answered); (2) the selector built its panel from a
  joined markup string, so wide **CJK** glyphs made Rich undercount lines and the
  stale highlight stayed on screen — it now renders a structured `Group`/`Table`,
  width-corrects with `rich.cells.cell_len`, bounds height with a selection-centered
  sliding window, and is `transient=True` (no stacking); (3) `_clarify` **silently
  swallowed** selector errors (recording `None` for every question) — it now surfaces
  the error and stops, and **Ctrl-C propagates** to cancel the plan instead of being
  caught as a per-question "skip".
- **`/plan` crashed the whole CLI with an uncaught `KeyboardInterrupt`** the instant the
  clarification menu opened (`biobank_agent/cli/interactive.py`). Root cause: the menu's
  own Rich `Live` defaults to `redirect_stdout=True`, which swaps `sys.stdout` for a Rich
  `FileProxy` whose `isatty()` returns `False` even on a real terminal (`Live` only
  redirects when `console.is_terminal`, so this fired live but never under the StringIO
  test console). `_read_single_key` gates on `_interactive_terminal()`
  (`sys.stdin.isatty() and sys.stdout.isatty()`); under the redirect that flipped to
  `False`, so the menu raised `KeyboardInterrupt` on itself before reading a key — and the
  interrupt propagated uncaught (`run()` only guarded the prompt read, not `handle_line`)
  to `main()` as a raw traceback. Fixes: (1) the menu `Live` now sets
  `redirect_stdout=False, redirect_stderr=False` so `sys.stdout` is never swapped while
  reading keys (the menu never writes to stdout, so nothing is lost); (2) `run()` now
  contains a `KeyboardInterrupt` raised inside command handling — a genuine Ctrl-C during a
  command prints `Cancelled` and returns to the prompt instead of crashing — with a final
  safety net in `main()`. New regression test drives the selector with a real-terminal
  console and the real `_interactive_terminal()` so the `FileProxy`-redirect path is
  actually exercised (red before the fix, green after) — closing the
  "passed-in-isolation / failed-live" coverage gap.
- **Clarification questions were silently dropped for many models** (`biobank_agent/runtime/planner.py`).
  `_identify_clarifications` only accepted `options` as dicts with a `label` key and the
  question under a `question` key. Real models vary: a live OpenRouter run showed the
  planner (kimi) emit `"options": ["A. …","B. …"]` (bare strings) with a `text` field —
  every option was filtered out, so zero questions survived and the whole clarification
  step no-opped (no menu). The parser is now tolerant: it accepts the question list under
  several keys (or a bare top-level list), each option as a **string or** a dict (label
  under `label`/`text`/`value`/`name`/…), and the question text under
  `question`/`text`/`prompt`/`title` — normalizing everything into the menu's
  `{label, description}` shape. Regression test reproduces the exact kimi shape.
- **`/plan` crashed the CLI with `rich.errors.MarkupError: closing tag '[/]' has nothing
  to close`** (`biobank_agent/progress.py`, `biobank_agent/cli/interactive.py`). Two
  cascading faults: (1) the dashboard built status lines as `f"[{style}]{icon}[/{style}]"`
  with `style = DASHBOARD_STATUS_STYLES.get(status, "")` — an unmapped status (notably the
  `"error"` that a failed council job emits, absent from the style map) made `style=""`, so
  the row rendered `[]…[/]` and the unmatched `[/]` killed the daemon refresh thread; (2) the
  error reporter `console.print(f"[red]Command failed:[/] {exc}")` re-parsed the exception
  text as markup — and since `{exc}` was *itself* that `MarkupError` (text contains `[/]`),
  reporting the error raised a **second, uncaught** `MarkupError` that took down the CLI.
  Fixes: a `_styled(text, style)` helper that emits a tag only when the style is non-empty
  (else escapes, so no empty/unbalanced tag is ever produced) used across every dynamic-style
  renderable; `"error"`/`"cancelled"` added to the status icon/style maps; the daemon refresh
  now wraps `live.update` in try/except so one bad frame can't kill the thread; and every
  markup `console.print` that interpolates exception/user/LLM text now escapes it with
  `_rich_escape` (so an error report can never itself crash). Plan/step text in
  `PlanProgressDisplay` is likewise escaped. Red-before-green regression tests reproduce both
  crashes (the empty-style render and the self-eating reporter).
- **Clarification menus now accept free-text answers** (`biobank_agent/cli/interactive.py`),
  matching Claude Code's "Type something" UX. Every clarification menu gains a final
  **"✎ Type a custom answer / 以上都不是"** row that opens a free-text prompt and forwards the
  typed text to the planner; and pressing **Tab** on any focused option opens the same editor
  **prefilled** with that option's label so the user can complete/correct it. Implemented as
  an opt-in `edit_handler` on `_live_single_select` (return type widened to `int | str`; the
  plan-review menu passes no handler and stays int-only), with the editor stopping the Rich
  `Live` before prompting, degrading to `console.input`/no-op when prompt_toolkit/tty are
  absent, and treating Ctrl-C/empty input as *cancel-edit-only* (the menu re-opens; the plan
  is not cancelled). 8 new pty-driven tests cover Tab-edit prefill, the custom row, empty/
  cancel re-entry, and non-interactive no-op.
- **`/plan` now runs genuine multi-model, multi-round debate** instead of three identical
  drafts (`biobank_agent/runtime/planner.py`, `council.py`, `types.py`, `config.py`). The
  three candidate slots use **distinct provider roles** (planner/primary/critic →
  kimi/deepseek/glm), so the dashboard shows three *different* models, each with its own live
  token stream. After critique, surviving drafts **debate** over bounded rounds: each model
  revises its plan having seen its peers' plans and reports a self-confidence; homogeneous
  drafts (Jaccard over a structured step-signature ≥ `consensus_threshold`) and low-confidence
  drafts are **pruned** (CONCAT/EVOCHAMBER), debate **short-circuits when models already
  agree** (the MAD "don't debate when they agree" finding), and an orchestrator (summarizer)
  synthesizes the winners. Bounded by round cap, a global time deadline (round-scoped
  `run_parallel` timeout), a call ceiling, and `cancel_event`; a failed revision keeps the
  prior plan (confidence decays, never deleted) so a transient error can't drop a model; the
  whole feature degrades to the single-shot pipeline when disabled or when <2 distinct models
  are configured. The dashboard gains a **Debate** phase and shows each model's per-round
  activity (e.g. `deepseek · revising R2 · bioinformatics`). New `RuntimeConfig` knobs:
  `enable_debate`, `debate_rounds`, `consensus_threshold`, `debate_confidence_floor`.
- **Clarification answers are now edited INLINE inside the menu panel**
  (`biobank_agent/cli/interactive.py`). Pressing Tab on an option (or selecting the
  "Type a custom answer / 以上都不是" row) used to `live.stop()` and drop out to a separate
  `refine answer ›` prompt line; it now turns the focused row into an editable text field
  *inside* the bordered panel — typed ASCII/CJK inserts at a visible block cursor,
  Backspace/←/→ edit, Enter submits, Esc cancels back to the menu — exactly Claude Code's
  "Type something" UX. The `✎` emoji was removed. The field is built as a styled `Text`
  (never `from_markup`), so typing `[`/`[/]` can't raise a `MarkupError`. Also fixed a
  latent `_read_single_key` bug: it greedily consumed the byte after a bare Esc, merging
  `Esc`+`Enter` into `"\x1b\r"` (which matched no handler and could hang an inline editor) —
  now a 1-byte pushback surfaces Esc and the next key separately.
- **Approving a plan now executes it autonomously with a live progress view**
  (`biobank_agent/cli/interactive.py`). Before, `/plan-approve` marked the steps "planned",
  printed a static "…awaiting executable work" panel, and returned to the prompt — there was
  no execution loop at all. Approval now restores the user's real permission profile (e.g.
  yolo — it no longer force-switches to `ask_before_edits`, which never actually gated tools
  since the scheduler has no `confirm_fn`) and runs every step autonomously via `run_turn`
  (the agent executes each step from its natural-language purpose + tool scope, since runtime
  plan steps carry no `skill`/`args`), in dependency (topological) order, under a single live
  `PlanRunDashboard` (or a line-logger on a non-TTY). No per-step input; a genuine Ctrl-C
  cancels cleanly ("cancelled at step k/n"); the live view is always torn down; completion
  prints one concise summary. Steps run bounded by `run_turn`'s own `max_tool_rounds` +
  anti-loop guard.
- **Concise post-plan output.** The post-draft wall is trimmed (the "Council activity" and
  "Plan Formulation Progress" preambles and a redundant third steps panel are dropped), and
  the Workflow Diagram is shown once instead of being reprinted at draft, approval, and view.

### Added
- **Multi-agent council planner** (`biobank_agent/runtime/council.py`,
  `biobank_agent/runtime/planner.py`): `/plan` runs a real, objective-specific,
  parallel pipeline — Clarification → N parallel candidate drafts (diverse personas)
  → parallel critics → merge → schema validation → review — over the runtime
  `ProviderRouter` (planner/critic/summarizer roles), with honest per-stage progress
  on `PlanRunDashboard`. It **fails loudly** on planning failure instead of emitting a
  generic template. Verified live against a vitiligo WGS objective: a 7-step
  QC→annotate→PCA→association→ML→visualization→report DAG.
- **Unified `/research` mode** (`biobank_agent/runtime/researcher.py`): the same
  council core powers a multi-source cited research brief — sub-query decomposition
  → parallel `deep_research` retrieval → synthesis with citation sanitization
  (phantom `[n]` citations stripped) → `verifier_mesh` claim checks.
- **Completion gates** (`biobank_agent/runtime/completion.py`): `/verify` with no
  argument assesses plan completion — report-contract enforcement, an LLM
  goal-acceptance judge (never auto-accepts on unparseable/unavailable output),
  completion warnings, and bounded LLM-proposed repair via the runtime's existing
  repair recording.
- **Claude-Code-style multi-model live dashboard** (`biobank_agent/progress.py`).
  While the council fans out, `PlanRunDashboard` now shows one live row per model
  (short name deepseek/kimi/glm · activity · persona · ticking timer), a compact
  phase pipeline, and a recent-milestones log; on completion each model moves to a
  history with its duration, and a static "Council activity" summary is printed
  after the (transient) live panel stops. The council emits `role`/`persona` per
  job (`council.py`); rows are tracked per-subagent under the dashboard lock; panel
  height is budgeted against the terminal (selection/active/recent caps) so a full
  fan-out never overflows the viewport; all model-derived text is markup-escaped.
  On a council timeout, outstanding rows get a per-subagent terminal event (both
  `run_parallel` and `map_parallel`) so their live timers stop instead of ticking
  forever.
- **Live token streaming into the dashboard** (the middle region shows what each
  model is writing, in real time). `ProviderRequest.stream_cb` (runtime-only, not
  serialised) lets `LLMProvider.complete` route plain-text council calls through
  `LLMClient.stream()`; streamed `usage` is captured via `stream_options.include_usage`
  with a creation- **and** iteration-time fallback for providers that reject it.
  Deltas flow on a dedicated channel (never recorded as events / never logged) and
  update only the active row's tail buffer (`note_partial`, throttled by the
  background refresh, not per token). Orphaned post-timeout/Ctrl-C deltas are dropped
  via a per-stage cancel + a global `cancel_event`, with `note_partial`'s row-active
  check (and `stop()` clearing rows) as the authoritative atomic guard. Tool-call
  turns and the `run_turn` chat path stay non-streaming.

### Changed
- **Single interaction mode**: `biobank` (optionally `biobank "<task>"`) is the only
  way to start the agent. The former `--tui` (Textual) and `--legacy-repl` modes were
  removed from dispatch.

### Notes
- The legacy `planner.py`, `plan_executor.py`, and `cli_legacy.py` modules remain on
  disk: they are still load-bearing for the eval harness (`eval/behavioral.py`),
  `plan_state.py`, and the CLI entry point, so removing them physically requires a
  dedicated follow-on cross-subsystem refactor rather than a blind delete.

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
