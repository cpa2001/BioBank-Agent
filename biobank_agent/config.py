"""Configuration loaded from .env via pydantic-settings.

All biobank-specific settings have sensible UK Biobank defaults but are
fully overridable for other biobanks (FinnGen, CKB, HPP, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ── LLM ──────────────────────────────────────────────────────
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek/deepseek-v4-pro"
    llm_request_timeout_s: float = 60.0
    llm_max_retries: int = 3
    llm_retry_base_delay_s: float = 2.0

    # ── Data Paths ───────────────────────────────────────────────
    # Accept both new (DATA_DIR/RAW_DIR) and legacy (UKB_PARQUET_DIR/UKB_RAW_DIR) env var names
    data_dir: Path = Field(
        default=Path("./milton_data"),
        validation_alias=AliasChoices("data_dir", "DATA_DIR", "UKB_PARQUET_DIR"),
    )
    raw_dir: Path = Field(
        default=Path("./UKB"),
        validation_alias=AliasChoices("raw_dir", "RAW_DIR", "UKB_RAW_DIR"),
    )
    full_ukb_feature_store_dir: str = Field(
        default="",
        validation_alias=AliasChoices(
            "full_ukb_feature_store_dir",
            "FULL_UKB_FEATURE_STORE_DIR",
            "UKB_FULL_PARQUET_DIR",
        ),
    )

    # Backward compatibility aliases
    @property
    def ukb_parquet_dir(self) -> Path:
        return self.data_dir

    @property
    def ukb_raw_dir(self) -> Path:
        return self.raw_dir

    # ── Biobank Identity ─────────────────────────────────────────
    bank_id: str = "ukb"
    biobank_name: str = "UK Biobank"
    biobank_abbreviation: str = "UKB"
    biobank_description: str = (
        "a large-scale prospective cohort study comprising over 500,000 "
        "participants aged 40-69 at recruitment"
    )
    biobank_caveats: str = (
        "healthy volunteer cohort with known selection biases"
    )

    # ── Column / Schema Identity ─────────────────────────────────
    subject_id_col: str = "eid"
    diagnoses_code_col: str = "diag_icd10"
    deaths_code_col: str = "cause_icd10"
    field_column_pattern: str = "{field_id}-{instance}.{array}"

    # ── Dataset File Names ───────────────────────────────────────
    biomarker_parquet_name: str = "ukb.parquet"
    diagnoses_parquet_name: str = "hesin_diag.parquet"
    deaths_parquet_name: str = "death_cause.parquet"
    catalog_fields_file: str = "field.txt"
    catalog_categories_file: str = "category.txt"
    catalog_encoding_file: str = "esimpint.txt"

    # ── Data Usage / Governance ─────────────────────────────────
    # The local UKB parquet release used by this project is already
    # de-identified. Agent skills should therefore use the full analytical
    # tables by default and must not discard/round/redact useful in-memory
    # data unless the user explicitly switches to an external disclosure mode.
    data_deidentified: bool = True
    data_use_full_dataset_default: bool = True
    max_train_rows_default: int = 0  # 0 means no cap; use every eligible row.
    default_analysis_sample_size: int = 0  # 0 means full table for descriptive skills.
    disclosure_control_mode: str = "internal"  # internal | external | strict

    # ── Output ───────────────────────────────────────────────────
    reports_dir: Path = Path("./reports")
    memory_dir: Path = Path.home() / ".biobank_agent"
    # When true, generated reports/outputs land under the ACTIVE workspace
    # (<cwd>/reports/<session>) so a `--workspace`/`/cd` switch keeps inputs and
    # outputs together; when false they always go under reports_dir.
    reports_follow_workspace: bool = True

    # ── Command execution (timeouts + background jobs) ───────────
    # Single source of truth consumed via biobank_agent.utils.exec_policy. A
    # command timeout of 0 means "auto" (scale for known long bio tools).
    exec_default_timeout_s: int = 120          # generic command default
    exec_long_tool_timeout_s: int = 7200       # baseline floor for known long tools
    exec_hard_ceiling_s: int = 86400           # clamp explicit foreground timeouts
    exec_allow_no_timeout: bool = True         # permit uncapped BACKGROUND jobs
    exec_background_threshold_s: int = 600      # expected duration >= this -> background
    exec_auto_background_long_tools: bool = True  # auto-background long tools when headless
    exec_stream_tail_chars: int = 5000          # LLM-facing stdout/stderr tail size
    jobs_dir_name: str = ".biobank_jobs"        # background job logs under the workspace
    # Execution backend for long jobs: auto-detects an available cluster
    # scheduler (slurm/sge/lsf/dxrun) else local; or force one explicitly.
    exec_backend: str = "auto"                  # auto | local | slurm | sge | lsf | dxrun
    scheduler_cpus: int = 1
    scheduler_mem_mb: int = 4096
    scheduler_time_min: int = 240
    scheduler_partition: str = ""

    # ── Lazy tool exposure ──────────────────────────────────
    # When true, only Direct-tier skills (skills/manifest.json) + native tools are
    # offered to the model each round; Deferred skills load on demand via the
    # skill_search tool. False ⇒ inject all tool schemas.
    lazy_tools_enabled: bool = True
    # Inject the hierarchical skill-tree node summaries (+ navigate_skill_tree) into the
    # executor prompt when manifest.json carries a tree. False ⇒ flat category summaries.
    skill_tree_enabled: bool = True

    # ── Evidence Contract ───────────────────────────────────
    # When true, a plan step that declared a verification but produced no observable
    # artifact is marked 'unverified' (never 'done').
    evidence_contract_enabled: bool = True
    # When true, the completion gate runs the methodology reviewer over the evidence and
    # hard-blocks goal acceptance on a consensus statistical/omics sin. Default OFF — a
    # deliberate, gated rollout (enable after confirming it does not over-block real goals).
    methodology_gate_enabled: bool = False
    # Notifications for finished background jobs / paused (awaiting-input) runs.
    notify_enabled: bool = True
    notify_command: str = ""                    # optional shell cmd; message piped on stdin

    # ── Web Search ───────────────────────────────────────────────
    search_provider: str = "duckduckgo"
    search_api_key: str = ""

    # ── Plan Mode ────────────────────────────────────────────────
    plans_dir: Path = Path("./plans")
    plan_repair_budget_per_step: int = 3
    plan_repair_budget_total: int = 8
    plan_code_mutation_mode: str = "review_only"
    plan_goal_acceptance_enabled: bool = True
    plan_external_council_enabled: bool = False  # opt-in: shells out to external coding-agent CLIs
    plan_external_council_policy: str = "requested"  # requested | always | never
    plan_external_council_timeout_s: int = 180
    plan_external_council_agents: str = "codex,claude,gemini"
    plan_heartbeat_interval_s: float = 1.0
    plan_build_timeout_s: float = 240.0      # max model SILENCE while drafting before it counts as stalled
    plan_step_max_retries: int = 2
    # Per-step deadline measures model SILENCE, not wall-clock: while the model streams tokens or tools
    # make progress, the step is never judged stalled (timeout exists to catch a hang, NOT to kill a
    # producing task). Only this many seconds with NO activity trips it. 0 disables.
    plan_step_timeout_s: float = 240.0
    # Absolute per-step backstop regardless of activity (anti-runaway only). A healthy streaming/tool
    # step is bounded by plan_step_timeout_s INACTIVITY; this ceiling just stops a step that keeps
    # "making progress" yet runs unreasonably long. 0 disables.
    plan_step_hard_ceiling_s: float = 3600.0
    # Token-cost guards — the REAL anti-runaway limiter (a timeout must never stop a producing task).
    # Counts completion tokens across a step / whole plan. On exhaustion the run stops GRACEFULLY
    # (checkpoint + summary, no crash). 0 = unlimited (timeout watchdogs still apply).
    token_budget_per_step: int = 0
    token_budget_per_plan: int = 0
    cli_refresh_per_second: float = 10.0
    plan_research_setup_enabled: bool = True
    plan_clarification_enabled: bool = True
    plan_clarification_policy: str = "critical_only"
    plan_env_repair_policy: str = "ask"
    plan_review_hook_mode: str = "ask"  # ask | all | both | codex | claude | gemini | never
    plan_review_hook_agents: str = "codex,claude,gemini"
    plan_review_repair_mode: str = "auto_safe"  # auto_safe | ask | never
    plan_review_repair_max_loops: int = 2

    # ── Custom Skills ────────────────────────────────────────────
    custom_skills_dir: Path = Path("./custom_skills")

    # ── External skill ingestion ─────────────────────────────────
    # Opt-in: pull community skill libraries from GitHub. Default OFF. v1 ingests
    # SKILL.md metadata as deferred KNOWLEDGE skills only (no third-party code runs);
    # ingested skills are tagged trust='external' and never auto-promote.
    external_skill_ingestion_enabled: bool = False
    external_skills_dir: Path = Path("./external_skills")
    enabled_external_corpora: list[str] = Field(default_factory=list)

    # ── Plugin marketplaces (Claude-Code-style) ──────────────────
    # Opt-in: consume third-party plugins from a marketplace repo (e.g. obra/superpowers). SKILL.md
    # skills load as USABLE knowledge skills; plugin HOOKS (executable shell commands) are recorded but
    # NEVER run without an explicit per-plugin opt-in (plugin_allow_hooks). Cloned under the memory dir.
    # Default OFF — third-party plugin code is never executed on install.
    plugins_enabled: bool = False
    plugin_allow_hooks: bool = False

    # ── Adversarial-game council ─────────────────────────────────
    # Route planning through the proposer/red-team/referee game instead of the symmetric
    # debate. Default OFF; kept only if it beats symmetric on the council A/B set.
    adversarial_council_enabled: bool = False

    # ── Paper → skill synthesis ──────────────────────────────────
    # Turn a method paper into a tree-filed skill through the methodology pre-gate, the
    # load-safety validator, and a REVIEW-BRANCH apply (never auto-merged). Default OFF.
    skill_synthesis_enabled: bool = False

    # ── External coding-agent task delegation ───────────────────
    # Opt-in: hand a coding subtask to codex/claude-code; the agent runs in an isolated worktree and
    # its diff lands on a REVIEW BRANCH only (never auto-merged). Default OFF.
    external_agent_delegation_enabled: bool = False

    # ── Agent ────────────────────────────────────────────────────
    max_tool_rounds: int = 30
    context_window: int = 180_000

    # ── Async runtime ────────────────────────────────────────────
    # Routes Agent.run() and the CLI through core/runtime.AsyncAgent
    # when the legacy path is safe to wrap. Unsafe states fall back to
    # the synchronous loop instead of pretending streaming is active.
    async_runtime_enabled: bool = True
    compaction_warn_pct: float = 0.6
    compaction_compact_pct: float = 0.75
    compaction_force_pct: float = 0.9

    # ── Local Telemetry ──────────────────────────────────────────
    # Local-only, privacy-preserving JSONL. Records event type, skill,
    # status and timing metadata only; never records user query text,
    # tool args, raw outputs, or identifiers.
    telemetry_enabled: bool = True
    telemetry_jsonl: str = ""
    otel_enabled: bool = False
    otel_service_name: str = "biobank-agent"
    otel_exporter: str = "none"  # none | console | otlp
    otel_endpoint: str = ""      # e.g. http://localhost:4318/v1/traces
    otel_policy: str = "local"   # local | production
    otel_allow_local_endpoint: bool = False

    # ── MCP extensions ────────────────────────────────────────
    # Empty uses BIOBANK_MCP_CONFIG when set, then
    # ~/.biobank_agent/mcp_servers.json.
    mcp_config_path: str = ""

    # ── Multi-Model Orchestration ────────────────────────────────
    multi_model_enabled: bool = True
    model_pool: str = ""       # comma-separated model IDs, e.g. "model-a,model-b"
    auto_discover_models: bool = True
    preferred_multi_models: str = "deepseek/deepseek-v4-pro,moonshotai/kimi-k2.6,z-ai/glm-5.1"
    max_auto_model_pool: int = 3
    debate_rounds: int = 2
    plan_debate_enabled: bool = True               # multi-model /plan debate rounds
    plan_consensus_threshold: float = 0.85         # plan-similarity above which two drafts are homogeneous
    plan_debate_confidence_floor: float = 0.45     # prune a draft whose self-confidence is below this
    complexity_threshold: float = 0.7   # score above this triggers multi-model
    enable_reflexion: bool = True       # structured self-correction on failure
    enable_tot: bool = False            # Tree-of-Thought for branching decisions
    tool_call_content_mode: str = "null"  # "null" | "empty" — how to send empty content with tool_calls

    # ── Derived Paths ────────────────────────────────────────────

    @property
    def biomarker_parquet(self) -> Path:
        return self.data_dir / self.biomarker_parquet_name

    @property
    def diagnoses_parquet(self) -> Path:
        return self.data_dir / self.diagnoses_parquet_name

    @property
    def deaths_parquet(self) -> Path:
        return self.data_dir / self.deaths_parquet_name

    @property
    def field_txt(self) -> Path:
        return self.data_dir / self.catalog_fields_file

    @property
    def category_txt(self) -> Path:
        return self.data_dir / self.catalog_categories_file

    @property
    def encoding_txt(self) -> Path:
        return self.data_dir / self.catalog_encoding_file

    @property
    def category_parquet_dir(self) -> Path:
        return self.data_dir / "categories"

    @property
    def raw_csv_dir(self) -> Path:
        return self.raw_dir / "UKB_info"

    @property
    def main_csv(self) -> Path:
        return self.raw_dir / "UKB" / "ukb672073.csv"

    @property
    def data_dict_csv(self) -> Path:
        return self.raw_dir / "UKB" / "Data_Dictionary_Showcase.csv"

    @property
    def full_ukb_feature_store(self) -> Path:
        if str(self.full_ukb_feature_store_dir).strip():
            return Path(self.full_ukb_feature_store_dir).expanduser()
        return self.raw_dir.parent / "ukb_full_parquet"

    def ensure_dirs(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)


def get_settings(**overrides) -> Settings:
    """Factory that finds .env walking up from cwd."""
    cwd = Path.cwd()
    for d in [cwd, *cwd.parents]:
        env = d / ".env"
        if env.exists():
            return Settings(_env_file=str(env), **overrides)
    return Settings(**overrides)
