<div align="center">

# Biobank Agent

**Local-first autonomous research agent for biobank, genomics, and biomedical discovery workflows**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Natural-language planning, data-aware tool use, WGS analysis, cited research, runtime audit, replay, and review-gated self-evolution in one CLI.

</div>

---

## What It Does

Biobank Agent is an LLM-powered scientific workflow agent for population-scale biobank research. It runs from a local interactive shell, discovers registered analysis skills, drafts and repairs multi-step plans, executes local tools under permission controls, records session trajectories, and produces auditable reports.

The current repository is a v3 runtime-oriented build with:

- **101 registered skills** discovered from `biobank_agent.skills`, including cohort analysis, modelling, WGS/VCF workflows, literature research, report writing, paper→skill synthesis, and self-evolution support.
- **71 slash commands** in the v3 command registry, including `/plan`, `/plan-diagnose`, `/plan-retry`, `/plan-use`, `/research`, `/doctor`, `/tools`, `/artifacts`, `/resume`, `/audit`, `/trace`, `/harness`, `/replay`, `/learn`, `/plugin`, and `/evolve`.
- **Runtime-backed sessions** with event logs, action graph references, plan state, trajectory replay, audit reports, and resume support.
- **VirtualCell/WGS support** for local VCF discovery, WGS dependency checks, exploratory VCF QC, PCA, kinship, association, burden testing, annotation, pathway enrichment, and WGS report polishing.
- **OpenAI-compatible providers** configured through `.env`, with multi-model planning and review routes controlled by settings.

The project is local-first: data paths, reports, memory, sessions, tool calls, and audit artifacts remain on the workstation unless a configured skill or provider explicitly uses a network service.

## Quick Start

```bash
git clone https://github.com/cpa2001/BioBank-Agent.git
cd BioBank-Agent

# Optional but recommended
conda create -n biobank-agent python=3.11 -y
conda activate biobank-agent

pip install -e ".[all,dev]"
cp .env.example .env
```

Edit `.env`:

```ini
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=<your-api-key>
LLM_MODEL=deepseek/deepseek-v4-pro

DATA_DIR=./data
RAW_DIR=./raw
REPORTS_DIR=./reports
PLANS_DIR=./plans
```

Start the interactive shell:

```bash
biobank
```

Run the first checks:

```text
biobank > /doctor
biobank > /tools
biobank > /skills
```

For a complete runnable setup path, use [docs/guides/QUICK_START.md](docs/guides/QUICK_START.md). For the full Agent + WGS workflow, use [docs/guides/END_TO_END_TUTORIAL.md](docs/guides/END_TO_END_TUTORIAL.md).

## API Connectivity Smoke Test

After editing `.env`, verify provider connectivity before running a long plan:

```bash
python - <<'PY'
from biobank_agent.config import get_settings
from biobank_agent.llm import LLMClient

settings = get_settings()
client = LLMClient(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    model=settings.llm_model,
)
response = client.chat([{"role": "user", "content": "Reply with exactly: API_OK"}])
print(response.text)
print(response.usage)
PY
```

Expected result: the text contains `API_OK`. If the call fails, fix `LLM_BASE_URL`, `LLM_API_KEY`, or `LLM_MODEL` before testing agent workflows.

## Core Workflows

### Interactive Planning

Use `/plan` for multi-step workflows that need visible structure, approvals, repair, and provenance.

```text
biobank > /plan Compare vitiligo cases and controls using the available WGS VCF files, then write an auditable report.
biobank > /plan-approve
```

If execution is blocked or a step fails, stay in the same shell:

```text
biobank > what is the problem?
biobank > /plan-diagnose
biobank > /plan-use vcf_dir=data/vc_wgs_vcf
biobank > /plan-use workflow_mode=exploratory
biobank > continue
```

Useful plan commands:

| Command | Use |
| --- | --- |
| `/plan <task>` | Draft a structured plan from a natural-language objective. |
| `/plan-approve` | Approve and execute the active draft. |
| `/plan-edit <feedback>` | Modify the draft or active plan with natural-language feedback. |
| `/plan-diagnose` | Explain why the current plan is blocked or failed. |
| `/plan-use key=value` | Add repair context such as `vcf_dir=data/vc_wgs_vcf`. |
| `/plan-retry [step_id]` | Retry a failed or named plan step. |
| `/plan-resume` | Resume a paused or repaired plan. |
| `/plan-skip <step_id>` | Record a step id to skip (advisory — does not yet alter execution; use `/plan-edit` to change the plan). |

Long-running bioinformatics steps should either stream progress through the
foreground tool log or run through the background job tools. Use `/jobs` to see
running jobs and logs, and `/artifacts` after a step or plan completes to see
full output paths. Tune long-run behavior in `.env`:

```bash
PLAN_BUILD_TIMEOUT_S=240
PLAN_STEP_TIMEOUT_S=240
EXEC_LONG_TOOL_TIMEOUT_S=7200
```

Relative `cwd` and output paths are resolved under the active workspace. Start
`biobank` from the project folder that should own the inputs, scripts, logs, and
outputs, or provide paths relative to that folder.

### WGS and VirtualCell

The repository can discover local VCFs under `data/vc_wgs_vcf`. The WGS skills include:

- `wgs_environment_check`
- `vcf_sample_list`
- `vcf_cohort_stats`
- `vcf_qc`
- `vcf_pca`
- `vcf_kinship`
- `vcf_association`
- `vcf_burden_test`
- `vcf_annotation`
- `pathway_enrichment`
- `virtualcell_data_inventory`
- `virtualcell_multimodal_link`

External genomics tools such as `bcftools`, `tabix`, and `plink2` are useful for standard workflows. The agent can still run exploratory Python-backed VCF analysis when those tools are missing, provided the VCF files and indexes are available.

### Research Mode

Use `/research` for a cited multi-source biomedical brief:

```text
biobank > /research What is the current evidence linking TYR, HLA, and immune regulation to vitiligo?
```

The research path uses configured web/literature skills and local report directories. It is best for background synthesis, target context, and study-design support, not for making causal claims from local data alone.

### Resume, Audit, Replay, and Evolution

Runtime sessions are persisted under the configured memory directory. Use:

| Command | Use |
| --- | --- |
| `/resume` or `/resume --last` | Resume or inspect saved sessions. |
| `/audit [session-id]` | Produce a read-only audit of events, tools, plans, and evidence. |
| `/replay [session-id|trajectory.jsonl]` | Replay a trajectory without model/tool execution. |
| `/harness <task.json>` | Run a versioned runtime harness task. |
| `/learn [--write]` | Mine the active trajectory for review-only improvement proposals. |
| `/evolve [--write|--apply]` | Review controlled self-evolution proposals. Persistent apply is approval-gated. |

Self-evolution is intentionally conservative: proposals are review-oriented by default, and persistent code mutation should be treated as a gated engineering workflow with tests and audit artifacts.

## Configuration

Configuration is loaded from `.env` through `biobank_agent.config.Settings`.

| Setting | Purpose |
| --- | --- |
| `LLM_BASE_URL` | OpenAI-compatible provider endpoint. |
| `LLM_API_KEY` | Provider API key. Keep this out of logs and commits. |
| `LLM_MODEL` | Primary model for normal turns. |
| `LLM_REQUEST_TIMEOUT_S` | Per-request provider timeout for model calls. |
| `LLM_MAX_RETRIES` | Retry count for transient provider errors outside plan-time timeout overrides. |
| `LLM_RETRY_BASE_DELAY_S` | Base retry backoff delay for provider calls. |
| `DATA_DIR` | Processed biobank or VirtualCell-style data directory. |
| `RAW_DIR` | Optional raw data fallback directory. |
| `REPORTS_DIR` | Generated reports and analysis artifacts. |
| `PLANS_DIR` | Saved plan checkpoints. |
| `MEMORY_DIR` | Runtime sessions, memory, trajectories, and action graph state. |
| `PLAN_BUILD_TIMEOUT_S` | Wall-clock cap for generating a plan before saving a diagnosis. |
| `PLAN_STEP_TIMEOUT_S` | Wall-clock cap for one autonomous plan step; set `0` to disable. |
| `EXEC_DEFAULT_TIMEOUT_S` | Default timeout for shell/Python tools. |
| `EXEC_LONG_TOOL_TIMEOUT_S` | Minimum timeout for known long bioinformatics tools. |
| `JOBS_DIR_NAME` | Workspace subdirectory for background job metadata and logs. |
| `MULTI_MODEL_ENABLED` | Enable multi-model routing when configured. |
| `MCP_CONFIG_PATH` | Optional MCP server configuration path. |

See [.env.example](.env.example) for the current template.

## Outputs and Provenance

Biobank Agent separates human-facing results from audit records:

- `reports/` contains generated analysis outputs, figures, reports, audit files, and research briefs.
- `plans/` contains plan checkpoints and saved plan artifacts.
- `MEMORY_DIR` contains runtime sessions, trajectories, action graph state, and long-term memory.
- `/audit` and `/replay` provide reproducibility checks after a run.

Generated reports should state method assumptions, data limitations, statistical caveats, and whether a workflow is exploratory or standard.

## Documentation

Recommended reading order:

1. [Quick Start](docs/guides/QUICK_START.md) - install, configure, launch, and run first checks.
2. [End-to-End Tutorial](docs/guides/END_TO_END_TUTORIAL.md) - full Agent + WGS path with repair, research, resume, audit, replay, and evolution.
3. [Documentation Index](docs/README.md) - all guides, architecture docs, examples, data references, and related works.
4. [Architecture Overview](docs/architecture/OVERVIEW.md) - system modules and runtime design.
5. [Custom Skills](docs/guides/CUSTOM_SKILLS.md) - write new `@skill` tools.
6. [Plugin Integration](docs/guides/PLUGIN_INTEGRATION.md) - install and use the repo-shipped plugin bundles.

## Development and Tests

Install development dependencies:

```bash
pip install -e ".[all,dev]"
```

Run focused checks:

```bash
python -m pytest tests/test_cli_v3_modules.py tests/test_interactive_cli_runtime.py tests/wgs/test_vcf_manifest.py -q
```

Run broader workflow checks:

```bash
python -m pytest tests/test_research_mode.py tests/test_self_evolve.py tests/test_runtime_audit_harness.py tests/wgs -q
```

Run the full suite when preparing a release:

```bash
python -m pytest tests/ -q
```

Before committing, keep the tracked tree clean:

```bash
git ls-files | grep -E '^[^/]+\.(md|txt)$' | grep -vE '^(README|CHANGELOG|LICENSE)' || echo OK
```

The command should print `OK`.

## Safety and Data Governance

Biobank Agent is a research assistant, not a clinical decision system. Treat outputs as scientific workflow artifacts that require review. For real biobank data, follow the governing data access agreement, local privacy policy, disclosure-control rules, and publication review process. Keep secrets in `.env`, avoid committing generated reports that contain sensitive information, and use `/doctor`, `/audit`, and `/replay` to verify readiness and provenance.

## Citation

```bibtex
@software{biobank_agent,
  title   = {Biobank Agent: Autonomous Scientific Discovery for Population-Scale Biobank Research},
  author  = {AIH Group, CUHK},
  year    = {2026},
  url     = {https://github.com/cpa2001/BioBank-Agent}
}
```

## License

[MIT](LICENSE)
