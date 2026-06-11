# Quick Start Guide

This guide gets Biobank Agent running from a fresh checkout and verifies the basic API, tool, skill, and WGS readiness paths. For the full workflow tutorial, continue with [END_TO_END_TUTORIAL.md](END_TO_END_TUTORIAL.md).

## 1. Create the Environment

From the repository root:

```bash
conda create -n biobank-agent python=3.11 -y
conda activate biobank-agent
pip install -e ".[all,dev]"
```

If you do not use conda, use any Python 3.10+ environment and run the same editable install command.

## 2. Configure `.env`

```bash
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

Use the provider, endpoint, and model that match your key. Biobank Agent uses an OpenAI-compatible client, so relay endpoints must expose a compatible `/v1/chat/completions` API.

## 3. Verify API Connectivity

Run this before any long agent workflow:

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

Expected result:

```text
API_OK
```

If this fails, fix `.env` first. Most failures are caused by a wrong base URL, missing key, unavailable model id, or a relay endpoint that is not OpenAI-compatible.

## 4. Launch the Agent

```bash
biobank
```

The startup banner should show the version, workspace, session id, active role, permission mode, provider roles, tool summary, WGS readiness summary, MCP tool count, and command palette.

Inside the shell, run:

```text
biobank > /doctor
biobank > /tools
biobank > /skills
```

Interpretation:

- `/doctor` checks provider config, data directories, report/memory paths, permission mode, tool readiness, and WGS dependency status.
- `/tools` and `/skills` are aliases: both print one **Available Tools** table grouped by category — currently ~126 entries (108 analysis skills plus the native runtime tools) — followed by a WGS readiness summary.

## 5. Run a First Natural-Language Turn

```text
biobank > What data and WGS inputs are currently available in this workspace?
```

For simple questions, ordinary text is enough. For multi-step workflows, use `/plan`.

## 6. Run a First Plan

```text
biobank > /plan Compare vitiligo cases and controls using the available WGS VCF files, then write a concise QC and association report.
biobank > /plan-approve
```

If the plan pauses or fails, do not exit the shell. Diagnose and repair in place:

```text
biobank > what is the problem?
biobank > /plan-diagnose
biobank > /plan-use vcf_dir=data/vc_wgs_vcf
biobank > /plan-use workflow_mode=exploratory
biobank > /plan-retry
```

The natural-language `continue` command also retries the current failed or paused plan step when the shell can infer the active repair path:

```text
biobank > continue
```

## 7. Resume, Audit, and Replay

Runtime sessions are persisted under `MEMORY_DIR`.

```text
biobank > /resume --last
biobank > /audit
biobank > /replay
```

Use `/audit` after any meaningful run to inspect the recorded trajectory, tools, plan steps, generated artifacts, and supported claims.

## 8. Useful Commands

| Command | Purpose |
| --- | --- |
| `/help` | Show registered slash commands. |
| `/status` | Show session, platform, memory, and token state. |
| `/doctor` | Run read-only readiness diagnostics. |
| `/tools` / `/skills` | List all registered tools by category (≈126: 108 analysis skills + native tools), with WGS readiness. Aliases — same output. |
| `/plan <task>` | Draft a structured plan. |
| `/plan-approve` | Execute the active plan. |
| `/plan-edit <feedback>` | Modify the active plan. |
| `/plan-diagnose` | Explain why the plan is blocked or failed. |
| `/plan-use key=value` | Add repair context such as `vcf_dir=data/vc_wgs_vcf`. |
| `/plan-retry [step_id]` | Retry a failed or named step. |
| `/research <question>` | Run a cited multi-source research brief. |
| `/resume [session-id|--last]` | Resume saved sessions. |
| `/audit [session-id]` | Audit runtime evidence and artifacts. |
| `/replay [session-id|trajectory.jsonl]` | Replay a trajectory without executing tools. |
| `/learn [--write]` | Mine the current trajectory for review-only improvement proposals. |
| `/evolve [--write|--apply]` | Review controlled self-evolution proposals. |

## 9. Troubleshooting

**API call fails**

- Recheck `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL`.
- Run the API smoke test again before using `/plan`.

**No data found**

- Run `/doctor`.
- Check `DATA_DIR` and `RAW_DIR`.
- For local WGS tests, check whether `data/vc_wgs_vcf` contains `.vcf.gz` files and `.tbi` indexes.

**WGS standard dependencies missing**

- Install `bcftools`, `tabix`, and `plink2` when you need a standard GWAS-style workflow.
- Use `/plan-use workflow_mode=exploratory` when exploratory Python-backed VCF analysis is acceptable.

**Plan failed without clear progress**

- Ask `what is the problem?`.
- Run `/plan-diagnose`.
- Use `/plan-edit`, `/plan-use`, or `/plan-retry`.
- Run `/audit` if you need the recorded execution evidence.

**Import errors**

- Reinstall in the active environment:

```bash
pip install -e ".[all,dev]"
```

## 10. Test the Checkout

Focused runtime and WGS checks:

```bash
python -m pytest tests/test_cli_v3_modules.py tests/test_interactive_cli_runtime.py tests/wgs/test_vcf_manifest.py -q
```

Broader workflow checks:

```bash
python -m pytest tests/test_research_mode.py tests/test_self_evolve.py tests/test_runtime_audit_harness.py tests/wgs -q
```

Full test suite:

```bash
python -m pytest tests/ -q
```
