# End-to-End Tutorial: Agent + WGS Workflow

This tutorial walks through a complete local Biobank Agent workflow: environment setup, API connectivity, runtime readiness, WGS/VCF analysis planning, plan repair, research mode, resume, audit, replay, and review-gated self-evolution.

The goal is not to force one fixed pipeline. The goal is to teach the operating pattern other users should follow when they bring their own data and research question.

## 0. What You Will Run

You will:

1. Install the package in an isolated Python environment.
2. Configure `.env` for an OpenAI-compatible model provider.
3. Verify the API before spending time on an agent run.
4. Launch the interactive shell.
5. Check tool, skill, and WGS readiness.
6. Draft and execute a WGS case-control plan.
7. Repair a blocked plan without restarting the session.
8. Run cited research for biological context.
9. Resume and audit the session.
10. Replay the trajectory and mine review-only self-evolution proposals.

The tutorial assumes commands are run from the repository root.

## 1. Install

```bash
conda create -n biobank-agent python=3.11 -y
conda activate biobank-agent
pip install -e ".[all,dev]"
```

Optional external genomics tools:

```bash
bcftools --version
tabix --version
plink2 --version
```

If these are unavailable, the agent can still run exploratory Python-backed VCF workflows for supported tasks. Standard GWAS-style execution is stronger when `bcftools`, `tabix`, and `plink2` are installed.

## 2. Configure `.env`

```bash
cp .env.example .env
```

Edit the provider and path settings:

```ini
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=<your-api-key>
LLM_MODEL=deepseek/deepseek-v4-pro

DATA_DIR=./data
RAW_DIR=./raw
REPORTS_DIR=./reports
PLANS_DIR=./plans
MEMORY_DIR=~/.biobank_agent
```

For local WGS work, place VCFs under one of the configured or discovered VCF directories. The current repo-local path is:

```text
data/vc_wgs_vcf/
```

Expected VCF layout:

```text
data/vc_wgs_vcf/<sample-id>.genotyper.vcf.gz
data/vc_wgs_vcf/<sample-id>.genotyper.vcf.gz.tbi
```

## 3. Verify API Connectivity

Run a direct model call before opening the agent:

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

Success criteria:

- The command exits with code 0.
- The printed text contains `API_OK`.
- Usage metadata is present or the provider returns a normal completion response.

If this step fails, do not continue to `/plan`. Fix `.env` first.

## 4. Launch and Inspect Startup

```bash
biobank
```

The startup banner should show:

- package version
- workspace
- session id
- active role
- permission mode
- provider roles
- tools summary
- WGS readiness summary
- MCP loaded tool count
- command palette

Then run:

```text
biobank > /doctor
biobank > /tools
biobank > /skills
```

Success criteria:

- `/doctor` reports provider configuration and local path readiness without exposing secrets.
- `/tools` and `/skills` are aliases: both print one **Available Tools** table by category — currently ~116 entries (105 analysis skills plus the native runtime tools).
- WGS readiness reports whether VCF files, indexes, and standard external tools are available.

## 5. Explore the Local WGS Inputs

Use ordinary natural language first:

```text
biobank > What WGS VCF samples and phenotype-like labels are available in this workspace?
```

You can also ask for a narrower diagnostic:

```text
biobank > Check the local WGS environment and list available VCF samples.
```

The agent should prefer safe inspection skills such as:

- `wgs_environment_check`
- `vcf_sample_list`
- `virtualcell_data_inventory`
- `virtualcell_multimodal_link`

Do not start a full association analysis until VCF discovery and phenotype assumptions are clear.

## 6. Draft a WGS Plan

Start with a concrete research objective:

```text
biobank > /plan Compare vitiligo cases and controls using the available WGS VCF files. Run QC, exploratory PCA or kinship checks if feasible, association or burden tests where sample size permits, annotate candidate loci, and write an auditable report.
```

Review the draft. A good draft should include:

- input readiness or VCF discovery
- sample and variant QC
- phenotype harmonization or explicit case-control assumptions
- PCA or kinship checks when feasible
- association or burden testing with caveats for small sample size
- annotation and pathway context
- report generation and statistical limitations

If the plan is wrong before execution, edit it:

```text
biobank > /plan-edit Use exploratory mode unless standard GWAS dependencies are available. Do not claim discovery power from small sample counts.
```

## 7. Execute the Plan

```text
biobank > /plan-approve
```

During execution, monitor:

- current plan step
- tool calls
- failures or pauses
- generated artifacts
- final report summary

If a step fails, do not restart the CLI. Use the repair flow below.

## 8. Repair a Blocked or Failed Plan

First ask in natural language:

```text
biobank > what is the problem?
```

Then run the explicit diagnostic command:

```text
biobank > /plan-diagnose
```

Common repairs:

### Missing or Wrong VCF Directory

```text
biobank > /plan-use vcf_dir=data/vc_wgs_vcf
biobank > /plan-retry
```

### Standard Tools Missing

```text
biobank > /plan-use workflow_mode=exploratory
biobank > /plan-retry
```

### Plan Needs Different Scientific Assumptions

```text
biobank > /plan-edit Treat this as an exploratory WGS readiness and candidate-context report, not a powered GWAS.
biobank > /plan-retry
```

### Continue From Current Failed Step

```text
biobank > continue
```

The interactive shell treats `continue` as a retry request when the active plan is paused or failed and the repair context is sufficient.

## 9. Inspect Outputs

After execution, use:

```text
biobank > /history
biobank > /figures
biobank > /audit
```

Expected artifact locations:

- generated reports and figures under `reports/`
- plan checkpoints under `plans/`
- session trajectories and action graph records under `MEMORY_DIR`

The final report should distinguish:

- data readiness findings
- exploratory QC and association results
- statistically powered claims, if any
- limitations from sample size, missing tools, missing phenotype labels, population structure, or local data layout

## 10. Add Cited Biological Context

Run research mode for literature grounding:

```text
biobank > /research Summarize current evidence linking vitiligo risk to HLA, TYR, NLRP1, immune regulation, pigmentation pathways, and WGS variant interpretation.
```

Use research mode to support background and interpretation. Do not use it as a substitute for local statistical evidence.

Good research-mode outputs should include:

- cited sources
- distinction between established findings and hypotheses
- relevance to the local WGS workflow
- caveats about ancestry, phenotype definition, sample size, and assay modality

## 11. Resume Later

Exit safely:

```text
biobank > /quit
```

Resume:

```bash
biobank
```

```text
biobank > /resume --last
```

Or list available sessions:

```text
biobank > /resume
```

After resuming, verify state:

```text
biobank > /status
biobank > /plans
biobank > /audit
```

## 12. Replay and Audit

Use replay to validate the recorded trajectory without re-running model or tool execution:

```text
biobank > /replay
```

Use audit for a human-readable evidence summary:

```text
biobank > /audit
```

Audit output should help answer:

- What did the user ask?
- What plan did the agent approve?
- Which tools and skills ran?
- What data paths were touched?
- What artifacts were generated?
- Which claims were supported, limited, or unsupported?
- How can the session be reproduced or reviewed?

## 13. Review-Only Self-Evolution

After a failed or imperfect run, mine the trajectory:

```text
biobank > /learn
```

Then review possible improvement proposals:

```text
biobank > /evolve
```

By default, use self-evolution as a review workflow. Persistent apply should be treated as a separate engineering change:

```text
biobank > /evolve --write
```

Only use approval-gated persistent apply when you are intentionally changing code and are prepared to run tests:

```text
biobank > /evolve --apply
```

Good self-evolution proposals should include:

- the observed failure pattern
- why the current workflow was insufficient
- a proposed skill, harness, prompt, or routing improvement
- tests or acceptance criteria
- no uncontrolled mutation of sensitive paths

## 14. Suggested Acceptance Checklist

For a complete local tutorial run, verify:

- API smoke test returns `API_OK`.
- `biobank` starts and shows a session id.
- `/doctor` runs without leaking secrets.
- `/tools` and `/skills` show registered capabilities.
- WGS readiness identifies local VCF files or clearly explains missing inputs.
- `/plan` creates a structured plan.
- `/plan-approve` executes or pauses with a clear diagnosis.
- Failed plans can be repaired with `/plan-diagnose`, `/plan-use`, `/plan-edit`, and `/plan-retry`.
- `/research` produces cited context.
- `/audit` summarizes the run.
- `/resume --last` restores the session.
- `/replay` validates the recorded trajectory path.
- `/learn` and `/evolve` produce review-only improvement proposals when there is enough trajectory evidence.

## 15. Common Failure Patterns

| Symptom | Likely cause | Repair |
| --- | --- | --- |
| API smoke test fails | Wrong endpoint, key, or model id | Fix `.env`, rerun smoke test. |
| Startup works but `/plan` fails early | Planner model unavailable or provider routing issue | Run `/doctor`, check provider roles, retry with available model. |
| WGS plan cannot find VCFs | Missing or wrong VCF directory | `/plan-use vcf_dir=data/vc_wgs_vcf`, then `/plan-retry`. |
| Standard GWAS step fails | Missing `bcftools`, `tabix`, or `plink2` | Install tools or use `/plan-use workflow_mode=exploratory`. |
| Association results are weak or absent | Small sample size or missing phenotype labels | Report as exploratory readiness, not powered discovery. |
| Plan failure is unclear | Need active diagnosis | Ask `what is the problem?`, then run `/plan-diagnose`. |
| Need to continue after restart | Existing session persisted | Start `biobank`, then `/resume --last`. |

## 16. Next Steps

After completing this tutorial:

- Read [CUSTOM_SKILLS.md](CUSTOM_SKILLS.md) to add new domain skills.
- Read [CLI_COMMAND_PLUGINS.md](CLI_COMMAND_PLUGINS.md) to add slash-command modules.
- Read [OBSERVABILITY.md](OBSERVABILITY.md) to inspect telemetry and OpenTelemetry settings.
- Read [../architecture/OVERVIEW.md](../architecture/OVERVIEW.md) to understand the runtime architecture.
- Read [../data/UKB_DATA_REFERENCE.md](../data/UKB_DATA_REFERENCE.md) for UK Biobank data layout conventions.
