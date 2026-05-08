# External Skills and Plugin Integration

This note records the integration choices for letting Biobank Agent use local
agent tooling and relevant open-source skill/plugin ecosystems.

## Survey

- `openai/codex-plugin-cc`: useful reference for a local companion pattern:
  commands delegate to a script, expose setup/status/result/review flows, and
  keep review output structured. Biobank Agent adopts the same separation of
  plugin UX from runtime execution, but implements the bridge in Python so it is
  callable from `@skill` tools.
- `sendbird/cc-plugin-codex`: relevant opposite-direction reference for running
  Claude Code from Codex with tracked review/rescue jobs. Biobank Agent uses the
  same idea at a smaller first layer: local `claude` calls for plan/review with
  read-only defaults.
- Codex Marketplace, mdskills.ai, SkillKit, and public Claude skill libraries:
  useful discovery surfaces, but broad vendoring is risky for biomedical work.
  The safe default is to curate domain-specific skills instead of importing a
  generic skill library wholesale.
- GraphPop MCP: relevant population-genomics tooling. The repo includes an
  optional `biobank_agent.mcp.graphpop_client` bridge and tests for the 21-tool
  registry without requiring a running server.

## Implemented In This Repo

- `biobank_agent/external_agents.py`: safe subprocess runner for local `codex`
  and `claude` CLIs. Defaults to read-only planning/review modes and reports
  unavailable binaries or timeouts as structured results.
- `biobank_agent/skills/external_agents.py`: agent-callable skills:
  `external_agent_status`, `codex_plan`, `claude_plan`,
  `codex_check_execution`, and `claude_check_execution`.
- `biobank_agent/skills/project_doc.py`: read-only access to curated README,
  data reference, architecture, guide, and plugin Markdown so the agent can
  inspect repository documentation without opening raw notes or generated
  outputs.
- `biobank_agent/skills/genetic_target_hypothesis.py`: genetics-first
  rare-variant burden target prioritization exposed as a normal agent skill.
- `biobank_agent/skills/target_annotation_context.py`: biobank target-context
  annotation from Open Targets, UniProt, GTEx, ClinicalTrials.gov, and optional
  CELLxGENE snapshots. These annotations support interpretation only and do not
  change target ranking.
- `biobank_agent/skills/target_enrichment.py`: local GMT target enrichment with
  optional GSEApy support when installed.
- CLI commands: `/external-agents`, `/codex-plan`, `/codex-check`,
  `/claude-plan`, and `/claude-check`.
- Codex plugin bundle: `plugins/biobank-agent` plus
  `.agents/plugins/marketplace.json`.
- Claude Code plugin bundle: `plugins/biobank-agent-claude` plus
  `.claude-plugin/marketplace.json`.

## Guardrails

- External-agent calls are argv-list subprocesses, never shell strings.
- Planning and review default to read-only modes.
- Claude Code review disallows edit tools by default.
- Codex execution uses `--sandbox read-only` and `--ask-for-approval never`
  unless a future caller explicitly opts into write-capable delegation.
- Tests mock the CLIs, so normal CI does not spend model quota.

## Deferred Integrations

- Do not vendor broad open-source skill packs automatically. Curate individual
  skills only after license, prompt-safety, and biomedical-governance review.
- A tracked background-job lifecycle, similar to `codex-plugin-cc`, can be added
  once the synchronous bridge is stable under real long-running workloads.
