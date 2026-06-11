---
name: biobank-agent-runtime
description: Use when Codex is working inside the Biobank Agent repository and needs to plan, run, or review population-biobank workflows, agent skills, report generation, guardrails, or tests.
---

# Biobank Agent Runtime

Use this skill when the task involves `biobank_agent/`, registered `@skill`
tools, cohort/report generation, statistical guardrails, external-agent
delegation, or tests in this repository.

## Operating Rules

- Treat participant-level data as sensitive. Prefer aggregate records, synthetic
  fixtures, and report artifacts unless the user explicitly authorizes raw data
  inspection.
- Preserve causal-language guardrails. Unsupported intervention or causal claims
  must be downgraded, blocked, or reported with explicit caveats.
- For therapeutic target prioritization, use `genetic_target_hypothesis` with
  GeneBass-like burden statistics and keep annotation-rich context separate
  from the genetics-first rank. Use `prefiltered=true` only when every supplied
  row has already been filtered to the requested phenotype.
- For target interpretation, use `target_annotation_context` and
  `target_enrichment` only as biobank evidence-context layers. They may explain
  target biology, expression, trials, and pathways, but must not alter a GWAS or
  burden-derived ranking.
- Prefer the repository extension path: helper module, `@skill` wrapper, CLI or
  agent integration, targeted tests, then report/safety verification.
- For report work, check scientific content, uncertainty, cohort provenance,
  privacy/small-count handling, figure references, and reproducibility metadata.
- Do not rely on passing tests alone if the prompt asks for report quality or
  scientific validity; inspect the generated artifact or the renderer code.
- Use the `project_doc` Biobank Agent skill when repository guidance, UKB data
  reference notes, custom skill instructions, or plugin behavior need to be
  loaded into the agent context.

## Useful Commands

```bash
python plugins/biobank-agent/scripts/biobank_agent_bridge.py status
python plugins/biobank-agent/scripts/biobank_agent_bridge.py external-status
python -m pytest tests -q -m "not integration"
python -m pytest tests/test_report_v2.py tests/test_agentic_cohort_discovery.py -q
biobank eval --suite research_eval_v1 --mode baseline
```

## Documentation Entry Points

- `README.md`: public overview, install path, CLI commands, verification.
- `docs/data/UKB_DATA_REFERENCE.md`: local UKB modality and field coverage.
- `docs/related_works/GENETIC_TARGET_PRIORITIZATION.md`: rare-variant burden
  target-prioritization design mapping.
- `docs/guides/PLUGIN_INTEGRATION.md`: target annotation, enrichment, and
  external-agent integration boundaries.
- `docs/architecture/SKILLS.md`: skill registration and invocation rules.

## Review Checklist

- New capabilities are visible through `@skill` discovery or CLI commands.
- Report and guardrail layers understand new record types.
- Tests cover success, invalid inputs, unavailable dependencies, and boundary
  values such as small cohorts, impossible metrics, and missing external CLIs.
- Public-paper references are used as calibration context, not as unsupported
  causal evidence.
