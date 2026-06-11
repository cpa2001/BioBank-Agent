---
description: Build a guarded Biobank Agent research or implementation plan
argument-hint: "<research or implementation goal>"
allowed-tools: Read, Glob, Grep, Bash(python:*), Bash(git:*)
---

Plan the requested Biobank Agent work.

Raw request:
$ARGUMENTS

Rules:

- Start by running the bridge status command:

```bash
python "${CLAUDE_PLUGIN_ROOT}/../biobank-agent/scripts/biobank_agent_bridge.py" status
```

- Inspect only files needed to build a concrete plan.
- Do not edit files.
- Include a skill/command sequence, data/privacy assumptions, statistical
  guardrails, report-quality gates, and the tests that should prove the work.
- Use `project_doc` as the Biobank Agent path for reading curated README,
  data-reference, architecture, guide, and plugin Markdown.
- For rare-variant therapeutic target prioritization, include
  `genetic_target_hypothesis` and require burden-statistic provenance,
  validation caveats, and report/safety review.
- For target interpretation, include `target_annotation_context` and
  `target_enrichment` only when they support a biobank GWAS, burden, or target
  list. Treat external annotation/enrichment as context, not rank evidence.
