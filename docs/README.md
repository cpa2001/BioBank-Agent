# Documentation

This directory contains user guides, architecture notes, data references, examples, plugin documentation, and related-work mappings for Biobank Agent.

## Recommended Reading Order

| Step | Document | Use |
| --- | --- | --- |
| 1 | [../README.md](../README.md) | Product overview, command surface, configuration, and workflow map. |
| 2 | [guides/QUICK_START.md](guides/QUICK_START.md) | Install, configure `.env`, verify API connectivity, launch the shell, and run first checks. |
| 3 | [guides/END_TO_END_TUTORIAL.md](guides/END_TO_END_TUTORIAL.md) | Complete Agent + WGS tutorial with plan repair, research, resume, audit, replay, and self-evolution. |
| 4 | [architecture/OVERVIEW.md](architecture/OVERVIEW.md) | System architecture, modules, runtime flow, and design principles. |
| 5 | [architecture/SKILLS.md](architecture/SKILLS.md) | Skill registration, invocation, and extension mechanics. |
| 6 | [data/UKB_DATA_REFERENCE.md](data/UKB_DATA_REFERENCE.md) | UK Biobank data modalities, field references, and file layout. |

## Quick Navigation

| Document | Description |
| --- | --- |
| [guides/QUICK_START.md](guides/QUICK_START.md) | Getting started with installation, configuration, API smoke test, and first CLI checks. |
| [guides/END_TO_END_TUTORIAL.md](guides/END_TO_END_TUTORIAL.md) | Full Agent + WGS workflow tutorial for real interactive use. |
| [guides/CUSTOM_SKILLS.md](guides/CUSTOM_SKILLS.md) | How to write new `@skill` tools. |
| [guides/PLUGIN_INTEGRATION.md](guides/PLUGIN_INTEGRATION.md) | External agent and plugin integration. |
| [guides/CLI_COMMAND_PLUGINS.md](guides/CLI_COMMAND_PLUGINS.md) | Add third-party slash commands through the v3 command registry. |
| [guides/OBSERVABILITY.md](guides/OBSERVABILITY.md) | Local telemetry and OpenTelemetry/Jaeger runbook. |
| [architecture/OVERVIEW.md](architecture/OVERVIEW.md) | System architecture and runtime design. |
| [architecture/V3.md](architecture/V3.md) | v3 foundation scope and verification contract. |
| [architecture/V3_REMAINING_IMPLEMENTATION.md](architecture/V3_REMAINING_IMPLEMENTATION.md) | Evidence-based remaining v3 implementation sequence. |
| [architecture/SKILLS.md](architecture/SKILLS.md) | Skills system deep dive. |
| [architecture/COMPATIBILITY.md](architecture/COMPATIBILITY.md) | Compatibility notes for the v3 runtime and CLI surface. |
| [design/REFERENCE.md](design/REFERENCE.md) | Technical reference tables for memory, config, skills, and verification. |
| [design/TECH_WHITEPAPER.md](design/TECH_WHITEPAPER.md) | Technical whitepaper and product rationale. |
| [data/UKB_DATA_REFERENCE.md](data/UKB_DATA_REFERENCE.md) | UK Biobank data modalities and field layout. |
| [examples/README.md](examples/README.md) | Manual plan, SDK, MCP, and replication smoke-test examples. |
| [related_works/MANIFEST.md](related_works/MANIFEST.md) | Concept to paper to integration mapping. |
| [related_works/GENETIC_TARGET_PRIORITIZATION.md](related_works/GENETIC_TARGET_PRIORITIZATION.md) | Rare-variant burden target-prioritization workflow mapping. |
| [related_works/TARGET_ANNOTATION_ENRICHMENT.md](related_works/TARGET_ANNOTATION_ENRICHMENT.md) | Target annotation and enrichment context boundaries. |

## Directory Structure

```text
docs/
|-- architecture/       System architecture, compatibility, release, and v3 notes
|-- data/               UK Biobank and data-layout references
|-- deep_research/      Research snapshots and long-form external research notes
|-- design/             Technical references and whitepaper material
|-- examples/           Manual runbooks and small integration examples
|-- guides/             User-facing setup, tutorial, skills, plugin, and observability guides
`-- related_works/      Paper traceability and research-to-implementation mappings
```

## Agent-Visible Docs

The `project_doc` skill can list, search, and read curated Markdown from the root README, `docs/`, and repo-local plugin guides. Generated reports, raw internal research notes, and temporary exploration outputs should not be added to the repository root.

## For New Contributors

1. Start with [QUICK_START.md](guides/QUICK_START.md).
2. Run the [END_TO_END_TUTORIAL.md](guides/END_TO_END_TUTORIAL.md) workflow.
3. Read [architecture/OVERVIEW.md](architecture/OVERVIEW.md) to understand the runtime.
4. Read [CUSTOM_SKILLS.md](guides/CUSTOM_SKILLS.md) before adding skills.
5. Keep generated reports in `reports/` and plans in `plans/`; do not place temporary summaries in the repository root.
