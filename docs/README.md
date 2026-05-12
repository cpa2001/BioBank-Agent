# Documentation

## Quick Navigation

| Document | Description |
|----------|-------------|
| [guides/QUICK_START.md](guides/QUICK_START.md) | Getting started — installation, configuration, first queries |
| [design/REFERENCE.md](design/REFERENCE.md) | Technical reference tables — memory, config, skills, verification |
| [architecture/OVERVIEW.md](architecture/OVERVIEW.md) | System architecture — modules, flows, design principles |
| [architecture/V3.md](architecture/V3.md) | v3 foundation scope and verification contract |
| [architecture/V3_REMAINING_IMPLEMENTATION.md](architecture/V3_REMAINING_IMPLEMENTATION.md) | Evidence-based remaining v3 implementation sequence |
| [architecture/SKILLS.md](architecture/SKILLS.md) | Skills system deep-dive — registration, invocation, custom skills |
| [guides/CUSTOM_SKILLS.md](guides/CUSTOM_SKILLS.md) | How to write your own skills |
| [guides/PLUGIN_INTEGRATION.md](guides/PLUGIN_INTEGRATION.md) | External agent and plugin integration |
| [guides/CLI_COMMAND_PLUGINS.md](guides/CLI_COMMAND_PLUGINS.md) | Add third-party slash commands through the v3 command registry |
| [guides/OBSERVABILITY.md](guides/OBSERVABILITY.md) | Local telemetry and OpenTelemetry/Jaeger runbook |
| [examples/README.md](examples/README.md) | Manual plan, SDK, MCP, and replication smoke-test examples |
| [design/TECH_WHITEPAPER.md](design/TECH_WHITEPAPER.md) | Technical whitepaper — design philosophy and roadmap (Chinese) |
| [data/UKB_DATA_REFERENCE.md](data/UKB_DATA_REFERENCE.md) | UK Biobank data modalities — coverage, field IDs, file layout |
| [related_works/GENETIC_TARGET_PRIORITIZATION.md](related_works/GENETIC_TARGET_PRIORITIZATION.md) | Rare-variant burden target-prioritization workflow mapping |
| [related_works/TARGET_ANNOTATION_ENRICHMENT.md](related_works/TARGET_ANNOTATION_ENRICHMENT.md) | Target annotation and enrichment context boundaries |

## Directory Structure

```
docs/
├── architecture/       System architecture and module maps
│   ├── OVERVIEW.md     High-level architecture
│   ├── V3.md           v3 foundation verification contract
│   └── SKILLS.md       Skill system deep-dive
├── design/             Design philosophy and reference
│   ├── REFERENCE.md    Quick-reference tables
│   └── TECH_WHITEPAPER.md  Technical whitepaper (中文)
├── guides/             User-facing guides
│   ├── QUICK_START.md  Getting started
│   ├── CUSTOM_SKILLS.md  Writing custom skills
│   ├── PLUGIN_INTEGRATION.md  External integrations
│   ├── CLI_COMMAND_PLUGINS.md  Slash command plugin modules
│   └── OBSERVABILITY.md  Telemetry and OpenTelemetry
├── data/               UK Biobank data documentation
│   └── UKB_DATA_REFERENCE.md  All modalities in one file
├── examples/           Human, SDK, MCP, and strict live-audit runbooks
└── related_works/      Paper traceability
    ├── MANIFEST.md     Concept → paper → integration mapping
    ├── GENETIC_TARGET_PRIORITIZATION.md
    └── TARGET_ANNOTATION_ENRICHMENT.md
```

## Agent-Visible Docs

The `project_doc` skill can list, search, and read curated Markdown from the
root README, `docs/`, and repo-local plugin guides. Raw research notes and
generated outputs are intentionally excluded from this skill and from published
documentation.

## For New Contributors

1. **Start with** [QUICK_START.md](guides/QUICK_START.md) — get the agent running
2. **Understand architecture** via [OVERVIEW.md](architecture/OVERVIEW.md) — how it all fits together
3. **Look up details** in [REFERENCE.md](design/REFERENCE.md) — tables for memory, config, verification
4. **Extend with skills** using [CUSTOM_SKILLS.md](guides/CUSTOM_SKILLS.md) — the `@skill` decorator pattern
