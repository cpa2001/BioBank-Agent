# Plugin Integration

Biobank Agent ships two repo-resident plugin bundles so the local CLI can be
launched from compatible agent runtimes that read marketplace manifests. The
bundles are user-facing surfaces and only expose status, planning, and review
commands that map to the Biobank Agent runtime — they do not install any
external services.

## Repo-Resident Plugin Bundles

- `plugins/biobank-agent/` with `.agents/plugins/marketplace.json`: the primary
  plugin bundle. A small Python bridge script
  (`plugins/biobank-agent/scripts/biobank_agent_bridge.py`) exposes
  `status` and `test` subcommands so the runtime is introspectable from the
  plugin host.
- `plugins/biobank-agent-claude/` with `.claude-plugin/marketplace.json`:
  alternative plugin manifest using the same bridge for plugin hosts that
  consume the `.claude-plugin/` marketplace shape.

Both manifests point at the same skill bundle in
`plugins/biobank-agent/skills/biobank-agent-runtime/SKILL.md`. Each plugin is
versioned alongside the rest of the source tree so reproducibility, audit, and
release tagging remain straightforward.

## Bridge Subcommands

`plugins/biobank-agent/scripts/biobank_agent_bridge.py` is the bridge entry
point invoked by the plugin host:

- `status`: dumps the registered skill catalog, repo root, pyproject path, and
  a preview of available skills as JSON.
- `test`: runs the local pytest suite (or a filtered subset) for plugin-host
  health checks.

## MCP Integration

For richer cross-tool integration — wrapping shell utilities, third-party LLM
runtimes, or other research tools — use the MCP manager (`/mcp-list`,
`/mcp-start`, `/mcp-health`, `/mcp-call`, `/mcp-stop`). MCP is the supported
integration substrate; the previous in-process bridge for external CLI agents
has been removed.

## Guardrails

- All plugin actions go through the registered skill set; the plugin host
  cannot bypass the workspace approval policy.
- Plugin tests run inside the standard CI pytest path with no extra credentials.
- The plugin manifests are tracked in source; everything else under
  `.agents/`, `.claude-plugin/`, `.claude/`, and `.codex/` is gitignored.
