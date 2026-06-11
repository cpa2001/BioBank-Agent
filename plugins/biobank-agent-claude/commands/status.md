---
description: Show Biobank Agent runtime status and local Codex/Claude bridge availability
argument-hint: ""
disable-model-invocation: true
allowed-tools: Bash(python:*), Bash(git:*)
---

Run both commands and present the JSON outputs compactly:

```bash
python "${CLAUDE_PLUGIN_ROOT}/../biobank-agent/scripts/biobank_agent_bridge.py" status
python "${CLAUDE_PLUGIN_ROOT}/../biobank-agent/scripts/biobank_agent_bridge.py" external-status
```

Do not modify files.
