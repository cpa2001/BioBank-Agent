"""Biobank Agent core framework (v3 scaffold).

Streaming-first, protocol-driven modules that wrap the legacy synchronous
Agent in agent.py. Only modules with real callers in the CLI/runtime should be
treated as production paths; empty subpackages are migration placeholders until
their entrypoints are wired and covered by integration tests.

Layers:
    runtime.py     - AsyncAgent main loop (async generator)
    events.py      - AgentEvent enum + AgentEventBus + PII scrubbing
    compaction.py  - smart context compaction (replaces [:8000])
    tools/         - ToolHandler protocol + registry + scheduler + approval
    memory/        - migration placeholder
    planning/      - migration placeholder
    orchestration/ - sub-agent prototype
    evolution/     - reflexion / patch_classifier / auto_merger (M4)
    safety/        - reproducibility auto-hook / data_fingerprint / disclosure
    llm/           - streaming-first client + incremental tool args parser
"""
