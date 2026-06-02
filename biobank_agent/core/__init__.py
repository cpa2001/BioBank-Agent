"""Biobank Agent core framework.

Streaming-first, protocol-driven modules shared by the runtime-backed CLI,
tool scheduler, approval layer, memory/action graph, and TUI renderers. Public
subpackages are expected to be wired through the runtime or covered by tests
before they are treated as release surfaces.

Layers:
    runtime.py     - AsyncAgent main loop (async generator)
    events.py      - AgentEvent enum + AgentEventBus + PII scrubbing
    compaction.py  - smart context compaction (replaces [:8000])
    tools/         - ToolHandler protocol + registry + scheduler + approval
    memory/        - action graph and memory-facing helpers
    planning/      - planning protocol helpers
    orchestration/ - sub-agent prototype
    evolution/     - reflexion / patch_classifier / auto_merger (M4)
    safety/        - reproducibility auto-hook / data_fingerprint / disclosure
    llm/           - streaming-first client + incremental tool args parser
"""
