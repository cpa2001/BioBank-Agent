"""In-process sub-agent forked from a parent ``AsyncAgent``.

A ``SubAgent`` shares the parent's ``ToolRegistry``, ``LongTermMemory``,
``LLMClient``, and active ``DataManager`` rather than spawning a separate
process, so a delegated investigation or verifier pass can reuse the
parent context without rebuilding state.

The API is intentionally minimal — ``mode``, ``last_n_turns``,
``forward_events_to`` — so the reflexion loop can spawn a "verifier"
sub-agent without bringing the whole agent state along.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Optional

from ..events import AgentEvent, AgentEventBus, AgentEventType
from ..runtime import AsyncAgent

logger = logging.getLogger(__name__)


class SubAgentMode(str, Enum):
    WORKER = "worker"        # delegated execution of a focused sub-task
    EXPLORER = "explorer"    # read-only investigation + report-back
    VERIFIER = "verifier"    # adversarial review of parent's claims


@dataclass
class SubAgentSpec:
    mode: SubAgentMode = SubAgentMode.WORKER
    last_n_turns: int = 4
    inherit_memory: bool = True
    label: str = ""
    parent_turn_id: Optional[str] = None


@dataclass
class SubAgentRun:
    spec: SubAgentSpec
    final_text: str = ""
    events: list[AgentEvent] = field(default_factory=list)


class SubAgent:
    """Light wrapper around ``AsyncAgent`` that forwards events upstream.

    Usage::

        sub = SubAgent.from_parent(parent_agent, SubAgentSpec(mode=...))
        run = await sub.run("review the cohort design")
    """

    def __init__(
        self,
        runtime: AsyncAgent,
        *,
        spec: SubAgentSpec,
        forward_to: Optional[AgentEventBus] = None,
        fork_messages: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        self.runtime = runtime
        self.spec = spec
        self.forward_to = forward_to
        self._fork_messages = list(fork_messages or [])

    @classmethod
    def from_parent(
        cls,
        parent: AsyncAgent,
        spec: SubAgentSpec,
        *,
        forward_to: Optional[AgentEventBus] = None,
    ) -> "SubAgent":
        legacy = parent.legacy
        # Sub-agents share the legacy Agent (registry, dm, memory, llm)
        # but get their own message list bounded to the last N turns.
        if spec.last_n_turns > 0:
            legacy_messages = list(legacy.messages[-spec.last_n_turns * 2:])
        else:
            legacy_messages = list(legacy.messages)
        # Shallow-copy the legacy agent identity-wise; the real
        # isolation comes from the new AsyncAgent owning its own bus.
        # (Full process isolation is M3 territory.)
        sub_runtime = AsyncAgent(legacy)
        return cls(
            sub_runtime,
            spec=spec,
            forward_to=forward_to or parent.bus,
            fork_messages=legacy_messages,
        )

    async def run(self, query: str) -> SubAgentRun:
        events: list[AgentEvent] = []
        legacy = self.runtime.legacy
        parent_messages = legacy.messages
        legacy.messages = list(self._fork_messages)
        try:
            async for ev in self.runtime.stream_events(query):
                tagged = AgentEvent.make(
                    ev.type,
                    turn_id=ev.turn_id,
                    tool_call_id=ev.tool_call_id,
                    model_id=ev.model_id,
                    **{
                        **ev.payload,
                        "subagent_mode": self.spec.mode.value,
                        "subagent_label": self.spec.label,
                        "parent_turn_id": self.spec.parent_turn_id,
                    },
                )
                if self.forward_to is not None:
                    await self.forward_to.publish(tagged)
                events.append(tagged)
        finally:
            legacy.messages = parent_messages

        final_text = ""
        for ev in events:
            if ev.type == AgentEventType.TURN_FINISHED:
                final_text = ev.payload.get("final_text", "") or final_text
            elif ev.type == AgentEventType.MESSAGE_COMPLETE:
                final_text = ev.payload.get("text", "") or final_text
        return SubAgentRun(spec=self.spec, final_text=final_text, events=events)


__all__ = ["SubAgentMode", "SubAgentSpec", "SubAgentRun", "SubAgent"]
