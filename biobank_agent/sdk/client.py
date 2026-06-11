"""Programmatic client facade for embedding Biobank Agent."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from biobank_agent.agent import Agent
from biobank_agent.config import Settings, get_settings
from biobank_agent.core.events import AgentEvent
from biobank_agent.core.runtime import AsyncAgent


@dataclass
class AsyncBiobankClient:
    """Async SDK wrapper around the production agent runtime.

    The SDK intentionally uses the same ``Agent`` and ``AsyncAgent`` code paths
    as the CLI so embedded use does not diverge from manual REPL behavior.
    """

    settings: Optional[Settings] = None
    agent: Optional[Agent] = None

    def __post_init__(self) -> None:
        if self.agent is None:
            settings = self.settings or get_settings()
            settings.ensure_dirs()
            self.agent = Agent(settings)
        self.runtime = AsyncAgent(self.agent)

    async def run(self, query: str) -> str:
        """Run one turn and return the final text response."""
        return await self.runtime.run_to_text(query)

    async def stream(self, query: str) -> AsyncIterator[AgentEvent]:
        """Stream raw ``AgentEvent`` objects for one turn."""
        async for event in self.runtime.stream_events(query):
            yield event

    def skills(self) -> list[dict]:
        """Return registered skill metadata."""
        return self.agent.registry.list_skills() if self.agent is not None else []

    def close(self) -> None:
        """Close local resources where the wrapped agent exposes them."""
        dm = getattr(self.agent, "dm", None)
        conn = getattr(dm, "conn", None)
        close = getattr(conn, "close", None)
        if callable(close):
            close()


class BiobankClient:
    """Synchronous convenience facade for scripts and notebooks."""

    def __init__(self, settings: Optional[Settings] = None, agent: Optional[Agent] = None) -> None:
        self._async = AsyncBiobankClient(settings=settings, agent=agent)

    @property
    def agent(self) -> Agent:
        return self._async.agent

    def run(self, query: str) -> str:
        """Run one turn from synchronous code."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._async.run(query))
        raise RuntimeError("BiobankClient.run() cannot be called from a running event loop; use AsyncBiobankClient")

    def skills(self) -> list[dict]:
        return self._async.skills()

    def close(self) -> None:
        self._async.close()


__all__ = ["AsyncBiobankClient", "BiobankClient"]
