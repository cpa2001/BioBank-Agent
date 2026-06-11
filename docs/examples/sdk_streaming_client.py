"""Minimal embedded Biobank Agent streaming example.

Run from the repository root:

    python docs/examples/sdk_streaming_client.py
"""

from __future__ import annotations

import asyncio

from biobank_agent.core.events import AgentEventType
from biobank_agent.sdk import AsyncBiobankClient


async def main() -> None:
    client = AsyncBiobankClient()
    try:
        async for event in client.stream(
            "List the available skills and suggest a safe first E11 analysis."
        ):
            if event.type == AgentEventType.MESSAGE_DELTA:
                print(event.payload.get("text", ""), end="", flush=True)
            elif event.type == AgentEventType.TOOL_STARTED:
                print(f"\n[tool] {event.payload.get('skill')} started")
            elif event.type == AgentEventType.TOOL_RESULT:
                print(f"\n[tool] {event.payload.get('skill')} done")
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())

