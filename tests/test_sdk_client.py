"""Tests for embeddable SDK client facades."""

from __future__ import annotations

import pytest

from biobank_agent.sdk import client as sdk_client
from biobank_agent.sdk import AsyncBiobankClient, BiobankClient


class FakeRegistry:
    def list_skills(self):
        return [{"name": "think", "description": "Reason"}]


class FakeConn:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeAgent:
    def __init__(self):
        self.registry = FakeRegistry()
        self.dm = type("DM", (), {"conn": FakeConn()})()


class FakeRuntime:
    def __init__(self, agent):
        self.agent = agent

    async def run_to_text(self, query):
        return f"answer:{query}"

    async def stream_events(self, query):
        yield {"query": query}


def test_async_client_uses_injected_agent_and_runtime(monkeypatch):
    monkeypatch.setattr(sdk_client, "AsyncAgent", FakeRuntime)
    agent = FakeAgent()
    client = AsyncBiobankClient(agent=agent)

    assert client.skills()[0]["name"] == "think"
    client.close()
    assert agent.dm.conn.closed is True


@pytest.mark.asyncio
async def test_async_client_run_and_stream(monkeypatch):
    monkeypatch.setattr(sdk_client, "AsyncAgent", FakeRuntime)
    client = AsyncBiobankClient(agent=FakeAgent())

    assert await client.run("hello") == "answer:hello"
    events = [event async for event in client.stream("hello")]
    assert events == [{"query": "hello"}]


def test_sync_client_run(monkeypatch):
    monkeypatch.setattr(sdk_client, "AsyncAgent", FakeRuntime)
    client = BiobankClient(agent=FakeAgent())

    assert client.run("hello") == "answer:hello"
    assert client.skills()[0]["name"] == "think"
