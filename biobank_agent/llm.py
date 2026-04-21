"""LLM client — OpenAI-compatible interface for all models.

All models (Claude, GPT, Gemini) are accessed via the same
OpenAI-compatible endpoint at api.shubiaobiao.cn.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMClient:
    """Unified LLM client via OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://api.shubiaobiao.cn",
        api_key: str = "",
        model: str = "claude-sonnet-4-6",
    ) -> None:
        # Ensure base URL has /v1 suffix for OpenAI SDK
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat completion request with optional tool definitions."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        return self._parse_response(response)

    def stream(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> Generator[str, None, LLMResponse]:
        """Stream a chat completion, yielding text chunks.

        Returns the final LLMResponse (with tool_calls if any) at the end.
        Tool calls are accumulated from streamed deltas.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        text_parts: list[str] = []
        tool_call_deltas: dict[int, dict] = {}  # index → {id, name, args_str}

        stream = self.client.chat.completions.create(**kwargs)
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            # Text content
            if delta.content:
                text_parts.append(delta.content)
                yield delta.content

            # Tool call deltas
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_deltas:
                        tool_call_deltas[idx] = {"id": "", "name": "", "args": ""}
                    if tc_delta.id:
                        tool_call_deltas[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_call_deltas[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_call_deltas[idx]["args"] += tc_delta.function.arguments

        # Build final response
        tool_calls = []
        for idx in sorted(tool_call_deltas):
            tc = tool_call_deltas[idx]
            try:
                args = json.loads(tc["args"]) if tc["args"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(id=tc["id"], name=tc["name"], args=args))

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
        )

    def _parse_response(self, response) -> LLMResponse:
        """Parse a non-streaming response."""
        msg = response.choices[0].message
        text = msg.content or ""

        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    args=args,
                ))

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(text=text, tool_calls=tool_calls, usage=usage)
