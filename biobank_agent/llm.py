"""LLM client — OpenAI-compatible interface for all models.

All models (Claude, GPT, Gemini) are accessed via the same
OpenAI-compatible endpoint at api.shubiaobiao.cn.
Includes retry logic with exponential backoff for transient errors.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError

logger = logging.getLogger(__name__)

# Retryable error types
_RETRYABLE_ERRORS = (APIConnectionError, RateLimitError, APITimeoutError)
_MAX_RETRIES = 3
_BASE_DELAY = 2.0  # seconds
_COMPAT_OPTIONAL_PARAMS = ("temperature", "max_tokens")
_DEPRECATED_PARAM_PATTERNS = (
    re.compile(r"`([a-zA-Z_][a-zA-Z0-9_]*)`\s+is\s+deprecated", re.I),
    re.compile(r"parameter\s+['`\"]?([a-zA-Z_][a-zA-Z0-9_]*)['`\"]?\s+is\s+deprecated", re.I),
    re.compile(r"does\s+not\s+support\s+(?:parameter\s+)?['`\"]?([a-zA-Z_][a-zA-Z0-9_]*)['`\"]?", re.I),
)
_UNAVAILABLE_CHANNEL_PATTERNS = (
    re.compile(r"no available channel for model", re.I),
    re.compile(r"model_not_found", re.I),
    re.compile(r"无可用渠道", re.I),
)


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
    diagnostics: dict = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class LLMClient:
    """Unified LLM client via OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://api.shubiaobiao.cn",
        api_key: str = "",
        model: str = "claude-opus-4-7",
        request_timeout_s: float = 60.0,
    ) -> None:
        # Ensure base URL has /v1 suffix for OpenAI SDK
        is_openrouter = "openrouter.ai" in base_url.lower()
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        default_headers = None
        if is_openrouter:
            default_headers = {
                "HTTP-Referer": "http://localhost/biobank-agent",
                "X-Title": "Biobank Agent",
            }
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=request_timeout_s,
            default_headers=default_headers,
        )
        self.model = model
        self.tool_call_content_mode = "null"  # "null" | "empty"
        self._model_cache: list[str] = []
        self._model_cache_ts: float = 0.0
        self._model_cache_ttl_s: float = 600.0
        self._deprecated_params: set[str] = set()

    @staticmethod
    def _usage_int(value: Any) -> int:
        """Normalize provider token counts to safe non-negative ints."""
        if value is None:
            return 0
        try:
            n = int(value)
        except (TypeError, ValueError):
            return 0
        return max(0, n)

    @staticmethod
    def sanitize_messages(
        messages: list[dict],
        tool_call_content_mode: str = "null",
    ) -> list[dict]:
        """Normalize messages for relay compatibility.

        Ensures:
        - Every message has a ``content`` key (str or None, never missing)
        - Assistant messages with tool_calls: empty content → null or ""
          (configurable via tool_call_content_mode)
        - Tool messages (role=tool): content always str
        - System messages: content always non-empty str
        """
        result = []
        null_content = None if tool_call_content_mode == "null" else ""
        for msg in messages:
            m = dict(msg)  # shallow copy — don't mutate caller's dict
            role = m.get("role", "")
            has_tool_calls = "tool_calls" in m

            if "content" not in m:
                m["content"] = null_content if has_tool_calls else ""
            elif m["content"] is None or m["content"] == "":
                if has_tool_calls:
                    m["content"] = null_content
                elif role == "tool":
                    m["content"] = m["content"] or ""  # tool results must be str
            result.append(m)
        return result

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """Send a chat completion request with optional tool definitions.

        Retries on transient errors (rate limit, connection, timeout) with
        exponential backoff up to _MAX_RETRIES times.
        """
        kwargs = self._build_chat_kwargs(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        t0 = time.time()
        response = self._create_chat_completion_with_compat(kwargs)
        elapsed = time.time() - t0
        result = self._parse_response(response)
        if self._should_retry_empty_reasoning_response(result, tools=tools):
            retry_kwargs = dict(kwargs)
            if "max_tokens" in retry_kwargs:
                retry_kwargs["max_tokens"] = max(
                    int(retry_kwargs.get("max_tokens") or 0) * 4,
                    1024,
                )
            logger.info(
                "LLM [%s] returned reasoning without final content; retrying once with more output budget.",
                self.model,
            )
            response = self._create_chat_completion_with_compat(retry_kwargs)
            retry_result = self._parse_response(response)
            retry_result.diagnostics["retried_empty_reasoning_response"] = True
            retry_result.diagnostics["initial_empty_reasoning_response"] = result.diagnostics
            result = retry_result
        if result.usage:
            logger.info(
                "LLM [%s] %d prompt + %d completion tokens (%.1fs)",
                self.model,
                self._usage_int(result.usage.get("prompt_tokens", 0)),
                self._usage_int(result.usage.get("completion_tokens", 0)),
                elapsed,
            )
        return result

    def list_models(self, refresh: bool = False) -> list[str]:
        """List models supported by the OpenAI-compatible relay.

        Uses an in-memory cache to avoid frequent /models calls.
        Returns an empty list on failure (non-fatal for the CLI/agent).
        """
        now = time.time()
        cache_valid = (
            self._model_cache
            and (now - self._model_cache_ts) < self._model_cache_ttl_s
        )
        if cache_valid and not refresh:
            return list(self._model_cache)

        last_error = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = self.client.models.list()
                data = getattr(resp, "data", []) or []
                ids = sorted({
                    str(getattr(item, "id", "")).strip()
                    for item in data
                    if getattr(item, "id", None)
                })
                self._model_cache = [m for m in ids if m]
                self._model_cache_ts = time.time()
                return list(self._model_cache)
            except _RETRYABLE_ERRORS as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    delay = _BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Model list call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1, _MAX_RETRIES + 1, e, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.warning("Model list call failed after %d attempts: %s", _MAX_RETRIES + 1, e)
            except APIError as e:
                if e.status_code and e.status_code >= 500:
                    last_error = e
                    if attempt < _MAX_RETRIES:
                        delay = _BASE_DELAY * (2 ** attempt)
                        logger.warning(
                            "Model list server error %d (attempt %d/%d): %s. Retrying in %.1fs...",
                            e.status_code, attempt + 1, _MAX_RETRIES + 1, e, delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.warning("Model list server error after %d attempts: %s", _MAX_RETRIES + 1, e)
                else:
                    last_error = e
                    break
            except Exception as e:
                last_error = e
                break

        if last_error:
            logger.warning("Unable to fetch model list from relay: %s", last_error)
        return list(self._model_cache)

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
        kwargs = self._build_chat_kwargs(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        text_parts: list[str] = []
        tool_call_deltas: dict[int, dict] = {}  # index → {id, name, args_str}

        stream = self._create_chat_completion_with_compat(kwargs)
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

    def _build_chat_kwargs(
        self,
        messages: list[dict],
        tools: Optional[list[dict]],
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict[str, Any]:
        """Build chat completion kwargs with per-model compatibility guards."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self.sanitize_messages(messages, self.tool_call_content_mode),
        }
        if stream:
            kwargs["stream"] = True
        if "temperature" not in self._deprecated_params:
            kwargs["temperature"] = temperature
        if "max_tokens" not in self._deprecated_params:
            kwargs["max_tokens"] = max_tokens
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        return kwargs

    def _create_chat_completion_with_compat(self, kwargs: dict[str, Any]):
        """Create completion with retries + deprecated-parameter adaptation."""
        last_error = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self.client.chat.completions.create(**kwargs)
            except _RETRYABLE_ERRORS as e:
                last_error = e
                if attempt < _MAX_RETRIES:
                    delay = _BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "LLM call failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        attempt + 1, _MAX_RETRIES + 1, e, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error("LLM call failed after %d attempts: %s", _MAX_RETRIES + 1, e)
            except APIError as e:
                # If a model deprecates a parameter, drop it and retry immediately.
                if self._adapt_deprecated_params(e, kwargs):
                    continue
                # Relay says this model has no active channel right now: fail fast.
                if self._is_unavailable_channel_error(e):
                    logger.error(
                        "LLM [%s] unavailable on relay (no channel/model_not_found): %s",
                        self.model,
                        e,
                    )
                    raise
                # Server errors (500/502/503) are retryable.
                if e.status_code and e.status_code >= 500:
                    last_error = e
                    if attempt < _MAX_RETRIES:
                        delay = _BASE_DELAY * (2 ** attempt)
                        logger.warning(
                            "LLM server error %d (attempt %d/%d): %s. Retrying in %.1fs...",
                            e.status_code, attempt + 1, _MAX_RETRIES + 1, e, delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error("LLM server error after %d attempts: %s", _MAX_RETRIES + 1, e)
                else:
                    raise  # Non-retryable API errors (400, 401, 403, etc.)
            except Exception as e:
                # Some relays may wrap 400s in non-APIError exception types.
                if self._adapt_deprecated_params(e, kwargs):
                    continue
                raise
        raise last_error  # type: ignore[misc]

    @staticmethod
    def _is_unavailable_channel_error(error: Exception) -> bool:
        """Return True if relay reports model has no serving channel."""
        text = str(error or "")
        if not text:
            return False
        return any(p.search(text) for p in _UNAVAILABLE_CHANNEL_PATTERNS)

    @staticmethod
    def _should_retry_empty_reasoning_response(
        result: LLMResponse,
        tools: Optional[list[dict]] = None,
    ) -> bool:
        """Retry reasoning-only responses once, without exposing reasoning text."""
        if result.text or result.has_tool_calls:
            return False
        if tools:
            return False
        return bool(result.diagnostics.get("empty_content_with_reasoning"))

    def _adapt_deprecated_params(self, error: Exception, kwargs: dict[str, Any]) -> bool:
        """Detect deprecated parameter errors and mutate kwargs in place."""
        text = str(error)
        if not text:
            return False
        lower_text = text.lower()

        matched_param = None
        for pattern in _DEPRECATED_PARAM_PATTERNS:
            match = pattern.search(text)
            if match:
                matched_param = match.group(1)
                break

        if not matched_param:
            for param in _COMPAT_OPTIONAL_PARAMS:
                if param in lower_text and "deprecated" in lower_text:
                    matched_param = param
                    break

        if not matched_param:
            return False

        param = matched_param.strip().lower()
        if param not in _COMPAT_OPTIONAL_PARAMS:
            return False
        if param not in kwargs:
            self._deprecated_params.add(param)
            return False

        kwargs.pop(param, None)
        self._deprecated_params.add(param)
        logger.warning(
            "LLM [%s] relay reports `%s` deprecated; retrying without it.",
            self.model,
            param,
        )
        return True

    def _parse_response(self, response) -> LLMResponse:
        """Parse a non-streaming response."""
        msg = response.choices[0].message
        text = msg.content or ""
        reasoning = getattr(msg, "reasoning", None)
        diagnostics = {}
        if reasoning:
            diagnostics = {
                "reasoning_present": True,
                "reasoning_chars": len(str(reasoning)),
                "empty_content_with_reasoning": not bool(msg.content),
            }

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
                "prompt_tokens": self._usage_int(getattr(response.usage, "prompt_tokens", 0)),
                "completion_tokens": self._usage_int(getattr(response.usage, "completion_tokens", 0)),
                "total_tokens": self._usage_int(getattr(response.usage, "total_tokens", 0)),
            }

        return LLMResponse(
            text=text,
            tool_calls=tool_calls,
            usage=usage,
            diagnostics=diagnostics,
        )
