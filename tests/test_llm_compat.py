"""Compatibility tests for relay-specific deprecated parameters in LLMClient."""

from types import SimpleNamespace

import pytest


class _FakeCreate:
    def __init__(self, effects):
        self.effects = list(effects)
        self.calls = []

    def __call__(self, **kwargs):
        # keep a shallow copy for assertions
        self.calls.append(dict(kwargs))
        if not self.effects:
            raise AssertionError("No more side effects configured")
        effect = self.effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect


def _fake_response(text: str = "ok"):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7)
    return SimpleNamespace(choices=[choice], usage=usage)


def _fake_stream(chunks: list[str]):
    out = []
    for content in chunks:
        delta = SimpleNamespace(content=content, tool_calls=None)
        out.append(SimpleNamespace(choices=[SimpleNamespace(delta=delta)]))
    return out


def _build_client(fake_create):
    from biobank_agent.llm import LLMClient

    llm = LLMClient(base_url="http://relay.test", api_key="test", model="gpt-5.4")
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
    )
    return llm


class TestDeprecatedParamCompat:
    """When relay rejects params (e.g., temperature), client should self-heal."""

    def test_chat_retries_without_temperature_on_deprecation(self):
        fake_create = _FakeCreate([
            Exception("Error code: 400 - {'message':'`temperature` is deprecated for this model.'}"),
            _fake_response("recovered"),
        ])
        llm = _build_client(fake_create)

        resp = llm.chat(messages=[{"role": "user", "content": "hello"}])

        assert resp.text == "recovered"
        assert len(fake_create.calls) == 2
        assert "temperature" in fake_create.calls[0]
        assert "temperature" not in fake_create.calls[1]

    def test_chat_caches_deprecated_temperature_for_later_calls(self):
        fake_create = _FakeCreate([
            Exception("`temperature` is deprecated for this model."),
            _fake_response("first"),
            _fake_response("second"),
        ])
        llm = _build_client(fake_create)

        r1 = llm.chat(messages=[{"role": "user", "content": "a"}])
        r2 = llm.chat(messages=[{"role": "user", "content": "b"}])

        assert r1.text == "first"
        assert r2.text == "second"
        assert len(fake_create.calls) == 3
        assert "temperature" in fake_create.calls[0]
        assert "temperature" not in fake_create.calls[1]
        assert "temperature" not in fake_create.calls[2]

    def test_chat_handles_multiple_deprecated_params(self):
        fake_create = _FakeCreate([
            Exception("`temperature` is deprecated for this model."),
            Exception("`max_tokens` is deprecated for this model."),
            _fake_response("done"),
        ])
        llm = _build_client(fake_create)

        resp = llm.chat(messages=[{"role": "user", "content": "x"}], max_tokens=1234)

        assert resp.text == "done"
        assert len(fake_create.calls) == 3
        assert "temperature" in fake_create.calls[0]
        assert "max_tokens" in fake_create.calls[0]
        assert "temperature" not in fake_create.calls[1]
        assert "max_tokens" in fake_create.calls[1]
        assert "temperature" not in fake_create.calls[2]
        assert "max_tokens" not in fake_create.calls[2]

    def test_stream_retries_without_temperature_on_deprecation(self):
        fake_create = _FakeCreate([
            Exception("Auto-blocked (>= 3 errors / 60s): `temperature` is deprecated for this model."),
            _fake_stream(["hello", " world"]),
        ])
        llm = _build_client(fake_create)

        chunks = list(llm.stream(messages=[{"role": "user", "content": "stream"}]))

        assert "".join(chunks) == "hello world"
        assert len(fake_create.calls) == 2
        assert "temperature" in fake_create.calls[0]
        assert "temperature" not in fake_create.calls[1]
        assert fake_create.calls[1].get("stream") is True

    def test_non_compat_error_is_raised(self):
        fake_create = _FakeCreate([
            Exception("Error code: 400 - invalid API key"),
        ])
        llm = _build_client(fake_create)

        with pytest.raises(Exception, match="invalid API key"):
            llm.chat(messages=[{"role": "user", "content": "hello"}])
