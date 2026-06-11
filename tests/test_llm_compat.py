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


def _fake_response_with_none_usage(text: str = "ok"):
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=None, completion_tokens=None, total_tokens=None)
    return SimpleNamespace(choices=[choice], usage=usage)


def _fake_reasoning_only_response():
    message = SimpleNamespace(
        content=None,
        tool_calls=None,
        reasoning="internal reasoning should not become visible answer",
    )
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=128, total_tokens=131)
    return SimpleNamespace(choices=[choice], usage=usage)


def _fake_stream(chunks: list[str]):
    out = []
    for content in chunks:
        delta = SimpleNamespace(content=content, tool_calls=None)
        out.append(SimpleNamespace(choices=[SimpleNamespace(delta=delta)]))
    return out


def _fake_tool_delta(index=0, tc_id=None, name=None, arguments=None):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=tc_id, function=function)


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

    def test_usage_none_values_are_normalized_to_zero(self):
        fake_create = _FakeCreate([
            _fake_response_with_none_usage("ok"),
        ])
        llm = _build_client(fake_create)

        resp = llm.chat(messages=[{"role": "user", "content": "hello"}])

        assert resp.text == "ok"
        assert resp.usage["prompt_tokens"] == 0
        assert resp.usage["completion_tokens"] == 0
        assert resp.usage["total_tokens"] == 0

    def test_chat_without_usage_skips_usage_logging_branch(self):
        message = SimpleNamespace(content="ok", tool_calls=None)
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)
        llm = _build_client(_FakeCreate([response]))

        resp = llm.chat(messages=[{"role": "user", "content": "hello"}])

        assert resp.text == "ok"
        assert resp.usage == {}

    def test_reasoning_only_response_retries_without_leaking_reasoning(self):
        fake_create = _FakeCreate([
            _fake_reasoning_only_response(),
            _fake_response("final"),
        ])
        llm = _build_client(fake_create)

        resp = llm.chat(messages=[{"role": "user", "content": "short answer"}], max_tokens=32)

        assert resp.text == "final"
        assert "internal reasoning" not in resp.text
        assert resp.diagnostics["retried_empty_reasoning_response"] is True
        assert fake_create.calls[0]["max_tokens"] == 32
        assert fake_create.calls[1]["max_tokens"] == 1024


class TestLLMClientEdges:
    def test_usage_int_and_sanitize_messages_modes(self):
        from biobank_agent.llm import LLMClient

        original = [
            {"role": "assistant", "tool_calls": [{"id": "t"}]},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t2"}]},
            {"role": "assistant", "content": ""},
            {"role": "tool", "content": None},
            {"role": "user"},
        ]

        null_mode = LLMClient.sanitize_messages(original, "null")
        empty_mode = LLMClient.sanitize_messages(original, "empty")

        assert null_mode[0]["content"] is None
        assert null_mode[1]["content"] is None
        assert null_mode[2]["content"] == ""
        assert null_mode[3]["content"] == ""
        assert null_mode[4]["content"] == ""
        assert empty_mode[0]["content"] == ""
        assert "content" not in original[0]
        assert LLMClient._usage_int("5") == 5
        assert LLMClient._usage_int(-3) == 0
        assert LLMClient._usage_int("bad") == 0

    def test_build_chat_kwargs_respects_deprecated_cache_tools_and_stream(self):
        from biobank_agent.llm import LLMClient

        llm = _build_client(_FakeCreate([]))
        llm._deprecated_params = {"temperature", "max_tokens"}
        llm.tool_call_content_mode = "empty"

        kwargs = llm._build_chat_kwargs(
            messages=[{"role": "assistant", "tool_calls": [{"id": "t"}]}],
            tools=[{"type": "function", "function": {"name": "x"}}],
            temperature=0.2,
            max_tokens=12,
            stream=True,
        )

        assert kwargs["stream"] is True
        assert "temperature" not in kwargs
        assert "max_tokens" not in kwargs
        assert kwargs["messages"][0]["content"] == ""
        assert kwargs["tool_choice"] == "auto"

    def test_adapt_deprecated_params_no_text_not_optional_and_missing_kwarg(self):
        llm = _build_client(_FakeCreate([]))
        kwargs = {"model": "m", "temperature": 0.1}

        assert llm._adapt_deprecated_params(Exception(""), kwargs) is False
        assert llm._adapt_deprecated_params(Exception("`top_p` is deprecated"), kwargs) is False
        assert llm._adapt_deprecated_params(Exception("parameter 'max_tokens' is deprecated"), kwargs) is False
        assert "max_tokens" in llm._deprecated_params
        assert llm._adapt_deprecated_params(Exception("does not support temperature"), kwargs) is True
        assert "temperature" not in kwargs
        kwargs = {"model": "m", "temperature": 0.2}
        assert llm._adapt_deprecated_params(Exception("temperature deprecated for this relay"), kwargs) is True

    def test_unavailable_channel_detection_and_retry_reasoning_predicate(self):
        from biobank_agent.llm import LLMResponse, ToolCall

        llm = _build_client(_FakeCreate([]))

        assert llm._is_unavailable_channel_error(Exception("no available channel for model")) is True
        assert llm._is_unavailable_channel_error(Exception("model_not_found")) is True
        assert llm._is_unavailable_channel_error(Exception("")) is False
        assert llm._should_retry_empty_reasoning_response(
            LLMResponse(diagnostics={"empty_content_with_reasoning": True})
        ) is True
        assert llm._should_retry_empty_reasoning_response(
            LLMResponse(text="ok", diagnostics={"empty_content_with_reasoning": True})
        ) is False
        assert llm._should_retry_empty_reasoning_response(
            LLMResponse(tool_calls=[ToolCall("t", "x", {})], diagnostics={"empty_content_with_reasoning": True})
        ) is False
        assert llm._should_retry_empty_reasoning_response(
            LLMResponse(diagnostics={"empty_content_with_reasoning": True}),
            tools=[{"type": "function"}],
        ) is False

    def test_create_chat_completion_retries_retryable_and_server_errors(self, monkeypatch):
        import biobank_agent.llm as llm_mod

        class FakeAPIError(Exception):
            def __init__(self, message, status_code=None):
                super().__init__(message)
                self.status_code = status_code

        monkeypatch.setattr(llm_mod, "_RETRYABLE_ERRORS", (RuntimeError,))
        monkeypatch.setattr(llm_mod, "APIError", FakeAPIError)
        monkeypatch.setattr(llm_mod.time, "sleep", lambda _delay: None)

        retryable = _build_client(_FakeCreate([RuntimeError("busy"), _fake_response("ok")]))
        assert retryable._create_chat_completion_with_compat({"model": "m"}).choices[0].message.content == "ok"

        server = _build_client(_FakeCreate([FakeAPIError("server", status_code=503), _fake_response("ok2")]))
        assert server._create_chat_completion_with_compat({"model": "m"}).choices[0].message.content == "ok2"

        deprecated_api = _build_client(_FakeCreate([
            FakeAPIError("`temperature` is deprecated", status_code=400),
            _fake_response("ok3"),
        ]))
        kwargs = {"model": "m", "temperature": 0.1}
        assert deprecated_api._create_chat_completion_with_compat(kwargs).choices[0].message.content == "ok3"
        assert "temperature" not in kwargs

        unavailable = _build_client(_FakeCreate([FakeAPIError("model_not_found", status_code=404)]))
        with pytest.raises(FakeAPIError, match="model_not_found"):
            unavailable._create_chat_completion_with_compat({"model": "m"})

        bad_request = _build_client(_FakeCreate([FakeAPIError("bad", status_code=400)]))
        with pytest.raises(FakeAPIError, match="bad"):
            bad_request._create_chat_completion_with_compat({"model": "m"})

        exhausted = _build_client(_FakeCreate([RuntimeError("busy")] * 4))
        with pytest.raises(RuntimeError, match="busy"):
            exhausted._create_chat_completion_with_compat({"model": "m"})

        server_exhausted = _build_client(_FakeCreate([FakeAPIError("server", status_code=503)] * 4))
        with pytest.raises(FakeAPIError, match="server"):
            server_exhausted._create_chat_completion_with_compat({"model": "m"})

    def test_list_models_retry_server_nonretry_and_cache_paths(self, monkeypatch):
        import biobank_agent.llm as llm_mod

        class FakeAPIError(Exception):
            def __init__(self, message, status_code=None):
                super().__init__(message)
                self.status_code = status_code

        monkeypatch.setattr(llm_mod, "_RETRYABLE_ERRORS", (RuntimeError,))
        monkeypatch.setattr(llm_mod, "APIError", FakeAPIError)
        monkeypatch.setattr(llm_mod.time, "sleep", lambda _delay: None)

        llm = _build_client(_FakeCreate([]))
        model_api = _FakeCreate([
            RuntimeError("busy"),
            SimpleNamespace(data=[SimpleNamespace(id="b"), SimpleNamespace(id="a"), SimpleNamespace(id="")]),
            FakeAPIError("server", status_code=503),
            SimpleNamespace(data=[SimpleNamespace(id="c")]),
            FakeAPIError("bad", status_code=400),
        ])
        llm.client = SimpleNamespace(models=SimpleNamespace(list=model_api))

        assert llm.list_models(refresh=True) == ["a", "b"]
        assert llm.list_models(refresh=True) == ["c"]
        assert llm.list_models(refresh=True) == ["c"]

        retry_exhausted = _build_client(_FakeCreate([]))
        retry_exhausted.client = SimpleNamespace(models=SimpleNamespace(list=_FakeCreate([RuntimeError("busy")] * 4)))
        retry_exhausted._model_cache = ["cached"]
        assert retry_exhausted.list_models(refresh=True) == ["cached"]

        server_exhausted = _build_client(_FakeCreate([]))
        server_exhausted.client = SimpleNamespace(models=SimpleNamespace(list=_FakeCreate([FakeAPIError("server", status_code=503)] * 4)))
        assert server_exhausted.list_models(refresh=True) == []

    def test_parse_response_tool_calls_reasoning_and_invalid_args(self):
        llm = _build_client(_FakeCreate([]))
        tool_calls = [
            SimpleNamespace(id="t1", function=SimpleNamespace(name="good", arguments="{\"x\": 1}")),
            SimpleNamespace(id="t2", function=SimpleNamespace(name="bad", arguments="{not json")),
        ]
        message = SimpleNamespace(content=None, reasoning="hidden", tool_calls=tool_calls)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens="3", completion_tokens="bad", total_tokens=-1),
        )

        parsed = llm._parse_response(response)

        assert parsed.text == ""
        assert parsed.diagnostics["reasoning_present"] is True
        assert parsed.diagnostics["empty_content_with_reasoning"] is True
        assert parsed.tool_calls[0].args == {"x": 1}
        assert parsed.tool_calls[1].args == {}
        assert parsed.usage == {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 0}

        no_usage = llm._parse_response(SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="plain", reasoning=None, tool_calls=None))],
            usage=None,
        ))
        assert no_usage.usage == {}

    def test_stream_accumulates_tool_call_deltas_and_invalid_json(self):
        from biobank_agent.llm import LLMClient

        chunks = [
            SimpleNamespace(choices=[]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi ", tool_calls=None))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[
                _fake_tool_delta(index=0, tc_id="t1", name="tool_a", arguments="{\"x\""),
            ]))]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[
                _fake_tool_delta(index=0, arguments=": 1}"),
                _fake_tool_delta(index=1, tc_id="t2", name="tool_b", arguments="{bad"),
            ]))]),
        ]
        llm = _build_client(_FakeCreate([chunks]))

        gen = llm.stream([{"role": "user", "content": "q"}])
        assert next(gen) == "hi "
        try:
            next(gen)
        except StopIteration as stop:
            final = stop.value
        else:
            raise AssertionError("stream should be exhausted")

        assert final.text == "hi "
        assert final.tool_calls[0].name == "tool_a"
        assert final.tool_calls[0].args == {"x": 1}
        assert final.tool_calls[1].args == {}

    def test_stream_ignores_tool_deltas_without_function_payloads(self):
        chunks = [
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[
                SimpleNamespace(index=0, id="t1", function=None),
                SimpleNamespace(index=1, id="t2", function=SimpleNamespace(name="", arguments="")),
            ]))]),
        ]
        llm = _build_client(_FakeCreate([chunks]))

        gen = llm.stream([{"role": "user", "content": "q"}])
        try:
            next(gen)
        except StopIteration as stop:
            final = stop.value
        else:
            raise AssertionError("stream should be exhausted")

        assert [(tc.id, tc.name, tc.args) for tc in final.tool_calls] == [
            ("t1", "", {}),
            ("t2", "", {}),
        ]

    def test_chat_does_not_retry_reasoning_when_tools_present(self):
        fake_create = _FakeCreate([_fake_reasoning_only_response()])
        llm = _build_client(fake_create)

        resp = llm.chat(
            messages=[{"role": "user", "content": "use tool"}],
            tools=[{"type": "function", "function": {"name": "x"}}],
        )

        assert resp.text == ""
        assert resp.diagnostics["empty_content_with_reasoning"] is True
        assert len(fake_create.calls) == 1

    def test_chat_reasoning_retry_without_max_tokens_kwarg(self):
        fake_create = _FakeCreate([
            _fake_reasoning_only_response(),
            _fake_response("final"),
        ])
        llm = _build_client(fake_create)
        llm._deprecated_params = {"max_tokens"}

        resp = llm.chat(messages=[{"role": "user", "content": "short answer"}], max_tokens=32)

        assert resp.text == "final"
        assert "max_tokens" not in fake_create.calls[0]
        assert "max_tokens" not in fake_create.calls[1]
