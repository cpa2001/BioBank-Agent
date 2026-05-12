"""Unit tests for biobank_agent.core.llm.stream_parser."""

from __future__ import annotations

from dataclasses import dataclass

from biobank_agent.core.llm.stream_parser import (
    ToolArgFragment,
    ToolArgumentStreamParser,
    _close_open_json,
    _try_parse_partial,
)


@dataclass
class _FakeFn:
    name: str = ""
    arguments: str = ""


@dataclass
class _FakeDelta:
    index: int
    id: str = ""
    function: _FakeFn | None = None


def test_close_open_json_balanced_passthrough():
    assert _close_open_json('{"a":1}') == '{"a":1}'


def test_close_open_json_unbalanced_object():
    assert _close_open_json('{"a":1,"b":2') == '{"a":1,"b":2}'


def test_close_open_json_inside_string_value_backs_off_to_comma():
    # Mid-string: back off to last comma boundary.
    text = '{"a":1,"b":"unter'
    closed = _close_open_json(text)
    # Either close at last boundary (works) or returns None (acceptable).
    assert closed in ('{"a":1}', None)


def test_try_parse_partial_returns_fallback_for_garbage():
    fallback = {"prev": "value"}
    out = _try_parse_partial("not json", fallback)
    assert out == fallback


def test_try_parse_partial_progresses_with_each_delta():
    fallback = {}
    a = _try_parse_partial('{"icd10_code"', fallback)
    # Mid-key, should fallback.
    assert a == {}
    b = _try_parse_partial('{"icd10_code":"E11"', fallback)
    assert b == {"icd10_code": "E11"}
    c = _try_parse_partial('{"icd10_code":"E11","top_n":', b)
    # Mid value: fallback to last good.
    assert c == {"icd10_code": "E11"}
    d = _try_parse_partial('{"icd10_code":"E11","top_n":5', c)
    assert d == {"icd10_code": "E11", "top_n": 5}
    e = _try_parse_partial('{"icd10_code":"E11","top_n":5}', d)
    assert e == {"icd10_code": "E11", "top_n": 5}


def test_parser_accumulates_across_deltas():
    parser = ToolArgumentStreamParser()
    deltas = [
        _FakeDelta(index=0, id="call_1", function=_FakeFn(name="prevalence", arguments='{"icd10')),
        _FakeDelta(index=0, function=_FakeFn(arguments='_code":"E11"')),
        _FakeDelta(index=0, function=_FakeFn(arguments=',"top_n":5}')),
    ]
    fragments: list[ToolArgFragment] = []
    for d in deltas:
        fragments.extend(parser.feed_openai_delta([d]))
    final = parser.finalize()

    # Three streaming fragments, all with the same call_id and name.
    assert all(f.call_id == "call_1" for f in fragments + final)
    assert fragments[0].name == "prevalence"
    # Last streamed fragment should already have parsed args.
    assert fragments[-1].partial_args == {"icd10_code": "E11", "top_n": 5}
    assert final[0].finalized is True
    assert final[0].partial_args == {"icd10_code": "E11", "top_n": 5}


def test_parser_handles_two_concurrent_calls():
    parser = ToolArgumentStreamParser()
    deltas_chunk_1 = [
        _FakeDelta(index=0, id="a", function=_FakeFn(name="x", arguments='{"k":')),
        _FakeDelta(index=1, id="b", function=_FakeFn(name="y", arguments='{"q":')),
    ]
    deltas_chunk_2 = [
        _FakeDelta(index=0, function=_FakeFn(arguments='1}')),
        _FakeDelta(index=1, function=_FakeFn(arguments='2}')),
    ]
    parser.feed_openai_delta(deltas_chunk_1)
    parser.feed_openai_delta(deltas_chunk_2)
    final = parser.finalize()
    by_id = {f.call_id: f for f in final}
    assert by_id["a"].partial_args == {"k": 1}
    assert by_id["b"].partial_args == {"q": 2}
