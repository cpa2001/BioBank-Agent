"""Phase 5: biobank-spawns-biobank child sessions with a structural depth guard.

Offline — a fake runtime injects create_session/run_turn. Pins: a child runs one level deeper, the depth
is stamped so nested spawns stay bounded by max_depth, and a child failure never crashes the parent.
"""

from __future__ import annotations

from types import SimpleNamespace

from biobank_agent.runtime.subruntime import (
    DEPTH_META_KEY,
    current_depth,
    spawn_biobank_subagent,
)


def _session(title="s", cwd="/tmp"):
    return SimpleNamespace(
        session_id=f"sid-{title}",
        title=title,
        cwd=cwd,
        state=SimpleNamespace(custom_data={}),
        turns=[SimpleNamespace(assistant_messages=[SimpleNamespace(text=f"done:{title}")])],
    )


class _FakeRuntime:
    def __init__(self):
        self.created = []

    def create_session(self, *, title, cwd):
        s = _session(title, cwd)
        self.created.append(s)
        return s

    def run_turn(self, session, task):
        return session  # real run_turn returns the (mutated) session


def test_child_runs_one_level_deeper_and_is_stamped():
    rt = _FakeRuntime()
    parent = _session("root")  # depth 0
    result = spawn_biobank_subagent(rt, parent, "do subtask", max_depth=2)

    assert result.ok and result.depth == 1
    assert result.text == "done:subtask"
    assert current_depth(rt.created[-1]) == 1  # depth stamped on the child


def test_depth_guard_bounds_nested_spawns():
    rt = _FakeRuntime()
    parent = _session("root")

    r1 = spawn_biobank_subagent(rt, parent, "d1", max_depth=2)
    child1 = rt.created[-1]
    r2 = spawn_biobank_subagent(rt, child1, "d2", max_depth=2)
    child2 = rt.created[-1]
    r3 = spawn_biobank_subagent(rt, child2, "d3", max_depth=2)  # would be depth 3

    assert r1.depth == 1 and r1.ok
    assert r2.depth == 2 and r2.ok
    assert not r3.ok and "exceeds max_depth" in r3.error


def test_child_turn_error_is_isolated():
    class _Boom(_FakeRuntime):
        def run_turn(self, session, task):
            raise RuntimeError("child blew up")

    result = spawn_biobank_subagent(_Boom(), _session("root"), "x", max_depth=2)
    assert not result.ok and "child blew up" in result.error


def test_missing_run_turn_is_reported_not_raised():
    rt = SimpleNamespace(create_session=lambda *, title, cwd: _session(title, cwd))
    result = spawn_biobank_subagent(rt, _session("root"), "x", max_depth=2, run_turn=None)
    assert not result.ok and "run_turn" in result.error
