"""Autonomous LLM patch generation feeding the transactional apply loop.

These tests use a STUB llm (an object with a ``.chat`` method returning a canned
``LLMResponse``) — no network. They pin the contract that generation produces a
concrete, applicable patch ONLY when the LLM output is parseable, in-scope, and
test-gated, and that the generated patch flows end-to-end through the real
transactional apply loop into a disposable git repo.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from biobank_agent.llm import LLMResponse
from biobank_agent.runtime.evolution import EvolutionProposal
from biobank_agent.runtime.patch_generation import (
    PatchGenerationResult,
    _compact_evidence,
    _extract_json,
    enrich_proposals_with_patches,
    generate_and_apply_proposals,
    generate_patch_for_proposal,
)


class StubLLM:
    """Minimal LLMClient stand-in: returns canned text and records the prompt."""

    def __init__(self, text: str = "", *, raises: Exception | None = None):
        self.text = text
        self.raises = raises
        self.calls: list[dict] = []

    def chat(self, messages, temperature=0.0, max_tokens=8192):
        self.calls.append({"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        if self.raises is not None:
            raise self.raises
        return LLMResponse(text=self.text)


def _proposal(**kw) -> EvolutionProposal:
    base = dict(proposal_id="p-1", category="skill", summary="Add a helper")
    base.update(kw)
    return EvolutionProposal(**base)


def _good_json(target="custom_skills/gen_demo.py", cmd="pytest custom_skills/gen_demo.py -q") -> str:
    body = "def add(a, b):\n    return a + b\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
    return json.dumps({
        "target_path": target,
        "diff": body,
        "test_commands": [cmd],
        "summary": "demo",
        "rationale": "verifies add",
    })


# ── JSON extraction ────────────────────────────────────────────


def test_extract_json_plain():
    obj = _extract_json('{"a": 1, "b": "x"}')
    assert obj == {"a": 1, "b": "x"}


def test_extract_json_fenced_with_prose():
    text = 'Sure, here it is:\n```json\n{"target_path": "x", "diff": "y"}\n```\nDone.'
    obj = _extract_json(text)
    assert obj["target_path"] == "x"


def test_extract_json_braces_and_escaped_quotes_in_diff():
    # A diff payload containing { } and escaped quotes must not break the scanner.
    text = '{"diff": "def f():\\n    return {\\"a\\": 1}\\n", "target_path": "custom_skills/x.py"}'
    obj = _extract_json(text)
    assert "{" in obj["diff"] and '"a"' in obj["diff"]
    assert obj["target_path"] == "custom_skills/x.py"


def test_extract_json_truncated_returns_none():
    # Truncated at max_tokens — unbalanced braces, must not raise, returns None.
    assert _extract_json('{"target_path": "custom_skills/x.py", "diff": "def f(') is None


def test_extract_json_prose_returns_none():
    assert _extract_json("I'm sorry, I cannot help with that.") is None


# ── Generation happy path + validation ─────────────────────────


def test_generates_valid_patch():
    llm = StubLLM(_good_json())
    res = generate_patch_for_proposal(_proposal(), llm=llm)
    assert res.status == "generated", res.error
    assert res.target_path == "custom_skills/gen_demo.py"
    assert "def add" in res.diff
    assert res.test_commands == ["pytest custom_skills/gen_demo.py -q"]
    assert res.has_patch
    # one LLM call, with the evidence section present and bounded
    assert len(llm.calls) == 1
    user_msg = llm.calls[0]["messages"][-1]["content"]
    assert "ALLOWED PATHS" in user_msg


def test_does_not_mutate_input_proposal():
    p = _proposal()
    generate_patch_for_proposal(p, llm=StubLLM(_good_json()))
    assert p.diff == "" and p.target_path == ""  # pure: caller decides to populate


def test_llm_graceful_decline():
    llm = StubLLM('{"status": "rejected", "reason": "nothing actionable"}')
    res = generate_patch_for_proposal(_proposal(), llm=llm)
    assert res.status == "rejected"
    assert "nothing actionable" in res.error


def test_llm_call_failure_becomes_error_not_crash():
    llm = StubLLM(raises=RuntimeError("connection reset"))
    res = generate_patch_for_proposal(_proposal(), llm=llm)
    assert res.status == "error"
    assert "connection reset" in res.error


def test_garbage_output_is_error():
    res = generate_patch_for_proposal(_proposal(), llm=StubLLM("no json here at all"))
    assert res.status == "error"


# ── Rejections: path scope, traversal, protected ───────────────


@pytest.mark.parametrize("target", [
    "biobank_agent/runtime/pwn.py",   # protected
    "biobank_agent/core/pwn.py",      # protected
    "tests/test_pwn.py",              # protected
    "docs/notes.py",                  # not in allow-list
    "/etc/passwd",                    # absolute
    "custom_skills/../biobank_agent/runtime/pwn.py",  # traversal
])
def test_rejects_out_of_scope_targets(target):
    cmd = "pytest custom_skills/x.py -q"
    llm = StubLLM(_good_json(target=target, cmd=cmd))
    res = generate_patch_for_proposal(_proposal(), llm=llm)
    assert res.status == "rejected", target


# ── Rejections: test-command gate (laundering + safety) ────────


@pytest.mark.parametrize("cmd", [
    "rm -rf /tmp/x",                          # not a runner
    "python evil_script.py",                  # bare python file
    'python -c "import os"',                  # arbitrary code
    "pytest x && curl http://evil",           # metacharacter
    "ruff check custom_skills/gen_demo.py",   # linter only, no pytest run
    "pytest custom_skills/gen_demo.py --collect-only",  # collect-only laundering
    "pytest custom_skills/gen_demo.py --collectonly",   # no-hyphen alias
    "pytest custom_skills/gen_demo.py --co",  # collect-only short
])
def test_rejects_bad_test_commands(cmd):
    llm = StubLLM(_good_json(cmd=cmd))
    res = generate_patch_for_proposal(_proposal(), llm=llm)
    assert res.status == "rejected", cmd


def test_allows_cov_flag_not_treated_as_collect_only():
    js = _good_json(cmd="pytest custom_skills/gen_demo.py --cov=custom_skills -q")
    res = generate_patch_for_proposal(_proposal(), llm=StubLLM(js))
    assert res.status == "generated", res.error


def test_extract_json_recovers_from_preamble_brace():
    # A stray balanced fragment before the real object must not sink the parse.
    text = (
        '{note: ignore me} The real object: '
        '{"target_path": "custom_skills/x.py", "diff": "x = 1\\n", '
        '"test_commands": ["pytest custom_skills/x.py -q"]}'
    )
    obj = _extract_json(text)
    assert obj is not None and obj["target_path"] == "custom_skills/x.py"


def test_rejects_diff_touching_out_of_scope_path():
    # A unified diff that declares an allowed target but ALSO edits a protected
    # file must be rejected at generation (defense-in-depth; apply re-checks too).
    diff = (
        "diff --git a/custom_skills/ok.py b/custom_skills/ok.py\n"
        "--- a/custom_skills/ok.py\n+++ b/custom_skills/ok.py\n"
        "@@ -0,0 +1 @@\n+x = 1\n"
        "diff --git a/biobank_agent/runtime/pwn.py b/biobank_agent/runtime/pwn.py\n"
        "--- a/biobank_agent/runtime/pwn.py\n+++ b/biobank_agent/runtime/pwn.py\n"
        "@@ -0,0 +1 @@\n+pwned = 1\n"
    )
    js = json.dumps({
        "target_path": "custom_skills/ok.py", "diff": diff,
        "test_commands": ["pytest custom_skills/ok.py -q"], "summary": "x",
    })
    res = generate_patch_for_proposal(_proposal(), llm=StubLLM(js))
    assert res.status == "rejected"
    assert "out-of-scope" in res.error or "protected" in res.error


def test_empty_diff_rejected():
    llm = StubLLM('{"target_path": "custom_skills/x.py", "diff": "   ", "test_commands": ["pytest custom_skills/x.py -q"]}')
    res = generate_patch_for_proposal(_proposal(), llm=llm)
    assert res.status == "rejected"


def test_advisory_warning_when_test_does_not_name_target():
    # Valid (a pytest run exists) but the command targets a different file → warn, not reject.
    js = _good_json(target="custom_skills/gen_demo.py", cmd="pytest custom_skills/other_suite.py -q")
    res = generate_patch_for_proposal(_proposal(), llm=StubLLM(js))
    assert res.status == "generated"
    assert any("may not exercise" in w for w in res.warnings)


# ── Evidence compaction (prompt-injection surface) ─────────────


def test_evidence_is_compacted_and_capped():
    big = "X" * 5000
    evidence = [{"kind": "tool_failure", "tool": "shell", "severity": "error",
                 "summary": big, "error": big} for _ in range(50)]
    out = _compact_evidence(evidence)
    assert len(out) <= 1500
    assert big not in out  # the 5000-char blobs never appear verbatim
    rows = json.loads(out) if out.startswith("[") else []
    assert len(rows) <= 10  # only the most recent rows are kept
    assert all("error" not in r for r in rows)  # free-text 'error' field is dropped
    assert all(len(r.get("summary", "")) <= 120 for r in rows)


# ── Idempotence / enrich ───────────────────────────────────────


def test_enrich_populates_proposal_and_marks_apply_requested():
    p = _proposal()
    [(p2, res)] = enrich_proposals_with_patches([p], llm=StubLLM(_good_json()))
    assert res.status == "generated"
    assert p2 is p
    assert p.diff and p.target_path == "custom_skills/gen_demo.py"
    assert p.apply_mode == "apply_requested"


def test_enrich_skips_already_patched_without_calling_llm():
    p = _proposal(target_path="custom_skills/x.py", diff="X = 1\n",
                  test_commands=["pytest custom_skills/x.py -q"])
    llm = StubLLM(raises=AssertionError("LLM must not be called for an already-patched proposal"))
    [(p2, res)] = enrich_proposals_with_patches([p], llm=llm)
    assert res.status == "already_generated"
    assert llm.calls == []


def test_enrich_force_regenerate_calls_llm():
    p = _proposal(target_path="custom_skills/x.py", diff="X = 1\n",
                  test_commands=["pytest custom_skills/x.py -q"])
    llm = StubLLM(_good_json())
    [(p2, res)] = enrich_proposals_with_patches([p], llm=llm, force_regenerate=True)
    assert res.status == "generated"
    assert llm.calls  # was called despite an existing patch


# ── Capstone: generate → apply into a real disposable git repo ─


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
    (repo / "custom_skills" / "__keep__").write_text("")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def test_generate_and_apply_loop_applies_verified_patch(tmp_path):
    """The autonomous loop: an LLM-shaped proposal is generated into a concrete
    self-contained pytest file, then APPLIED through the real transactional gate;
    because its tests pass and it is allow-listed, it auto-merges into the repo."""
    repo = _init_repo(tmp_path)
    llm = StubLLM(_good_json())
    records = generate_and_apply_proposals([_proposal()], llm=llm, repo_root=repo)

    assert len(records) == 1
    rec = records[0]
    assert rec["generation"]["status"] == "generated"
    assert rec["apply"] is not None
    assert rec["apply"]["status"] == "applied", rec["apply"]
    assert (repo / "custom_skills" / "gen_demo.py").exists()
    assert "def add" in (repo / "custom_skills" / "gen_demo.py").read_text()


def test_generate_and_apply_loop_failing_tests_roll_back(tmp_path):
    """If the generated patch's own tests FAIL, the apply loop rolls back: no
    file lands and no evolve branch remains."""
    repo = _init_repo(tmp_path)
    js = json.dumps({
        "target_path": "custom_skills/gen_bad.py",
        "diff": "def test_bad():\n    assert 1 == 2\n",
        "test_commands": ["pytest custom_skills/gen_bad.py -q"],
        "summary": "bad",
    })
    records = generate_and_apply_proposals([_proposal()], llm=StubLLM(js), repo_root=repo)
    rec = records[0]
    assert rec["generation"]["status"] == "generated"
    assert rec["apply"]["status"] == "tests_failed"
    assert not (repo / "custom_skills" / "gen_bad.py").exists()
    assert "evolve/" not in _git(repo, "branch", "--list")


def test_generate_and_apply_skips_apply_when_generation_rejected(tmp_path):
    repo = _init_repo(tmp_path)
    # Out-of-scope target → generation rejected → no apply attempted.
    js = _good_json(target="biobank_agent/runtime/pwn.py")
    records = generate_and_apply_proposals([_proposal()], llm=StubLLM(js), repo_root=repo)
    rec = records[0]
    assert rec["generation"]["status"] == "rejected"
    assert rec["apply"] is None
