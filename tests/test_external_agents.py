"""External coding-agent invocation seam: detection, safe invocation, and the injected runner."""

from __future__ import annotations

import subprocess

from biobank_agent.runtime.external_agents import (
    DEFAULT_SPECS,
    ExternalAgentResult,
    _safe_env,
    detect_available,
    parse_agent_list,
    run_external_agent,
    spec_for,
)


def test_parse_agent_list_handles_string_and_sequence():
    assert parse_agent_list("codex, claude ,gemini") == ["codex", "claude", "gemini"]
    assert parse_agent_list(["codex", " claude "]) == ["codex", "claude"]
    assert parse_agent_list("") == []


def test_build_argv_passes_prompt_as_single_token():
    spec = DEFAULT_SPECS["codex"]
    argv = spec.build_argv("review this; rm -rf / && echo hi")
    # The prompt is one argv element — no shell, so metacharacters cannot split or inject.
    assert argv == ["codex", "exec", "review this; rm -rf / && echo hi"]
    assert argv[-1].count(" ") >= 1


def test_detect_available_only_returns_on_path():
    present = {"codex"}
    specs = detect_available(["codex", "claude", "gemini"], which=lambda b: "/usr/bin/" + b if b in present else None)
    assert [s.name for s in specs] == ["codex"]


def test_detect_available_skips_unknown_names():
    specs = detect_available(["not-a-real-agent"], which=lambda b: "/usr/bin/" + b)
    assert specs == []


def test_run_external_agent_success_with_fake_runner():
    seen = {}

    def fake_runner(argv, cwd, timeout_s, env):
        seen["argv"] = argv
        seen["env"] = env
        return 0, "REVIEW: the plan omits a QC step.\n", ""

    res = run_external_agent(spec_for("codex"), "critique this plan", runner=fake_runner)
    assert isinstance(res, ExternalAgentResult) and res.ok is True
    assert "omits a QC step" in res.text
    assert seen["argv"][0] == "codex"


def test_run_external_agent_nonzero_exit_is_not_ok():
    res = run_external_agent(spec_for("codex"), "x", runner=lambda *a: (2, "partial", "boom"))
    assert res.ok is False and res.returncode == 2 and "boom" in res.error


def test_run_external_agent_timeout_is_caught():
    def boom(argv, cwd, timeout_s, env):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout_s)

    res = run_external_agent(spec_for("codex"), "x", timeout_s=0.1, runner=boom)
    assert res.ok is False and "timed out" in res.error


def test_run_external_agent_missing_binary_is_caught():
    def missing(argv, cwd, timeout_s, env):
        raise FileNotFoundError(argv[0])

    res = run_external_agent(spec_for("claude"), "x", runner=missing)
    assert res.ok is False and "not found" in res.error


def test_run_external_agent_output_is_capped():
    res = run_external_agent(spec_for("codex"), "x", runner=lambda *a: (0, "z" * 500_000, ""))
    assert res.ok is True and len(res.text) <= 200_000


def test_safe_env_strips_credentials(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-secret-123")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    monkeypatch.setenv("MY_PASSWORD", "hunter2")
    monkeypatch.setenv("PATH_KEEPER", "/usr/bin")  # 'PATH_KEEPER' has no secret token -> kept
    env = _safe_env()
    assert "LLM_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert "MY_PASSWORD" not in env
    assert env.get("PATH_KEEPER") == "/usr/bin"


def test_run_external_agent_receives_secret_stripped_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-leak")
    captured = {}

    def fake_runner(argv, cwd, timeout_s, env):
        captured["env"] = env
        return 0, "ok", ""

    run_external_agent(spec_for("codex"), "x", runner=fake_runner)
    assert "OPENAI_API_KEY" not in captured["env"]
