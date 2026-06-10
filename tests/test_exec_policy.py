"""Tests for command timeout / background policy (issue #1)."""

from __future__ import annotations

from biobank_agent.utils.exec_policy import (
    is_long_tool,
    resolve_timeout,
    should_background,
    tool_key,
)


def test_tool_key_from_argv_and_command():
    assert tool_key(argv=["/usr/bin/plink2", "--bfile", "x"]) == "plink2"
    assert tool_key(command="bcftools view a.vcf | head") == "bcftools"
    assert tool_key(command="FOO=bar gatk HaplotypeCaller") == "gatk"
    assert tool_key(command="./run.sh") == "run.sh"
    assert tool_key() == ""


def test_is_long_tool():
    assert is_long_tool(command="plink --bfile x")
    assert is_long_tool(argv=["gatk", "HaplotypeCaller"])
    assert not is_long_tool(command="ls -la")
    assert not is_long_tool(command="python -c 'print(1)'")


def test_resolve_timeout_explicit_positive_honored_and_clamped():
    # explicit positive value honored verbatim (foreground clamps to ceiling)
    assert resolve_timeout(command="ls", override=300) == 300
    assert resolve_timeout(command="ls", override=10**9) == 86400  # hard ceiling
    # background is not clamped
    assert resolve_timeout(command="ls", override=10**9, background=True) == 10**9


def test_resolve_timeout_auto_scales_known_long_tools():
    # 0/None -> auto: unknown command falls back to the generic default
    assert resolve_timeout(command="ls", override=0) == 120
    assert resolve_timeout(command="ls", override=None) == 120
    # known long tools get their (longer) baseline
    assert resolve_timeout(command="plink --bfile x", override=0) == 7200
    assert resolve_timeout(argv=["gatk", "HaplotypeCaller"], override=0) == 14400


def test_resolve_timeout_background_uncapped_when_allowed():
    # explicit "no cap" (<=0) is honored only for background jobs
    assert resolve_timeout(command="gatk x", override=0, background=True) == 14400  # auto, not None
    assert resolve_timeout(command="gatk x", override=-1, background=True) is None  # explicit no-cap
    # foreground can never be uncapped
    assert resolve_timeout(command="gatk x", override=-1, background=False) == 14400


def test_should_background_precedence():
    # explicit flag wins
    assert should_background(command="ls", explicit=True) is True
    assert should_background(command="plink x", explicit=False) is False
    # expected duration over threshold
    assert should_background(command="ls", expected_s=601) is True
    assert should_background(command="ls", expected_s=10) is False
    # long-tool heuristic only auto-backgrounds when NOT interactive
    assert should_background(command="plink x", interactive=True) is False
    assert should_background(command="plink x", interactive=False) is True
    assert should_background(command="ls", interactive=False) is False


class _Settings:
    exec_default_timeout_s = 60
    exec_long_tool_timeout_s = 9999
    exec_hard_ceiling_s = 100000
    exec_allow_no_timeout = True
    exec_background_threshold_s = 300


def test_resolve_timeout_reads_settings():
    s = _Settings()
    assert resolve_timeout(command="ls", override=0, settings=s) == 60          # configured default
    # long-tool baseline is max(configured floor, per-tool) => max(9999, 7200)
    assert resolve_timeout(command="plink x", override=0, settings=s) == 9999
    assert should_background(command="ls", expected_s=300, settings=s) is True
