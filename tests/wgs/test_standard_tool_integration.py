"""Tests for standard WGS tool command integration."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_parse_plink2_glm_additive_rows(tmp_path):
    from biobank_agent.skills.vcf_association import _parse_plink2_glm

    glm = tmp_path / "assoc.PHENO.glm.logistic.hybrid"
    glm.write_text(
        "#CHROM\tPOS\tID\tREF\tALT\tA1\tTEST\tOBS_CT\tOR\tP\tA1_FREQ\n"
        "22\t16050075\trs1\tA\tG\tG\tADD\t16\t1.8\t0.012\t0.22\n"
        "22\t16050100\trs2\tC\tT\tT\tDOMDEV\t16\t1.1\t0.5\t0.10\n",
        encoding="utf-8",
    )

    rows = _parse_plink2_glm(glm)
    assert len(rows) == 1
    assert rows[0]["chrom"] == "chr22"
    assert rows[0]["pos"] == 16050075
    assert rows[0]["p_value"] == 0.012
    assert rows[0]["source"] == "plink2_glm"


def test_run_external_writes_log_for_redirected_stdout(tmp_path):
    from biobank_agent.utils.wgs import run_external

    out = tmp_path / "out.txt"
    log = tmp_path / "tool.log"

    result = run_external(
        ["bash", "-lc", "echo redirected; echo err >&2"],
        timeout=10,
        stdout_path=out,
        log_path=log,
    )

    assert result["ok"] is True
    assert result["log_path"] == str(log)
    assert out.read_text(encoding="utf-8").strip() == "redirected"
    text = log.read_text(encoding="utf-8")
    assert "[stderr]" in text and "err" in text


def test_run_plink2_association_builds_glm_command(monkeypatch, tmp_path):
    import biobank_agent.skills.vcf_association as mod

    merged = tmp_path / "merged.vcf.gz"
    merged.write_text("", encoding="utf-8")
    pheno_df = pd.DataFrame({
        "sample_id": ["J1", "J2", "V1", "V2"],
        "age": [21, 22, 40, 41],
        "sex": ["F", "M", "M", "F"],
    })
    captured = {}

    monkeypatch.setattr(mod, "find_executable", lambda name: "/bin/plink2")
    monkeypatch.setattr(mod, "tool_version", lambda *a, **k: "PLINK v2")

    def fake_run(cmd, timeout=0, cwd=None, stdout_path=None, settings=None, line_sink=None, log_path=None):
        captured["cmd"] = cmd
        captured["line_sink"] = line_sink
        captured["log_path"] = log_path
        out_prefix = Path(cmd[cmd.index("--out") + 1])
        glm = out_prefix.with_suffix(".PHENO.glm.logistic.hybrid")
        glm.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\tA1\tTEST\tOBS_CT\tOR\tP\n"
            "22\t16050075\trs1\tA\tG\tG\tADD\t4\t2.0\t0.04\n",
            encoding="utf-8",
        )
        return {"ok": True, "returncode": 0, "stdout": "", "stderr": "", "cmd": cmd}

    monkeypatch.setattr(mod, "run_external", fake_run)

    result = mod._run_plink2_association(
        merged, pheno_df, ["J1", "J2"], ["V1", "V2"], "sample_id", tmp_path, "chr22"
    )

    assert result["ok"] is True
    assert "--glm" in captured["cmd"]
    assert "firth-fallback" in captured["cmd"]
    assert "--new-id-max-allele-len" in captured["cmd"]
    assert "--pheno" in captured["cmd"]
    assert result["records"][0]["p_value"] == 0.04


def test_run_plink2_association_streams_to_log(monkeypatch, tmp_path):
    import biobank_agent.skills.vcf_association as mod

    merged = tmp_path / "merged.vcf.gz"
    merged.write_text("", encoding="utf-8")
    pheno_df = pd.DataFrame({"sample_id": ["J1", "J2", "V1", "V2"]})
    seen = {}

    monkeypatch.setattr(mod, "find_executable", lambda name: "/bin/plink2")
    monkeypatch.setattr(mod, "tool_version", lambda *a, **k: "PLINK v2")

    def fake_run(cmd, timeout=0, cwd=None, stdout_path=None, settings=None, line_sink=None, log_path=None):
        seen["line_sink"] = line_sink
        seen["log_path"] = log_path
        out_prefix = Path(cmd[cmd.index("--out") + 1])
        glm = out_prefix.with_suffix(".PHENO.glm.logistic.hybrid")
        glm.write_text(
            "#CHROM\tPOS\tID\tREF\tALT\tA1\tTEST\tOBS_CT\tOR\tP\n"
            "22\t16050075\trs1\tA\tG\tG\tADD\t4\t2.0\t0.04\n",
            encoding="utf-8",
        )
        return {"ok": True, "returncode": 0, "stdout": "", "stderr": "", "cmd": cmd, "log_path": str(log_path)}

    ctx = type("Ctx", (), {"emit_line": lambda *a, **k: None, "tool_call_id": "call1", "settings": object()})()
    monkeypatch.setattr(mod, "run_external", fake_run)

    result = mod._run_plink2_association(
        merged, pheno_df, ["J1", "J2"], ["V1", "V2"], "sample_id", tmp_path, "chr22", ctx=ctx
    )

    assert result["ok"] is True
    assert seen["line_sink"] is not None
    assert str(seen["log_path"]).endswith(".log")
    assert result["run"]["log_path"] == str(seen["log_path"])


def test_snpeff_command_writes_expected_outputs(monkeypatch, tmp_path):
    import biobank_agent.skills.vcf_annotation as mod

    merged = tmp_path / "merged.vcf.gz"
    merged.write_text("", encoding="utf-8")
    captured = {}

    monkeypatch.setattr(mod, "find_executable", lambda name: "/bin/snpEff")
    monkeypatch.setattr(mod, "snpeff_database_ready", lambda genome: True)
    monkeypatch.setattr(mod, "tool_version", lambda *a, **k: "SnpEff 5.4")
    monkeypatch.setattr(mod, "_parse_snpeff_annotations", lambda path: (
        [{"chrom": "chr22", "pos": 16050075, "gene": "SOX10", "effect": "missense_variant"}],
        {"missense_variant": 1},
    ))
    monkeypatch.setattr(mod, "_parse_snpeff_stats", lambda path: {"number_of_variants": 1})

    def fake_run(cmd, timeout=0, cwd=None, stdout_path=None, settings=None, line_sink=None, log_path=None):
        captured["cmd"] = cmd
        captured["log_path"] = log_path
        assert stdout_path is not None
        Path(stdout_path).write_text("##fileformat=VCFv4.2\n", encoding="utf-8")
        return {"ok": True, "returncode": 0, "stdout": "", "stderr": "", "cmd": cmd}

    monkeypatch.setattr(mod, "run_external", fake_run)

    result = mod._run_snpeff_annotation(merged, tmp_path, genome="hg38", label="chr22")

    assert result["ok"] is True
    assert captured["cmd"][:2] == ["/bin/snpEff", "-noLog"]
    assert "hg38" in captured["cmd"]
    assert captured["log_path"] is None
    assert Path(result["annotation_table"]).exists()
    assert result["effect_counts"]["missense_variant"] == 1
