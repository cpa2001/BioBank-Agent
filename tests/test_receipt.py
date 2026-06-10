"""Tests for M7: reproducibility receipt."""

from __future__ import annotations

import hashlib
import json

from biobank_agent.runtime.receipt import (
    build_receipt, data_fingerprint, format_receipt_markdown, write_receipt, SCHEMA,
)


def test_data_fingerprint_hashes_small_file(tmp_path):
    f = tmp_path / "cohort.csv"
    payload = b"eid,age\n1,50\n2,60\n"
    f.write_bytes(payload)
    fp = data_fingerprint(f)
    assert fp["exists"] is True
    assert fp["size_bytes"] == len(payload)
    assert fp["sha256"] == hashlib.sha256(payload).hexdigest()
    assert "mtime" in fp


def test_data_fingerprint_missing_file(tmp_path):
    fp = data_fingerprint(tmp_path / "nope.parquet")
    assert fp["exists"] is False
    assert "sha256" not in fp


def test_build_receipt_captures_reproducibility_fields(tmp_path):
    f = tmp_path / "geno.bed"
    f.write_bytes(b"\x00\x01\x02")
    receipt = build_receipt(
        objective="GWAS on T2D",
        skill="vcf_association",
        command="plink2 --bfile geno --glm",
        params={"n_pcs": 10},
        data_paths=[f],
        seeds={"numpy": 42},
        timestamp=1700000000.0,
    )
    assert receipt["schema"] == SCHEMA
    assert receipt["timestamp"] == 1700000000.0          # verbatim (deterministic)
    assert receipt["skill"] == "vcf_association"
    assert receipt["seeds"] == {"numpy": 42}
    assert receipt["params"] == {"n_pcs": 10}
    assert isinstance(receipt["code_version"], str)      # git SHA or ''
    assert receipt["python_version"]
    assert isinstance(receipt["packages"], dict)
    assert receipt["data"] and receipt["data"][0]["exists"] is True


def test_format_receipt_markdown_has_key_lines():
    receipt = build_receipt(objective="x", command="plink2 --glm",
                            seeds={"numpy": 1}, timestamp=1.0)
    md = format_receipt_markdown(receipt)
    assert "Reproducibility receipt" in md
    assert "Code version" in md
    assert "plink2 --glm" in md
    assert "Seeds" in md


def test_write_receipt_roundtrips(tmp_path):
    receipt = build_receipt(objective="x", timestamp=1.0)
    path = write_receipt(receipt, tmp_path / "out")
    assert path.exists() and path.name == "receipt.json"
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == SCHEMA
