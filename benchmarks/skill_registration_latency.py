"""Regression guard for schema-first skill registration latency."""

from __future__ import annotations

import json
import subprocess
import sys


def test_builtin_skill_registration_is_lazy_and_fast():
    code = r"""
import json
import sys
import time
from biobank_agent.registry import autodiscover_skills, get_registry

start = time.perf_counter()
autodiscover_skills()
elapsed = time.perf_counter() - start
reg = get_registry()
names = {item["name"] for item in reg.list_skills()}
imported = sorted(
    name
    for name in sys.modules
    if name.startswith("biobank_agent.skills.") and name != "biobank_agent.skills"
)
print(json.dumps({
    "elapsed_s": elapsed,
    "skill_count": len(reg),
    "has_train_model": "train_model" in names,
    "has_generate_report": "generate_report" in names,
    "imported_skill_modules": imported,
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["skill_count"] >= 55
    assert payload["has_train_model"]
    assert payload["has_generate_report"]
    assert payload["imported_skill_modules"] == []
    assert payload["elapsed_s"] < 0.25
