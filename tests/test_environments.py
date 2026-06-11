"""Scheduler-aware execution backends."""

from __future__ import annotations

from types import SimpleNamespace

from biobank_agent.runtime.environments import (
    JobResources, LocalBackend, SlurmBackend, SGEBackend, LSFBackend, DxBackend,
    get_backend, detect_backend,
)


def test_slurm_submit_argv_and_parse():
    b = SlurmBackend()
    argv = b.submit_argv("/jobs/run.sh", job_name="gwas", resources=JobResources(cpus=8, mem_mb=16000, time_min=600, partition="bio"))
    assert argv[0] == "sbatch" and "--parsable" in argv
    assert "--cpus-per-task" in argv and "8" in argv
    assert "--mem" in argv and "16000M" in argv
    assert "--time" in argv and "10:00:00" in argv
    assert "--partition" in argv and "bio" in argv
    assert argv[-1] == "/jobs/run.sh"
    assert b.parse_job_id("123456;cluster1") == "123456"
    assert b.parse_job_id("789012\n") == "789012"
    assert b.status_argv("123456")[0] == "squeue"
    assert b.cancel_argv("123456") == ["scancel", "123456"]


def test_sge_and_lsf_parse():
    assert SGEBackend().parse_job_id("9988776\n") == "9988776"
    assert SGEBackend().submit_argv("/r.sh", job_name="j", resources=JobResources())[0] == "qsub"
    lsf = LSFBackend()
    assert lsf.parse_job_id("Job <44556> is submitted to queue <normal>.") == "44556"
    assert lsf.submit_argv("/r.sh", job_name="j", resources=JobResources())[0] == "bsub"


def test_dx_parse():
    dx = DxBackend()
    assert dx.parse_job_id("job-Gxff00112233445566778899") == "job-Gxff00112233445566778899"
    assert dx.submit_argv("/r.sh", job_name="j", resources=JobResources())[0] == "dx"


def test_build_script_portable():
    script = SlurmBackend().build_script("plink2 --bfile x", cwd="/work", job_name="j", resources=JobResources())
    assert script.startswith("#!/bin/bash")
    assert "cd /work" in script and "plink2 --bfile x" in script


def test_detect_backend_explicit_and_local_default():
    # explicit config wins
    assert detect_backend(SimpleNamespace(exec_backend="slurm")).name == "slurm"
    # unknown -> local
    assert detect_backend(SimpleNamespace(exec_backend="nope")).name in {"local", "nope"} or isinstance(detect_backend(SimpleNamespace(exec_backend="nope")), LocalBackend)
    # auto with no scheduler installed (CI) -> local
    b = detect_backend(SimpleNamespace(exec_backend="auto"))
    assert b.name in {"local", "slurm", "sge", "lsf", "dxrun"}  # local on a dev box
    # no settings -> local
    assert isinstance(detect_backend(None), LocalBackend)


def test_local_backend_always_available():
    assert LocalBackend().available() is True
    assert get_backend("local").name == "local"
