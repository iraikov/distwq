"""
Check that a worker init function that raises aborts the whole MPI job.

These tests start their own MPI jobs (tests/init_failure_driver.py), so
run them in a single process, not under mpirun.  Like
tests/test_examples.py, they are skipped unless DISTWQ_MPIRUN gives the
launcher command, for example:

    DISTWQ_MPIRUN="/usr/bin/mpirun.openmpi --use-hwthread-cpus --bind-to none" \\
        python -m pytest tests/test_init_failure.py

A job that hangs instead of aborting fails the test after
DISTWQ_INIT_FAILURE_TIMEOUT seconds (default 120).
"""

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from run_guards import outside_mpi_job_only

ROOT = Path(__file__).resolve().parent.parent
DRIVER = Path(__file__).resolve().parent / "init_failure_driver.py"
MPIRUN = os.environ.get("DISTWQ_MPIRUN")
TIMEOUT = float(os.environ.get("DISTWQ_INIT_FAILURE_TIMEOUT", "120"))

pytestmark = [
    pytest.mark.skipif(
        not MPIRUN, reason="set DISTWQ_MPIRUN to the MPI launcher command"
    ),
    outside_mpi_job_only,
]


def run_driver(mode, fail_rank, n_ranks=3):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(ROOT), env.get("PYTHONPATH", "")) if p
    )
    command = [
        *shlex.split(MPIRUN),
        "-x",
        "PYTHONPATH",
        "-n",
        str(n_ranks),
        sys.executable,
        str(DRIVER),
        mode,
        str(fail_rank),
    ]
    try:
        completed = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"the job hung instead of aborting:\n{exc.stdout}{exc.stderr}")
    return completed.returncode, completed.stdout + completed.stderr


@pytest.mark.parametrize(
    "mode, fail_rank",
    [
        ("none", 1),
        ("none", 2),
        # Rank 1 is the group's broker, which also works; rank 2 is the
        # other member of the group.
        ("split", 1),
        ("split", 2),
    ],
)
def test_failing_worker_init_aborts_the_job(mode, fail_rank):
    returncode, output = run_driver(mode, fail_rank)
    assert returncode != 0, output
    assert "driver finished normally" not in output
    assert f"init failed on purpose on rank {fail_rank}" in output
    assert "raised an exception; aborting the job" in output


def test_driver_finishes_when_init_succeeds():
    # No rank has this number, so no init function raises.
    returncode, output = run_driver("none", -1)
    assert returncode == 0, output
    assert output.count("driver finished normally") == 3
