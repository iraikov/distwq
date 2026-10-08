"""
Run the example scripts under MPI and check their output.

These tests start their own MPI jobs, so run them in a single process,
not under mpirun.  They are skipped unless DISTWQ_MPIRUN gives the
launcher command, for example:

    DISTWQ_MPIRUN="/usr/bin/mpirun.openmpi --use-hwthread-cpus --bind-to none" \\
        python -m pytest tests/test_examples.py

The repository root is put at the front of PYTHONPATH for the jobs; the
NEURON example also needs NEURON's Python package on PYTHONPATH, and is
skipped if neuron cannot be imported.  Each job is stopped after
DISTWQ_EXAMPLE_TIMEOUT seconds (default 300), so a hang fails the test
instead of blocking it.
"""

import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from run_guards import outside_mpi_job_only

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
MPIRUN = os.environ.get("DISTWQ_MPIRUN")
TIMEOUT = float(os.environ.get("DISTWQ_EXAMPLE_TIMEOUT", "300"))

pytestmark = [
    pytest.mark.skipif(
        not MPIRUN, reason="set DISTWQ_MPIRUN to the MPI launcher command"
    ),
    outside_mpi_job_only,
]


def run_example(script, n_ranks):
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
        str(EXAMPLES / script),
    ]
    completed = subprocess.run(
        command, env=env, capture_output=True, text=True, timeout=TIMEOUT, check=False
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    return output


@pytest.mark.parametrize("n_ranks", [2, 4])
def test_task_interface_example(n_ranks):
    output = run_example("example_task_interface.py", n_ranks)
    assert "A failing task raised on the controller: ValueError" in output
    assert "pi = 3.14" in output


@pytest.mark.parametrize("n_ranks", [2, 4])
def test_functional_pipeline_example(n_ranks):
    output = run_example("example_functional_pipeline.py", n_ranks)
    assert "9 of 12 recordings have a clear peak" in output
    assert "quick_map result on every rank: [0, 1, 4, 9" in output


@pytest.mark.skipif(
    importlib.util.find_spec("neuron") is None, reason="NEURON is not importable"
)
def test_neuron_fi_curve_example():
    output = run_example("example_neuron_fi_curve.py", 4)
    assert "Rheobase at 6.3 C: between" in output
    assert "Step duration chosen for the fine curve:" in output


@pytest.mark.parametrize("n_ranks", [3, 5])
def test_worker_group_ensembles_example(n_ranks):
    output = run_example("example_worker_group_ensembles.py", n_ranks)
    assert f"{(n_ranks - 1) // 2} worker groups of 2 processes" in output
    assert "noiseless rate" in output
