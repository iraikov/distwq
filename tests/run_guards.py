"""
Skip markers that keep each test file to the kind of run it was written for.

The tests come in two kinds that cannot share a run.  Single-process tests
assume that there are no workers.  Under mpirun they fail on the worker
ranks only, the ranks stop making the same MPI calls in the same order, and
later MPI tests then fail or hang.  Tests that start their own MPI jobs
would start one job from every rank.  These markers skip such files, so
that ``mpirun -n N python -m pytest tests`` runs only the MPI tests and a
plain ``python -m pytest tests`` runs only the others.

single_process_only reads the world size from distwq.mpi, which every
single-process test file imports anyway.  outside_mpi_job_only looks only
at the environment that MPI launchers set for their ranks, so that a file
that starts MPI jobs does not have to initialize MPI itself.
"""

import os

import pytest

# Variables that a launcher sets in the environment of each rank it starts:
# Open MPI's, any PMIx-based launcher (srun with PMIx too), and PMI-based
# ones such as MPICH's Hydra.
LAUNCHER_VARIABLES = (
    "OMPI_COMM_WORLD_SIZE",
    "PMIX_RANK",
    "PMI_SIZE",
    "PMI_RANK",
)


def mpi_world_size() -> int:
    """The size of this process's MPI world: 1 without MPI or a launcher."""
    from distwq import mpi

    return mpi.size


def launched_by_mpi() -> bool:
    """True if this process is a rank started by an MPI launcher."""
    return any(name in os.environ for name in LAUNCHER_VARIABLES)


single_process_only = pytest.mark.skipif(
    mpi_world_size() > 1,
    reason="single-process test: run it without mpirun",
)

outside_mpi_job_only = pytest.mark.skipif(
    launched_by_mpi(),
    reason="starts its own MPI jobs: run it without mpirun",
)
