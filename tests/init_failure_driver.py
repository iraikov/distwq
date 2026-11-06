"""
MPI program whose worker init function raises on one rank.

Started by tests/test_init_failure.py as its own MPI job:

    mpirun -n 3 python tests/init_failure_driver.py MODE FAIL_RANK

MODE is "none" (no worker grouping) or "split" (split groups of 2, with
working brokers).  worker_init raises on world rank FAIL_RANK.  The job
must then end with a non-zero exit status instead of hanging.
"""

import logging
import sys

from mpi4py import MPI

from distwq.config import Config
from distwq.future import DistributedEnvironment, WorkerGroupConfig
from distwq.mpi import GroupingMethod

logging.basicConfig(level=logging.WARNING)


def failing_init(server, fail_rank):
    if MPI.COMM_WORLD.Get_rank() == fail_rank:
        raise RuntimeError(f"init failed on purpose on rank {fail_rank}")


def square(x):
    return x * x


def main(mode, fail_rank):
    worker_config = None
    if mode == "split":
        worker_config = WorkerGroupConfig(
            grouping_method=GroupingMethod.GroupSplit,
            nprocs_per_worker=2,
            broker_is_worker=True,
        )
    with DistributedEnvironment(
        Config(), worker_config, worker_init=failing_init, init_args=(fail_rank,)
    ) as env:
        for controller in env.controller_session():
            for f in [controller.submit(square, i) for i in range(4)]:
                f.result()
    print("driver finished normally", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
