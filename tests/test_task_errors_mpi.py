"""
MPI tests for task exceptions raised on workers.

Run on several ranks, for example:

    mpirun.openmpi -n 3 python -m pytest tests/test_task_errors_mpi.py

A task's exception is sent back to the controller and raised from its
future's result(), with the worker's traceback as its cause.  The workers
keep serving, so later tasks, later sessions and leaving the environment
all work as usual.  The split-group tests need (size - 1) divisible by 2.
"""

import pytest

try:
    from mpi4py import MPI

    RANK = MPI.COMM_WORLD.Get_rank()
    SIZE = MPI.COMM_WORLD.Get_size()
except ImportError:
    pytest.skip("MPI not available", allow_module_level=True)

from distwq.functional import map_distributed
from distwq.future import (
    Config,
    DistributedEnvironment,
    RemoteTraceback,
    WorkerGroupConfig,
)
from distwq.mpi import GroupingMethod

pytestmark = pytest.mark.skipif(SIZE < 2, reason="Requires at least 2 MPI processes")

IS_CONTROLLER = RANK == 0

split_groups = pytest.mark.skipif(
    SIZE < 3 or (SIZE - 1) % 2 != 0,
    reason="split groups of 2 need (size - 1) divisible by 2",
)


def square(x):
    return x * x


def fail_on_three(x):
    if x == 3:
        raise ValueError(f"cannot handle {x}")
    return x


def fail_on_even_world_rank(x):
    # In split groups of 2 with broker_is_worker=True, the brokers are the
    # odd world ranks, so this fails only on the non-broker processes.
    if MPI.COMM_WORLD.Get_rank() % 2 == 0:
        raise KeyError(f"rank {MPI.COMM_WORLD.Get_rank()}")
    return x


def test_failed_task_raises_and_workers_keep_serving():
    with DistributedEnvironment(Config()) as env:
        for ctl in env.controller_session():
            bad = ctl.submit(fail_on_three, 3)
            with pytest.raises(ValueError, match="cannot handle 3") as info:
                bad.result()
            assert isinstance(info.value.__cause__, RemoteTraceback)
            assert "fail_on_three" in info.value.__cause__.tb
            assert [ctl.submit(square, i).result() for i in range(4)] == [0, 1, 4, 9]
        for ctl in env.controller_session():
            assert map_distributed(ctl, square, [5, 6]) == [25, 36]


def test_failure_inside_map_then_next_session():
    with DistributedEnvironment(Config()) as env:
        for ctl in env.controller_session():
            with pytest.raises(ValueError, match="cannot handle 3"):
                map_distributed(ctl, fail_on_three, range(6))
        # The session closed cleanly, with the other tasks drained.
        for ctl in env.controller_session():
            assert map_distributed(ctl, fail_on_three, [1, 2]) == [1, 2]


def test_failure_escaping_the_session_body():
    if IS_CONTROLLER:
        with (
            pytest.raises(ValueError, match="cannot handle 3"),
            DistributedEnvironment(Config()) as env,
        ):
            for ctl in env.controller_session():
                ctl.submit(square, 2)
                ctl.submit(fail_on_three, 3).result()
    else:
        with DistributedEnvironment(Config()) as env:
            for _ctl in env.controller_session():
                pass
    # Everything was released: a new environment works.
    with DistributedEnvironment(Config()) as env:
        for ctl in env.controller_session():
            assert ctl.submit(square, 7).result() == 49


def _split_config():
    return WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSplit,
        nprocs_per_worker=2,
        broker_is_worker=True,
    )


@split_groups
def test_failure_on_every_group_member():
    with DistributedEnvironment(Config(), _split_config()) as env:
        for ctl in env.controller_session():
            with pytest.raises(ValueError, match="cannot handle 3"):
                ctl.submit(fail_on_three, 3).result()
            assert ctl.submit(square, 4).result() == [16, 16]


@split_groups
def test_failure_on_non_broker_member_only():
    with DistributedEnvironment(Config(), _split_config()) as env:
        for ctl in env.controller_session():
            with pytest.raises(KeyError, match="rank") as info:
                ctl.submit(fail_on_even_world_rank, 1).result()
            assert isinstance(info.value.__cause__, RemoteTraceback)
            assert ctl.submit(square, 3).result() == [9, 9]
