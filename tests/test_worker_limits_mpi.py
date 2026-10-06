"""
MPI tests for the max_workers limit and the controller statistics of
DistributedEnvironment.

Run on several ranks, for example:

    mpirun.openmpi -n 5 python -m pytest tests/test_worker_limits_mpi.py

Every rank runs every test in the same order.  A limit of max_workers
keeps the worker ranks with the lowest worker index, that is, the first
max_workers ranks other than the controller; the others are idle for the
length of the environment.  Tests that need ranks left over are skipped
on smaller worlds.
"""

import re

import pytest

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except ImportError:
    pytest.skip("MPI not available", allow_module_level=True)

import distwq
import distwq.mpi as distwq_mpi
from distwq.config import Config, ResourceConfig
from distwq.future import DistributedEnvironment, WorkerGroupConfig
from distwq.mpi import CollectiveTaskResult, GroupingMethod

pytestmark = pytest.mark.skipif(SIZE < 2, reason="Requires at least 2 MPI processes")

MODULE = __name__
IS_CONTROLLER = distwq_mpi.is_controller
# The world ranks of the workers, in order of their worker index.
WORKER_RANKS = [r for r in range(SIZE) if r != distwq_mpi.controller_rank]


def world_rank():
    return MPI.COMM_WORLD.Get_rank()


def add(a, b):
    return a + b


def limited(max_workers):
    return Config(resources=ResourceConfig(max_workers=max_workers))


def summary_worker_count(text):
    match = re.search(r"workers:\s+(\d+)", text)
    assert match, text
    return int(match.group(1))


@pytest.mark.skipif(SIZE < 4, reason="needs ranks left over by max_workers=2")
def test_limit_leaves_other_ranks_idle():
    kept = WORKER_RANKS[:2]
    ranks = None
    stats = None
    with DistributedEnvironment(limited(2)) as env:
        if RANK in kept:
            assert env.role == "worker"
            assert distwq_mpi.is_worker
        elif not IS_CONTROLLER:
            assert env.role == "idle"
            assert not distwq_mpi.is_worker
        assert distwq_mpi.n_workers == 2
        for controller in env.controller_session():
            assert env.worker_count == 2
            futures = [controller.submit(world_rank) for _ in range(8)]
            ranks = {f.result() for f in futures}
            stats = controller.stats
        value = env.broadcast("done")
    if IS_CONTROLLER:
        assert ranks <= set(kept)
        assert len(stats) == 8
        # A worker reports its world rank.
        assert {s["rank"] for s in stats} <= set(kept)
        assert value == "done"
    elif RANK in kept:
        assert value == "done"
    else:
        assert value is None


def test_limit_above_world_uses_every_worker():
    with DistributedEnvironment(limited(SIZE + 3)) as env:
        assert env.role != "idle"
        assert distwq_mpi.n_workers == SIZE - 1
        for controller in env.controller_session():
            assert env.worker_count == SIZE - 1
            assert controller.submit(add, 1, 2).result() == 3


@pytest.mark.skipif(SIZE < 3, reason="needs ranks left over by max_workers=1")
def test_limit_is_recomputed_for_each_environment_and_run():
    with DistributedEnvironment(limited(1)) as env:
        for controller in env.controller_session():
            assert env.worker_count == 1
            assert controller.submit(add, 1, 1).result() == 2
    with DistributedEnvironment(Config()) as env:
        assert env.role != "idle"
        for controller in env.controller_session():
            assert env.worker_count == SIZE - 1
            assert controller.submit(add, 2, 2).result() == 4
    fun_name = "ctl_count_workers" if distwq.is_controller else None
    assert distwq.run(fun_name=fun_name, module_name=MODULE) == (
        SIZE - 1 if distwq.is_controller else None
    )


def ctl_count_workers(controller):
    return controller.comm.size - 1


@pytest.mark.skipif(SIZE < 4, reason="needs ranks left over by max_workers=2")
def test_split_groups_use_only_the_limited_ranks():
    kept = WORKER_RANKS[:2]
    worker_config = WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSplit,
        nprocs_per_worker=2,
        broker_is_worker=True,
    )
    result = text = None
    with DistributedEnvironment(limited(2), worker_config) as env:
        if RANK == kept[0]:
            assert env.role == "broker"
        elif RANK == kept[1]:
            assert env.role == "collective_worker"
        elif not IS_CONTROLLER:
            assert env.role == "idle"
        for controller in env.controller_session():
            assert env.worker_count == 1
            result = controller.submit(add, 3, 4).result()
            text = controller.info()
    if IS_CONTROLLER:
        assert result == CollectiveTaskResult([7, 7])
        assert summary_worker_count(text) == 1


@pytest.mark.skipif(SIZE < 5, reason="needs at least 3 worker ranks to limit to 3")
def test_uneven_split_of_limited_ranks_fails_on_every_rank():
    worker_config = WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSplit,
        nprocs_per_worker=2,
        broker_is_worker=True,
    )
    with (
        pytest.raises(RuntimeError, match="unable to split workers evenly"),
        DistributedEnvironment(limited(3), worker_config),
    ):
        pass  # pragma: no cover
    # The failed environment left nothing behind: a new one works.
    with DistributedEnvironment(Config()) as env:
        for controller in env.controller_session():
            assert controller.submit(add, 2, 2).result() == 4


def test_stats_and_info_count_every_task():
    n_tasks = 2 * SIZE
    stats = text = None
    with DistributedEnvironment(Config()) as env:
        for controller in env.controller_session():
            futures = [controller.submit(add, i, 1) for i in range(n_tasks)]
            assert [f.result() for f in futures] == [i + 1 for i in range(n_tasks)]
            stats = controller.stats
            text = controller.info()
            n_workers = env.worker_count
    if IS_CONTROLLER:
        assert sorted(s["id"] for s in stats) == sorted(f.task_id for f in futures)
        assert not any(s.get("error") for s in stats)
        assert summary_worker_count(text) == n_workers
        assert f"results collected:         {n_tasks}" in text
