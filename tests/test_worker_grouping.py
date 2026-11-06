"""
MPI tests for worker groups in DistributedEnvironment.

Run on several ranks; each layout runs only at world sizes it fits, so
use a few sizes to cover them all, for example:

    mpirun.openmpi -n 3 python -m pytest tests/test_worker_grouping.py
    mpirun.openmpi -n 4 python -m pytest tests/test_worker_grouping.py
    mpirun.openmpi -n 5 python -m pytest tests/test_worker_grouping.py

A worker group is nprocs_per_worker processes that each run every task
sent to the group, plus one broker process that relays tasks from the
controller.  With broker_is_worker=True the broker is one of those
processes; otherwise it is an extra process that only relays.  A group
task's future returns a CollectiveTaskResult, whose values are the results
from the nprocs_per_worker processes that ran it.

Spawned groups are tested only when DISTWQ_TEST_SPAWN=1, since spawning
does not work with every MPI installation.
"""

import os

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
    WorkerGroupConfig,
    create_split_environment,
)
from distwq.mpi import CollectiveTaskResult, GroupingMethod

pytestmark = pytest.mark.skipif(SIZE < 3, reason="Requires at least 3 MPI processes")

IS_CONTROLLER = RANK == 0


def square(x):
    return x * x


def nothing(x):
    return None


def values_of(results):
    """The values of each CollectiveTaskResult in results."""
    assert all(isinstance(r, CollectiveTaskResult) for r in results), results
    return [r.values for r in results]


def processes_per_group(nprocs_per_worker, broker_is_worker):
    return nprocs_per_worker if broker_is_worker else nprocs_per_worker + 1


def fits(nprocs_per_worker, broker_is_worker):
    return (SIZE - 1) % processes_per_group(nprocs_per_worker, broker_is_worker) == 0


def run_two_sessions(worker_config):
    """Run a submit and a map in two sessions; return what the controller saw."""
    seen = {}
    with DistributedEnvironment(Config(), worker_config) as env:
        for ctl in env.controller_session():
            seen["worker_count"] = env.worker_count
            seen["submit"] = values_of([ctl.submit(square, 3).result()])[0]
            # A task returning None still gives one value per process.
            seen["none"] = values_of([ctl.submit(nothing, 0).result()])[0]
        for ctl in env.controller_session():
            seen["map"] = values_of(map_distributed(ctl, square, range(4)))
    return seen


# (nprocs_per_worker, broker_is_worker, collective_mode)
LAYOUTS = [
    (1, False, "gather"),
    (1, True, "gather"),
    (2, True, "gather"),
    (2, False, "gather"),
    (2, True, "sendrecv"),
    (2, False, "sendrecv"),
    (3, True, "gather"),
]


@pytest.mark.parametrize(("nprocs", "broker_is_worker", "collective_mode"), LAYOUTS)
def test_split_layout(nprocs, broker_is_worker, collective_mode):
    if not fits(nprocs, broker_is_worker):
        pytest.skip(f"world size {SIZE} does not fit this layout")
    worker_config = WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSplit,
        nprocs_per_worker=nprocs,
        broker_is_worker=broker_is_worker,
        collective_mode=collective_mode,
    )
    seen = run_two_sessions(worker_config)
    if IS_CONTROLLER:
        n_groups = (SIZE - 1) // processes_per_group(nprocs, broker_is_worker)
        assert seen["worker_count"] == n_groups
        assert seen["submit"] == [9] * nprocs
        assert seen["none"] == [None] * nprocs
        assert seen["map"] == [[x * x] * nprocs for x in range(4)]


def test_single_process_workers_return_plain_values():
    # Without grouping a future returns the task's value itself.
    with DistributedEnvironment(Config(), WorkerGroupConfig()) as env:
        results = None
        for ctl in env.controller_session():
            results = [ctl.submit(square, 3).result(), ctl.submit(nothing, 0).result()]
            results.append(map_distributed(ctl, square, [1, 2]))
    if IS_CONTROLLER:
        assert results == [9, None, [1, 4]]


def test_no_grouping_with_several_processes_means_split():
    # NoGrouping with nprocs_per_worker > 1 is treated as split grouping.
    if not fits(2, True):
        pytest.skip(f"world size {SIZE} does not fit groups of 2")
    worker_config = WorkerGroupConfig(nprocs_per_worker=2, broker_is_worker=True)
    seen = run_two_sessions(worker_config)
    if IS_CONTROLLER:
        assert seen["submit"] == [9, 9]


def test_create_split_environment():
    if not fits(2, True):
        pytest.skip(f"world size {SIZE} does not fit groups of 2")
    results = None
    with create_split_environment(nprocs_per_worker=2, broker_is_worker=True) as env:
        for ctl in env.controller_session():
            results = values_of(map_distributed(ctl, square, [1, 2]))
        assert env.total_cores == env.worker_count * 2
    if IS_CONTROLLER:
        assert results == [[1, 1], [4, 4]]


def test_broadcast_reaches_every_group_member():
    if not fits(2, True):
        pytest.skip(f"world size {SIZE} does not fit groups of 2")
    worker_config = WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSplit,
        nprocs_per_worker=2,
        broker_is_worker=True,
    )
    with DistributedEnvironment(Config(), worker_config) as env:
        results = None
        for ctl in env.controller_session():
            results = values_of(map_distributed(ctl, square, [2, 3]))
        # Brokers and the other group members alike receive the value.
        assert env.broadcast(results) == [[4, 4], [9, 9]]


@pytest.mark.skipif(
    os.environ.get("DISTWQ_TEST_SPAWN") != "1",
    reason="set DISTWQ_TEST_SPAWN=1 to test spawned worker groups",
)
@pytest.mark.parametrize("nprocs", [2, 3])
def test_spawn_layout(nprocs):
    worker_config = WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSpawn,
        nprocs_per_worker=nprocs,
        broker_is_worker=True,
    )
    seen = run_two_sessions(worker_config)
    if IS_CONTROLLER:
        assert seen["worker_count"] == SIZE - 1
        assert seen["submit"] == [9] * nprocs
        assert seen["none"] == [None] * nprocs
