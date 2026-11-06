"""
MPI tests for the init functions of DistributedEnvironment.

Run on several ranks, for example:

    mpirun.openmpi -n 5 python -m pytest tests/test_init_hooks_mpi.py

The init functions record each call in INIT_CALLS, a list in this
module: which hook ran, the type of server it was given, the world rank
and the arguments.  A task returns the records of the process that ran
it, so the controller can check the workers.  Each rank checks its own
records after leaving the environment, so that a wrong record cannot
leave the other ranks waiting.

The split-group tests need (size - 1) divisible by 2, or by 3 when the
broker does not work.  The spawn tests run only with DISTWQ_TEST_SPAWN=1,
with tests/ on PYTHONPATH so that the spawned processes can import this
module.
"""

import os

import pytest

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except ImportError:
    pytest.skip("MPI not available", allow_module_level=True)

import distwq.mpi as distwq_mpi
from distwq.config import Config, ResourceConfig
from distwq.future import DistributedEnvironment, WorkerGroupConfig
from distwq.mpi import CollectiveTaskResult, GroupingMethod

pytestmark = pytest.mark.skipif(SIZE < 2, reason="Requires at least 2 MPI processes")

IS_CONTROLLER = distwq_mpi.is_controller
WORKER_RANKS = [r for r in range(SIZE) if r != distwq_mpi.controller_rank]
ARGS = ("model", 3)

INIT_CALLS = []


def record_worker(server, *args):
    INIT_CALLS.append(("worker", type(server).__name__, RANK, args))


def record_broker(server, *args):
    INIT_CALLS.append(("broker", type(server).__name__, RANK, args))


def init_records():
    """Task: the init records of the process that runs it."""
    return list(INIT_CALLS)


def rank_and_records():
    return RANK, list(INIT_CALLS)


@pytest.fixture(autouse=True)
def clear_records():
    INIT_CALLS.clear()
    yield
    INIT_CALLS.clear()


def test_worker_init_runs_once_on_each_worker():
    seen = {}
    with DistributedEnvironment(
        Config(), worker_init=record_worker, init_args=ARGS
    ) as env:
        server_type = type(env.server).__name__ if env.server is not None else None
        for controller in env.controller_session():
            for f in [controller.submit(rank_and_records) for _ in range(3 * SIZE)]:
                rank, records = f.result()
                seen[rank] = records
    if IS_CONTROLLER:
        assert server_type is None
        assert INIT_CALLS == []
        for rank, records in seen.items():
            assert records == [("worker", "MPIWorker", rank, ARGS)]
    else:
        assert server_type == "MPIWorker"
        assert INIT_CALLS == [("worker", "MPIWorker", RANK, ARGS)]


@pytest.mark.skipif(SIZE < 4, reason="needs ranks left over by max_workers=2")
def test_idle_ranks_do_not_run_worker_init():
    kept = WORKER_RANKS[:2]
    with DistributedEnvironment(
        Config(resources=ResourceConfig(max_workers=2)), worker_init=record_worker
    ) as env:
        role = env.role
        idle_server = env.server
        for controller in env.controller_session():
            controller.submit(init_records).result()
    if RANK in kept:
        assert INIT_CALLS == [("worker", "MPIWorker", RANK, ())]
    else:
        assert INIT_CALLS == []
        if not IS_CONTROLLER:
            assert role == "idle"
            assert idle_server is None


def test_worker_init_runs_again_in_each_environment():
    for _ in range(2):
        with DistributedEnvironment(
            Config(), worker_init=record_worker, init_args=ARGS
        ) as env:
            for controller in env.controller_session():
                controller.submit(init_records).result()
    if IS_CONTROLLER:
        assert INIT_CALLS == []
    else:
        assert INIT_CALLS == [("worker", "MPIWorker", RANK, ARGS)] * 2


@pytest.mark.skipif(
    SIZE < 3 or (SIZE - 1) % 2 != 0,
    reason="split groups of 2 need (size - 1) divisible by 2",
)
def test_split_groups_with_working_broker():
    worker_config = WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSplit,
        nprocs_per_worker=2,
        broker_is_worker=True,
    )
    result = None
    with DistributedEnvironment(
        Config(), worker_config, worker_init=record_worker, init_args=ARGS
    ) as env:
        role = env.role
        server_type = type(env.server).__name__
        for controller in env.controller_session():
            result = controller.submit(init_records).result()
    if IS_CONTROLLER:
        assert INIT_CALLS == []
        assert isinstance(result, CollectiveTaskResult)
        broker_records, worker_records = result.values
        assert [r[:2] for r in broker_records] == [("worker", "MPICollectiveBroker")]
        assert [r[:2] for r in worker_records] == [("worker", "MPICollectiveWorker")]
    elif role == "broker":
        assert server_type == "MPICollectiveBroker"
        assert INIT_CALLS == [("worker", "MPICollectiveBroker", RANK, ARGS)]
    else:
        assert role == "collective_worker"
        assert server_type == "MPICollectiveWorker"
        assert INIT_CALLS == [("worker", "MPICollectiveWorker", RANK, ARGS)]


@pytest.mark.skipif(
    SIZE < 4 or (SIZE - 1) % 3 != 0,
    reason="split groups of 2 plus a broker need (size - 1) divisible by 3",
)
def test_split_groups_with_relaying_broker():
    worker_config = WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSplit,
        nprocs_per_worker=2,
        broker_is_worker=False,
    )
    result = None
    with DistributedEnvironment(
        Config(),
        worker_config,
        worker_init=record_worker,
        broker_init=record_broker,
        init_args=ARGS,
    ) as env:
        role = env.role
        for controller in env.controller_session():
            result = controller.submit(init_records).result()
    if IS_CONTROLLER:
        assert INIT_CALLS == []
        # Only the two workers of the group compute.
        assert [[r[:2] for r in v] for v in result.values] == [
            [("worker", "MPICollectiveWorker")]
        ] * 2
    elif role == "broker":
        assert INIT_CALLS == [("broker", "MPICollectiveBroker", RANK, ARGS)]
    else:
        assert INIT_CALLS == [("worker", "MPICollectiveWorker", RANK, ARGS)]


spawn_only = pytest.mark.skipif(
    os.environ.get("DISTWQ_TEST_SPAWN") != "1",
    reason="set DISTWQ_TEST_SPAWN=1 to test spawned worker groups",
)


@spawn_only
def test_spawned_workers_run_worker_init():
    worker_config = WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSpawn,
        nprocs_per_worker=3,
        broker_is_worker=True,
    )
    result = None
    with DistributedEnvironment(
        Config(), worker_config, worker_init=record_worker, init_args=ARGS
    ) as env:
        for controller in env.controller_session():
            result = controller.submit(init_records).result()
    if IS_CONTROLLER:
        assert INIT_CALLS == []
        broker_records, *spawned_records = result.values
        assert [r[:2] for r in broker_records] == [("worker", "MPICollectiveBroker")]
        assert len(spawned_records) == 2
        for records in spawned_records:
            assert [(r[0], r[1], r[3]) for r in records] == [
                ("worker", "MPICollectiveWorker", ARGS)
            ]
    else:
        assert INIT_CALLS == [("worker", "MPICollectiveBroker", RANK, ARGS)]


@spawn_only
def test_spawned_group_with_relaying_broker():
    worker_config = WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSpawn,
        nprocs_per_worker=2,
        broker_is_worker=False,
    )
    result = None
    with DistributedEnvironment(
        Config(),
        worker_config,
        worker_init=record_worker,
        broker_init=record_broker,
        init_args=ARGS,
    ) as env:
        for controller in env.controller_session():
            result = controller.submit(init_records).result()
    if IS_CONTROLLER:
        assert INIT_CALLS == []
        assert [[(r[0], r[1], r[3]) for r in v] for v in result.values] == [
            [("worker", "MPICollectiveWorker", ARGS)]
        ] * 2
    else:
        assert INIT_CALLS == [("broker", "MPICollectiveBroker", RANK, ARGS)]
