"""
MPI tests for the session protocol of DistributedEnvironment.

Run on several ranks, for example:

    mpirun.openmpi -n 3 python -m pytest tests/test_session_protocol.py

Every rank runs every test in the same order, as SIMD code would.  On worker
ranks, controller_session() serves calls until the controller ends the
session, and leaving the environment serves until the controller sends
EXIT.  The tests check that each of these hand-overs completes, so a
failure usually shows as a hang (run under a timeout) rather than an
assertion.

The split-group test needs a world size with (size - 1) divisible by 2,
for example 5.  The spawn-group test runs only when DISTWQ_TEST_SPAWN=1.
"""

import os
import threading

import pytest

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except ImportError:
    pytest.skip("MPI not available", allow_module_level=True)

from distwq.future import (
    Config,
    ControllerExitedError,
    DistributedEnvironment,
    WorkerGroupConfig,
)
from distwq.mpi import GroupingMethod

pytestmark = pytest.mark.skipif(SIZE < 2, reason="Requires at least 2 MPI processes")

IS_CONTROLLER = RANK == 0


def add(a, b):
    return a + b


def scale(x, factor=2):
    return x * factor


def test_environment_without_session():
    with DistributedEnvironment(Config()) as env:
        assert env.is_active()
    assert not env.is_active()


def test_session_without_tasks():
    yielded = 0
    with DistributedEnvironment(Config()) as env:
        for _controller in env.controller_session():
            yielded += 1
    assert yielded == (1 if IS_CONTROLLER else 0)


def test_two_sessions_in_one_environment():
    results = []
    sessions = 0
    with DistributedEnvironment(Config()) as env:
        for controller in env.controller_session():
            futures = [controller.submit(add, i, 1) for i in range(5)]
            results.append([f.result() for f in futures])
        sessions += 1
        for controller in env.controller_session():
            results.append(list(controller.map(scale, range(4))))
        sessions += 1
    assert sessions == 2
    if IS_CONTROLLER:
        assert results == [[1, 2, 3, 4, 5], [0, 2, 4, 6]]
    else:
        assert results == []


def test_pending_futures_are_delivered_when_session_ends():
    futures = []
    with DistributedEnvironment(Config()) as env:
        for controller in env.controller_session():
            futures = [controller.submit(add, i, i) for i in range(6)]
            # Leave the session without waiting for any result.
    if IS_CONTROLLER:
        assert all(f._done for f in futures)
        assert [f.result() for f in futures] == [0, 2, 4, 6, 8, 10]


def test_exception_inside_session():
    if IS_CONTROLLER:
        with (
            pytest.raises(ValueError, match="inside session"),
            DistributedEnvironment(Config()) as env,
        ):
            for controller in env.controller_session():
                controller.submit(add, 1, 2)
                raise ValueError("inside session")
    else:
        with DistributedEnvironment(Config()) as env:
            for _controller in env.controller_session():
                pass


def test_exception_before_session():
    # The workers are serving inside controller_session() when the
    # controller, which never opened a session, tells them to exit.
    if IS_CONTROLLER:
        with (
            pytest.raises(ValueError, match="before session"),
            DistributedEnvironment(Config()),
        ):
            raise ValueError("before session")
    else:
        with DistributedEnvironment(Config()) as env:
            for _controller in env.controller_session():
                pass


def test_broadcast_reaches_every_rank():
    with DistributedEnvironment(Config()) as env:
        total = None
        for controller in env.controller_session():
            total = sum(controller.map(scale, range(4)))
        total = env.broadcast(total)
        assert total == 12
        # A second broadcast in the same environment, with no session.
        assert env.broadcast({"rank0": True} if IS_CONTROLLER else None) == {
            "rank0": True
        }


def test_controller_error_before_broadcast_releases_workers():
    # The workers wait in broadcast() while the controller, which raised,
    # leaves the environment; they must be released, not left waiting.
    if IS_CONTROLLER:
        with (
            pytest.raises(ValueError, match="before broadcast"),
            DistributedEnvironment(Config()) as env,
        ):
            for controller in env.controller_session():
                controller.submit(add, 1, 2).result()
                raise ValueError("before broadcast")
            env.broadcast("never sent")
    else:
        with (
            pytest.raises(ControllerExitedError),
            DistributedEnvironment(Config()) as env,
        ):
            for _controller in env.controller_session():
                pass
            env.broadcast()


def test_environments_back_to_back():
    totals = []
    for _ in range(3):
        with DistributedEnvironment(Config()) as env:
            for controller in env.controller_session():
                totals.append(sum(controller.map(scale, range(3))))
    if IS_CONTROLLER:
        assert totals == [6, 6, 6]


def test_nested_environment_is_rejected():
    with DistributedEnvironment(Config()) as env:
        # Rejected on every rank before any MPI call, so the outer
        # environment is unaffected.
        with (
            pytest.raises(RuntimeError, match="cannot be nested"),
            DistributedEnvironment(Config()),
        ):
            pass  # pragma: no cover
        for controller in env.controller_session():
            assert controller.submit(add, 1, 1).result() == 2


@pytest.mark.skipif(
    (SIZE - 1) % 2 == 0, reason="needs a world size that cannot be split in pairs"
)
def test_invalid_split_config_fails_on_every_rank():
    # Rejected either when the configuration is created or when the
    # environment is entered; in both cases before any collective call.
    with pytest.raises(RuntimeError, match="unable to split workers evenly"):
        worker_config = WorkerGroupConfig(
            grouping_method=GroupingMethod.GroupSplit,
            nprocs_per_worker=2,
            broker_is_worker=True,
        )
        with DistributedEnvironment(Config(), worker_config):
            pass  # pragma: no cover
    # The failed environment left nothing behind: a new one works.
    with DistributedEnvironment(Config()) as env:
        for controller in env.controller_session():
            assert controller.submit(add, 2, 2).result() == 4


def test_no_thread_is_started():
    before = threading.active_count()
    with DistributedEnvironment(Config()) as env:
        assert threading.active_count() == before
        for controller in env.controller_session():
            controller.submit(add, 2, 3).result()
            assert threading.active_count() == before
    assert threading.active_count() == before


@pytest.mark.skipif(
    SIZE < 3 or (SIZE - 1) % 2 != 0,
    reason="split groups of 2 need (size - 1) divisible by 2",
)
def test_split_groups_two_sessions():
    worker_config = WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSplit,
        nprocs_per_worker=2,
        broker_is_worker=True,
    )
    results = []
    with DistributedEnvironment(Config(), worker_config) as env:
        for controller in env.controller_session():
            assert env.worker_count == (SIZE - 1) // 2
            results.append(controller.submit(add, 3, 4).result())
        for controller in env.controller_session():
            results.append(controller.submit(scale, 5).result())
    if IS_CONTROLLER:
        # A broker returns the list of results from every process in its
        # group.
        assert results == [[7, 7], [10, 10]]


@pytest.mark.skipif(
    os.environ.get("DISTWQ_TEST_SPAWN") != "1",
    reason="set DISTWQ_TEST_SPAWN=1 to test spawned worker groups",
)
def test_spawn_groups_two_sessions():
    worker_config = WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSpawn,
        nprocs_per_worker=2,
        broker_is_worker=True,
    )
    results = []
    with DistributedEnvironment(Config(), worker_config) as env:
        for controller in env.controller_session():
            results.append(controller.submit(add, 3, 4).result())
        for controller in env.controller_session():
            results.append(controller.submit(scale, 5).result())
    if IS_CONTROLLER:
        assert results == [[7, 7], [10, 10]]
