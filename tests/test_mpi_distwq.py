"""
MPI tests for the controller/worker interface, distwq.run().

Run on several ranks, for example:

    mpirun.openmpi -n 3 python -m pytest tests/test_mpi_distwq.py

Every rank calls run() in each test.  The controller rank passes the name
of a function in this module, which run() calls with the MPIController and
whose return value run() returns; worker ranks pass no function, so they
just serve calls until the controller exits.  The split-group test needs
(size - 1) divisible by 2, and the max_workers test needs at least 3 ranks.
"""

import pytest

try:
    from mpi4py import MPI

    SIZE = MPI.COMM_WORLD.Get_size()
except ImportError:
    pytest.skip("MPI not available", allow_module_level=True)

import distwq
import distwq.mpi as distwq_mpi

pytestmark = pytest.mark.skipif(SIZE < 2, reason="Requires at least 2 MPI processes")

MODULE = __name__


def square(x):
    return x * x


def add(a, b=0):
    return a + b


def plain(result):
    """The value or values inside a TaskResult or CollectiveTaskResult."""
    if isinstance(result, distwq.TaskResult):
        return result.value
    assert isinstance(result, distwq.CollectiveTaskResult), result
    return result.values


def run_on_all_ranks(controller_fun, **kwargs):
    """Call run() on every rank, with controller_fun on the controller only."""
    fun_name = controller_fun if distwq.is_controller else None
    return distwq.run(fun_name=fun_name, module_name=MODULE, **kwargs)


# Controller functions, called by run() with the MPIController.


def ctl_submit_and_collect(controller):
    ids = [controller.submit_call("square", (i,), module_name=MODULE) for i in range(8)]
    results = dict(controller.get_next_result() for _ in ids)
    assert all(isinstance(r, distwq.TaskResult) for r in results.values())
    return [plain(results[i]) for i in ids]


def ctl_get_result_in_order(controller):
    ids = [
        controller.submit_call("add", (i,), {"b": 10}, module_name=MODULE)
        for i in range(5)
    ]
    return [plain(controller.get_result(i)[1]) for i in ids]


def ctl_submit_multiple(controller):
    ids = controller.submit_multiple(
        "add", args=[(i, i) for i in range(6)], module_name=MODULE
    )
    collected = {}
    while len(collected) < len(ids):
        for task_id, value in controller.probe_all_next_results():
            collected[task_id] = plain(value)
    return [collected[i] for i in ids]


def ctl_single_task(controller):
    # One call for possibly many workers: most workers are never given a
    # call, and must still be told to exit.
    controller.submit_call("square", (7,), module_name=MODULE)
    return plain(controller.get_next_result()[1])


def ctl_no_tasks(controller):
    return "done"


def ctl_raise_with_calls_outstanding(controller, exc_type):
    # Raise while calls are still running, so that run() has results to
    # discard before it can tell the workers to exit.
    for i in range(6):
        controller.submit_call("square", (i,), module_name=MODULE)
    raise exc_type("controller failed")


def ctl_count_workers(controller):
    ids = [controller.submit_call("square", (i,), module_name=MODULE) for i in range(4)]
    results = [controller.get_next_result()[1] for _ in ids]
    kinds = {type(r).__name__ for r in results}
    return controller.comm.size - 1, sorted(plain(r) for r in results), kinds


# Tests


def test_submit_call_and_get_next_result():
    result = run_on_all_ranks("ctl_submit_and_collect")
    if distwq.is_controller:
        assert result == [i * i for i in range(8)]
    else:
        assert result is None


def test_get_result_with_kwargs():
    result = run_on_all_ranks("ctl_get_result_in_order")
    if distwq.is_controller:
        assert result == [10, 11, 12, 13, 14]


def test_submit_multiple_and_probe_all_next_results():
    result = run_on_all_ranks("ctl_submit_multiple")
    if distwq.is_controller:
        assert result == [0, 2, 4, 6, 8, 10]


def test_fewer_tasks_than_workers_still_exits():
    result = run_on_all_ranks("ctl_single_task")
    if distwq.is_controller:
        assert result == 49


def test_no_tasks_still_exits():
    result = run_on_all_ranks("ctl_no_tasks")
    if distwq.is_controller:
        assert result == "done"


@pytest.mark.parametrize("exc_type", [ValueError, KeyError])
@pytest.mark.parametrize("grouping", ["none", "split"])
def test_controller_exception_releases_workers(exc_type, grouping):
    kwargs = {}
    if grouping == "split":
        if SIZE < 3 or (SIZE - 1) % 2 != 0:
            pytest.skip("split groups of 2 need (size - 1) divisible by 2")
        kwargs = {
            "worker_grouping_method": "split",
            "nprocs_per_worker": 2,
            "broker_is_worker": True,
        }
    if distwq.is_controller:
        with pytest.raises(exc_type, match="controller failed"):
            run_on_all_ranks(
                "ctl_raise_with_calls_outstanding", args=(exc_type,), **kwargs
            )
    else:
        # The workers are told to exit and return from run() as usual.
        assert run_on_all_ranks(None, **kwargs) is None
    # Every rank left run() cleanly, so the next run() works.
    result = run_on_all_ranks("ctl_single_task", **kwargs)
    if distwq.is_controller:
        # A group of 2 returns the result of each of its processes.
        assert result == (49 if grouping == "none" else [49, 49])


@pytest.mark.skipif(
    SIZE < 3 or (SIZE - 1) % 2 != 0,
    reason="split groups of 2 need (size - 1) divisible by 2",
)
def test_split_groups():
    result = run_on_all_ranks(
        "ctl_count_workers",
        worker_grouping_method="split",
        nprocs_per_worker=2,
        broker_is_worker=True,
    )
    if distwq.is_controller:
        n_brokers, values, kinds = result
        assert n_brokers == (SIZE - 1) // 2
        # Each broker returns the results of both processes in its group.
        assert kinds == {"CollectiveTaskResult"}
        assert values == [[i * i, i * i] for i in range(4)]


@pytest.mark.skipif(SIZE < 3, reason="needs ranks left over for max_workers")
def test_max_workers_leaves_other_ranks_idle(monkeypatch):
    # run() changes these module globals when max_workers is given; restore
    # them afterwards so that later tests see every worker again.
    monkeypatch.setattr(distwq_mpi, "n_workers", distwq_mpi.n_workers)
    monkeypatch.setattr(distwq_mpi, "is_worker", distwq_mpi.is_worker)
    result = run_on_all_ranks("ctl_count_workers", max_workers=1)
    if distwq.is_controller:
        assert result == (1, [0, 1, 4, 9], {"TaskResult"})
    else:
        assert result is None
