"""
Single-process tests for the result types and for task errors on the
controller when no workers are available.

Every result that MPIController stores is a TaskResult, a
CollectiveTaskResult, or, for a failed call, the task_failure()
description of its exception.  Without workers the controller runs each
call itself; with capture_errors=True it stores a failed call as a
failure instead of raising from submit_call().  The futures layer turns a
TaskResult into its plain value, or into a CollectiveTaskResult under
worker grouping, and passes a CollectiveTaskResult through.

The controller tests use MPIController(None), which has no workers on any
world size.  The DistributedEnvironment tests need a world of one rank and
skip themselves otherwise.
"""

import dataclasses
import pickle
import signal
from types import SimpleNamespace

import pytest

import distwq
from distwq.future import (
    Config,
    DistributedEnvironment,
    RemoteTraceback,
    TaskController,
    WorkerGroupConfig,
    requests_grouping,
)
from distwq.mpi import (
    CollectiveTaskResult,
    GroupingMethod,
    MPIController,
    TaskResult,
    _group_outcome,
    _split_group_data,
    task_failure,
    world_comm,
)

MODULE = __name__

single_rank = pytest.mark.skipif(
    world_comm.size > 1,
    reason="needs a world of one rank, so that there are no workers",
)


def square(x):
    return x * x


def nothing():
    return None


def fails(x):
    raise ValueError(f"bad value {x}")


def fails_on_two(x):
    if x == 2:
        raise ValueError("two")
    return x


def looks_like_failure():
    return {"distwq_failure": True, "message": "only a value"}


def stat(this_time, error=False):
    return {"id": 0, "rank": 0, "this_time": this_time, "error": error}


def failure_of(exc):
    try:
        raise exc
    except type(exc) as caught:
        return task_failure(caught)


# The result types


def test_result_types_survive_pickling():
    for result in (TaskResult([1, 2]), CollectiveTaskResult([{"a": 1}, None])):
        assert pickle.loads(pickle.dumps(result)) == result


def test_result_types_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        TaskResult(1).value = 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        CollectiveTaskResult([1]).values = [2]


def test_result_types_compare_by_type_and_content():
    assert TaskResult(None) != None
    assert TaskResult([1]) != CollectiveTaskResult([1])
    assert CollectiveTaskResult([1, 2]) == CollectiveTaskResult([1, 2])


def test_result_types_are_exported():
    assert distwq.TaskResult is TaskResult
    assert distwq.CollectiveTaskResult is CollectiveTaskResult


# The controller without workers


def test_controller_without_communicator_has_no_workers():
    controller = MPIController(None)
    assert not controller.workers_available
    assert controller.process(block=True) is None


def test_submit_call_stores_task_result():
    controller = MPIController(None)
    task_id = controller.submit_call("square", (3,), module_name=MODULE)
    assert controller.get_result(task_id) == (task_id, TaskResult(9))
    assert controller.results == {}
    assert controller.result_queue == []


def test_none_result_is_wrapped():
    controller = MPIController(None)
    controller.submit_call("nothing", module_name=MODULE)
    assert controller.get_next_result()[1] == TaskResult(None)


def test_queue_call_runs_at_once_and_stores_task_result():
    controller = MPIController(None)
    task_id = controller.queue_call("square", (4,), module_name=MODULE)
    assert controller.probe_all_next_results() == [(task_id, TaskResult(16))]


def test_submit_multiple_stores_task_results():
    controller = MPIController(None)
    ids = controller.submit_multiple(
        "square", args=[(i,) for i in range(4)], module_name=MODULE
    )
    assert dict(controller.probe_all_next_results()) == {
        task_id: TaskResult(i * i) for i, task_id in enumerate(ids)
    }


def test_submit_multiple_ignores_requested_workers():
    # There are no workers to request; only this process's estimate grows.
    controller = MPIController(None)
    controller.total_time_est[0] = 0.0
    controller.submit_multiple(
        "square", args=[(1,), (2,)], module_name=MODULE, workers=[3, 4], time_est=2
    )
    assert list(controller.total_time_est) == [4.0]


def test_submit_multiple_without_requested_workers_updates_one_estimate():
    controller = MPIController(None)
    controller.total_time_est[0] = 0.0
    controller.submit_multiple("square", args=[(1,), (2,), (3,)], module_name=MODULE)
    assert list(controller.total_time_est) == [3.0]


def test_local_calls_record_stats():
    controller = MPIController(None, capture_errors=True)
    controller.submit_call("square", (2,), module_name=MODULE)
    controller.submit_call("fails", (2,), module_name=MODULE)
    assert [s["error"] for s in controller.stats] == [False, True]
    assert [s["n_processed"] for s in controller.stats] == [1, 2]


def test_exception_propagates_without_capture_errors():
    controller = MPIController(None)
    with pytest.raises(ValueError, match="bad value 1"):
        controller.submit_call("fails", (1,), module_name=MODULE)
    # Nothing is stored for the failed call.
    assert controller.results == {}
    assert controller.result_queue == []
    assert controller.failed_tasks == set()


def test_captured_exception_is_stored_as_failure():
    controller = MPIController(None, capture_errors=True)
    task_id = controller.submit_call("fails", (1,), module_name=MODULE)
    assert task_id in controller.failed_tasks
    _, failure = controller.get_result(task_id)
    assert failure["distwq_failure"]
    assert failure["type"] == "builtins.ValueError"
    assert failure["message"] == "bad value 1"


def test_unknown_task_name_is_captured():
    controller = MPIController(None, capture_errors=True)
    task_id = controller.submit_call("no_such_function", module_name=MODULE)
    assert task_id in controller.failed_tasks
    assert controller.get_result(task_id)[1]["type"] == "builtins.NameError"


def test_queue_call_captures_exception():
    controller = MPIController(None, capture_errors=True)
    task_id = controller.queue_call("fails", (5,), module_name=MODULE)
    assert controller.failed_tasks == {task_id}


def test_one_failure_in_submit_multiple_leaves_the_others():
    controller = MPIController(None, capture_errors=True)
    ids = controller.submit_multiple(
        "fails_on_two", args=[(i,) for i in range(4)], module_name=MODULE
    )
    assert len(ids) == 4
    assert controller.failed_tasks == {ids[2]}
    results = dict(controller.probe_all_next_results())
    assert [results[ids[i]] for i in (0, 1, 3)] == [
        TaskResult(0),
        TaskResult(1),
        TaskResult(3),
    ]


def test_returned_failure_lookalike_is_a_value():
    controller = MPIController(None, capture_errors=True)
    task_id = controller.submit_call("looks_like_failure", module_name=MODULE)
    assert controller.failed_tasks == set()
    assert controller.get_result(task_id)[1] == TaskResult(looks_like_failure())


# How a broker assembles a group's result


def test_group_data_keeps_none_results():
    sub_data = [(None, stat(0.1)), (None, stat(0.2))]
    assert _split_group_data(sub_data) == ([None, None], [stat(0.1), stat(0.2)])


def test_group_data_drops_the_non_working_broker():
    # A broker that does not work contributes (None, None) to the gather.
    sub_data = [(None, None), (4, stat(0.1)), (None, stat(0.2))]
    assert _split_group_data(sub_data) == ([4, None], [stat(0.1), stat(0.2)])


def test_group_outcome_wraps_values_and_reports_slowest_stat():
    payload, slowest = _group_outcome([1, None, 3], [stat(0.1), stat(0.5), stat(0.2)])
    assert payload == CollectiveTaskResult([1, None, 3])
    assert slowest == stat(0.5)


def test_group_outcome_picks_failure_by_its_stat():
    failure = failure_of(KeyError("member"))
    payload, flagged = _group_outcome(
        [looks_like_failure(), failure], [stat(0.3), stat(0.1, error=True)]
    )
    # The returned dict that only looks like a failure is not chosen.
    assert payload is failure
    assert flagged == dict(stat(0.3), error=True)


def test_group_outcome_reports_first_failure():
    first = failure_of(ValueError("first"))
    second = failure_of(ValueError("second"))
    payload, _ = _group_outcome(
        [5, first, second], [stat(0.1), stat(0.1, error=True), stat(0.1, error=True)]
    )
    assert payload is first


def test_group_outcome_with_failure_lookalike_only_is_a_value():
    payload, flagged = _group_outcome([looks_like_failure()], [stat(0.1)])
    assert payload == CollectiveTaskResult([looks_like_failure()])
    assert not flagged["error"]


# The futures layer

# A TaskController can only be made on the controller rank.
controller_only = pytest.mark.skipif(
    not distwq.is_controller, reason="TaskController runs on the controller only"
)


class FakeController:
    """Holds results put in by the test, as the controller would."""

    def __init__(self):
        self.results = {}
        self.result_queue = []
        self.failed_tasks = set()
        self.waiting = {}
        self.assigned = {}
        self.count = 0

    def submit_call(self, name, args=(), kwargs=None, module_name="", time_est=1):
        task_id = self.count
        self.count += 1
        self.assigned[task_id] = 1
        return task_id

    def finish(self, task_id, value):
        self.assigned.pop(task_id, None)
        self.results[task_id] = value
        self.result_queue.append(task_id)

    def process(self, block=False):
        return []

    def probe_all_next_results(self):
        ready = [(t, self.results.pop(t)) for t in self.result_queue]
        self.result_queue.clear()
        return ready


def make_task_controller(worker_config=None):
    fake = FakeController()
    env = SimpleNamespace(_controller=fake, worker_config=worker_config)
    return TaskController(env), fake


@controller_only
@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (TaskResult(7), 7),
        (TaskResult(None), None),
        (CollectiveTaskResult([1, 2]), CollectiveTaskResult([1, 2])),
        # A plain value, as from an older controller, is passed as it is.
        ([1, 2], [1, 2]),
    ],
)
def test_future_unwraps_task_result(result, expected):
    ctl, fake = make_task_controller()
    future = ctl.submit(square, 1)
    fake.finish(future.task_id, result)
    assert future.result() == expected


@controller_only
def test_orphan_result_is_unwrapped_when_its_future_registers():
    ctl, fake = make_task_controller()
    fake.finish(0, TaskResult(3))
    # The result arrives before any future for task 0 exists.
    ctl._pump(block=False)
    assert ctl._orphans
    future = ctl.submit(square, 1)
    assert future.task_id == 0
    assert future.result() == 3


def group_config(grouping_method=GroupingMethod.NoGrouping, nprocs_per_worker=1):
    """Stands in for a WorkerGroupConfig, which checks the layout against
    the world size when it is made."""
    return SimpleNamespace(
        grouping_method=grouping_method, nprocs_per_worker=nprocs_per_worker
    )


@controller_only
def test_grouped_controller_wraps_task_result_as_collective():
    ctl, fake = make_task_controller(group_config(GroupingMethod.GroupSplit, 2))
    future = ctl.submit(square, 1)
    fake.finish(future.task_id, TaskResult(5))
    assert future.result() == CollectiveTaskResult([5])


@pytest.mark.parametrize(
    ("worker_config", "grouped"),
    [
        (None, False),
        (group_config(), False),
        (group_config(nprocs_per_worker=2), True),
        (group_config(GroupingMethod.GroupSplit), True),
        (group_config("spawn", 3), True),
    ],
)
def test_requests_grouping(worker_config, grouped):
    assert requests_grouping(worker_config) is grouped


# A DistributedEnvironment on one rank


@single_rank
def test_task_error_raises_from_result_not_submit():
    with DistributedEnvironment(Config()) as env:
        for ctl in env.controller_session():
            future = ctl.submit(fails, 4)
            with pytest.raises(ValueError, match="bad value 4") as info:
                future.result()
            assert isinstance(info.value.__cause__, RemoteTraceback)
            assert "in fails" in info.value.__cause__.tb
            assert isinstance(future.exception(), ValueError)
            # The session carries on after the failure.
            assert ctl.submit(square, 5).result() == 25
            assert list(ctl.map(square, [1, 2, 3])) == [1, 4, 9]


@single_rank
def test_task_error_inside_map_raises_from_iteration():
    with DistributedEnvironment(Config()) as env:
        for ctl in env.controller_session():
            results = ctl.map(fails_on_two, range(4))
            assert next(results) == 0
            assert next(results) == 1
            with pytest.raises(ValueError, match="two"):
                next(results)


@single_rank
def test_single_rank_returns_plain_values():
    with DistributedEnvironment(Config()) as env:
        for ctl in env.controller_session():
            assert ctl.submit(nothing).result() is None
            assert ctl.submit(looks_like_failure).result() == looks_like_failure()


@single_rank
@pytest.mark.parametrize(
    "config_args",
    [
        {"grouping_method": GroupingMethod.GroupSplit, "nprocs_per_worker": 2},
        {"nprocs_per_worker": 2, "broker_is_worker": True},
    ],
)
def test_grouped_config_on_single_rank_returns_collective_result(config_args):
    with DistributedEnvironment(Config(), WorkerGroupConfig(**config_args)) as env:
        for ctl in env.controller_session():
            assert ctl.submit(square, 3).result() == CollectiveTaskResult([9])
            with pytest.raises(ValueError, match="bad value 1"):
                ctl.submit(fails, 1).result()


@single_rank
def test_run_without_workers_returns_task_results():
    assert distwq.run(fun_name="_submit_square", module_name=MODULE) == TaskResult(36)


@single_rank
@pytest.mark.parametrize("exc_type", [ValueError, KeyError, SystemExit])
def test_run_without_workers_still_raises_task_errors(exc_type):
    # run() does not capture errors: the task's exception leaves
    # submit_call() and run(), whatever its type.
    with pytest.raises(exc_type, match="missing"):
        distwq.run(fun_name="_submit_failing", module_name=MODULE, args=(exc_type,))


@single_rank
def test_run_restores_sigint_handler_after_an_exception():
    def handler(signum, frame):
        pass

    previous = signal.signal(signal.SIGINT, handler)
    try:
        with pytest.raises(ValueError, match="controller failed"):
            distwq.run(fun_name="_raise_in_controller", module_name=MODULE)
        assert signal.getsignal(signal.SIGINT) is handler
    finally:
        signal.signal(signal.SIGINT, previous)


def fails_with(exc_type):
    raise exc_type("missing")


def _submit_failing(controller, exc_type):
    controller.submit_call("fails_with", (exc_type,), module_name=MODULE)


def _raise_in_controller(controller):
    raise ValueError("controller failed")


def _submit_square(controller):
    controller.submit_call("square", (6,), module_name=MODULE)
    return controller.get_next_result()[1]
