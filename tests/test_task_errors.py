"""
Single-process tests for reporting task exceptions to the controller.

call_task() and task_failure() run where a worker would run them, and
exception_from_failure() rebuilds the exception on the controller.  A fake
controller then checks that a failed task's future raises the exception,
whether it arrives before or after the future is registered.
"""

import pickle
from types import SimpleNamespace

import pytest

from distwq import future as fut_mod
from distwq.future import (
    RemoteTaskError,
    RemoteTraceback,
    TaskController,
    exception_from_failure,
)
from distwq.mpi import call_task


class NeedsTwoArgs(Exception):
    """Pickles but cannot be unpickled: its __init__ needs two arguments."""

    def __init__(self, a, b):
        super().__init__(f"{a}/{b}")


class Unpicklable(Exception):
    def __reduce__(self):
        raise TypeError("not picklable")


def ok(x):
    return x + 1


def fails(x):
    raise ValueError(f"bad value {x}")


def fails_two_args():
    raise NeedsTwoArgs(1, 2)


def fails_unpicklable():
    raise Unpicklable("nope")


def run_on_worker(name, *args):
    return call_task(name, __name__, args, {}, capture_errors=True)


def test_successful_call_is_not_marked_failed():
    assert run_on_worker("ok", 1) == (2, False)


def test_exception_is_rebuilt_with_remote_traceback():
    failure, failed = run_on_worker("fails", 3)
    assert failed
    # The description survives the trip to the controller.
    failure = pickle.loads(pickle.dumps(failure))
    exc = exception_from_failure(failure)
    assert isinstance(exc, ValueError)
    assert str(exc) == "bad value 3"
    assert isinstance(exc.__cause__, RemoteTraceback)
    assert "in fails" in exc.__cause__.tb


@pytest.mark.parametrize(
    ("name", "exc_class"),
    [("fails_two_args", NeedsTwoArgs), ("fails_unpicklable", Unpicklable)],
)
def test_exception_that_cannot_travel_becomes_remote_task_error(name, exc_class):
    failure, failed = run_on_worker(name)
    assert failed
    exc = exception_from_failure(pickle.loads(pickle.dumps(failure)))
    assert isinstance(exc, RemoteTaskError)
    assert exc.exc_type == f"{__name__}.{exc_class.__qualname__}"
    assert isinstance(exc.__cause__, RemoteTraceback)


def test_lookup_failure_is_captured():
    failure, failed = run_on_worker("no_such_function", 1)
    assert failed
    assert isinstance(exception_from_failure(failure), NameError)


def test_without_capture_the_exception_propagates():
    with pytest.raises(ValueError):
        call_task("fails", __name__, (1,), {}, capture_errors=False)


class FailingController:
    """Delivers each submitted task's call_task() outcome on process()."""

    def __init__(self):
        self.count = 0
        self.waiting = {}
        self.assigned = {}
        self.results = {}
        self.result_queue = []
        self.failed_tasks = set()
        self.ready = []

    def submit_call(self, name_to_call, args=(), kwargs=None, module_name="", **kw):
        task_id = self.count
        self.count += 1
        self.assigned[task_id] = 1
        self.ready.append(
            (task_id, call_task(name_to_call, module_name, args, {}, True))
        )
        return task_id

    def process(self, limit=1000, block=False):
        for task_id, (value, failed) in self.ready:
            self.assigned.pop(task_id)
            self.results[task_id] = value
            self.result_queue.append(task_id)
            if failed:
                self.failed_tasks.add(task_id)
        self.ready = []
        return []

    def probe_all_next_results(self):
        ret = []
        while self.result_queue:
            task_id = self.result_queue.pop(0)
            ret.append((task_id, self.results.pop(task_id)))
        return ret

    def end_session(self):
        pass


@pytest.fixture
def ctl(monkeypatch):
    monkeypatch.setattr(fut_mod, "is_controller", True)
    return TaskController(
        SimpleNamespace(_controller=FailingController(), worker_config=None)
    )


def test_failed_task_raises_from_result_and_others_still_succeed(ctl):
    good, bad, also_good = ctl.submit(ok, 1), ctl.submit(fails, 2), ctl.submit(ok, 3)
    with pytest.raises(ValueError, match="bad value 2") as info:
        bad.result()
    assert isinstance(info.value.__cause__, RemoteTraceback)
    assert isinstance(bad.exception(), ValueError)
    assert good.result() == 2
    assert also_good.result() == 4
    assert ctl._controller.failed_tasks == set()


def test_failure_arriving_before_registration(ctl):
    task_id = ctl._controller.submit_call("fails", (5,), module_name=__name__)
    ctl._pump(block=False)
    future = fut_mod.DistributedFuture(task_id, ctl)
    ctl._setup_future_completion(future)
    with pytest.raises(ValueError, match="bad value 5"):
        future.result()
