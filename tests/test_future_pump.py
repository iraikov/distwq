"""
Single-process tests for the synchronous result delivery in distwq.future.

A fake controller stands in for MPIController, so these tests need neither
MPI nor workers.  The fake keeps the same queues that TaskController reads
(waiting, assigned, results, result_queue) and lets each test decide which
results arrive on each call to process().
"""

import threading
from collections import deque
from types import SimpleNamespace

import pytest

from distwq import future as fut_mod
from distwq.future import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    DistributedFuture,
    TaskController,
    as_completed,
    wait,
)


def square(x):
    return x * x


class FakeController:
    """Minimal MPIController stand-in driven by a script of arrivals.

    Each call to process() delivers the next batch from `arrivals`, a list
    of lists of task ids.  An empty batch means nothing arrived.  The value
    of a finished task is looked up in `values`.
    """

    def __init__(self, values=None):
        self.count = 0
        self.waiting = {}
        self.assigned = {}
        self.results = {}
        self.result_queue = []
        self.arrivals = deque()
        self.values = values or {}
        self.process_calls = []
        self.end_session_calls = 0

    def submit_call(self, name_to_call, args=(), kwargs=None, module_name="", **kw):
        task_id = self.count
        self.count += 1
        self.assigned[task_id] = 1
        self.values.setdefault(task_id, ("value", task_id))
        return task_id

    def process(self, limit=1000, block=False):
        self.process_calls.append(block)
        batch = self.arrivals.popleft() if self.arrivals else []
        if block and not batch:
            raise AssertionError("blocking process() called with nothing to deliver")
        for task_id in batch:
            self.assigned.pop(task_id, None)
            self.results[task_id] = self.values[task_id]
            self.result_queue.append(task_id)
        return []

    def probe_all_next_results(self):
        ret = []
        while self.result_queue:
            task_id = self.result_queue.pop(0)
            ret.append((task_id, self.results.pop(task_id)))
        return ret

    def end_session(self):
        self.end_session_calls += 1


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.setattr(fut_mod, "is_controller", True)
    fake = FakeController()
    env = SimpleNamespace(_controller=fake, worker_config=None)
    return TaskController(env), fake


def test_results_routed_by_task_id_out_of_order(controller):
    ctl, fake = controller
    futures = [ctl.submit(square, i) for i in range(3)]
    fake.arrivals.extend([[2], [0], [1]])

    assert futures[1].result() == ("value", 1)
    assert futures[0].result() == ("value", 0)
    assert futures[2].result() == ("value", 2)


def test_result_arriving_before_registration_is_kept(controller):
    ctl, fake = controller
    # Deliver task 0 before any future exists for it.
    task_id = fake.submit_call("square")
    fake.arrivals.append([task_id])
    ctl._pump(block=False)
    assert ctl._orphans == {task_id: ("value", task_id)}

    f = DistributedFuture(task_id, ctl)
    ctl._setup_future_completion(f)
    assert f._done
    assert f.result() == ("value", task_id)
    assert ctl._orphans == {}


def test_as_completed_yields_in_arrival_order(controller):
    ctl, fake = controller
    futures = [ctl.submit(square, i) for i in range(4)]
    fake.arrivals.extend([[3, 1], [0], [2]])

    order = [f.task_id for f in as_completed(futures)]
    assert order == [3, 1, 0, 2]


def test_as_completed_rejects_futures_from_two_controllers(controller, monkeypatch):
    ctl, _ = controller
    other_env = SimpleNamespace(_controller=FakeController(), worker_config=None)
    other = TaskController(other_env)
    with pytest.raises(ValueError):
        list(as_completed([ctl.submit(square, 1), other.submit(square, 2)]))


def test_wait_first_completed_returns_after_first_result(controller):
    ctl, fake = controller
    futures = [ctl.submit(square, i) for i in range(3)]
    fake.arrivals.append([1])

    done, not_done = wait(futures, return_when=FIRST_COMPLETED)
    assert {f.task_id for f in done} == {1}
    assert {f.task_id for f in not_done} == {0, 2}


def test_wait_all_completed_with_timeout_reports_unfinished(controller):
    ctl, fake = controller
    futures = [ctl.submit(square, i) for i in range(2)]
    fake.arrivals.append([0])
    ctl.poll_interval = 0.001

    done, not_done = wait(futures, timeout=0.05, return_when=ALL_COMPLETED)
    assert {f.task_id for f in done} == {0}
    assert {f.task_id for f in not_done} == {1}


def test_result_timeout_raises(controller):
    ctl, fake = controller
    f = ctl.submit(square, 1)
    ctl.poll_interval = 0.001
    with pytest.raises(TimeoutError):
        f.result(timeout=0.05)
    # Only non-blocking probes are used when a timeout is given.
    assert not any(fake.process_calls)


def test_done_pumps_without_blocking(controller):
    ctl, fake = controller
    f = ctl.submit(square, 1)
    assert not f.done()
    fake.arrivals.append([f.task_id])
    assert f.done()
    assert fake.process_calls == [False, False]


def test_callbacks_run_once_in_order_on_calling_thread(controller):
    ctl, fake = controller
    f = ctl.submit(square, 1)
    calls = []
    f.add_done_callback(lambda x: calls.append(("a", threading.get_ident())))
    f.add_done_callback(lambda x: calls.append(("b", threading.get_ident())))
    fake.arrivals.append([f.task_id])

    f.result()
    f.result()
    me = threading.get_ident()
    assert calls == [("a", me), ("b", me)]

    # A callback added after completion runs immediately.
    f.add_done_callback(lambda x: calls.append(("c", threading.get_ident())))
    assert calls[-1] == ("c", me)


def test_close_resolves_pending_and_is_idempotent(controller):
    ctl, fake = controller
    futures = [ctl.submit(square, i) for i in range(3)]
    fake.arrivals.extend([[0, 2], [1]])

    ctl._close()
    assert all(f._done for f in futures)
    assert ctl._pending == {}
    ctl._close()


def test_close_ends_session_once_after_results(controller):
    ctl, fake = controller
    f = ctl.submit(square, 1)
    fake.arrivals.append([f.task_id])

    ctl._close()
    assert f._done
    assert fake.end_session_calls == 1
    ctl._close()
    assert fake.end_session_calls == 1


def test_close_fails_futures_unknown_to_controller(controller):
    ctl, fake = controller
    f = ctl.submit(square, 1)
    fake.assigned.clear()

    ctl._close()
    with pytest.raises(RuntimeError):
        f.result()


def test_result_raises_if_task_unknown(controller):
    ctl, fake = controller
    f = ctl.submit(square, 1)
    fake.assigned.clear()
    with pytest.raises(RuntimeError):
        f.result()


def test_no_thread_started(controller):
    before = threading.active_count()
    ctl, fake = controller
    futures = [ctl.submit(square, i) for i in range(3)]
    fake.arrivals.append([0, 1, 2])
    for f in futures:
        f.result()
    assert threading.active_count() == before


def test_map_returns_results_in_submission_order(controller):
    ctl, fake = controller
    fake.arrivals.extend([[2], [1], [0]])
    assert list(ctl.map(square, [10, 11, 12])) == [
        ("value", 0),
        ("value", 1),
        ("value", 2),
    ]
