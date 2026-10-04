"""
Single-process tests for distwq.functional and distwq.workflow.

A fake controller stands in for MPIController.  It runs each submitted
call itself, looking the callable up by module and name as a worker does,
and delivers the results in an order the test chooses: "fifo" in
submission order, or "reverse" with the last submitted first.  Reverse
delivery is what separates ordered from arrival-order behaviour.
"""

import functools
import operator
import sys
from types import SimpleNamespace

import pytest

from distwq import future as fut_mod
from distwq.functional import (
    FunctionalPipeline,
    adaptive_map,
    conditional_map,
    fold_distributed,
    imap_distributed,
    map_distributed,
    map_reduce,
    parallel_map,
    pipeline,
    reduce_distributed,
    starmap_distributed,
    zip_distributed,
)
from distwq.future import TaskController, task_reference
from distwq.workflow import adaptive_parameter_exploration, parameter_sweep


def square(x):
    return x * x


def negate(x):
    return -x


def inc(x):
    return x + 1


def add(a, b):
    return a + b


def twice(x):
    return [x, x]


def as_text(x):
    return str(x)


def model(params):
    return params["a"] * 10 + params["b"]


class Tools:
    @staticmethod
    def triple(x):
        return 3 * x


class ExecutingController:
    """Runs submitted calls locally and delivers results in a chosen order."""

    def __init__(self, order="fifo"):
        self.order = order
        self.count = 0
        self.waiting = {}
        self.assigned = {}
        self.results = {}
        self.result_queue = []
        self.computed = []
        self.end_session_calls = 0

    def submit_call(self, name_to_call, args=(), kwargs=None, module_name="", **kw):
        func = eval(name_to_call, sys.modules[module_name].__dict__)
        task_id = self.count
        self.count += 1
        self.assigned[task_id] = 1
        self.computed.append((task_id, func(*args, **(kwargs or {}))))
        return task_id

    def process(self, limit=1000, block=False):
        batch = self.computed if self.order == "fifo" else self.computed[::-1]
        self.computed = []
        for task_id, value in batch:
            self.assigned.pop(task_id)
            self.results[task_id] = value
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


def make_controller(monkeypatch, order="fifo"):
    monkeypatch.setattr(fut_mod, "is_controller", True)
    fake = ExecutingController(order)
    env = SimpleNamespace(_controller=fake, worker_config=None)
    return TaskController(env), fake


@pytest.fixture
def ctl(monkeypatch):
    return make_controller(monkeypatch, "reverse")[0]


# Mapping


def test_map_returns_input_order_despite_reverse_arrival(ctl):
    assert map_distributed(ctl, square, range(5)) == [0, 1, 4, 9, 16]


def test_imap_unordered_yields_in_arrival_order(ctl):
    assert list(imap_distributed(ctl, square, range(4), ordered=False)) == [9, 4, 1, 0]


def test_starmap_zip_and_parallel_map(ctl):
    assert starmap_distributed(ctl, add, [(1, 2), (3, 4)]) == [3, 7]
    assert zip_distributed(ctl, add, [1, 2, 3], [10, 20, 30]) == [11, 22, 33]
    assert parallel_map(ctl, [square, negate, inc], 4) == [16, -4, 5]


# Reduction


def test_ordered_reduction_follows_submission_order(ctl):
    assert map_reduce(ctl, as_text, operator.add, range(4)) == "0123"


def test_unordered_reduction_follows_arrival_order(ctl):
    assert map_reduce(ctl, as_text, operator.add, range(4), ordered=False) == "3210"


def test_fold_with_initial_value(ctl):
    assert fold_distributed(ctl, square, operator.add, range(4), 100) == 114


def test_empty_reduction(ctl):
    assert map_reduce(ctl, square, operator.add, [], 0) == 0
    with pytest.raises(TypeError):
        map_reduce(ctl, square, operator.add, [])


def test_reduce_over_existing_futures(ctl):
    futures = [ctl.submit(square, i) for i in range(4)]
    assert reduce_distributed(max, futures, ordered=False) == 9


# Pipelines


def test_pipeline_map_filter_flatmap(ctl):
    p = pipeline(square).filter(lambda x: x % 2 == 0).flatmap(twice).map(inc)
    assert p.run(ctl, range(5)) == [1, 1, 5, 5, 17, 17]


def test_pipeline_is_immutable_and_composable(ctl):
    base = pipeline(square)
    longer = base.map(inc)
    assert len(base.steps) == 1
    assert base.compose(longer).run(ctl, [2]) == [17]


def test_pipeline_rejects_lambda_map_step():
    with pytest.raises(TypeError, match="module level"):
        FunctionalPipeline().map(lambda x: x)


# Control flow


def test_conditional_map(ctl):
    assert conditional_map(ctl, [1, 2], lambda: False, square, negate) == [-1, -2]


def test_adaptive_map_chooses_lowest_cost(ctl):
    func, results = adaptive_map(
        ctl,
        [square, negate],
        range(6),
        sample_size=2,
        cost=lambda results, elapsed: sum(results),
    )
    assert func is negate
    assert results == [0, -1, -2, -3, -4, -5]


# Callables that cannot be sent


def test_submit_rejects_lambda_nested_and_partial(monkeypatch):
    ctl, fake = make_controller(monkeypatch)

    def nested(x):
        return x

    for bad in (lambda x: x, nested, functools.partial(add, 1)):
        with pytest.raises(TypeError):
            ctl.submit(bad, 1)
    with pytest.raises(TypeError):
        ctl.batch_submit([(square, (1,), {}), (lambda x: x, (2,), {})])
    assert fake.count == 0


def test_static_method_of_module_level_class_is_accepted(monkeypatch):
    ctl, _ = make_controller(monkeypatch)
    assert task_reference(Tools.triple) == ("Tools.triple", __name__)
    assert ctl.submit(Tools.triple, 2).result() == 6


# Workflows


def test_parameter_sweep_order(ctl):
    sweep = parameter_sweep(ctl, model, {"a": [1, 2], "b": [0, 5]})
    assert sweep == [
        ({"a": 1, "b": 0}, 10),
        ({"a": 1, "b": 5}, 15),
        ({"a": 2, "b": 0}, 20),
        ({"a": 2, "b": 5}, 25),
    ]


def test_adaptive_parameter_exploration_stops_on_empty_refinement(ctl):
    def refine(results):
        return [{"a": max(results), "b": 1}] if max(results) < 100 else []

    params, results = adaptive_parameter_exploration(
        ctl, model, [{"a": 1, "b": 0}, {"a": 2, "b": 0}], refine, max_iterations=5
    )
    assert results == [10, 20, 201]
    assert params[-1] == {"a": 20, "b": 1}
