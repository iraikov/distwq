"""
MPI tests for distwq.functional and distwq.workflow.

Run on several ranks, for example:

    mpirun.openmpi -n 3 python -m pytest tests/test_functional_mpi.py

Every rank runs every test in the same order.  The combinators run on the
controller only; the one-shot helpers return their result on every rank.
"""

import operator

import pytest

try:
    from mpi4py import MPI

    RANK = MPI.COMM_WORLD.Get_rank()
    SIZE = MPI.COMM_WORLD.Get_size()
except ImportError:
    pytest.skip("MPI not available", allow_module_level=True)

from distwq.functional import (
    adaptive_map,
    map_distributed,
    map_reduce,
    pipeline,
    quick_map,
    run_session,
)
from distwq.future import Config, DistributedEnvironment, WorkerGroupConfig
from distwq.mpi import GroupingMethod
from distwq.workflow import adaptive_parameter_exploration, parameter_sweep

pytestmark = pytest.mark.skipif(SIZE < 2, reason="Requires at least 2 MPI processes")

IS_CONTROLLER = RANK == 0


def square(x):
    return x * x


def negate(x):
    return -x


def twice(x):
    return [x, x]


def model(params):
    return params["a"] * 10 + params["b"]


def test_quick_map_returns_result_on_every_rank():
    assert quick_map(square, range(8)) == [x * x for x in range(8)]


def test_run_session_without_broadcast():
    result = run_session(
        lambda ctl: map_reduce(ctl, square, operator.add, range(10)),
        broadcast_result=False,
    )
    assert result == (285 if IS_CONTROLLER else None)


def test_many_combinators_in_one_session():
    out = {}
    with DistributedEnvironment(Config()) as env:
        for ctl in env.controller_session():
            out["map"] = map_distributed(ctl, square, range(5))
            out["sum"] = map_reduce(ctl, square, operator.add, range(5), ordered=False)
            out["pipe"] = (
                pipeline(square).filter(lambda x: x > 1).flatmap(twice).run(ctl, [1, 2])
            )
            out["sweep"] = parameter_sweep(ctl, model, {"a": [1, 2], "b": [3]})
        out = env.broadcast(out)
    assert out["map"] == [0, 1, 4, 9, 16]
    assert out["sum"] == 30
    assert out["pipe"] == [4, 4]
    assert out["sweep"] == [({"a": 1, "b": 3}, 13), ({"a": 2, "b": 3}, 23)]


def test_branching_on_results_across_two_sessions():
    # The controller branches on results; the workers do not follow.
    with DistributedEnvironment(Config()) as env:
        chosen = None
        for ctl in env.controller_session():
            chosen, _ = adaptive_map(
                ctl,
                [square, negate],
                range(4),
                sample_size=2,
                cost=lambda results, elapsed: sum(results),
            )
        rounds = None
        for ctl in env.controller_session():

            def refine(results):
                return [{"a": max(results), "b": 0}] if max(results) < 50 else []

            _, rounds = adaptive_parameter_exploration(
                ctl, model, [{"a": 1, "b": 0}], refine, max_iterations=4
            )
    if IS_CONTROLLER:
        assert chosen is negate
        assert rounds == [10, 100]


@pytest.mark.skipif(
    SIZE < 3 or (SIZE - 1) % 2 != 0,
    reason="split groups of 2 need (size - 1) divisible by 2",
)
def test_quick_map_with_split_groups():
    worker_config = WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSplit,
        nprocs_per_worker=2,
        broker_is_worker=True,
    )
    # Each broker returns the results of both processes in its group.
    assert quick_map(square, [3, 4], worker_config=worker_config) == [[9, 9], [16, 16]]
