"""
Distributed scientific workflows built on distwq.functional.

Like the combinators in distwq.functional, these run on the controller,
inside a session, and take the session's TaskController as their first
argument.  Model functions are sent to workers and must be defined at
module level; refinement functions run on the controller and can be any
callable.
"""

import itertools
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from .functional import map_distributed
from .future import TaskController

R = TypeVar("R")


def parameter_sweep(
    ctl: TaskController,
    model_func: Callable[[dict[str, Any]], R],
    parameter_space: Mapping[str, Sequence[Any]],
    *,
    timeout: float | None = None,
) -> list[tuple[dict[str, Any], R]]:
    """Evaluate model_func at every combination of parameter values.

    parameter_space maps each parameter name to the values to try.  The
    combinations are generated in the order of itertools.product over the
    names in parameter_space, each is passed to model_func as a dict, and
    the evaluations run on the workers.  Returns (parameters, result)
    pairs in that order.
    """
    names = list(parameter_space)
    combinations = [
        dict(zip(names, values))
        for values in itertools.product(*(parameter_space[n] for n in names))
    ]
    results = map_distributed(ctl, model_func, combinations, timeout=timeout)
    return list(zip(combinations, results))


def adaptive_parameter_exploration(
    ctl: TaskController,
    model_func: Callable[[dict[str, Any]], R],
    initial_parameters: Sequence[dict[str, Any]],
    refinement_func: Callable[[list[R]], list[dict[str, Any]]],
    *,
    max_iterations: int = 5,
    timeout: float | None = None,
) -> tuple[list[dict[str, Any]], list[R]]:
    """Evaluate parameter sets in rounds, refining them between rounds.

    Each round evaluates the current parameter sets on the workers.
    Between rounds, refinement_func(results) runs on the controller and
    returns the parameter sets for the next round; an empty list ends the
    exploration early.  At most max_iterations rounds are run.  Returns
    every parameter set evaluated and its result, in the order they were
    evaluated.
    """
    current = list(initial_parameters)
    all_params: list[dict[str, Any]] = []
    all_results: list[R] = []
    for iteration in range(max_iterations):
        if not current:
            break
        results = map_distributed(ctl, model_func, current, timeout=timeout)
        all_params.extend(current)
        all_results.extend(results)
        if iteration < max_iterations - 1:
            current = list(refinement_func(results))
    return all_params, all_results


__all__ = ["adaptive_parameter_exploration", "parameter_sweep"]
