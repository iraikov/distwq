"""Functional combinators for distributed work.

Every combinator runs on the controller, inside a session, and takes
the session's TaskController as its first argument:

    with DistributedEnvironment(config) as env:
        for ctl in env.controller_session():
            squares = map_distributed(ctl, square, range(100))
            total = map_reduce(ctl, square, operator.add, range(100))

Because all coordination happens on the controller, the controller's
code can branch on results freely: the worker ranks remain in their
serve loop and do need to follow those branches.  Several combinators
can share one session, and one environment can hold several sessions.

Functions sent to workers (the mapped functions) must be defined at module
level, since workers look them up by module and name; see
distwq.future.task_reference().  Functions that run on the controller,
such as reducers, filter predicates and conditions, can be any callable,
including lambdas.

For a single operation with no surrounding environment, run_session()
and quick_map() open an environment, run one session, and return the
result on every rank.

"""

import itertools
import time
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import reduce
from typing import Any, TypeVar

from .config import Config, WorkerGroupConfig
from .future import (
    DistributedEnvironment,
    DistributedFuture,
    TaskController,
    as_completed,
    task_reference,
)

T = TypeVar("T")
U = TypeVar("U")
R = TypeVar("R")

# Marks a missing initializer, since None is a valid initial value.
_MISSING = object()


# =============================================================================
# Mapping
# =============================================================================


def imap_distributed(
    ctl: TaskController,
    func: Callable[[T], R],
    data: Iterable[T],
    *,
    ordered: bool = True,
    timeout: float | None = None,
) -> Iterator[R]:
    """Apply func to every item on the workers and yield the results.

    All items are submitted first.  With ordered=True results are yielded
    in input order, and timeout bounds the wait for each result.  With
    ordered=False they are yielded as they arrive, and timeout bounds the
    wait for all of them.
    """
    futures = ctl.batch_submit([(func, (item,), {}) for item in data])
    if ordered:
        for future in futures:
            yield future.result(timeout=timeout)
    else:
        for future in as_completed(futures, timeout=timeout):
            yield future.result()


def map_distributed(
    ctl: TaskController,
    func: Callable[[T], R],
    data: Iterable[T],
    *,
    timeout: float | None = None,
) -> list[R]:
    """Return [func(x) for x in data], computed on the workers."""
    return list(imap_distributed(ctl, func, data, timeout=timeout))


def starmap_distributed(
    ctl: TaskController,
    func: Callable[..., R],
    arg_tuples: Iterable[Sequence[Any]],
    *,
    timeout: float | None = None,
) -> list[R]:
    """Return [func(*args) for args in arg_tuples], computed on the workers."""
    futures = ctl.batch_submit([(func, tuple(args), {}) for args in arg_tuples])
    return [f.result(timeout=timeout) for f in futures]


def zip_distributed(
    ctl: TaskController,
    func: Callable[..., R],
    *iterables: Iterable[Any],
    timeout: float | None = None,
) -> list[R]:
    """Return [func(*items) for items in zip(*iterables)], computed on the workers."""
    return starmap_distributed(ctl, func, zip(*iterables), timeout=timeout)


def parallel_map(
    ctl: TaskController,
    funcs: Sequence[Callable[[T], R]],
    data: T,
    *,
    timeout: float | None = None,
) -> list[R]:
    """Apply each function in funcs to the same data, all at once.

    Returns the results in the order of funcs.
    """
    futures = ctl.batch_submit([(func, (data,), {}) for func in funcs])
    return [f.result(timeout=timeout) for f in futures]


# =============================================================================
# Reduction
# =============================================================================


def reduce_distributed(
    func: Callable[[Any, R], Any],
    futures: Iterable[DistributedFuture],
    initializer: Any = _MISSING,
    *,
    ordered: bool = True,
    timeout: float | None = None,
) -> Any:
    """Combine the results of futures with func, on the controller.

    With ordered=True results are combined in the order of futures, so the
    answer is the same on every run.  With ordered=False each result is
    combined as soon as it arrives, so no result waits behind a slower
    one.  This gives the same answer only when func is associative and
    commutative, for example addition or max.

    With an initializer this is a left fold: func(func(init, r0), r1)...
    Without one, an empty sequence of futures raises TypeError, as
    functools.reduce does.
    """
    futures = list(futures)
    if ordered:
        results = (f.result(timeout=timeout) for f in futures)
    else:
        results = (f.result() for f in as_completed(futures, timeout=timeout))
    if initializer is _MISSING:
        return reduce(func, results)
    return reduce(func, results, initializer)


def fold_distributed(
    ctl: TaskController,
    map_func: Callable[[T], U],
    fold_func: Callable[[Any, U], Any],
    data: Iterable[T],
    initial: Any,
    *,
    ordered: bool = True,
    timeout: float | None = None,
) -> Any:
    """Map map_func over data on the workers, then fold the results with
    fold_func on the controller, starting from initial.

    See reduce_distributed() for the meaning of ordered.
    """
    futures = ctl.batch_submit([(map_func, (item,), {}) for item in data])
    return reduce_distributed(
        fold_func, futures, initial, ordered=ordered, timeout=timeout
    )


def map_reduce(
    ctl: TaskController,
    map_func: Callable[[T], U],
    reduce_func: Callable[[U, U], U],
    data: Iterable[T],
    initializer: Any = _MISSING,
    *,
    ordered: bool = True,
    timeout: float | None = None,
) -> U:
    """Map map_func over data on the workers, then reduce the results with
    reduce_func on the controller.

    See reduce_distributed() for the meaning of ordered and initializer.
    """
    futures = ctl.batch_submit([(map_func, (item,), {}) for item in data])
    return reduce_distributed(
        reduce_func, futures, initializer, ordered=ordered, timeout=timeout
    )


# =============================================================================
# Pipelines
# =============================================================================


class FunctionalPipeline:
    """An immutable sequence of steps applied to a list of items.

    Each method returns a new pipeline with one more step:

    - map(func): func is applied to every item on the workers;
    - flatmap(func): func is applied to every item on the workers, and the
      iterables it returns are concatenated on the controller;
    - filter(predicate): items for which predicate is false are dropped,
      on the controller.

    A pipeline is only a description.  run(ctl, data) carries it out in a
    session, step by step, and returns the final list.
    """

    def __init__(self, steps: Sequence[tuple[str, Callable]] = ()):
        self.steps: tuple[tuple[str, Callable], ...] = tuple(steps)

    def _with(self, kind: str, func: Callable) -> "FunctionalPipeline":
        return FunctionalPipeline((*self.steps, (kind, func)))

    def map(self, func: Callable[[Any], Any]) -> "FunctionalPipeline":
        """Add a step applying func to every item on the workers."""
        task_reference(func)
        return self._with("map", func)

    def flatmap(self, func: Callable[[Any], Iterable[Any]]) -> "FunctionalPipeline":
        """Add a step applying func on the workers and concatenating the results."""
        task_reference(func)
        return self._with("flatmap", func)

    def filter(self, predicate: Callable[[Any], bool]) -> "FunctionalPipeline":
        """Add a step keeping only the items for which predicate is true."""
        return self._with("filter", predicate)

    def compose(self, other: "FunctionalPipeline") -> "FunctionalPipeline":
        """Return a pipeline running this one's steps, then other's."""
        return FunctionalPipeline(self.steps + other.steps)

    def run(
        self,
        ctl: TaskController,
        data: Iterable[Any],
        *,
        timeout: float | None = None,
    ) -> list[Any]:
        """Carry out the pipeline on data in the session of ctl."""
        items = list(data)
        for kind, func in self.steps:
            if kind == "map":
                items = map_distributed(ctl, func, items, timeout=timeout)
            elif kind == "flatmap":
                pieces = map_distributed(ctl, func, items, timeout=timeout)
                items = list(itertools.chain.from_iterable(pieces))
            else:
                items = [item for item in items if func(item)]
        return items

    def __repr__(self) -> str:
        steps = ", ".join(
            f"{kind}({getattr(f, '__name__', f)!s})" for kind, f in self.steps
        )
        return f"FunctionalPipeline([{steps}])"


def pipeline(*funcs: Callable[[Any], Any]) -> FunctionalPipeline:
    """Return a pipeline of map steps, one for each function, in order."""
    result = FunctionalPipeline()
    for func in funcs:
        result = result.map(func)
    return result


# =============================================================================
# Control flow on the controller
# =============================================================================


def conditional_map(
    ctl: TaskController,
    data: Iterable[T],
    condition: Callable[[], bool],
    if_true: Callable[[T], R],
    if_false: Callable[[T], R],
    *,
    timeout: float | None = None,
) -> list[R]:
    """Map if_true over data if condition() is true, otherwise if_false.

    condition is evaluated once, on the controller.
    """
    func = if_true if condition() else if_false
    return map_distributed(ctl, func, data, timeout=timeout)


def _elapsed_seconds(_results: list[Any], elapsed: float) -> float:
    return elapsed


def adaptive_map(
    ctl: TaskController,
    candidates: Sequence[Callable[[T], R]],
    data: Iterable[T],
    *,
    sample_size: int = 10,
    cost: Callable[[list[R], float], float] = _elapsed_seconds,
    timeout: float | None = None,
) -> tuple[Callable[[T], R], list[R]]:
    """Choose among candidate functions on a sample, then map the best one.

    Each candidate is mapped over the first sample_size items, one
    candidate after another.  cost(results, elapsed_seconds) scores each
    attempt, by default its wall-clock time, and the candidate with the
    lowest cost is mapped over the remaining items.  Returns the chosen
    function and its results for all of data, in input order.

    The choice is made from results on the controller, so the workers are
    not involved in it.  Wall-clock costs vary from run to run, so the
    default choice can too; use a cost based on the results for a
    reproducible choice.
    """
    if not candidates:
        raise ValueError("adaptive_map needs at least one candidate function")
    items = list(data)
    sample = items[:sample_size]
    best = None
    for func in candidates:
        start = time.monotonic()
        results = map_distributed(ctl, func, sample, timeout=timeout)
        score = cost(results, time.monotonic() - start)
        if best is None or score < best[0]:
            best = (score, func, results)
    _, func, sample_results = best
    rest = map_distributed(ctl, func, items[sample_size:], timeout=timeout)
    return func, sample_results + rest


# =============================================================================
# One-shot helpers
# =============================================================================


def run_session(
    func: Callable[..., R],
    *args: Any,
    config: Config | None = None,
    worker_config: WorkerGroupConfig | None = None,
    broadcast_result: bool = True,
    **kwargs: Any,
) -> R | None:
    """Open an environment, call func(ctl, *args, **kwargs) in one session,
    and return its result.

    Must be called on every rank.  func runs on the controller only.  With
    broadcast_result=True the result is sent to every rank, so all ranks
    return it; otherwise worker ranks return None.  func itself may be a
    lambda, since it runs on the controller.
    """
    result = None
    with DistributedEnvironment(config, worker_config) as env:
        for ctl in env.controller_session():
            result = func(ctl, *args, **kwargs)
        if broadcast_result:
            result = env.broadcast(result)
    return result


def quick_map(
    func: Callable[[T], R],
    data: Iterable[T],
    *,
    config: Config | None = None,
    worker_config: WorkerGroupConfig | None = None,
    broadcast_result: bool = True,
) -> list[R] | None:
    """Return [func(x) for x in data] computed on the workers, in a new
    environment.  Must be called on every rank; see run_session()."""
    return run_session(
        map_distributed,
        func,
        data,
        config=config,
        worker_config=worker_config,
        broadcast_result=broadcast_result,
    )


__all__ = [
    "FunctionalPipeline",
    "adaptive_map",
    "conditional_map",
    "fold_distributed",
    "imap_distributed",
    "map_distributed",
    "map_reduce",
    "parallel_map",
    "pipeline",
    "quick_map",
    "reduce_distributed",
    "run_session",
    "starmap_distributed",
    "zip_distributed",
]
