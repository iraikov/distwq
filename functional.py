# Functional Interface

from typing import Callable, Iterable, Iterator, TypeVar, List
from functools import partial, reduce
from distwq.config import Config
from distwq.future import DistributedWorkQueue

T = TypeVar("T")
U = TypeVar("U")
R = TypeVar("R")

# Combinator functions for distributed computation


def map_distributed(
    func: Callable[[T], R], data: Iterable[T], config: Config = None
) -> List[R]:
    """Functional map - no side effects, returns all results"""
    config = config or Config()

    with DistributedWorkQueue(config) as executor:
        return list(executor.map(func, data))


def imap_distributed(
    func: Callable[[T], R], data: Iterable[T], config: Config = None
) -> Iterator[R]:
    """Lazy distributed map - returns iterator"""
    config = config or Config()

    with DistributedWorkQueue(config) as executor:
        yield from executor.imap_unordered(func, data)


def reduce_distributed(
    func: Callable[[T, T], T],
    data: Iterable[T],
    initializer: T = None,
    config: Config = None,
) -> T:
    """Distributed reduce operation"""
    config = config or Config()

    return reduce(func, map_distributed(lambda x: x, data, config), initializer)


# Function composition for pipelines
class Pipeline:
    """Functional pipeline for distributed computation"""

    def __init__(self, steps: List[Callable], config: Config = None):
        self.steps = steps
        self.config = config or Config()

    def __call__(self, data: Iterable[T]) -> List[R]:
        """Execute pipeline on data"""
        current_data = data

        for step in self.steps:
            current_data = map_distributed(step, current_data, self.config)

        return current_data

    def compose(self, other: "Pipeline") -> "Pipeline":
        """Compose two pipelines"""
        return Pipeline(self.steps + other.steps, self.config)

    def map(self, func: Callable) -> "Pipeline":
        """Add a map step to the pipeline"""
        return Pipeline(self.steps + [func], self.config)

    def filter(self, predicate: Callable[[T], bool]) -> "Pipeline":
        """Add a filter step to the pipeline"""

        def filter_step(data):
            return [item for item in data if predicate(item)]

        return Pipeline(self.steps + [filter_step], self.config)


# Combinators for common patterns
def parallel_map(
    funcs: List[Callable[[T], R]], data: T, config: Config = None
) -> List[R]:
    """Apply multiple functions to the same data in parallel"""
    with DistributedWorkQueue(config) as executor:
        futures = [executor.submit(func, data) for func in funcs]
        return [f.result() for f in futures]


def map_reduce(
    map_func: Callable[[T], U],
    reduce_func: Callable[[U, U], U],
    data: Iterable[T],
    config: Config = None,
) -> U:
    """Classic map-reduce pattern"""
    mapped_data = map_distributed(map_func, data, config)
    return reduce(reduce_func, mapped_data)


def parallel_pipeline(
    pipelines: List[Pipeline], data: Iterable[T], config: Config = None
) -> List[List[R]]:
    """Run multiple pipelines in parallel on the same data"""

    with DistributedWorkQueue(config) as executor:
        futures = [executor.submit(pipeline, data) for pipeline in pipelines]
        return [f.result() for f in futures]


# Currying and partial application helpers
def curry_distributed(func: Callable) -> Callable:
    """Create a curried version of a function for distributed execution"""

    def curried(*args, **kwargs):
        if len(args) + len(kwargs) >= func.__code__.co_argcount:
            return func(*args, **kwargs)
        else:
            return partial(func, *args, **kwargs)

    return curried


# Higher-order combinators
def compose(*functions: Callable) -> Callable:
    """Compose functions right-to-left"""
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)


def pipe(*functions: Callable) -> Callable:
    """Compose functions left-to-right (pipe)"""
    return reduce(lambda f, g: lambda x: g(f(x)), functions, lambda x: x)
