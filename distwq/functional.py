# Functional Interface with Two-Phase Iterator Design

from typing import Callable, Iterable, Iterator, TypeVar, List, Optional, Tuple
from functools import reduce
from dataclasses import dataclass
from enum import Enum

try:
    from .future import (
        DistributedEnvironment,
        TaskController,
        DistributedFuture,
        create_spawn_environment,
        create_split_environment,
    )
    from .config import Config, WorkerGroupConfig, SessionPhase
except ImportError:
    from future import (
        DistributedEnvironment,
    )
    from config import Config, WorkerGroupConfig

    # Define SessionPhase if not available in config
    class SessionPhase(Enum):
        PREPROCESSING = "preprocessing"
        EXPLORATION = "exploration"
        OPTIMIZATION = "optimization"
        VALIDATION = "validation"
        POSTPROCESSING = "postprocessing"


T = TypeVar("T")
U = TypeVar("U")
R = TypeVar("R")
S = TypeVar("S")

# =============================================================================
# FUNCTIONAL SESSION CONTEXT
# =============================================================================


@dataclass(frozen=True)
class FunctionalContext:
    """Immutable context for functional operations"""

    phase: SessionPhase = SessionPhase.EXPLORATION
    worker_config: Optional[WorkerGroupConfig] = None
    timeout: Optional[float] = None
    chunk_size: Optional[int] = None
    ordered: bool = True
    retry_attempts: int = 0

    def with_phase(self, phase: SessionPhase) -> "FunctionalContext":
        """Create new context with different phase"""
        return FunctionalContext(
            phase=phase,
            worker_config=self.worker_config,
            timeout=self.timeout,
            chunk_size=self.chunk_size,
            ordered=self.ordered,
            retry_attempts=self.retry_attempts,
        )

    def with_workers(self, worker_config: WorkerGroupConfig) -> "FunctionalContext":
        """Create new context with worker configuration"""
        return FunctionalContext(
            phase=self.phase,
            worker_config=worker_config,
            timeout=self.timeout,
            chunk_size=self.chunk_size,
            ordered=self.ordered,
            retry_attempts=self.retry_attempts,
        )

    def with_options(self, **kwargs) -> "FunctionalContext":
        """Create new context with updated options"""
        return FunctionalContext(
            phase=self.phase,
            worker_config=self.worker_config,
            timeout=kwargs.get("timeout", self.timeout),
            chunk_size=kwargs.get("chunk_size", self.chunk_size),
            ordered=kwargs.get("ordered", self.ordered),
            retry_attempts=kwargs.get("retry_attempts", self.retry_attempts),
        )


# =============================================================================
# CORE FUNCTIONAL COMBINATORS
# =============================================================================


def map_distributed(
    func: Callable[[T], R],
    data: Iterable[T],
    config: Optional[Config] = None,
    context: Optional[FunctionalContext] = None,
) -> List[R]:
    """
    Functional map - no side effects, returns all results.

    Uses the Two-Phase Iterator design for clean separation of concerns.
    """
    config = config or Config()
    context = context or FunctionalContext()

    env_kwargs = {}
    if context.worker_config:
        env_kwargs["worker_config"] = context.worker_config

    with DistributedEnvironment(config, **env_kwargs) as env:
        for controller in env.controller_session():
            if context.ordered:
                return list(
                    controller.map(
                        func,
                        data,
                        timeout=context.timeout,
                        chunk_size=context.chunk_size,
                    )
                )
            else:
                return list(
                    controller.imap_unordered(func, data, chunk_size=context.chunk_size)
                )


def imap_distributed(
    func: Callable[[T], R],
    data: Iterable[T],
    config: Optional[Config] = None,
    context: Optional[FunctionalContext] = None,
) -> Iterator[R]:
    """
    Lazy distributed map - returns iterator.

    Note: This is conceptually tricky with the Two-Phase Iterator since
    the environment needs to stay alive. In practice, you'd want to use
    this within an active environment context.
    """
    config = config or Config()
    context = context or FunctionalContext()

    env_kwargs = {}
    if context.worker_config:
        env_kwargs["worker_config"] = context.worker_config

    with DistributedEnvironment(config, **env_kwargs) as env:
        for controller in env.controller_session():
            yield from controller.imap_unordered(
                func, data, chunk_size=context.chunk_size
            )


def reduce_distributed(
    func: Callable[[T, T], T],
    data: Iterable[T],
    initializer: Optional[T] = None,
    config: Optional[Config] = None,
    context: Optional[FunctionalContext] = None,
) -> T:
    """Distributed reduce operation using map-reduce pattern"""
    mapped_data = map_distributed(lambda x: x, data, config, context)

    if initializer is not None:
        return reduce(func, mapped_data, initializer)
    else:
        return reduce(func, mapped_data)


def fold_distributed(
    func: Callable[[U, T], U],
    data: Iterable[T],
    initial: U,
    config: Optional[Config] = None,
    context: Optional[FunctionalContext] = None,
) -> U:
    """Distributed fold (left reduce) operation"""
    mapped_data = map_distributed(lambda x: x, data, config, context)
    return reduce(func, mapped_data, initial)


# =============================================================================
# PIPELINE ABSTRACTION
# =============================================================================


class FunctionalPipeline:
    """
    Functional pipeline for distributed computation using Two-Phase Iterator design.

    Supports method chaining and functional composition with immutable transformations.
    """

    def __init__(
        self,
        steps: List[Callable] = None,
        config: Optional[Config] = None,
        context: Optional[FunctionalContext] = None,
    ):
        self.steps = steps or []
        self.config = config or Config()
        self.context = context or FunctionalContext()

    def __call__(self, data: Iterable[T]) -> List[R]:
        """Execute pipeline on data using Two-Phase Iterator"""
        current_data = data

        # Execute pipeline steps sequentially
        with DistributedEnvironment(self.config) as env:
            for controller in env.controller_session():
                for step in self.steps:
                    if isinstance(current_data, (list, tuple)):
                        current_data = list(controller.map(step, current_data))
                    else:
                        # Handle single value
                        future = controller.submit(step, current_data)
                        current_data = future.result()

        return current_data

    def compose(self, other: "FunctionalPipeline") -> "FunctionalPipeline":
        """Compose two pipelines functionally (immutable)"""
        return FunctionalPipeline(self.steps + other.steps, self.config, self.context)

    def map(self, func: Callable[[T], R]) -> "FunctionalPipeline":
        """Add a map step to the pipeline (immutable)"""
        return FunctionalPipeline(self.steps + [func], self.config, self.context)

    def filter(self, predicate: Callable[[T], bool]) -> "FunctionalPipeline":
        """Add a filter step to the pipeline (immutable)"""

        def filter_step(data):
            if isinstance(data, (list, tuple)):
                return [item for item in data if predicate(item)]
            else:
                return data if predicate(data) else None

        return FunctionalPipeline(self.steps + [filter_step], self.config, self.context)

    def flatmap(self, func: Callable[[T], Iterable[R]]) -> "FunctionalPipeline":
        """Add a flatmap step to the pipeline (immutable)"""

        def flatmap_step(data):
            if isinstance(data, (list, tuple)):
                result = []
                for item in data:
                    mapped = func(item)
                    if hasattr(mapped, "__iter__") and not isinstance(mapped, str):
                        result.extend(mapped)
                    else:
                        result.append(mapped)
                return result
            else:
                return func(data)

        return FunctionalPipeline(
            self.steps + [flatmap_step], self.config, self.context
        )

    def with_context(self, context: FunctionalContext) -> "FunctionalPipeline":
        """Create new pipeline with different context (immutable)"""
        return FunctionalPipeline(self.steps, self.config, context)

    def with_config(self, config: Config) -> "FunctionalPipeline":
        """Create new pipeline with different config (immutable)"""
        return FunctionalPipeline(self.steps, config, self.context)

    def parallel_map(self, func: Callable[[T], R]) -> "FunctionalPipeline":
        """Add a parallel map step using worker groups"""

        def parallel_map_step(data):
            # Use worker groups for parallel processing
            context_with_workers = self.context.with_workers(
                WorkerGroupConfig(nprocs_per_worker=2)
            )
            return map_distributed(func, data, self.config, context_with_workers)

        return FunctionalPipeline(
            self.steps + [parallel_map_step], self.config, self.context
        )


# =============================================================================
# ADVANCED FUNCTIONAL COMBINATORS
# =============================================================================


def parallel_map(
    funcs: List[Callable[[T], R]],
    data: T,
    config: Optional[Config] = None,
    context: Optional[FunctionalContext] = None,
) -> List[R]:
    """Apply multiple functions to the same data in parallel"""
    config = config or Config()
    context = context or FunctionalContext()

    env_kwargs = {}
    if context.worker_config:
        env_kwargs["worker_config"] = context.worker_config

    with DistributedEnvironment(config, **env_kwargs) as env:
        for controller in env.controller_session():
            futures = [controller.submit(func, data) for func in funcs]
            return [f.result(timeout=context.timeout) for f in futures]


def map_reduce(
    map_func: Callable[[T], U],
    reduce_func: Callable[[U, U], U],
    data: Iterable[T],
    config: Optional[Config] = None,
    context: Optional[FunctionalContext] = None,
) -> U:
    """Classic map-reduce pattern with Two-Phase Iterator"""
    mapped_data = map_distributed(map_func, data, config, context)
    return reduce(reduce_func, mapped_data)


def parallel_pipelines(
    pipelines: List[FunctionalPipeline],
    data: Iterable[T],
    config: Optional[Config] = None,
) -> List[List[R]]:
    """Run multiple pipelines in parallel on the same data"""
    config = config or Config()

    with DistributedEnvironment(config) as env:
        for controller in env.controller_session():
            futures = [controller.submit(pipeline, data) for pipeline in pipelines]
            return [f.result() for f in futures]


def zip_distributed(
    *iterables: Iterable[T],
    func: Callable[..., R] = tuple,
    config: Optional[Config] = None,
    context: Optional[FunctionalContext] = None,
) -> List[R]:
    """Distributed zip operation with optional combining function"""
    zipped_data = list(zip(*iterables))
    return map_distributed(lambda args: func(*args), zipped_data, config, context)


def starmap_distributed(
    func: Callable[..., R],
    iterable: Iterable[Tuple],
    config: Optional[Config] = None,
    context: Optional[FunctionalContext] = None,
) -> List[R]:
    """Distributed starmap - unpack arguments for function"""
    return map_distributed(lambda args: func(*args), iterable, config, context)


# =============================================================================
# FUNCTIONAL COMBINATORS FOR SESSIONS
# =============================================================================


def adaptive_map(
    func: Callable[[T], R],
    data: Iterable[T],
    performance_threshold: float = 0.8,
    config: Optional[Config] = None,
) -> List[R]:
    """
    Adaptive map that switches strategies based on performance.

    Starts with simple strategy, escalates to more powerful strategies if needed.
    """
    config = config or Config()
    data_list = list(data)

    # Start with simple context
    simple_context = FunctionalContext(
        phase=SessionPhase.EXPLORATION, worker_config=None, ordered=True
    )

    # Try simple approach first
    import time

    start_time = time.time()

    try:
        # Sample performance with small subset
        sample_size = min(10, len(data_list))
        sample_data = data_list[:sample_size]
        sample_results = map_distributed(func, sample_data, config, simple_context)
        sample_time = time.time() - start_time

        # Estimate full execution time
        estimated_time = sample_time * (len(data_list) / sample_size)

        if estimated_time < performance_threshold:
            # Simple approach is fast enough
            remaining_data = data_list[sample_size:]
            remaining_results = map_distributed(
                func, remaining_data, config, simple_context
            )
            return sample_results + remaining_results
        else:
            # Need more powerful approach
            powerful_context = FunctionalContext(
                phase=SessionPhase.OPTIMIZATION,
                worker_config=WorkerGroupConfig(nprocs_per_worker=4),
                ordered=False,  # Allow unordered for speed
            )
            return map_distributed(func, data_list, config, powerful_context)

    except Exception:
        # Fallback to powerful approach
        powerful_context = FunctionalContext(
            phase=SessionPhase.OPTIMIZATION,
            worker_config=WorkerGroupConfig(nprocs_per_worker=2),
            ordered=True,
        )
        return map_distributed(func, data_list, config, powerful_context)


def conditional_map(
    func: Callable[[T], R],
    data: Iterable[T],
    condition: Callable[[], bool],
    true_context: FunctionalContext,
    false_context: FunctionalContext,
    config: Optional[Config] = None,
) -> List[R]:
    """Conditional map based on runtime condition"""
    chosen_context = true_context if condition() else false_context
    return map_distributed(func, data, config, chosen_context)


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_exploration_pipeline(
    steps: List[Callable], config: Optional[Config] = None
) -> FunctionalPipeline:
    """Create pipeline optimized for exploration phase"""
    exploration_context = FunctionalContext(
        phase=SessionPhase.EXPLORATION,
        ordered=False,  # Allow unordered for speed
        chunk_size=10,
    )
    return FunctionalPipeline(steps, config, exploration_context)


def create_optimization_pipeline(
    steps: List[Callable], config: Optional[Config] = None
) -> FunctionalPipeline:
    """Create pipeline optimized for optimization phase"""
    optimization_context = FunctionalContext(
        phase=SessionPhase.OPTIMIZATION,
        worker_config=WorkerGroupConfig(nprocs_per_worker=4),
        ordered=True,
        timeout=300.0,
    )
    return FunctionalPipeline(steps, config, optimization_context)


def create_validation_pipeline(
    steps: List[Callable], config: Optional[Config] = None
) -> FunctionalPipeline:
    """Create pipeline optimized for validation phase"""
    validation_context = FunctionalContext(
        phase=SessionPhase.VALIDATION, ordered=True, retry_attempts=3
    )
    return FunctionalPipeline(steps, config, validation_context)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def quick_map(func: Callable[[T], R], data: Iterable[T]) -> List[R]:
    """Quick distributed map with sensible defaults"""
    return map_distributed(func, data, Config.for_development())


def quick_pipeline(*steps: Callable) -> FunctionalPipeline:
    """Quick pipeline creation with sensible defaults"""
    return FunctionalPipeline(list(steps), Config.for_development())


def parallel_quick_map(func: Callable[[T], R], data: Iterable[T]) -> List[R]:
    """Quick parallel map using worker groups"""
    context = FunctionalContext(worker_config=WorkerGroupConfig(nprocs_per_worker=2))
    return map_distributed(func, data, Config.for_development(), context)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================


def example_functional_patterns():
    """Example usage of functional patterns with Two-Phase Iterator"""

    # Simple functional operations
    data = list(range(100))

    # Basic map
    squared = quick_map(lambda x: x**2, data)

    # Pipeline with method chaining
    pipeline = (
        quick_pipeline(lambda x: x**2)
        .map(lambda x: x + 1)
        .filter(lambda x: x % 2 == 0)
        .map(lambda x: x / 2)
    )

    result = pipeline(data)

    # Parallel execution of multiple functions
    results = parallel_map([lambda x: sum(x), lambda x: max(x), lambda x: len(x)], data)

    # Adaptive execution
    def expensive_computation(x):
        import time

        time.sleep(0.01)  # Simulate expensive computation
        return x**0.5

    adaptive_results = adaptive_map(expensive_computation, data[:50])

    return {
        "squared": squared[:5],  # First 5 results
        "pipeline": result[:5],
        "parallel": results,
        "adaptive": adaptive_results[:5],
    }


# Export main functional interface
__all__ = [
    "map_distributed",
    "imap_distributed",
    "reduce_distributed",
    "fold_distributed",
    "FunctionalPipeline",
    "FunctionalContext",
    "parallel_map",
    "map_reduce",
    "parallel_pipelines",
    "zip_distributed",
    "starmap_distributed",
    "session_map",
    "adaptive_map",
    "conditional_map",
    "compose",
    "pipe",
    "curry_distributed",
    "memoize_distributed",
    "parameter_sweep",
    "adaptive_parameter_exploration",
    "create_exploration_pipeline",
    "create_optimization_pipeline",
    "create_validation_pipeline",
    "quick_map",
    "quick_pipeline",
    "parallel_quick_map",
]
