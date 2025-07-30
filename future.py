# Future-based distributed task management

import time
from concurrent.futures import Future
from typing import (
    Iterable,
    Iterator,
    TypeVar,
    Generic,
    Callable,
    List,
    Tuple,
    Dict,
    Any,
    Optional,
)
from distwq.config import Config

T = TypeVar("T")
R = TypeVar("R")


class DistributedFuture(Future, Generic[R]):
    """Enhanced Future with distributed computing features"""

    def __init__(self, task_id: int, context: "DistributedTaskExecution"):
        super().__init__()
        self.task_id = task_id
        self._context = context
        self._submitted_at = time.time()

    def progress(self) -> Optional[float]:
        """Get task progress if available (0.0 to 1.0)"""
        return self._context._get_task_progress(self.task_id)

    def worker_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the worker processing this task"""
        return self._context._get_worker_info(self.task_id)

    def estimated_completion(self) -> Optional[float]:
        """Get estimated completion time in seconds"""
        return self._context._get_estimated_completion(self.task_id)


class DistributedTaskExecution:
    """Main interface for distributed task execution"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._controller = None
        self._active = False

    def __enter__(self) -> "DistributedTaskExecution":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def start(self):
        """Initialize the distributed environment"""
        if not self._active:
            self._controller = self._create_controller()
            self._active = True

    def shutdown(self):
        """Clean shutdown of distributed environment"""
        if self._active and self._controller:
            self._controller.exit()
            self._active = False

    def submit(
        self, func: Callable[..., R], *args, timeout: Optional[int] = None, **kwargs
    ) -> DistributedFuture[R]:
        """Submit a single task for execution"""
        if not self._active:
            raise RuntimeError(
                "DistributedTaskExecution not started. Use with statement or call start()"
            )

        task_id = self._controller.submit_call(
            func.__name__,
            args=args,
            kwargs=kwargs,
            module_name=func.__module__,
            time_est=self._estimate_task_time(func, args, kwargs),
        )

        future = DistributedFuture(task_id, self)
        self._setup_future_completion(future)
        return future

    def map(
        self,
        func: Callable[[T], R],
        iterable: Iterable[T],
        chunk_size: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Iterator[R]:
        """Map function over iterable, returning results in order"""

        futures = []
        for item in iterable:
            future = self.submit(func, item, timeout=timeout)
            futures.append(future)

        # Return results in original order
        for future in futures:
            yield future.result(timeout=timeout)

    def imap_unordered(
        self,
        func: Callable[[T], R],
        iterable: Iterable[T],
        chunk_size: Optional[int] = None,
    ) -> Iterator[R]:
        """Map function over iterable, yielding results as they complete"""

        futures = []
        for item in iterable:
            futures.append(self.submit(func, item))

        # Yield results as they become available
        from concurrent.futures import as_completed

        for future in as_completed(futures):
            yield future.result()

    def batch_submit(
        self, tasks: List[Tuple[Callable, tuple, dict]]
    ) -> List[DistributedFuture]:
        """Submit multiple heterogeneous tasks efficiently"""
        futures = []

        # Use submit_multiple for efficiency
        names = [task[0].__name__ for task in tasks]
        args_list = [task[1] for task in tasks]
        kwargs_list = [task[2] for task in tasks]
        modules = [task[0].__module__ for task in tasks]

        task_ids = self._controller.submit_multiple(
            names[0]
            if all(n == names[0] for n in names)
            else None,  # Optimization for same function
            args_list,
            kwargs_list,
            modules[0] if all(m == modules[0] for m in modules) else "__main__",
        )

        for task_id in task_ids:
            future = DistributedFuture(task_id, self)
            self._setup_future_completion(future)
            futures.append(future)

        return futures


# Context manager for automatic resource management
class DistributedWorkQueue:
    """High-level distributed work queue interface"""

    def __init__(self, config: Config = None):
        self.config = config or Config()
        self._workqueue = None

    def __enter__(self) -> DistributedTaskExecution:
        self._context = DistributedTaskExecution(self.config)
        return self._context.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._context:
            self._context.__exit__(exc_type, exc_val, exc_tb)
