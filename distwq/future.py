# Future-based distributed task management with Two-Phase Iterator design

import logging
import pickle
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from typing import (
    Any,
    Generic,
    Optional,
    TypeVar,
)

# Import from mpi.py to access MPI functionality
try:
    from .mpi import (
        GroupingMethod,
        MessageTag,
        MPIController,
        Topology,
        controller_rank,
        establish_topology,
        has_mpi,
        is_controller,
        is_worker,
        rank,
        worker_main,
        world_comm,
    )
except ImportError:
    # Try absolute import
    from mpi import (
        GroupingMethod,
        MessageTag,
        MPIController,
        Topology,
        controller_rank,
        establish_topology,
        has_mpi,
        is_controller,
        is_worker,
        rank,
        worker_main,
        world_comm,
    )

# Import config if available, otherwise provide basic fallback
try:
    from .config import Config, WorkerGroupConfig
except ImportError:
    try:
        from config import Config, WorkerGroupConfig
    except ImportError:
        # Fallback config implementation
        class Config:
            def __init__(self):
                self.resources = type("ResourceConfig", (), {"max_workers": -1})()
                self.execution = type(
                    "ExecutionConfig",
                    (),
                    {
                        "timeout": None,
                        "task_timeout": None,
                        "retry_attempts": 0,
                        "fail_fast": True,
                    },
                )()
                self.logging = type(
                    "LoggingConfig", (), {"verbose": False, "log_level": "INFO"}
                )()
                self.mpi = type(
                    "MPIConfig",
                    (),
                    {"collective_mode": "gather", "broker_is_worker": False},
                )()

        class WorkerGroupConfig:
            def __init__(
                self,
                grouping_method=None,
                nprocs_per_worker=1,
                broker_is_worker=False,
                **kwargs,
            ):
                self.grouping_method = grouping_method or GroupingMethod.NoGrouping
                self.nprocs_per_worker = nprocs_per_worker
                self.broker_is_worker = broker_is_worker
                self.collective_mode = kwargs.get("collective_mode", "gather")
                self.enable_worker_service = kwargs.get("enable_worker_service", False)
                self.worker_service_name = kwargs.get(
                    "worker_service_name", "distwq.init"
                )
                self.sequential_spawn = kwargs.get("sequential_spawn", False)
                self.spawn_startup_wait = kwargs.get("spawn_startup_wait", None)
                self.spawn_executable = kwargs.get("spawn_executable", None)
                self.spawn_args = kwargs.get("spawn_args", [])


T = TypeVar("T")
R = TypeVar("R")

logger = logging.getLogger(__name__)


FIRST_COMPLETED = "FIRST_COMPLETED"
ALL_COMPLETED = "ALL_COMPLETED"


class RemoteTraceback(Exception):
    """Carries the traceback of an exception raised on a worker.

    Set as the __cause__ of the exception that result() raises, so that
    the worker's traceback is printed after the controller's.
    """

    def __init__(self, tb: str):
        super().__init__(tb)
        self.tb = tb

    def __str__(self) -> str:
        return f'\n"""\n{self.tb}"""'


class RemoteTaskError(Exception):
    """Raised for a task whose exception could not be sent back as itself,
    because it could not be pickled on the worker or unpickled on the
    controller.

    exc_type is the qualified name of the original exception's class.
    """

    def __init__(self, exc_type: str, message: str, world_rank: int | None = None):
        super().__init__(f"{exc_type}: {message}")
        self.exc_type = exc_type
        self.world_rank = world_rank


def exception_from_failure(failure: dict[str, Any]) -> BaseException:
    """Rebuild the exception described by distwq.mpi.task_failure().

    Returns the original exception if it survives pickling, and a
    RemoteTaskError otherwise.  Either way the worker's traceback is
    attached as a RemoteTraceback in __cause__.
    """
    exc = None
    if failure.get("pickled") is not None:
        try:
            exc = pickle.loads(failure["pickled"])
        except Exception:  # noqa: BLE001 - fall back to a description
            exc = None
    if not isinstance(exc, BaseException):
        exc = RemoteTaskError(
            failure.get("type", "Exception"),
            failure.get("message", ""),
            failure.get("world_rank"),
        )
    exc.__cause__ = RemoteTraceback(failure.get("traceback", ""))
    return exc


def task_reference(func: Callable) -> tuple[str, str]:
    """Return (name, module) by which a worker can look up func.

    Tasks are sent to workers by name, and each worker finds the callable
    again by evaluating that name in the module's namespace.  This only
    works for functions, classes and static methods defined at module
    level, or as attributes of module-level classes.  Anything else, such
    as a lambda, a function defined inside another function, or a
    functools.partial, raises TypeError here, on the controller, instead
    of failing later on a worker.
    """
    name = getattr(func, "__qualname__", None)
    module_name = getattr(func, "__module__", None)
    if name is None or module_name is None:
        raise TypeError(
            f"{func!r} cannot be sent to workers: only module-level functions "
            "and classes can be submitted"
        )
    if "<lambda>" in name or "<locals>" in name:
        raise TypeError(
            f"{name} cannot be sent to workers: lambdas and functions defined "
            "inside other functions cannot be looked up by name; define it at "
            f"module level in {module_name}"
        )
    found = sys.modules.get(module_name)
    for part in name.split("."):
        found = getattr(found, part, None)
    if found is not func and getattr(found, "__func__", None) is not func:
        raise TypeError(
            f"{name} cannot be sent to workers: it is not reachable as "
            f"{module_name}.{name}"
        )
    return name, module_name


class DistributedFuture(Generic[R]):
    """Result of a task submitted through a TaskController.

    A future does not wait on a background thread.  Waiting for it, by
    calling result(), exception(), done(), or the module-level
    as_completed() and wait(), advances the controller on the calling
    thread until the result arrives.  Done callbacks therefore run on that
    same thread, in the order the results arrive.
    """

    def __init__(
        self,
        task_id: int,
        context: "TaskController",
        worker_group_size: int = 1,
    ):
        self.task_id = task_id
        self._context = context
        self._submitted_at = time.time()
        self.worker_group_size = worker_group_size
        self._done = False
        self._result: Any = None
        self._exception: BaseException | None = None
        self._callbacks: list[Callable[[DistributedFuture], Any]] = []
        self._completion_index: int | None = None

    def done(self) -> bool:
        """Return True once the result is available, checking without blocking."""
        if not self._done:
            self._context._pump(block=False)
        return self._done

    def result(self, timeout: float | None = None) -> R:
        """Return the task's result, waiting up to timeout seconds for it."""
        self._context._wait(self, timeout)
        if self._exception is not None:
            raise self._exception
        return self._result

    def exception(self, timeout: float | None = None) -> BaseException | None:
        """Return the exception raised by the task, or None if it succeeded."""
        self._context._wait(self, timeout)
        return self._exception

    def add_done_callback(self, fn: Callable[["DistributedFuture"], Any]) -> None:
        """Call fn(future) when the future completes, or now if it already has."""
        if self._done:
            fn(self)
        else:
            self._callbacks.append(fn)

    def cancel(self) -> bool:
        """Tasks cannot be withdrawn once submitted; always returns False."""
        return False

    def cancelled(self) -> bool:
        return False

    def _resolve(self, value: Any, completion_index: int) -> None:
        self._result = value
        self._finish(completion_index)

    def _fail(self, exc: BaseException, completion_index: int) -> None:
        self._exception = exc
        self._finish(completion_index)

    def _finish(self, completion_index: int) -> None:
        self._done = True
        self._completion_index = completion_index
        callbacks, self._callbacks = self._callbacks, []
        for fn in callbacks:
            fn(self)

    def worker_info(self) -> dict[str, Any] | None:
        """Get information about the worker or worker group processing this task"""
        return self._context._get_worker_info(self.task_id)

    def estimated_completion(self) -> float | None:
        """Get estimated completion time in seconds"""
        return self._context._get_estimated_completion(self.task_id)

    def progress(self) -> float | None:
        """Get task progress as a fraction between 0.0 and 1.0"""
        return self._context._get_task_progress(self.task_id)

    def is_group_task(self) -> bool:
        """Check if this task is being processed by a worker group"""
        return self.worker_group_size > 1


class TaskController:
    """Interface for coordinating distributed tasks - controller rank only"""

    def __init__(self, execution_context: "DistributedEnvironment"):
        if not is_controller:
            raise RuntimeError(
                "TaskController can only be instantiated on controller rank"
            )

        self._env = execution_context
        self._controller = execution_context._controller
        # Futures still waiting for their result, by task id.
        self._pending: dict[int, DistributedFuture] = {}
        # Results that arrived for a task id with no registered future,
        # as (value, failed) pairs.
        self._orphans: dict[int, tuple[Any, bool]] = {}
        self._completion_counter = 0
        # Seconds to sleep between non-blocking checks when a wait has a timeout.
        self.poll_interval = 0.01
        self._closed = False

    def submit(self, func: Callable[..., R], *args, **kwargs) -> DistributedFuture[R]:
        """Submit func(*args, **kwargs) for distributed execution.

        func must be defined at module level (see task_reference()).
        """
        return self._submit_one(func, args, kwargs)

    def _submit_one(
        self, func: Callable[..., R], args: tuple, kwargs: dict
    ) -> DistributedFuture[R]:
        if not self._controller:
            raise RuntimeError("No controller available for task submission")
        name, module_name = task_reference(func)
        group_size = (
            self._env.worker_config.nprocs_per_worker if self._env.worker_config else 1
        )
        task_id = self._controller.submit_call(
            name,
            args=args,
            kwargs=kwargs,
            module_name=module_name,
            time_est=self._estimate_task_time(func, args, kwargs),
        )
        logger.info(f"Submitted task id {task_id}")
        future = DistributedFuture(task_id, self, worker_group_size=group_size)
        self._setup_future_completion(future)
        return future

    def map(
        self,
        func: Callable[[T], R],
        iterable: Iterable[T],
        chunk_size: int | None = None,
        timeout: int | None = None,
        use_worker_groups: bool = False,
    ) -> Iterator[R]:
        """Map function over iterable using distributed workers"""
        tasks = [(func, (item,), {}) for item in iterable]
        futures = self.batch_submit(tasks)

        # Return results in original order
        for future in futures:
            yield future.result(timeout=timeout)

    def imap_unordered(
        self,
        func: Callable[[T], R],
        iterable: Iterable[T],
        chunk_size: int | None = None,
    ) -> Iterator[R]:
        """Map function over iterable, yielding results as they complete"""
        tasks = [(func, (item,), {}) for item in iterable]
        futures = self.batch_submit(tasks)

        # Yield results as they become available
        for future in as_completed(futures):
            yield future.result()

    def batch_submit(
        self, tasks: list[tuple[Callable, tuple, dict]]
    ) -> list[DistributedFuture]:
        """Submit (func, args, kwargs) tasks; return their futures in order.

        Every callable is checked before any task is submitted, so a bad
        one submits nothing.
        """
        for func, _args, _kwargs in tasks:
            task_reference(func)
        return [self._submit_one(func, args, kwargs) for func, args, kwargs in tasks]

    def active_worker_count(self) -> int:
        """Get the number of active workers"""
        if self._controller and hasattr(self._controller, "n_workers"):
            return self._controller.n_workers
        return 0

    def _estimate_task_time(self, func: Callable, args: tuple, kwargs: dict) -> int:
        """Estimate task execution time for scheduling"""
        # Simple heuristic - can be made more sophisticated
        return 1

    def _setup_future_completion(self, future: DistributedFuture):
        """Register future so that _pump() can deliver its result."""
        task_id = future.task_id
        if task_id in self._orphans:
            value, failed = self._orphans.pop(task_id)
            self._deliver(future, value, failed)
        else:
            self._pending[task_id] = future

    def _deliver(self, future: DistributedFuture, value: Any, failed: bool) -> None:
        if failed:
            future._fail(exception_from_failure(value), self._next_completion_index())
        else:
            future._resolve(value, self._next_completion_index())

    def _next_completion_index(self) -> int:
        index = self._completion_counter
        self._completion_counter += 1
        return index

    def _pump(self, block: bool) -> None:
        """Handle incoming worker messages and deliver any finished results.

        With block=True, waits in a blocking MPI probe until at least one
        message arrives; otherwise returns at once if nothing is pending.
        """
        if not self._controller:
            return
        self._controller.process(block=block)
        failed_tasks = getattr(self._controller, "failed_tasks", set())
        for task_id, value in self._controller.probe_all_next_results():
            failed = task_id in failed_tasks
            failed_tasks.discard(task_id)
            future = self._pending.pop(task_id, None)
            if future is None:
                self._orphans[task_id] = (value, failed)
            else:
                self._deliver(future, value, failed)

    def _is_outstanding(self, task_id: int) -> bool:
        """True if the controller still has this task queued, running or finished."""
        c = self._controller
        return (
            task_id in c.waiting
            or task_id in c.assigned
            or task_id in c.results
            or task_id in c.result_queue
        )

    def _wait(self, future: DistributedFuture, timeout: float | None = None) -> None:
        """Advance the controller until future is done or timeout seconds pass."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while not future._done:
            if not self._is_outstanding(future.task_id):
                raise RuntimeError(
                    f"task {future.task_id} is not known to the controller "
                    "and its result was never delivered"
                )
            if deadline is None:
                self._pump(block=True)
            else:
                self._pump(block=False)
                if future._done:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"task {future.task_id} did not finish within {timeout} s"
                    )
                time.sleep(self.poll_interval)

    def _close(self) -> None:
        """End the session.

        Waits for every pending future to receive its result, then tells
        the workers to return from serve().  Only the first call has any
        effect.
        """
        if self._closed:
            return
        self._closed = True
        while self._pending:
            for task_id in [t for t in self._pending if not self._is_outstanding(t)]:
                self._pending.pop(task_id)._fail(
                    RuntimeError(f"task {task_id} is not known to the controller"),
                    self._next_completion_index(),
                )
            if self._pending:
                self._pump(block=True)
        if self._controller:
            self._controller.end_session()

    def _get_worker_info(self, task_id: int) -> dict[str, Any] | None:
        """Get information about the worker or worker group processing this task"""
        if not self._controller:
            return None

        # Check if task is assigned
        if (
            hasattr(self._controller, "assigned")
            and task_id in self._controller.assigned
        ):
            worker_id = self._controller.assigned[task_id]
            info = {
                "worker_id": worker_id,
                "worker_rank": worker_id,
                "n_processed": (
                    self._controller.n_processed[worker_id]
                    if hasattr(self._controller, "n_processed")
                    and worker_id < len(self._controller.n_processed)
                    else 0
                ),
                "total_time": (
                    self._controller.total_time[worker_id]
                    if hasattr(self._controller, "total_time")
                    and worker_id < len(self._controller.total_time)
                    else 0.0
                ),
                "is_group_worker": (
                    self._env.worker_config.nprocs_per_worker > 1
                    if self._env.worker_config
                    else False
                ),
                "group_size": (
                    self._env.worker_config.nprocs_per_worker
                    if self._env.worker_config
                    else 1
                ),
                "grouping_method": (
                    self._env.worker_config.grouping_method.name
                    if self._env.worker_config
                    else "NoGrouping"
                ),
            }
            return info
        return None

    def _get_estimated_completion(self, task_id: int) -> float | None:
        """Get estimated completion time in seconds"""
        worker_info = self._get_worker_info(task_id)
        if worker_info:
            # Simple estimate based on worker's current load
            return worker_info.get("total_time", 0) + 1.0  # Add 1 second buffer
        return None

    def _get_task_progress(self, task_id: int) -> float | None:
        """Get task progress as a fraction between 0.0 and 1.0"""
        # Placeholder for progress tracking - would need enhancement
        # in the underlying MPI controller to support progress reporting
        if task_id in self._pending:
            elapsed = time.time() - self._pending[task_id]._submitted_at
            # Simple heuristic: assume 50% progress after 5 seconds
            return min(elapsed / 10.0, 0.95)  # Cap at 95% until actually complete
        return None


def _common_controller(futures: list[DistributedFuture]) -> TaskController | None:
    controllers = {id(f._context): f._context for f in futures}
    if len(controllers) > 1:
        raise ValueError("all futures must come from the same TaskController")
    return next(iter(controllers.values()), None)


def as_completed(
    futures: Iterable[DistributedFuture], timeout: float | None = None
) -> Iterator[DistributedFuture]:
    """Yield futures as they complete, in the order their results arrive.

    Futures that are already done are yielded first.  While waiting, the
    controller is advanced on the calling thread.  Raises TimeoutError if
    timeout seconds pass before every future is done.
    """
    remaining = list(dict.fromkeys(futures))
    controller = _common_controller(remaining)
    deadline = None if timeout is None else time.monotonic() + timeout
    while remaining:
        finished = sorted(
            (f for f in remaining if f._done), key=lambda f: f._completion_index
        )
        for f in finished:
            remaining.remove(f)
            yield f
        if not remaining:
            break
        for f in remaining:
            if not controller._is_outstanding(f.task_id):
                raise RuntimeError(
                    f"task {f.task_id} is not known to the controller "
                    "and its result was never delivered"
                )
        if deadline is None:
            controller._pump(block=True)
        else:
            controller._pump(block=False)
            if any(f._done for f in remaining):
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{len(remaining)} futures did not finish within {timeout} s"
                )
            time.sleep(controller.poll_interval)


def wait(
    futures: Iterable[DistributedFuture],
    timeout: float | None = None,
    return_when: str = ALL_COMPLETED,
) -> tuple[set, set]:
    """Wait for futures and return (done, not_done) sets.

    return_when is FIRST_COMPLETED or ALL_COMPLETED.  Unlike as_completed(),
    reaching the timeout is not an error: the futures still running are
    returned in not_done.
    """
    if return_when not in (FIRST_COMPLETED, ALL_COMPLETED):
        raise ValueError(f"unsupported return_when value {return_when!r}")
    pending = set(futures)
    done = set()
    try:
        for f in as_completed(pending, timeout=timeout):
            done.add(f)
            if return_when == FIRST_COMPLETED:
                break
    except TimeoutError:
        pass
    done |= {f for f in pending if f._done}
    return done, pending - done


# The environment that is currently entered on this rank, if any.
_active_environment: "DistributedEnvironment | None" = None


class DistributedEnvironment:
    """Distributed computing environment that coordinates MPI setup and task execution.

    All ranks enter the environment together.  Every rank runs on a single
    thread: the controller drives the work from controller_session(), and
    each worker rank serves calls on its main thread for the length of
    each session.  The controller ends a session by sending SESSION_END and
    ends the environment by sending EXIT, so every rank leaves each session
    and the environment at a well-defined point.

    Only one environment can be entered at a time.  Entering a second one
    inside the first raises RuntimeError on every rank, before any MPI
    call is made.
    """

    def __init__(
        self,
        config: Config | None = None,
        worker_config: WorkerGroupConfig | None = None,
    ):
        self.config = config or Config()
        self.worker_config = worker_config

        self._topology: Topology | None = None
        self._controller = None
        self._session: TaskController | None = None
        # Worker-side server: an MPIWorker, MPICollectiveBroker or
        # MPICollectiveWorker, depending on this rank's role.
        self._server = None
        # Set once the controller has told this worker rank to exit.
        self._exit_received = False
        self._active = False

    def __enter__(self) -> "DistributedEnvironment":  # noqa: PYI034
        """Set up distributed environment - ALL ranks participate"""
        self._setup_environment()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up distributed environment - ALL ranks participate"""
        self._cleanup_environment()
        return False

    @property
    def role(self) -> str | None:
        """This rank's role, as set by establish_topology(), or None
        before the environment is entered."""
        return self._topology.role if self._topology is not None else None

    def controller_session(self) -> Iterator[TaskController]:
        """Run one session of distributed work; all ranks must call this.

        On the controller rank, yields a single TaskController.  When the
        loop body finishes, by any means, the session waits for every
        pending result and then releases the workers.

        On worker ranks, yields nothing: the rank serves calls until the
        controller ends the session, then returns, so the loop body is
        skipped.
        """
        if not self._active:
            raise RuntimeError(
                "controller_session() must be called inside the environment"
            )
        if self.role == "controller":
            session = TaskController(self)
            self._session = session
            try:
                yield session
            finally:
                session._close()
                self._session = None
        else:
            self._serve()

    def _serve(self) -> MessageTag | None:
        """Serve calls on this worker rank until the controller says stop.

        Returns the tag that ended serving, or None if there is nothing to
        serve: on idle ranks, or because the controller has already sent
        EXIT.
        """
        if self._server is None or self._exit_received:
            return None
        tag = self._server.serve()
        if tag == MessageTag.EXIT:
            self._exit_received = True
        return tag

    def broadcast(self, value: Any = None) -> Any:
        """Return the controller's value on every rank; all ranks must call.

        Call it inside the environment but outside any session, for
        example right after the session loop, so that every rank reaches
        it.  The value given on worker ranks is ignored.
        """
        if not self._active:
            raise RuntimeError("broadcast() must be called inside the environment")
        if not has_mpi or world_comm.size < 2:
            return value
        payload = value if self.role == "controller" else None
        return world_comm.bcast(payload, root=controller_rank)

    def is_active(self) -> bool:
        """Check if the distributed environment is active"""
        return self._active

    @property
    def worker_count(self) -> int:
        """Get the number of workers in the environment"""
        if self._controller is not None and self._controller.workers_available:
            # Under worker grouping the controller's workers are the brokers.
            return self._controller.comm.size - 1
        if has_mpi and world_comm:
            return world_comm.size - 1  # Total ranks minus controller
        return 0

    @property
    def total_cores(self) -> int:
        """Get the total number of cores available"""
        worker_cores = self.worker_config.nprocs_per_worker if self.worker_config else 1
        return self.worker_count * worker_cores

    def _setup_environment(self):
        """Set up the distributed computing environment"""
        global _active_environment
        if self._active:
            return
        if _active_environment is not None:
            raise RuntimeError(
                "a DistributedEnvironment is already active; environments "
                "cannot be nested"
            )

        logger.info(
            "Setting up distributed environment: "
            f"is_controller={is_controller}, is_worker={is_worker}"
        )

        wc = self.worker_config or WorkerGroupConfig()
        self._topology = establish_topology(
            worker_grouping_method=wc.grouping_method,
            nprocs_per_worker=wc.nprocs_per_worker,
            broker_is_worker=wc.broker_is_worker,
            collective_mode=wc.collective_mode,
            enable_worker_service=wc.enable_worker_service,
            worker_service_name=wc.worker_service_name,
            sequential_spawn=wc.sequential_spawn,
            spawn_startup_wait=wc.spawn_startup_wait,
            spawn_executable=wc.spawn_executable,
            spawn_args=wc.spawn_args,
            verbose=getattr(self.config.logging, "verbose", False),
            # A task's exception is raised from its future's result()
            # instead of aborting the job.
            capture_errors=True,
        )
        logger.info(f"Topology established: role={self._topology.role}")

        if self._topology.role == "controller":
            self._controller = MPIController(
                self._topology.controller_worker_comm,
                time_limit=getattr(self.config.execution, "timeout", None),
            )
        else:
            self._server = self._topology.server

        self._active = True
        _active_environment = self

    def _cleanup_environment(self):
        """Clean up the distributed environment"""
        global _active_environment
        if not self._active:
            return

        try:
            if self._controller:
                # A session left open, for example by an exception raised
                # while its generator was suspended, is closed first.
                if self._session is not None:
                    self._session._close()
                    self._session = None
                self._controller.shutdown()
                self._controller = None
            else:
                # Serve once more so that the controller hears from this rank
                # and can tell it to exit.
                tag = self._serve()
                if tag == MessageTag.SESSION_END:
                    raise RuntimeError(
                        "worker received SESSION_END while leaving the "
                        "environment: the controller ran more sessions than "
                        "this rank"
                    )
        finally:
            self._active = False
            if _active_environment is self:
                _active_environment = None
        logger.info("Distributed environment cleanup complete")


# Convenience factory functions following functional programming patterns
def create_spawn_environment(
    nprocs_per_worker: int = 2,
    broker_is_worker: bool = True,
    config: Config | None = None,
) -> DistributedEnvironment:
    """Create distributed environment with spawn-based worker grouping"""
    worker_config = WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSpawn,
        nprocs_per_worker=nprocs_per_worker,
        broker_is_worker=broker_is_worker,
    )
    return DistributedEnvironment(config, worker_config)


def create_split_environment(
    nprocs_per_worker: int = 2,
    broker_is_worker: bool = True,
    config: Config | None = None,
) -> DistributedEnvironment:
    """Create distributed environment with split-based worker grouping"""
    worker_config = WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSplit,
        nprocs_per_worker=nprocs_per_worker,
        broker_is_worker=broker_is_worker,
    )
    return DistributedEnvironment(config, worker_config)


# Auto-initialization for workers
if __name__ == "__main__":
    # When this module is run directly (e.g., by spawned workers)
    if is_worker and has_mpi:
        worker_main()
