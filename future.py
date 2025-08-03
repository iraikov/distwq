# Future-based distributed task management with worker grouping support

import time
import threading
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
    Union,
)

# Import from mpi.py to access MPI functionality
try:
    from .mpi import (
        MPIController,
        MPIWorker,
        GroupingMethod,
        is_controller,
        is_worker,
        rank,
        world_comm,
        has_mpi,
        worker_main,
        run_spawn_group,
        run_split_group,
        get_group_comm,
        check_spawn_config,
        check_split_config,
    )
except ImportError:
    # Try absolute import
    from mpi import (
        MPIController,
        MPIWorker,
        GroupingMethod,
        is_controller,
        is_worker,
        rank,
        world_comm,
        has_mpi,
        worker_main,
        run_spawn_group,
        run_split_group,
        get_group_comm,
        check_spawn_config,
        check_split_config,
    )

# Import config if available, otherwise provide basic fallback
try:
    from .config import Config
except ImportError:
    # Fallback config implementation
    class Config:
        def __init__(self):
            self.time_limit = None
            self.verbose = False
            self.max_workers = -1
            self.worker_grouping_method = GroupingMethod.NoGrouping
            self.nprocs_per_worker = 1
            self.broker_is_worker = False
            self.collective_mode = "gather"
            self.enable_worker_service = False
            self.worker_service_name = "distwq.init"
            self.sequential_spawn = False
            self.spawn_startup_wait = None
            self.spawn_executable = None
            self.spawn_args = []


T = TypeVar("T")
R = TypeVar("R")


class WorkerGroupConfig:
    """Configuration for worker grouping"""

    def __init__(
        self,
        grouping_method: Union[str, GroupingMethod] = GroupingMethod.NoGrouping,
        nprocs_per_worker: int = 1,
        broker_is_worker: bool = False,
        collective_mode: str = "gather",
        enable_worker_service: bool = False,
        worker_service_name: str = "distwq.init",
        sequential_spawn: bool = False,
        spawn_startup_wait: Optional[int] = None,
        spawn_executable: Optional[str] = None,
        spawn_args: List[Any] = None,
    ):
        if isinstance(grouping_method, str):
            if grouping_method.lower() == "spawn":
                self.grouping_method = GroupingMethod.GroupSpawn
            elif grouping_method.lower() == "split":
                self.grouping_method = GroupingMethod.GroupSplit
            elif grouping_method.lower() == "none":
                self.grouping_method = GroupingMethod.NoGrouping
            else:
                raise ValueError(f"Unknown grouping method: {grouping_method}")
        else:
            self.grouping_method = grouping_method

        self.nprocs_per_worker = nprocs_per_worker
        self.broker_is_worker = broker_is_worker
        self.collective_mode = collective_mode
        self.enable_worker_service = enable_worker_service
        self.worker_service_name = worker_service_name
        self.sequential_spawn = sequential_spawn
        self.spawn_startup_wait = spawn_startup_wait
        self.spawn_executable = spawn_executable
        self.spawn_args = spawn_args or []

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate the worker group configuration"""
        if self.grouping_method == GroupingMethod.GroupSpawn:
            check_spawn_config(
                spawn_workers=True,
                nprocs_per_worker=self.nprocs_per_worker,
                broker_is_worker=self.broker_is_worker,
                enable_worker_service=self.enable_worker_service,
                broker_fun=None,  # We don't use broker_fun in futures interface
            )
        elif self.grouping_method == GroupingMethod.GroupSplit:
            if has_mpi:
                check_split_config(
                    split_workers=True,
                    nprocs_per_worker=self.nprocs_per_worker,
                    broker_is_worker=self.broker_is_worker,
                    broker_fun=None,  # We don't use broker_fun in futures interface
                    world_size=world_comm.size if world_comm else 1,
                )


class DistributedFuture(Future, Generic[R]):
    """Enhanced Future with distributed computing features and worker group support"""

    def __init__(
        self,
        task_id: int,
        context: "DistributedTaskExecution",
        worker_group_size: int = 1,
    ):
        super().__init__()
        self.task_id = task_id
        self._context = context
        self._submitted_at = time.time()
        self.worker_group_size = worker_group_size

    def worker_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the worker or worker group processing this task"""
        return self._context._get_worker_info(self.task_id)

    def estimated_completion(self) -> Optional[float]:
        """Get estimated completion time in seconds"""
        return self._context._get_estimated_completion(self.task_id)

    def is_group_task(self) -> bool:
        """Check if this task is being processed by a worker group"""
        return self.worker_group_size > 1


class DistributedTaskExecution:
    """Main interface for distributed task execution with worker grouping support"""

    def __init__(
        self, config: Config = None, worker_group_config: WorkerGroupConfig = None
    ):
        self.config = config or Config()
        self.worker_group_config = worker_group_config or WorkerGroupConfig()

        self._controller = None
        self._worker = None
        self._broker = None
        self._active = False
        self._pending_futures = {}  # Map task_id to future for completion tracking
        self._monitoring_thread = None
        self._shutdown_requested = False

        # Worker grouping state
        self._is_broker = False
        self._worker_id = None
        self._group_comm = None
        self._merged_comm = None

        # Extract grouping config from main config if present
        if hasattr(self.config, "worker_grouping_method"):
            self.worker_group_config.grouping_method = getattr(
                self.config, "worker_grouping_method", GroupingMethod.NoGrouping
            )
        if hasattr(self.config, "nprocs_per_worker"):
            self.worker_group_config.nprocs_per_worker = getattr(
                self.config, "nprocs_per_worker", 1
            )
        if hasattr(self.config, "broker_is_worker"):
            self.worker_group_config.broker_is_worker = getattr(
                self.config, "broker_is_worker", False
            )

    def __enter__(self) -> "DistributedTaskExecution":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

    def start(self):
        """Initialize the distributed environment with worker grouping support"""
        if not self._active:
            if is_controller:
                self._controller = self._create_controller()
            elif is_worker:
                self._setup_worker_groups()

            self._active = True

            # Start monitoring thread for futures completion
            if self._controller:
                self._start_monitoring_thread()

    def _setup_worker_groups(self):
        """Set up worker groups based on configuration"""
        if not has_mpi:
            return

        group_id, self._group_comm = get_group_comm(
            world_comm, world_comm.size, rank, is_controller, spawned=False
        )

        if self.worker_group_config.grouping_method == GroupingMethod.GroupSpawn:
            self._setup_spawn_groups()
        elif self.worker_group_config.grouping_method == GroupingMethod.GroupSplit:
            self._setup_split_groups()
        else:
            # Regular worker setup
            self._worker = self._create_worker()
            if self._worker:
                # Start worker main loop in background
                worker_thread = threading.Thread(target=self._run_worker)
                worker_thread.daemon = True
                worker_thread.start()

    def _setup_spawn_groups(self):
        """Set up worker groups using spawn method"""
        if not self._group_comm:
            return

        self._worker_id = self._group_comm.rank + 1
        self._is_broker = True

        # Create controller-worker communication
        color = 1 if self._is_broker or is_controller else 2
        controller_worker_comm = world_comm.Split(color, key=0 if is_controller else 1)

        # Set up spawn group in background thread
        spawn_thread = threading.Thread(
            target=self._run_spawn_group, args=(controller_worker_comm,)
        )
        spawn_thread.daemon = True
        spawn_thread.start()

    def _setup_split_groups(self):
        """Set up worker groups using split method"""
        if not self._group_comm:
            return

        # Determine worker configuration
        if self.worker_group_config.broker_is_worker:
            n_workers = (
                self._group_comm.size // self.worker_group_config.nprocs_per_worker
            )
            self._worker_id = (
                self._group_comm.rank // self.worker_group_config.nprocs_per_worker
            ) + 1
            broker_set = {
                (x * self.worker_group_config.nprocs_per_worker)
                for x in range(n_workers)
            }
        else:
            n_workers = self._group_comm.size // (
                self.worker_group_config.nprocs_per_worker + 1
            )
            self._worker_id = (
                self._group_comm.rank
                // (self.worker_group_config.nprocs_per_worker + 1)
            ) + 1
            broker_set = {
                (x * (self.worker_group_config.nprocs_per_worker + 1))
                for x in range(n_workers)
            }

        self._is_broker = self._group_comm.rank in broker_set

        # Create controller-worker communication
        color = 1 if self._is_broker or is_controller else 2
        controller_worker_comm = world_comm.Split(color, key=0 if is_controller else 1)

        # Set up split group in background thread
        split_thread = threading.Thread(
            target=self._run_split_group,
            args=(controller_worker_comm, broker_set, n_workers),
        )
        split_thread.daemon = True
        split_thread.start()

    def _run_spawn_group(self, controller_worker_comm):
        """Run spawn group setup and management"""
        run_spawn_group(
            worker_id=self._worker_id,
            spawn_workers=True,
            nprocs_per_worker=self.worker_group_config.nprocs_per_worker,
            broker_is_worker=self.worker_group_config.broker_is_worker,
            broker_fun=None,  # No separate broker function for futures interface
            fun=None,  # No initialization function
            fun_name=None,
            module_name="__main__",
            args=(),
            world_comm=controller_worker_comm,
            group_comm=self._group_comm,
            n_workers=self._group_comm.size,
            collective_mode=self.worker_group_config.collective_mode,
            enable_worker_service=self.worker_group_config.enable_worker_service,
            worker_service_name=self.worker_group_config.worker_service_name,
            spawn_args=self.worker_group_config.spawn_args,
            sequential_spawn=self.worker_group_config.sequential_spawn,
            spawn_startup_wait=self.worker_group_config.spawn_startup_wait,
            spawn_executable=self.worker_group_config.spawn_executable,
            verbose=getattr(self.config, "verbose", False),
        )

    def _run_split_group(self, controller_worker_comm, broker_set, n_workers):
        """Run split group setup and management"""
        run_split_group(
            worker_id=self._worker_id,
            n_workers=n_workers,
            split_workers=True,
            is_broker=self._is_broker,
            broker_ranks=tuple(broker_set),
            nprocs_per_worker=self.worker_group_config.nprocs_per_worker,
            broker_is_worker=self.worker_group_config.broker_is_worker,
            broker_fun=None,  # No separate broker function for futures interface
            fun=None,  # No initialization function
            fun_name=None,
            module_name="__main__",
            args=(),
            world_comm=world_comm,
            controller_worker_comm=controller_worker_comm,
            group_comm=self._group_comm,
            collective_mode=self.worker_group_config.collective_mode,
            verbose=getattr(self.config, "verbose", False),
        )

    def shutdown(self):
        """Clean shutdown of distributed environment"""
        if self._active:
            self._shutdown_requested = True

            if self._controller:
                self._controller.exit()

            if self._monitoring_thread:
                self._monitoring_thread.join(timeout=5.0)

            self._active = False

    def _create_controller(self):
        """Create controller instance with worker grouping support"""
        if not is_controller:
            raise RuntimeError(
                "_create_controller() can only be called by controller rank"
            )

        if not has_mpi:
            # Single process mode
            return MPIController(
                world_comm, time_limit=getattr(self.config, "time_limit", None)
            )

        # MPI mode with potential worker grouping
        controller_comm = world_comm

        # If using worker grouping, create separate controller-worker communication
        if (
            self.worker_group_config.grouping_method
            in [GroupingMethod.GroupSpawn, GroupingMethod.GroupSplit]
            and self.worker_group_config.nprocs_per_worker > 1
        ):
            color = 1 if is_controller else 2
            controller_comm = world_comm.Split(color, key=0 if is_controller else 1)

        return MPIController(
            controller_comm, time_limit=getattr(self.config, "time_limit", None)
        )

    def _create_worker(self):
        """Create worker instance - only invoked by worker ranks"""
        if not is_worker:
            raise RuntimeError("create_worker() can only be called by worker ranks")

        if not has_mpi:
            return None  # No workers in single process mode

        # For regular workers (no grouping), create simple worker
        return MPIWorker(world_comm, self._group_comm or world_comm, ready_data=None)

    def _run_worker(self):
        """Run worker main loop - called by workers at initialization time"""
        if self._worker:
            self._worker.serve()

    def submit(
        self, func: Callable[..., R], *args, timeout: Optional[int] = None, **kwargs
    ) -> DistributedFuture[R]:
        """Submit a single task for execution"""
        if not self._active:
            raise RuntimeError(
                "DistributedTaskExecution not started. Use with statement or call start()"
            )

        if not self._controller:
            raise RuntimeError("No controller available for task submission")

        # Determine effective worker group size
        effective_group_size = self.worker_group_config.nprocs_per_worker

        task_id = self._controller.submit_call(
            func.__name__,
            args=args,
            kwargs=kwargs,
            module_name=func.__module__,
            time_est=self._estimate_task_time(func, args, kwargs),
        )

        future = DistributedFuture(
            task_id, self, worker_group_size=effective_group_size
        )
        self._setup_future_completion(future)
        return future

    def map(
        self,
        func: Callable[[T], R],
        iterable: Iterable[T],
        chunk_size: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Iterator[R]:
        """Map function over iterable, optionally using worker groups"""

        tasks = []
        for item in iterable:
            tasks.append((func, (item,), {}))  # func, args tuple, kwargs dict

        futures = self.batch_submit(tasks)

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

        tasks = []
        for item in iterable:
            tasks.append((func, (item,), {}))  # func, args tuple, kwargs dict

        futures = self.batch_submit(tasks)

        # Yield results as they become available
        from concurrent.futures import as_completed

        for future in as_completed(futures):
            yield future.result()

    def batch_submit(
        self, tasks: List[Tuple[Callable, tuple, dict]]
    ) -> List[DistributedFuture]:
        """Submit multiple heterogeneous tasks efficiently"""
        if not self._controller:
            raise RuntimeError("No controller available for task submission")

        futures = []

        # Handle heterogeneous tasks - submit each individually for now
        # TODO: Optimize with submit_multiple for homogeneous tasks
        for task in tasks:
            func, args, kwargs = task
            task_id = self._controller.submit_call(
                func.__name__,
                args=args,
                kwargs=kwargs,
                module_name=func.__module__,
                time_est=self._estimate_task_time(func, args, kwargs),
            )

            future = DistributedFuture(task_id, self, worker_group_size=1)
            self._setup_future_completion(future)
            futures.append(future)

        return futures

    def _estimate_task_time(self, func: Callable, args: tuple, kwargs: dict) -> int:
        """Estimate task execution time for scheduling"""
        # Simple heuristic - can be made more sophisticated
        # For now, return a default estimate
        return 1

    def _setup_future_completion(self, future: DistributedFuture):
        """Set up future to be completed when task finishes"""
        self._pending_futures[future.task_id] = future

    def _start_monitoring_thread(self):
        """Start background thread to monitor task completion"""
        if self._monitoring_thread is None:
            self._monitoring_thread = threading.Thread(target=self._monitor_completions)
            self._monitoring_thread.daemon = True
            self._monitoring_thread.start()

    def _monitor_completions(self):
        """Monitor task completions and update futures"""
        while not self._shutdown_requested and self._controller:
            try:
                # Check for completed tasks
                result = self._controller.probe_next_result()

                if result is not None:
                    task_id, task_result = result

                    # Complete the corresponding future
                    if task_id in self._pending_futures:
                        future = self._pending_futures[task_id]
                        future.set_result(task_result)
                        del self._pending_futures[task_id]
                else:
                    # No results ready, sleep briefly
                    time.sleep(0.1)

            except Exception as e:
                # Handle errors by setting exception on all pending futures
                for future in self._pending_futures.values():
                    if not future.done():
                        future.set_exception(e)
                self._pending_futures.clear()
                break

    def _get_worker_info(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Get information about the worker or worker group processing this task"""
        if not self._controller:
            return None

        # Check if task is assigned
        if task_id in self._controller.assigned:
            worker_id = self._controller.assigned[task_id]
            info = {
                "worker_id": worker_id,
                "worker_rank": worker_id,
                "n_processed": self._controller.n_processed[worker_id]
                if worker_id < len(self._controller.n_processed)
                else 0,
                "total_time": self._controller.total_time[worker_id]
                if worker_id < len(self._controller.total_time)
                else 0.0,
                "is_group_worker": self.worker_group_config.nprocs_per_worker > 1,
                "group_size": self.worker_group_config.nprocs_per_worker,
                "grouping_method": self.worker_group_config.grouping_method.name,
            }
            return info
        return None

    def _get_estimated_completion(self, task_id: int) -> Optional[float]:
        """Get estimated completion time in seconds"""
        worker_info = self._get_worker_info(task_id)
        if worker_info:
            # Simple estimate based on worker's current load
            # This is a rough heuristic
            return worker_info.get("total_time", 0) + 1.0  # Add 1 second buffer
        return None


# Context manager for automatic resource management
class DistributedWorkQueue:
    """High-level distributed work queue interface with worker grouping support"""

    def __init__(
        self, config: Config = None, worker_group_config: WorkerGroupConfig = None
    ):
        self.config = config or Config()
        self.worker_group_config = worker_group_config or WorkerGroupConfig()
        self._context = None

    def __enter__(self) -> DistributedTaskExecution:
        self._context = DistributedTaskExecution(self.config, self.worker_group_config)
        return self._context.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._context:
            self._context.__exit__(exc_type, exc_val, exc_tb)


# Convenience factory functions
def create_spawn_queue(
    nprocs_per_worker: int = 2, broker_is_worker: bool = True, config: Config = None
) -> DistributedWorkQueue:
    """Create a distributed work queue using spawn-based worker grouping"""
    worker_config = WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSpawn,
        nprocs_per_worker=nprocs_per_worker,
        broker_is_worker=broker_is_worker,
    )
    return DistributedWorkQueue(config, worker_config)


def create_split_queue(
    nprocs_per_worker: int = 2, broker_is_worker: bool = True, config: Config = None
) -> DistributedWorkQueue:
    """Create a distributed work queue using split-based worker grouping"""
    worker_config = WorkerGroupConfig(
        grouping_method=GroupingMethod.GroupSplit,
        nprocs_per_worker=nprocs_per_worker,
        broker_is_worker=broker_is_worker,
    )
    return DistributedWorkQueue(config, worker_config)


# Ensure worker_main is called for workers at initialization time
def initialize_distributed_environment():
    """Initialize the distributed environment for workers"""
    if is_worker and has_mpi:
        # This will be called automatically when the module is imported by workers
        worker_main()


# Auto-initialization for workers
if __name__ == "__main__":
    # When this module is run directly (e.g., by spawned workers)
    initialize_distributed_environment()
