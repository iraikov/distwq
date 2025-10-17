# Future-based distributed task management with Two-Phase Iterator design

import sys
import time
import threading
import logging
from concurrent.futures import Future, as_completed
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
    from .config import Config, WorkerGroupConfig
except ImportError:
    try:
        from config import Config, WorkerGroupConfig
    except ImportError:
        # Fallback config implementation
        class Config:
            def __init__(self):
                self.resources = type('ResourceConfig', (), {'max_workers': -1})()
                self.execution = type('ExecutionConfig', (), {
                    'timeout': None, 'task_timeout': None, 'retry_attempts': 0, 'fail_fast': True
                })()
                self.logging = type('LoggingConfig', (), {'verbose': False, 'log_level': 'INFO'})()
                self.mpi = type('MPIConfig', (), {
                    'collective_mode': 'gather', 'broker_is_worker': False
                })()
        
        class WorkerGroupConfig:
            def __init__(self, grouping_method=None, nprocs_per_worker=1, broker_is_worker=False, **kwargs):
                self.grouping_method = grouping_method or GroupingMethod.NoGrouping
                self.nprocs_per_worker = nprocs_per_worker
                self.broker_is_worker = broker_is_worker
                self.collective_mode = kwargs.get('collective_mode', 'gather')
                self.enable_worker_service = kwargs.get('enable_worker_service', False)
                self.worker_service_name = kwargs.get('worker_service_name', 'distwq.init')
                self.sequential_spawn = kwargs.get('sequential_spawn', False)
                self.spawn_startup_wait = kwargs.get('spawn_startup_wait', None)
                self.spawn_executable = kwargs.get('spawn_executable', None)
                self.spawn_args = kwargs.get('spawn_args', [])


T = TypeVar("T")
R = TypeVar("R")

logger = logging.getLogger(__name__)

class DistributedFuture(Future, Generic[R]):
    """Enhanced Future with distributed computing features and progress monitoring"""

    def __init__(
        self,
        task_id: int,
        context: "TaskController",
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

    def progress(self) -> Optional[float]:
        """Get task progress as a fraction between 0.0 and 1.0"""
        return self._context._get_task_progress(self.task_id)

    def is_group_task(self) -> bool:
        """Check if this task is being processed by a worker group"""
        return self.worker_group_size > 1


class TaskController:
    """Interface for coordinating distributed tasks - controller rank only"""

    def __init__(self, execution_context: "DistributedEnvironment"):
        if not is_controller:
            raise RuntimeError("TaskController can only be instantiated on controller rank")
        
        self._env = execution_context
        self._controller = execution_context._controller
        self._pending_futures = {}  # Map task_id to future for completion tracking
        self._monitoring_thread = None
        self._shutdown_requested = False

        self.exit_timeout = 5.0
        # Start monitoring thread for futures completion
        self._start_monitoring_thread()

    def submit(self, func: Callable[..., R], *args, timeout: Optional[int] = None, **kwargs) -> DistributedFuture[R]:
        """Submit a single task for distributed execution"""
        if not self._controller:
            raise RuntimeError("No controller available for task submission")

        # Determine effective worker group size
        effective_group_size = (
            self._env.worker_config.nprocs_per_worker 
            if self._env.worker_config 
            else 1
        )

        task_id = self._controller.submit_call(
            func.__name__,
            args=args,
            kwargs=kwargs,
            module_name=func.__module__,
            time_est=self._estimate_task_time(func, args, kwargs),
        )
        logger.info(f"Submitted task id {task_id}")
        
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
        chunk_size: Optional[int] = None,
    ) -> Iterator[R]:
        """Map function over iterable, yielding results as they complete"""
        tasks = [(func, (item,), {}) for item in iterable]
        futures = self.batch_submit(tasks)

        # Yield results as they become available
        for future in as_completed(futures):
            yield future.result()

    def batch_submit(
        self, tasks: List[Tuple[Callable, tuple, dict]]
    ) -> List[DistributedFuture]:
        """Submit multiple tasks efficiently"""
        if not self._controller:
            raise RuntimeError("No controller available for task submission")

        futures = []
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

    def active_worker_count(self) -> int:
        """Get the number of active workers"""
        if self._controller and hasattr(self._controller, 'n_workers'):
            return self._controller.n_workers
        return 0

    def _estimate_task_time(self, func: Callable, args: tuple, kwargs: dict) -> int:
        """Estimate task execution time for scheduling"""
        # Simple heuristic - can be made more sophisticated
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
                logger.debug("Checking for completed tasks...")
                result = self._controller.probe_next_result()
                logger.debug(f"Result: {result}")

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
                logger.error(f"Error in monitoring thread: {e}")
                # Handle errors by setting exception on all pending futures
                for future in self._pending_futures.values():
                    if not future.done():
                        future.set_exception(e)
                self._pending_futures.clear()
                break

    def _shutdown(self):
        """Internal shutdown method"""
        self._shutdown_requested = True
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=self.exit_timeout)

    def _get_worker_info(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Get information about the worker or worker group processing this task"""
        if not self._controller:
            return None

        # Check if task is assigned
        if hasattr(self._controller, 'assigned') and task_id in self._controller.assigned:
            worker_id = self._controller.assigned[task_id]
            info = {
                "worker_id": worker_id,
                "worker_rank": worker_id,
                "n_processed": (
                    self._controller.n_processed[worker_id]
                    if hasattr(self._controller, 'n_processed') and worker_id < len(self._controller.n_processed)
                    else 0
                ),
                "total_time": (
                    self._controller.total_time[worker_id]
                    if hasattr(self._controller, 'total_time') and worker_id < len(self._controller.total_time)
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

    def _get_estimated_completion(self, task_id: int) -> Optional[float]:
        """Get estimated completion time in seconds"""
        worker_info = self._get_worker_info(task_id)
        if worker_info:
            # Simple estimate based on worker's current load
            return worker_info.get("total_time", 0) + 1.0  # Add 1 second buffer
        return None

    def _get_task_progress(self, task_id: int) -> Optional[float]:
        """Get task progress as a fraction between 0.0 and 1.0"""
        # Placeholder for progress tracking - would need enhancement
        # in the underlying MPI controller to support progress reporting
        if task_id in self._pending_futures:
            elapsed = time.time() - self._pending_futures[task_id]._submitted_at
            # Simple heuristic: assume 50% progress after 5 seconds
            return min(elapsed / 10.0, 0.95)  # Cap at 95% until actually complete
        return None


class DistributedEnvironment:
    """Distributed computing environment that coordinates MPI setup and task execution"""

    def __init__(
        self, 
        config: Optional[Config] = None, 
        worker_config: Optional[WorkerGroupConfig] = None
    ):
        self.config = config or Config()
        self.worker_config = worker_config
        
        self._controller = None
        self._worker = None
        self._active = False
        self._group_comm = None
        self._worker_threads = []

        # For worker grouping
        self._is_broker = False
        self._worker_id = None

        self.exit_timeout = 2.0
        
    def __enter__(self) -> "DistributedEnvironment":
        """Set up distributed environment - ALL ranks participate"""
        self._setup_environment()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up distributed environment - ALL ranks participate"""
        self._cleanup_environment()
        return False

    def controller_session(self) -> Iterator[TaskController]:
        """Return iterator that yields TaskController only on controller rank"""
        if is_controller:
            # Only controller gets the controller task
            yield TaskController(self)
        else:
            return iter([])
#        elif is_worker and self._worker is not None:
#            self._worker.serve()  # Workers block here
#            return iter([])

    def is_active(self) -> bool:
        """Check if the distributed environment is active"""
        return self._active

    @property
    def worker_count(self) -> int:
        """Get the number of workers in the environment"""
        if has_mpi and world_comm:
            return world_comm.size - 1  # Total ranks minus controller
        return 0

    @property
    def total_cores(self) -> int:
        """Get the total number of cores available"""
        worker_cores = (
            self.worker_config.nprocs_per_worker 
            if self.worker_config 
            else 1
        )
        return self.worker_count * worker_cores

    def _setup_environment(self):
        """Set up the distributed computing environment"""
        if self._active:
            return

        logger.info(f"Setting up distributed environment: is_controller={is_controller}, is_worker={is_worker}")

        # All ranks participate in MPI collective setup
        if has_mpi:
            self.group_id, self._group_comm = get_group_comm(
                world_comm, world_comm.size, rank, is_controller, spawned=False
            )
            logger.info(f"Group setup complete: group_id={self.group_id}")

        # Set up controller
        if is_controller:
            self._controller = self._create_controller()
            logger.info("Controller created")

        # Set up workers
        elif is_worker:
            self._setup_workers()
            logger.info("Worker setup complete")

        self._active = True

    def _setup_workers(self):
        """Set up workers based on configuration"""
        if not has_mpi:
            return

        if self.worker_config and self.worker_config.grouping_method == GroupingMethod.GroupSpawn:
            self._setup_spawn_groups()
        elif self.worker_config and self.worker_config.grouping_method == GroupingMethod.GroupSplit:
            self._setup_split_groups()
        else:
            # Regular worker setup
            self._setup_regular_worker()

    def _setup_regular_worker(self):
        """Set up a regular worker (no grouping)"""
        logger.info("Creating regular worker...")
        self._worker = self._create_worker()
        if self._worker:
            # Start worker main loop in background
            worker_thread = threading.Thread(target=self._run_worker, daemon=True)
            worker_thread.start()
            self._worker_threads.append(worker_thread)

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
            target=self._run_spawn_group, args=(controller_worker_comm,), daemon=True
        )
        spawn_thread.start()
        self._worker_threads.append(spawn_thread)

    def _setup_split_groups(self):
        """Set up worker groups using split method"""
        if not self._group_comm:
            return

        # Determine worker configuration
        if self.worker_config.broker_is_worker:
            n_workers = self._group_comm.size // self.worker_config.nprocs_per_worker
            self._worker_id = (
                self._group_comm.rank // self.worker_config.nprocs_per_worker
            ) + 1
            broker_set = {
                (x * self.worker_config.nprocs_per_worker) for x in range(n_workers)
            }
        else:
            n_workers = self._group_comm.size // (self.worker_config.nprocs_per_worker + 1)
            self._worker_id = (
                self._group_comm.rank // (self.worker_config.nprocs_per_worker + 1)
            ) + 1
            broker_set = {
                (x * (self.worker_config.nprocs_per_worker + 1)) for x in range(n_workers)
            }

        self._is_broker = self._group_comm.rank in broker_set

        # Create controller-worker communication
        color = 1 if self._is_broker or is_controller else 2
        controller_worker_comm = world_comm.Split(color, key=0 if is_controller else 1)

        # Set up split group in background thread
        split_thread = threading.Thread(
            target=self._run_split_group,
            args=(controller_worker_comm, broker_set, n_workers),
            daemon = True
        )
        split_thread.start()
        self._worker_threads.append(split_thread)

    def _run_spawn_group(self, controller_worker_comm):
        """Run spawn group setup and management"""
        run_spawn_group(
            worker_id=self._worker_id,
            spawn_workers=True,
            nprocs_per_worker=self.worker_config.nprocs_per_worker,
            broker_is_worker=self.worker_config.broker_is_worker,
            broker_fun=None,
            fun=None,
            fun_name=None,
            module_name="__main__",
            args=(),
            world_comm=controller_worker_comm,
            group_comm=self._group_comm,
            n_workers=self._group_comm.size,
            collective_mode=self.worker_config.collective_mode,
            enable_worker_service=self.worker_config.enable_worker_service,
            worker_service_name=self.worker_config.worker_service_name,
            spawn_args=self.worker_config.spawn_args,
            sequential_spawn=self.worker_config.sequential_spawn,
            spawn_startup_wait=self.worker_config.spawn_startup_wait,
            spawn_executable=self.worker_config.spawn_executable,
            verbose=getattr(self.config.logging, "verbose", False),
        )

    def _run_split_group(self, controller_worker_comm, broker_set, n_workers):
        """Run split group setup and management"""
        run_split_group(
            worker_id=self._worker_id,
            n_workers=n_workers,
            split_workers=True,
            is_broker=self._is_broker,
            broker_ranks=tuple(broker_set),
            nprocs_per_worker=self.worker_config.nprocs_per_worker,
            broker_is_worker=self.worker_config.broker_is_worker,
            broker_fun=None,
            fun=None,
            fun_name=None,
            module_name="__main__",
            args=(),
            world_comm=world_comm,
            controller_worker_comm=controller_worker_comm,
            group_comm=self._group_comm,
            collective_mode=self.worker_config.collective_mode,
            verbose=getattr(self.config.logging, "verbose", False),
        )

    def _create_controller(self):
        """Create controller instance"""
        if not is_controller:
            raise RuntimeError("_create_controller() can only be called by controller rank")

        if not has_mpi:
            # Single process mode
            return MPIController(
                world_comm, 
                time_limit=getattr(self.config.execution, "timeout", None)
            )

        # MPI mode with potential worker grouping
        controller_comm = world_comm

        # If using worker grouping, create separate controller-worker communication
        if (
            self.worker_config 
            and self.worker_config.grouping_method in [GroupingMethod.GroupSpawn, GroupingMethod.GroupSplit]
            and self.worker_config.nprocs_per_worker > 1
        ):
            color = 1 if is_controller else 2
            controller_comm = world_comm.Split(color, key=0 if is_controller else 1)

        return MPIController(
            controller_comm, 
            time_limit=getattr(self.config.execution, "timeout", None)
        )

    def _create_worker(self):
        """Create worker instance - only invoked by worker ranks"""
        if not is_worker:
            raise RuntimeError("_create_worker() can only be called by worker ranks")

        if not has_mpi:
            return None  # No workers in single process mode

        return MPIWorker(world_comm, self._group_comm or world_comm, ready_data=None)

    def _run_worker(self):
        """Run worker main loop - called by workers at initialization time"""
        if self._worker:
            self._worker.serve()

    def _cleanup_environment(self):
        """Clean up the distributed environment"""
        if not self._active:
            return

        # Clean up controller
        if self._controller:
            # Shutdown any active task controllers
            if hasattr(self, '_active_controllers'):
                for controller in getattr(self, '_active_controllers', []):
                    controller._shutdown()
            
            self._controller.exit()
            self._controller = None

        # Wait for worker threads to complete
        for thread in self._worker_threads:
            if thread.is_alive():
                thread.join(timeout=self.exit_timeout)
                    

        self._active = False
        logger.info("Distributed environment cleanup complete")


# Convenience factory functions following functional programming patterns
def create_spawn_environment(
    nprocs_per_worker: int = 2, 
    broker_is_worker: bool = True, 
    config: Optional[Config] = None
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
    config: Optional[Config] = None
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
