# Hierarchical Configuration Design

import sys
import importlib
from dataclasses import dataclass, replace
from typing import Optional, Callable
from enum import Enum


def import_object_by_path(path):
    module_path, _, obj_name = path.rpartition(".")
    if module_path == "__main__" or module_path == "":
        module = sys.modules["__main__"]
    else:
        module = importlib.import_module(module_path)
    return getattr(module, obj_name)


class SpawnMethod(Enum):
    AUTO = "auto"  # Let distwq choose best method
    SPAWN = "spawn"  # MPI_Comm_spawn
    SPLIT = "split"  # MPI_Comm_split
    THREADS = "threads"  # Thread-based fallback


class CollectiveMode(Enum):
    GATHER = "gather"
    SENDRECV = "sendrecv"


@dataclass(frozen=True)  # Immutable for functional programming
class ResourceConfig:
    """Resource allocation configuration"""

    max_workers: int = -1  # -1 = use all available
    cores_per_worker: int = 1
    memory_per_worker: Optional[str] = None  # e.g., "2GB", "512MB"
    gpu_per_worker: int = 0

    def with_workers(self, n: int) -> "ResourceConfig":
        return replace(self, max_workers=n)

    def with_cores(self, n: int) -> "ResourceConfig":
        return replace(self, cores_per_worker=n)


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution behavior configuration"""

    timeout: Optional[int] = None  # Total execution timeout
    task_timeout: Optional[int] = None  # Per-task timeout
    retry_attempts: int = 0  # Retry failed tasks
    fail_fast: bool = True  # Stop on first failure
    progress_callback: Optional[Callable] = None

    def with_timeout(self, seconds: int) -> "ExecutionConfig":
        return replace(self, timeout=seconds)

    def with_retries(self, attempts: int) -> "ExecutionConfig":
        return replace(self, retry_attempts=attempts)


@dataclass(frozen=True)
class MPIConfig:
    """MPI implementation configuration"""

    spawn_method: SpawnMethod = SpawnMethod.AUTO
    collective_mode: CollectiveMode = CollectiveMode.GATHER
    broker_is_worker: bool = False
    sequential_spawn: bool = False
    spawn_startup_wait: Optional[int] = None
    worker_service_name: str = "distwq.service"

    def with_spawn_method(self, method: SpawnMethod) -> "MPIConfig":
        return replace(self, spawn_method=method)


@dataclass(frozen=True)
class LoggingConfig:
    """Logging and development configuration"""

    verbose: bool = False
    profile: bool = False
    log_level: str = "INFO"
    stats_callback: Optional[Callable] = None

    def with_verbose(self, enabled: bool = True) -> "LoggingConfig":
        return replace(self, verbose=enabled)


@dataclass(frozen=True)
class Config:
    """Main configuration object"""

    resources: ResourceConfig = ResourceConfig()
    execution: ExecutionConfig = ExecutionConfig()
    mpi: MPIConfig = MPIConfig()
    logging: LoggingConfig = LoggingConfig()

    # Convenience methods for common operations
    def with_workers(self, n: int) -> "Config":
        return replace(self, resources=self.resources.with_workers(n))

    def with_timeout(self, seconds: int) -> "Config":
        return replace(self, execution=self.execution.with_timeout(seconds))

    def with_verbose(self, enabled: bool = True) -> "Config":
        return replace(self, debug=self.debug.with_verbose(enabled))

    # Factory methods for common scenarios
    @classmethod
    def for_development(cls) -> "Config":
        """Configuration optimized for development/testing"""
        return cls(
            resources=ResourceConfig(max_workers=2),
            execution=ExecutionConfig(fail_fast=True),
            debug=LoggingConfig(verbose=True, log_level="DEBUG"),
        )

    @classmethod
    def for_production(cls, workers: int = -1) -> "Config":
        """Configuration optimized for production runs"""
        return cls(
            resources=ResourceConfig(max_workers=workers),
            execution=ExecutionConfig(retry_attempts=3, fail_fast=False),
            debug=LoggingConfig(verbose=False, log_level="WARN"),
        )
