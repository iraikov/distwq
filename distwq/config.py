"""Settings for DistributedEnvironment.

Config holds the settings of the environment as a whole, and
WorkerGroupConfig those of worker groups.  Both are passed to
DistributedEnvironment.  Config and its parts are frozen dataclasses, so a
setting that does not exist raises TypeError when it is given.
"""

from dataclasses import dataclass, replace
from typing import Any

try:
    from .mpi import GroupingMethod, check_spawn_config
except ImportError:
    from mpi import GroupingMethod, check_spawn_config


@dataclass(frozen=True)
class ResourceConfig:
    """Which ranks to use.

    max_workers: the largest number of worker ranks to use; the others stay
    idle.  Zero or less means use every worker rank.  Every rank must give
    the same value.
    """

    max_workers: int = -1

    def with_workers(self, n: int) -> "ResourceConfig":
        return replace(self, max_workers=n)


@dataclass(frozen=True)
class LoggingConfig:
    """Logging of the worker processes.

    verbose: spawned worker processes log at INFO level instead of WARNING.
    """

    verbose: bool = False

    def with_verbose(self, enabled: bool = True) -> "LoggingConfig":
        return replace(self, verbose=enabled)


class WorkerGroupConfig:
    """Configuration for worker grouping"""

    def __init__(
        self,
        grouping_method: str | GroupingMethod = GroupingMethod.NoGrouping,
        nprocs_per_worker: int = 1,
        broker_is_worker: bool = False,
        collective_mode: str = "gather",
        enable_worker_service: bool = False,
        worker_service_name: str = "distwq.init",
        sequential_spawn: bool = False,
        spawn_startup_wait: int | None = None,
        spawn_executable: str | None = None,
        spawn_args: list[Any] | None = None,
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
        # A split configuration is checked against the number of worker
        # ranks when the environment is entered (establish_topology()),
        # because a max_workers limit can make it smaller than the world
        # size implies.


@dataclass(frozen=True)
class Config:
    """Settings of a DistributedEnvironment.

    resources: which ranks to use (see ResourceConfig).
    logging: logging of the worker processes (see LoggingConfig).

    Worker groups are set with the environment's worker_config argument,
    not here.
    """

    resources: ResourceConfig = ResourceConfig()
    logging: LoggingConfig = LoggingConfig()

    def with_workers(self, n: int) -> "Config":
        return replace(self, resources=self.resources.with_workers(n))

    def with_verbose(self, enabled: bool = True) -> "Config":
        return replace(self, logging=self.logging.with_verbose(enabled))
