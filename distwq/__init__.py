"""
Distributed work queue operations using mpi4py.

The controller/worker interface lives in :mod:`distwq.mpi`; its public
names are re-exported here, so that ``import distwq`` followed by
``distwq.run(...)`` or ``distwq.is_controller`` works as it did when
distwq was a single module.  The task-centric interface is in
:mod:`distwq.future`, and the functional interface in
:mod:`distwq.functional`; import those modules directly.

The role flags (``is_controller``, ``is_worker``, ``spawned``, ``rank``,
``size`` and so on) are set once, when :mod:`distwq.mpi` is first
imported, and are copied here at that time.
"""

from .mpi import (
    CollectiveMode,
    GroupingMethod,
    MessageTag,
    MPICollectiveBroker,
    MPICollectiveWorker,
    MPIController,
    MPIWorker,
    Topology,
    call_task,
    check_spawn_config,
    check_split_config,
    check_worker_grouping_method,
    do_spawn_workers,
    do_split_workers,
    establish_topology,
    get_fun,
    get_group_comm,
    has_mpi,
    is_controller,
    is_worker,
    multiple_task_arguments,
    my_args,
    my_config,
    n_workers,
    rank,
    run,
    run_spawn_group,
    run_split_group,
    setup_spawn_group,
    setup_split_group,
    size,
    spawned,
    start_time,
    task_failure,
    worker_main,
    workers_available,
)

__all__ = [
    "CollectiveMode",
    "GroupingMethod",
    "MPICollectiveBroker",
    "MPICollectiveWorker",
    "MPIController",
    "MPIWorker",
    "MessageTag",
    "Topology",
    "call_task",
    "check_spawn_config",
    "check_split_config",
    "check_worker_grouping_method",
    "do_spawn_workers",
    "do_split_workers",
    "establish_topology",
    "get_fun",
    "get_group_comm",
    "has_mpi",
    "is_controller",
    "is_worker",
    "multiple_task_arguments",
    "my_args",
    "my_config",
    "n_workers",
    "rank",
    "run",
    "run_spawn_group",
    "run_split_group",
    "setup_spawn_group",
    "setup_split_group",
    "size",
    "spawned",
    "start_time",
    "task_failure",
    "worker_main",
    "workers_available",
]

if has_mpi:
    from .mpi import controller_rank, world_comm

    __all__ += ["controller_rank", "world_comm"]
