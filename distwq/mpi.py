#!/usr/bin/python
#
# Distributed work queue operations using mpi4py.
#
# Copyright (C) 2020-2026 Ivan Raikov and distwq authors.
#
# Based on mpi.py from the pyunicorn project.
# Copyright (C) 2008--2019 Jonathan F. Donges and pyunicorn authors
# URL: <http://www.pik-potsdam.de/members/donges/software>
# License: BSD (3-clause)
#
# Please acknowledge and cite the use of this software and its authors
# when results are used in publications or published elsewhere.
#
# You can use the following reference:
# J.F. Donges, J. Heitzig, B. Beronov, M. Wiedermann, J. Runge, Q.-Y. Feng,
# L. Tupikina, V. Stolbova, R.V. Donner, N. Marwan, H.A. Dijkstra,
# and J. Kurths, "Unified functional network and nonlinear time series analysis
# for complex systems science: The pyunicorn package"

"""
Distributed work queue operations using mpi4py.

Allows for easy parallelization in controller/worker mode with one
controller submitting function or method calls to workers.  Supports
multiple ranks per worker (collective workers). Uses mpi4py if
available, otherwise processes calls sequentially in one process.

"""
#
#  Imports
#

import importlib
import json
import logging
import os
import pickle
import random
import signal
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from mpi4py.MPI import Intercomm, Intracomm
from numpy import float64


class CollectiveMode(IntEnum):
    Gather = 1
    SendRecv = 2


class MessageTag(IntEnum):
    READY = 0
    DONE = 1
    TASK = 2
    EXIT = 3
    SESSION_END = 4
    BROADCAST = 5


class GroupingMethod(IntEnum):
    NoGrouping = 0
    GroupSpawn = 1
    GroupSplit = 2


logger = logging.getLogger(__name__)

# Largest payload sent as a single physical MPI
# message. Point-to-point Send/Recv counts are limited to a 32-bit
# signed range by the underlying MPI implementation.
_MAX_MSG_BYTES = 1 << 30


def _isend_bytes(comm: Intracomm, data: bytes, dest: int, tag: int) -> None:
    """Send *data* to *dest* tagged *tag*, blocking until fully sent.

    Splits *data* into multiple physical messages when it exceeds a single
    message's size limit, so a payload of any length can be sent reliably.
    The receiver reconstructs it with :func:`_recv_bytes`, called with the
    same communicator, source, and tag.

    Wire format: one 8-byte little-endian length header, followed by
    however many chunks of at most ``_MAX_MSG_BYTES`` bytes it takes to
    carry the rest, all on *tag*. MPI preserves the order of messages sent
    between the same ordered pair of ranks on the same tag, so the
    receiver can read this sequence back in the order it was sent.
    """
    n = len(data)
    req = comm.Isend([n.to_bytes(8, "little"), MPI.BYTE], dest=dest, tag=tag)
    req.wait()
    # Slicing a memoryview sends each chunk without copying it.
    view = memoryview(data)
    offset = 0
    while offset < n:
        end = min(offset + _MAX_MSG_BYTES, n)
        req = comm.Isend([view[offset:end], MPI.BYTE], dest=dest, tag=tag)
        req.wait()
        offset = end


def _recv_bytes(comm: Intracomm, source: int, tag: int) -> bytes | bytearray:
    """Receive a payload sent with :func:`_isend_bytes` from *source*
    tagged *tag*.

    Returns the receive buffer itself rather than a copy, so a large
    payload is held in memory only once before it is unpickled.

    Only call this once a matching message is known to be pending (e.g.
    after a successful ``Iprobe`` for this exact ``(source, tag)`` pair) --
    receives must match the source and tag.
    """
    header = bytearray(8)
    comm.Recv([header, 8, MPI.BYTE], source=source, tag=tag)
    n = int.from_bytes(bytes(header), "little")
    if n == 0:
        return b""
    buf = bytearray(n)
    view = memoryview(buf)
    offset = 0
    while offset < n:
        end = min(offset + _MAX_MSG_BYTES, n)
        comm.Recv([view[offset:end], end - offset, MPI.BYTE], source=source, tag=tag)
        offset = end
    return buf


# try to get the communicator object to see whether mpi is available:
try:
    from mpi4py import MPI

    world_comm = MPI.COMM_WORLD
    has_mpi = True
except ImportError:
    has_mpi = False


def mpi_excepthook(type, value, traceback):
    """

    :param type:
    :param value:
    :param traceback:
    :return:
    """
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    MPI.COMM_WORLD.Abort(1)


has_args = "-" in sys.argv
my_args_start_index = sys.argv.index("-") + 1 if has_args else 0
my_args: list[str] | None = sys.argv[my_args_start_index:] if has_args else None
my_config: dict[Any, Any] | None = None

# initialize:
workers_available = True
spawned = False
if has_mpi:
    spawned = (my_args[0] == "distwq:spawned") if my_args is not None else False
    size = world_comm.size
    rank = world_comm.rank
    controller_rank = int(os.environ.get("DISTWQ_CONTROLLER_RANK", "0"))
    if controller_rank < 0:
        controller_rank = size - 1
    if controller_rank >= size:
        raise RuntimeError(
            f"Invalid controller rank {controller_rank} specified "
            f"when world size is {size}"
        )
    is_controller = (not spawned) and (rank == controller_rank)
    is_worker = not is_controller
    if size < 2:
        workers_available = False
        is_worker = True
else:
    size = 1
    rank = 0
    is_controller = True
    is_worker = True

if has_mpi and (size > 1):
    sys_excepthook = sys.excepthook
    sys.excepthook = mpi_excepthook

if spawned:
    my_config = json.loads(my_args[1]) if isinstance(my_args, list) else None

n_workers = (
    int(my_config["n_workers"]) if spawned and isinstance(my_config, dict) else size - 1
)
# The number of worker ranks before any max_workers limit; limit_workers()
# always starts again from this value.
_world_workers = n_workers
start_time = time.time()


def _limited_roles(
    max_workers: int | None,
    world_workers: int,
    rank: int,
    controller_rank: int,
    is_controller: bool,
) -> tuple[int, bool]:
    """Return (n_workers, is_worker) for this rank under a max_workers limit.

    A max_workers of None, zero or less means no limit.  The ranks kept as
    workers are those with the lowest worker index, where a rank's worker
    index is its position among the ranks other than the controller.  The
    controller is never a worker.
    """
    if max_workers is None or max_workers <= 0:
        limited = world_workers
    else:
        limited = min(max_workers, world_workers)
    worker_index = rank if rank < controller_rank else rank - 1
    return limited, (not is_controller) and worker_index < limited


def limit_workers(max_workers: int | None) -> int:
    """Use at most max_workers worker ranks; return the number used.

    Sets the module values n_workers and is_worker, which
    establish_topology() reads.  Ranks left out become idle ranks.  A
    max_workers of None, zero or less means every worker rank is used.
    Each call starts again from the full set of worker ranks, so a later
    call can raise the limit again.

    Every rank must call this with the same value before
    establish_topology().  Spawned processes, runs without MPI and runs
    on one rank are left unchanged.
    """
    global n_workers, is_worker
    if max_workers is not None and (
        isinstance(max_workers, bool) or not isinstance(max_workers, int)
    ):
        raise TypeError(
            f"max_workers must be an int or None, not {type(max_workers).__name__}"
        )
    if spawned or not has_mpi or size < 2:
        return n_workers
    n_workers, is_worker = _limited_roles(
        max_workers, _world_workers, rank, controller_rank, is_controller
    )
    return n_workers


def multiple_task_arguments(N, args, kwargs, task_ids, workers):
    """
    Helper function to set default arguments, task ids, and workers for multiple tasks.
    """
    if (args is None) or (len(args) == 0):
        args = [{} for _ in range(N)]
    if (kwargs is None) or (len(kwargs) == 0):
        kwargs = [{} for _ in range(N)]
    if task_ids is None:
        task_ids = [None for _ in range(N)]
    if workers is None:
        workers = [None for _ in range(N)]

    return args, kwargs, task_ids, workers


def task_failure(exc: BaseException) -> dict[str, Any]:
    """Describe a task's exception in built-in types only.

    Sent to the controller in place of the task's result.  Only built-in
    types are used, so the description can be unpickled anywhere, even
    when the worker runs its code as __main__.  "pickled" holds the
    exception itself when it can be pickled, and None otherwise.
    """
    try:
        pickled = pickle.dumps(exc)
    except Exception:  # noqa: BLE001 - any pickling failure means "not picklable"
        pickled = None
    return {
        "distwq_failure": True,
        "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "message": str(exc),
        "traceback": traceback.format_exc(),
        "pickled": pickled,
        "world_rank": world_comm.rank if has_mpi else 0,
    }


def call_task(
    name_to_call: str,
    module: str,
    args: tuple[Any],
    kwargs: dict[Any, Any],
    capture_errors: bool,
) -> tuple[Any, bool]:
    """Look up a task's callable by module and name, call it, and return
    (result, failed).

    With capture_errors=True an exception raised while looking up or
    calling the task is caught and returned as (task_failure(exc), True),
    so that the serving loop can report it and carry on.  Otherwise the
    exception propagates, and the module's exception hook aborts the job
    as it always has.
    """
    try:
        if module not in sys.modules:
            importlib.import_module(module)
        try:
            object_to_call = eval(name_to_call, sys.modules[module].__dict__)
        except NameError:
            logger.error(str(sys.modules[module].__dict__.keys()))
            raise
        return object_to_call(*args, **kwargs), False
    except Exception as exc:
        if not capture_errors:
            raise
        logger.error(f"task {name_to_call} in {module} failed: {exc!r}")
        return task_failure(exc), True


@dataclass(frozen=True)
class TaskResult:
    """Result of a task run by one process.

    Sent by an MPIWorker, or stored by the controller when it runs a task
    itself because no workers are available.  value is the task's return
    value.
    """

    value: Any


@dataclass(frozen=True)
class CollectiveTaskResult:
    """Results of a task run by every process of a worker group.

    Sent by an MPICollectiveBroker.  values[i] is the return value on the
    i-th process of the group that ran the task, in the order of the
    group's merged communicator; when the broker also works, its own value
    comes first.  A group has nprocs_per_worker processes that run tasks,
    so len(values) == nprocs_per_worker whether or not the broker is one
    of them.
    """

    values: list[Any]


def _split_group_data(
    sub_data: list[tuple[Any, dict[str, Any] | None]],
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Separate the (result, stat) pairs gathered from a worker group.

    A process that ran the task always sends its stat, even when the task
    returned None.  Only a broker that does not work sends (None, None),
    so pairs are kept or dropped by their stat, which keeps results[i] and
    stats[i] from the same process.
    """
    kept = [(result, stat) for result, stat in sub_data if stat is not None]
    return [result for result, _ in kept], [stat for _, stat in kept]


def _group_outcome(
    results: list[Any], stats: list[dict[str, Any]]
) -> tuple[Any, dict[str, Any] | None]:
    """Return what a broker sends to the controller for one task.

    The payload is a CollectiveTaskResult of every process's value or, if
    the task failed on any process, the task_failure() description from
    the first process whose stat reports an error.  The stat sent with it
    is the one of the slowest process, marked as an error on failure.
    """
    stat = max(stats, key=lambda s: s["this_time"]) if stats else None
    for result, this_stat in zip(results, stats):
        if this_stat.get("error"):
            return result, dict(stat, error=True)
    return CollectiveTaskResult(list(results)), stat


class MPIController:
    def __init__(
        self,
        comm: Intracomm | None = None,
        time_limit: Any = None,
        capture_errors: bool = False,
    ) -> None:
        # Without a communicator there are no workers, and every call is
        # run on this process.
        size = comm.size if comm is not None else 1

        self.comm = comm
        self.workers_available = size > 1
        # Report the exception of a call run on this process as a failed
        # result instead of raising it from submit_call() and the like.
        self.capture_errors = capture_errors

        self.count = 0

        self.start_time = start_time
        self.time_limit = time_limit
        self.total_time_est = np.ones(size)
        """
        (numpy array of ints)
        total_time_est[i] is the current estimate of the total time
        MPI worker i will work on already submitted calls.
        On worker i, only total_time_est[i] is available.
        """
        self.total_time_est[0] = np.inf
        self.result_queue: list[int] = []
        self.task_queue: list[int] = []
        self.wait_queue: list[int] = []
        self.waiting: dict[
            int,
            tuple[
                str,
                list[Any] | tuple[Any],
                dict[Any, Any],
                str,
                int | None,
                int | None,
            ],
        ] = {}
        self.ready_workers: list[int] = []
        self.ready_workers_data: dict[int, Any] = {}
        """(list) ids of submitted calls"""
        self.assigned = {}
        """
        (dictionary)
        assigned[id] is the worker assigned to the call with that id.
        """
        self.worker_queue = [[] for i in range(size)]
        """
        (list of lists)
        worker_queue[i] contains the ids of calls assigned to worker i.
        """
        self.active_workers = set()
        """
        (set) active_workers contains the ids of workers that have
        communicated with the controller
        """
        self.n_processed = np.zeros(size).astype(int)
        """
        (list of ints)
        n_processed[rank] is the total number of calls processed by MPI node rank.
        On worker i, only total_time[i] is available.
        """
        self.total_time = np.zeros(size).astype(np.float32)
        """
        (list of floats)
        total_time[rank] is the total wall time until that node finished its last
        call.  On worker i, only total_time[i] is available.
        """
        self.results = {}
        self.failed_tasks: set[int] = set()
        """
        (set) ids of finished calls whose result is a task_failure()
        description rather than a return value; only workers created with
        capture_errors=True, or this controller when created with
        capture_errors=True and running calls itself, report failures.
        """
        """
        (dictionary)
        if mpi is not available, the result of submit_call(..., id=a) will be
        cached in results[a] until get_result(a).
        """
        self.stats = []

        """
        (list of dictionaries)
        stats[id] contains processing statistics for the last call with this id. Keys:

        - "id": id of the call
        - "rank": MPI node who processed the call
        - "this_time": wall time for processing the call
        - "time_over_est": quotient of actual over estimated wall time
        - "n_processed": no. of calls processed so far by this worker, including this
        - "total_time": total wall time until this call was finished
        """

    def process(self, limit: int = 1000, block: bool = False) -> list[int | Any]:
        """
        Process incoming messages.

        :arg int limit: stop after handling this many DONE messages in one
            call.
        :arg bool block: if True and nothing is immediately pending, wait
            for the next message with a blocking MPI probe before giving
            up. Default False: return promptly if nothing is pending right now.
        """
        if not self.workers_available:
            return
        count = 0
        status = MPI.Status()
        have_message = self.comm.Iprobe(
            source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status
        )
        if block and not have_message:
            self.comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            have_message = True
        while have_message:
            if (limit is not None) and (limit < count):
                break

            worker = status.Get_source()
            tag = status.Get_tag()
            payload = _recv_bytes(self.comm, source=worker, tag=tag)

            if tag == MessageTag.READY.value:
                if worker not in self.ready_workers:
                    data = pickle.loads(payload) if payload else None
                    self.ready_workers.append(worker)
                    self.ready_workers_data[worker] = data
                    self.active_workers.add(worker)
                logger.info(
                    f"MPI controller : received READY message from worker {worker}"
                )
            elif tag == MessageTag.DONE.value:
                (task_id, results, stats) = pickle.loads(payload)
                logger.info(
                    f"MPI controller : received DONE message for task {task_id} "
                    f"from worker {worker}"
                )

                self.results[task_id] = results
                if stats is not None and stats.get("error"):
                    self.failed_tasks.add(task_id)
                self.stats.append(stats)
                self.n_processed[worker] = stats["n_processed"]
                self.total_time[worker] = stats["total_time"]
                self.task_queue.remove(task_id)
                self.result_queue.append(task_id)
                self.worker_queue[worker].remove(task_id)
                self.assigned.pop(task_id)
                count += 1
            else:
                raise RuntimeError(f"MPI controller : invalid message tag {tag}")
            have_message = self.comm.Iprobe(
                source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status
            )

        return self.submit_waiting()

    def _call_locally(
        self,
        name_to_call: str,
        args: list[Any] | tuple[Any],
        kwargs: dict[Any, Any],
        module_name: str,
        time_est: int,
        task_id: int,
    ) -> None:
        """Run a call on this process and store its result under task_id.

        Used when no workers are available.  The result is stored as a
        TaskResult, as a worker would send it.  With capture_errors=True a
        failed call is stored as its task_failure() description and its id
        is added to failed_tasks; otherwise the exception propagates and
        nothing is stored.
        """
        worker = 0
        logger.info(f"MPI controller : calling {name_to_call} {args} {kwargs} ...")
        call_time = time.time()
        result, failed = call_task(
            name_to_call, module_name, args, kwargs, self.capture_errors
        )
        this_time = time.time() - call_time
        self.results[task_id] = result if failed else TaskResult(result)
        if failed:
            self.failed_tasks.add(task_id)
        self.result_queue.append(task_id)
        self.n_processed[worker] += 1
        self.total_time[worker] = time.time() - start_time
        self.stats.append(
            {
                "id": task_id,
                "rank": worker,
                "this_time": this_time,
                "time_over_est": this_time / time_est,
                "n_processed": self.n_processed[worker],
                "total_time": self.total_time[worker],
                "error": failed,
            }
        )

    def submit_call(
        self,
        name_to_call: str,
        args: tuple[Any] = (),
        kwargs: dict[Any, Any] | None = None,
        module_name: str = "__main__",
        time_est: int = 1,
        task_id: int | None = None,
        worker: int | None = None,
    ) -> int:
        """
        Submit a call for parallel execution.

        If called by the controller and workers are available, the call is submitted
        to a worker for asynchronous execution.

        If called by a worker or if no workers are available, the call is instead
        executed synchronously on this MPI node.

        **Examples:**

            1. Provide ids and time estimate explicitly:

               .. code-block:: python

                  for n in range(0,10):
                      distwq.submit_call("doit", (n,A[n]), id=n, time_est=n**2)

                  for n in range(0,10):
                      result[n] = distwq.get_result(n)

            2. Use generated ids stored in a list:

               .. code-block:: python

                  for n in range(0,10):
                      ids.append(distwq.submit_call("doit", (n,A[n])))

                  for n in range(0,10):
                      results.append(distwq.get_result(ids.pop()))

            3. Ignore ids altogether:

               .. code-block:: python

                  for n in range(0,10):
                      distwq.submit_call("doit", (n,A[n]))

                  for n in range(0,10):
                      results.append(distwq.get_next_result())

            4. Call a module function and use keyword arguments:

               .. code-block:: python

                  distwq.submit_call("solve", (), {"a":a, "b":b},
                                       module="numpy.linalg")


        :arg str name_to_call: name of callable object (usually a function or
            static method of a class) as contained in the namespace specified
            by module.
        :arg tuple args: the positional arguments to provide to the callable
            object.  Tuples of length 1 must be written (arg,).  Default: ()
        :arg dict kwargs: the keyword arguments to provide to the callable
            object.  Default: {}
        :arg str module: optional name of the imported module or submodule in
            whose namespace the callable object is contained. For objects
            defined on the script level, this is "__main__", for objects
            defined in an imported package, this is the package name. Must be a
            key of the dictionary sys.modules (check there after import if in
            doubt).  Default: "__main__"
        :arg float time_est: estimated relative completion time for this call;
            used to find a suitable worker. Default: 1
        :type id: object or None
        :arg  id: unique id for this call. Must be a possible dictionary key.
            If None, a random id is assigned and returned. Can be re-used after
            get_result() for this is. Default: None
        :type worker: int > 0 and < comm.size, or None
        :arg  worker: optional no. of worker to assign the call to. If None, the
            call is assigned to the worker with the smallest current total time
            estimate. Default: None
        :return object: id of call, to be used in get_result().
        """
        if kwargs is None:
            kwargs = {}
        if task_id is None:
            task_id = self.count
            self.count += 1
        self.check_valid_task_id(task_id)
        if self.workers_available:
            self.process()
            if len(self.ready_workers) > 0:
                if worker is None:
                    ready_total_time_est = np.asarray(
                        [self.total_time_est[worker] for worker in self.ready_workers]
                    )
                    worker = self.ready_workers[np.argmin(ready_total_time_est)]
                else:
                    if worker not in self.ready_workers:
                        raise RuntimeError(f"worker {worker} is not in ready queue!")

                # send name to call, args, time_est to worker:
                logger.info(
                    f"MPI controller : assigning call with id {task_id} to worker "
                    f"{worker}: {name_to_call} {args} {kwargs} ..."
                )
                req = self.comm.isend(
                    (name_to_call, args, kwargs, module_name, time_est, task_id),
                    dest=worker,
                    tag=MessageTag.TASK.value,
                )
                req.wait()
                self.ready_workers.remove(worker)
                del self.ready_workers_data[worker]
                self.task_queue.append(task_id)
                self.worker_queue[worker].append(task_id)
                self.assigned[task_id] = worker
            else:
                self.queue_call(
                    name_to_call,
                    args=args,
                    kwargs=kwargs,
                    module_name=module_name,
                    time_est=time_est,
                    task_id=task_id,
                    requested_worker=worker,
                )

        else:
            # perform call on this rank if no workers are available:
            worker = 0
            self._call_locally(
                name_to_call, args, kwargs, module_name, time_est, task_id
            )

        self.total_time_est[worker] += time_est
        return task_id

    def queue_call(
        self,
        name_to_call: str,
        args: list[Any] | tuple[Any] = (),
        kwargs: dict[Any, Any] | None = None,
        module_name: str = "__main__",
        time_est: int = 1,
        task_id: int | None = None,
        requested_worker: int | None = None,
    ) -> int:
        """Submit a call for later execution.

        If called by the controller and workers are available, the
        call is put on the wait queue and submitted to a worker when
        it is available. Method process() checks the wait queue and
        submits calls on the wait queue.

        If called by a worker or if no workers are available, the call is instead
        executed synchronously on this MPI node.

        :arg str name_to_call: name of callable object (usually a function or
            static method of a class) as contained in the namespace specified
            by module.
        :arg tuple args: the positional arguments to provide to the callable
            object.  Tuples of length 1 must be written (arg,).  Default: ()
        :arg dict kwargs: the keyword arguments to provide to the callable
            object.  Default: {}
        :arg str module: optional name of the imported module or submodule in
            whose namespace the callable object is contained. For objects
            defined on the script level, this is "__main__", for objects
            defined in an imported package, this is the package name. Must be a
            key of the dictionary sys.modules (check there after import if in
            doubt).  Default: "__main__"
        :arg float time_est: estimated relative completion time for this call;
            used to find a suitable worker. Default: 1
        :type id: object or None
        :arg  id: unique id for this call. Must be a possible dictionary key.
            If None, a random id is assigned and returned. Can be re-used after
            get_result() for this is. Default: None
        :return object: id of call, to be used in get_result().
        :type requested_worker: int > 0 and < comm.size, or None
        :arg  requested_worker: optional no. of worker to assign the call to.
            If None, or the worker is not available, the call is assigned to
            the worker with the smallest current total time estimate.
            Default: None

        """
        if kwargs is None:
            kwargs = {}
        if task_id is None:
            task_id = self.count
            self.count += 1
        self.check_valid_task_id(task_id)
        if self.workers_available:
            self.wait_queue.append(task_id)
            self.waiting[task_id] = (
                name_to_call,
                args,
                kwargs,
                module_name,
                time_est,
                requested_worker,
            )
        else:
            # perform call on this rank if no workers are available:
            self._call_locally(
                name_to_call, args, kwargs, module_name, time_est, task_id
            )

        return task_id

    def submit_waiting(self) -> list[int | Any]:
        """
        Submit waiting tasks if workers are available.

        :return object: ids of calls, to be used in get_result().
        """
        task_ids = []
        if self.workers_available:  # noqa: SIM102
            if (len(self.waiting) > 0) and len(self.ready_workers) > 0:
                reqs = []
                status = []
                for i in range(len(self.ready_workers)):
                    if len(self.waiting) == 0:
                        break
                    task_id = self.wait_queue.pop(0)
                    (
                        name_to_call,
                        args,
                        kwargs,
                        module_name,
                        time_est,
                        requested_worker,
                    ) = self.waiting[task_id]
                    if (requested_worker is None) or (
                        requested_worker not in self.ready_workers
                    ):
                        ready_total_time_est = np.asarray(
                            [
                                self.total_time_est[worker]
                                for worker in self.ready_workers
                            ]
                        )
                        worker = self.ready_workers[np.argmin(ready_total_time_est)]
                    else:
                        worker = requested_worker

                    # send name to call, args, time_est to worker:
                    logger.info(
                        f"MPI controller : assigning waiting call with id {task_id} "
                        f"to worker {worker}: {name_to_call} {args} {kwargs} ..."
                    )
                    req = self.comm.isend(
                        (name_to_call, args, kwargs, module_name, time_est, task_id),
                        dest=worker,
                        tag=MessageTag.TASK.value,
                    )
                    reqs.append(req)
                    status.append(MPI.Status())
                    self.ready_workers.remove(worker)
                    del self.ready_workers_data[worker]
                    self.task_queue.append(task_id)
                    self.worker_queue[worker].append(task_id)
                    self.assigned[task_id] = worker
                    del self.waiting[task_id]
                    self.total_time_est[worker] += time_est
                    task_ids.append(task_id)
                MPI.Request.waitall(reqs, status)
        return task_ids

    def submit_multiple(
        self,
        name_to_call: str,
        args: list[list[Any]] | None = None,
        kwargs: list[dict[Any, Any]] | None = None,
        module_name: str = "__main__",
        time_est: int = 1,
        task_ids: int | None = None,
        workers: int | None = None,
    ) -> list[int]:
        """Submit multiple calls for parallel execution.

        Analogous to submit_call, but accepts lists of arguments and
        submits to multiple workers for asynchronous execution.

        If called by a worker or if no workers are available, the call is instead
        executed synchronously on this MPI node.

        :arg str name_to_call: name of callable object (usually a function or
            static method of a class) as contained in the namespace specified
            by module.
        :arg list args: the positional arguments to provide to the callable
            object for each task, as a list of tuples.  Default: []
        :arg list kwargs: the keyword arguments to provide to the callable
            object for each task, as a list of dictionaries.  Default: []
        :arg str module: optional name of the imported module or submodule in
            whose namespace the callable object is contained. For objects
            defined on the script level, this is "__main__", for objects
            defined in an imported package, this is the package name. Must be a
            key of the dictionary sys.modules (check there after import if in
            doubt).  Default: "__main__"
        :arg float time_est: estimated relative completion time for this call;
            used to find a suitable worker. Default: 1
        :type task_ids: list or None
        :arg task_ids: unique ids for each call. Must be a possible dictionary key.
            If None, a random id is assigned and returned. Can be re-used after
            get_result() for this id. Default: None
        :type workers: list of int > 0 and < comm.size, or None
        :arg  worker: optional worker ids to assign the tasks to. If None, the
            tasks are assigned in order to the workers with the smallest
            current total time estimate. Default: None
        :return object: id of call, to be used in get_result().

        """
        if args is None:
            args = []
        if kwargs is None:
            kwargs = []
        if len(kwargs) > 0:
            assert (len(args) == 0) or (len(args) == len(kwargs))
        submitted_task_ids = []
        N = len(args) if (len(args) > 0) else len(kwargs)
        if self.workers_available:
            self.process()
            args, kwargs, task_ids, workers = multiple_task_arguments(
                N, args, kwargs, task_ids, workers
            )
            for this_args, this_kwargs, this_task_id, this_worker in zip(
                args, kwargs, task_ids, workers
            ):
                self.check_valid_task_id(this_task_id)
                this_task_id = self.queue_call(
                    name_to_call,
                    args=this_args,
                    kwargs=this_kwargs,
                    module_name=module_name,
                    time_est=time_est,
                    task_id=this_task_id,
                    requested_worker=this_worker,
                )
                submitted_task_ids.append(this_task_id)
        else:
            # perform calls on this rank if no workers are available; the
            # requested workers do not exist and are ignored.
            worker = 0
            args, kwargs, task_ids, workers = multiple_task_arguments(
                N, args, kwargs, task_ids, workers
            )
            for this_args, this_kwargs, this_task_id in zip(args, kwargs, task_ids):
                if this_task_id is None:
                    this_task_id = self.count
                    self.count += 1
                self.check_valid_task_id(this_task_id)
                self._call_locally(
                    name_to_call,
                    this_args,
                    this_kwargs,
                    module_name,
                    time_est,
                    this_task_id,
                )
                self.total_time_est[worker] += time_est
                submitted_task_ids.append(this_task_id)

        return submitted_task_ids

    def get_ready_worker(self):
        """
        Returns the id and data of a ready worker.
        If there are no workers, or no worker is ready, returns (None, None)
        """
        if self.workers_available:
            self.process()
            if len(self.ready_workers) > 0:
                ready_total_time_est = np.asarray(
                    [self.total_time_est[worker] for worker in self.ready_workers]
                )
                worker = self.ready_workers[np.argmin(ready_total_time_est)]
                return worker, self.ready_workers_data[worker]
            else:
                return None, None
        else:
            return None, None

    def get_result(self, task_id: int) -> tuple[int, Any]:
        """
        Return result of earlier submitted call.

        Can only be called by the controller.

        If the call is not yet finished, waits for it to finish.
        Results should be collected in the same order as calls were submitted.
        For each worker, the results of calls assigned to that worker must be
        collected in the same order as those calls were submitted.
        Can only be called once per call.

        :type id: object
        :arg  id: id of an earlier submitted call, as provided to or returned
                  by submit_call().

        :rtype:  object
        :return: id and result of call.  The result is a TaskResult from a
                 single worker or from this process, a CollectiveTaskResult
                 from a worker group, or, if the id is in failed_tasks, the
                 task_failure() description of the call's exception.
        """
        if task_id in self.results:
            if task_id in self.result_queue:
                self.result_queue.remove(task_id)
            return task_id, self.results.pop(task_id)
        # A call still waiting for a free worker is dispatched by process()
        # once a worker reports READY.
        while task_id in self.waiting:
            self.process(block=True)
        if task_id in self.results:
            self.result_queue.remove(task_id)
            return task_id, self.results.pop(task_id)
        source = self.assigned[task_id]
        if self.workers_available:
            if self.worker_queue[source][0] != task_id:
                raise RuntimeError(
                    f"get_result({task_id})) called before get_result("
                    f"{self.worker_queue[source][0]})"
                )
            logger.info(
                f"MPI controller : retrieving result for call with id {task_id} "
                f"from worker {source} ..."
            )

            while task_id not in self.results:
                self.process(block=True)

            logger.info(
                f"MPI controller : received result for call with id {task_id} "
                f"from worker {source}."
            )

        else:
            logger.info(
                f"MPI controller : returning result for call with id {task_id} ..."
            )
        result = self.results.pop(task_id)
        self.result_queue.remove(task_id)
        return task_id, result

    def get_next_result(self) -> tuple[int, Any] | None:
        """
        Return result of next earlier submitted call whose result has not yet
        been obtained.

        Can only be called by the controller.

        If the call is not yet finished, waits for it to finish. This
        includes a call submitted so recently that it has not yet been
        assigned to a worker.

        :rtype:  object
        :return: id and result of call, as for get_result(), or None if
                 there are no calls outstanding at all.
        """
        while True:
            # Non-blocking first: pick up anything already arrived and
            # dispatch any waiting call to a worker that is already ready,
            # before deciding whether there is anything left to wait for.
            self.process()
            if len(self.result_queue) > 0:
                task_id = self.result_queue.pop(0)
                return task_id, self.results.pop(task_id)
            elif len(self.task_queue) > 0:
                task_id = self.task_queue[0]
                # Dispatched but not yet finished -- get_result() blocks
                # (via a blocking probe, not polling) until it is.
                return task_id, self.get_result(task_id)[1]
            elif len(self.waiting) > 0:
                # Submitted but not yet dispatched to any worker -- wait
                # for the next message (presumably a worker reporting
                # ready) and re-check, rather than reporting nothing
                # available while a call is only temporarily unassigned.
                self.process(block=True)
                continue
            else:
                return None

    def probe_next_result(self):
        """
        Return result of next earlier submitted call whose result has not yet
        been obtained.

        Can only be called by the controller.

        If no result is available, returns none.

        :rtype:  object
        :return: id and result of call, as for get_result(), or None if
                 there are no results ready.
        """
        self.process()
        if len(self.result_queue) > 0:
            task_id = self.result_queue.pop(0)
            logger.info(
                f"MPI controller : received result for call with id {task_id} ..."
            )
            return task_id, self.results.pop(task_id)
        else:
            return None

    def probe_all_next_results(self):
        """
        Return all available results of earlier submitted calls whose result has not yet
        been obtained.

        Can only be called by the controller.

        If no result is available, returns empty list.

        :rtype:  object
        :return: list of (id, result of call), as for get_result()
        """
        self.process()
        ret = []
        if len(self.result_queue) > 0:
            for i in range(len(self.result_queue)):
                task_id = self.result_queue.pop(0)
                logger.info(
                    f"MPI controller : received result for call with id {task_id} ..."
                )
                ret.append((task_id, self.results.pop(task_id)))

        return ret

    def summary(self) -> str:
        """
        Return processing statistics as text, or "" if no call has finished.

        Can only be called by the controller.
        """

        if len(self.stats) == 0:
            return ""

        call_times = np.array([s["this_time"] for s in self.stats])
        call_quotients = np.array([s["time_over_est"] for s in self.stats])
        cvar_call_quotients = call_quotients.std() / call_quotients.mean()

        if self.workers_available:
            worker_quotients = self.total_time / self.total_time_est
            cvar_worker_quotients = worker_quotients.std() / worker_quotients.mean()
            return (
                "\n"
                "distwq run statistics\n"
                "==========================\n"
                "     results collected:         "
                f"{self.n_processed[1:].sum()}\n"
                "     results not yet collected: "
                f"{len(self.task_queue)}\n"
                "     total reported time:       "
                f"{call_times.sum():.04f}\n"
                "     mean time per call:        "
                f"{call_times.mean():.04f}\n"
                "     std.dev. of time per call: "
                f"{call_times.std():.04f}\n"
                "     coeff. of var. of actual over estd. time per call: "
                f"{cvar_call_quotients:.04f}\n"
                "     workers:                      "
                # The workers of this controller, which are the brokers
                # under worker grouping.
                f"{self.comm.size - 1}\n"
                "     mean calls per worker:        "
                f"{self.n_processed[1:].mean():.04f}\n"
                "     std.dev. of calls per worker: "
                f"{self.n_processed[1:].std():.04f}\n"
                "     min calls per worker:         "
                f"{self.n_processed[1:].min()}\n"
                "     max calls per worker:         "
                f"{self.n_processed[1:].max()}\n"
                "     mean time per worker:        "
                f"{self.total_time.mean():.04f}\n"
                "     std.dev. of time per worker: "
                f"{self.total_time.std():.04f}\n"
                "     coeff. of var. of actual over estd. time per worker: "
                f"{cvar_worker_quotients:.04f}\n"
            )
        return (
            "\n"
            "distwq run statistics\n"
            "==========================\n"
            "     results collected:         "
            f"{self.n_processed[0]}\n"
            "     results not yet collected: "
            f"{len(self.task_queue)}\n"
            "     total reported time:       "
            f"{call_times.sum():.04f}\n"
            "     mean time per call:        "
            f"{call_times.mean():.04f}\n"
            "     std.dev. of time per call: "
            f"{call_times.std():.04f}\n"
            "     coeff. of var. of actual over estd. time per call: "
            f"{cvar_call_quotients:.04f}\n"
        )

    def info(self) -> None:
        """
        Print processing statistics (see summary()).

        Can only be called by the controller.
        """
        text = self.summary()
        if text:
            print(text)

    def exit(self) -> None:
        """
        Collect any results not yet collected, then tell all workers to exit.

        Every worker is told, including one the controller has not heard
        from yet because it was never given a call: shutdown() first waits
        for each worker to report READY.  Uncollected results are
        discarded.

        Can only be called by the controller.
        """
        if self.workers_available:
            while self.get_next_result() is not None:
                pass
            self.shutdown()

    def wait_all_ready(self) -> None:
        """
        Block until every worker is idle and has reported READY.

        A worker reports READY each time it enters serve() and after each
        call it finishes, so once all of them are in the ready list no
        worker is busy and none is still starting up.  All submitted calls
        must already have been collected, since a worker holding a call
        would never become ready.

        Can only be called by the controller.
        """
        if not self.workers_available:
            return
        if self.task_queue or self.waiting:
            raise RuntimeError(
                "MPI controller : cannot wait for idle workers while calls "
                f"are outstanding (running: {self.task_queue}, "
                f"waiting: {list(self.waiting)})"
            )
        n_workers = self.comm.size - 1
        while len(self.ready_workers) < n_workers:
            self.process(block=True)

    def _send_to_all_workers(self, tag: MessageTag, payload: Any = None) -> None:
        """Send payload with tag to every worker, then forget that they
        were ready: each will report READY again when it next enters
        serve()."""
        reqs = [
            self.comm.isend(payload, dest=worker, tag=tag.value)
            for worker in range(1, self.comm.size)
        ]
        MPI.Request.Waitall(reqs)
        self.ready_workers = []
        self.ready_workers_data = {}

    def broadcast_to_workers(self, value: Any) -> None:
        """
        Wait until every worker is idle in serve(), then send value to all
        of them.

        Each worker's serve() returns MessageTag.BROADCAST and leaves the
        value in its broadcast_value attribute; brokers pass it on to their
        collective workers.  Can only be called by the controller.
        """
        if not self.workers_available:
            return
        self.wait_all_ready()
        self._send_to_all_workers(MessageTag.BROADCAST, value)

    def end_session(self) -> None:
        """
        End the current session: wait until every worker is idle, then
        tell each one to return from serve() without exiting.

        The workers can serve another session afterwards.  Can only be
        called by the controller.
        """
        if not self.workers_available:
            return
        self.wait_all_ready()
        logger.info("MPI controller : ending session on all workers...")
        self._send_to_all_workers(MessageTag.SESSION_END)

    def shutdown(self) -> None:
        """
        Wait until every worker is idle, then tell all of them to exit.

        Unlike exit(), which tells only the workers it has already heard
        from, this waits to hear from every worker first, so none is left
        waiting for orders.  Can only be called by the controller.
        """
        if not self.workers_available:
            return
        self.wait_all_ready()
        logger.info("MPI controller : telling all workers to exit...")
        self._send_to_all_workers(MessageTag.EXIT)

    def abort(self):
        """
        Abort execution on all MPI nodes immediately, with exit status 1.

        Can be called by controller and workers.
        """
        traceback.print_exc()
        logger.error("MPI controller : aborting...")
        self.comm.Abort(1)

    def check_valid_task_id(self, task_id: int) -> bool:
        """
        Given a new task id, check that it is not already assigned or waiting.
        """
        if task_id in self.assigned:
            raise RuntimeError(f"task id {task_id} already in queue!")
        if task_id in self.waiting:
            raise RuntimeError(f"task id {task_id} already in wait queue!")
        return True


class MPIWorker:
    def __init__(
        self,
        comm: Intracomm,
        group_comm: Intracomm,
        ready_data: Any | None = None,
        capture_errors: bool = False,
    ) -> None:
        size = comm.size
        rank = comm.rank

        self.comm = comm
        self.group_comm = group_comm
        self.worker_id = group_comm.rank + 1
        self.total_time_est = np.zeros(size) * np.nan
        self.total_time_est[rank] = 0
        self.n_processed = np.zeros(size) * np.nan
        self.n_processed[rank] = 0
        self.start_time = start_time
        self.total_time = np.zeros(size) * np.nan
        self.total_time[rank] = 0
        self.stats = []
        self.ready_data = None
        # Report task exceptions to the controller instead of aborting.
        self.capture_errors = capture_errors

        logger.info(f"MPI worker {self.worker_id}: initialized.")

    def serve(self) -> MessageTag:
        """
        Serve submitted calls until told to finish.

        Returns the tag of the message that ended serving:
        MessageTag.SESSION_END when the controller ends a session and
        will later ask this worker to serve again, or MessageTag.EXIT
        when the controller is finished with it.

        Call this function if workers need to perform initialization
        different from the controller, like this:

        >>> def workerfun(worker):
        >>>     do = whatever + initialization - is * necessary
        >>>     worker.serve()
        >>>     do = whatever + cleanup - is * necessary

        If you don't define workerfun(), serve() will be called automatically by
        run().
        """
        rank = self.comm.rank

        logger.info(f"MPI worker {rank}: waiting for calls.")

        # wait for orders:
        status = MPI.Status()
        while True:
            # signal the controller this worker is ready
            _isend_bytes(
                self.comm,
                pickle.dumps(self.ready_data),
                dest=0,
                tag=MessageTag.READY.value,
            )

            # Block until the controller sends a task or exit signal.
            self.comm.Probe(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            data = self.comm.recv(source=0, tag=tag)

            if tag == MessageTag.EXIT.value:
                logger.info(f"MPI worker {self.worker_id}: exiting...")
                return MessageTag.EXIT
            elif tag == MessageTag.SESSION_END.value:
                logger.info(f"MPI worker {self.worker_id}: session ended.")
                return MessageTag.SESSION_END
            elif tag == MessageTag.BROADCAST.value:
                self.broadcast_value = data
                return MessageTag.BROADCAST
            elif tag == MessageTag.TASK.value:
                (name_to_call, args, kwargs, module, time_est, task_id) = data
            else:
                raise RuntimeError(
                    f"MPI worker {self.worker_id}: unknown message tag {tag}"
                )
            self.total_time_est[rank] += time_est
            call_time = time.time()
            result, failed = call_task(
                name_to_call, module, args, kwargs, self.capture_errors
            )
            this_time = time.time() - call_time
            self.n_processed[rank] += 1
            self.stats.append(
                {
                    "id": task_id,
                    "rank": rank,
                    "this_time": this_time,
                    "time_over_est": this_time / time_est,
                    "n_processed": self.n_processed[rank],
                    "total_time": time.time() - start_time,
                    "error": failed,
                }
            )
            # A failure is sent as its task_failure() description, flagged
            # in the stats; a return value is sent as a TaskResult.
            payload = result if failed else TaskResult(result)
            _isend_bytes(
                self.comm,
                pickle.dumps((task_id, payload, self.stats[-1])),
                dest=0,
                tag=MessageTag.DONE.value,
            )
            payload = None
            # Drop the sent result now rather than holding it, which may be
            # large, until the next call returns.
            result = None

    def abort(self):
        traceback.print_exc()
        logger.info(f"MPI worker {self.worker_id}: aborting...")
        self.comm.Abort(1)


def _leave_group(merged_comm: Intracomm, spawn_comm: Intercomm | None) -> None:
    """Disconnect from a worker group; every member of the group calls this.

    A split group lives in one job, and its merged communicator is
    disconnected.  A spawned group joins two jobs; there the merged
    communicator is freed and the intercommunicator from MPI_Comm_spawn
    is disconnected instead, because disconnecting a merged communicator
    that spans jobs does not complete under Open MPI.
    """
    if spawn_comm is None:
        merged_comm.Disconnect()
    else:
        merged_comm.Free()
        spawn_comm.Disconnect()


class MPICollectiveWorker:
    def __init__(
        self,
        comm: Intracomm,
        merged_comm: Intracomm,
        worker_id: int,
        n_workers: int,
        worker_service_name: str = "distwq.init",
        collective_mode: CollectiveMode = CollectiveMode.Gather,
        capture_errors: bool = False,
        spawn_comm: Intercomm | None = None,
    ):
        size = comm.size
        rank = comm.rank

        self.collective_mode = collective_mode
        # Report task exceptions to the broker instead of aborting; the
        # broker passes them on to the controller.
        self.capture_errors = capture_errors
        self.n_workers = n_workers
        self.worker_id = worker_id
        self.comm = comm
        self.merged_comm = merged_comm
        # The intercommunicator to the broker, for a spawned worker; None
        # in a split group.
        self.spawn_comm = spawn_comm

        self.worker_port = None
        self.server_worker_comm = None
        self.client_worker_comms = []
        self.worker_service_name = worker_service_name
        self.service_published = False
        self.is_worker = True
        self.is_broker = False

        self.start_time = start_time
        self.total_time_est = np.zeros(size) * np.nan
        self.total_time_est[rank] = 0
        self.n_processed = np.zeros(size) * np.nan
        self.n_processed[rank] = 0
        self.total_time = np.zeros(size) * np.nan
        self.total_time[rank] = 0
        self.stats = []
        logger.info(f"MPI collective worker {self.worker_id}: initialized.")

    def publish_service(self) -> None:
        if not self.service_published:
            if self.comm.rank == 0:
                try:
                    found = MPI.Lookup_name(self.worker_service_name)
                    if found:
                        MPI.Unpublish_name(self.worker_service_name, found)
                except MPI.Exception:
                    pass
                self.worker_port = MPI.Open_port()
                info = MPI.INFO_NULL
                MPI.Publish_name(self.worker_service_name, info, self.worker_port)
                self.comm.bcast(self.worker_port, root=0)
            else:
                self.worker_port = self.comm.bcast(None, root=0)
            self.comm.barrier()
            self.service_published = True

    def connect_service(self, n_lookup_attempts: int = 5) -> None:
        info = MPI.INFO_NULL
        if not self.service_published:
            if self.comm.rank == 0:
                attempt = 0
                while attempt < n_lookup_attempts:
                    try:
                        self.worker_port = MPI.Lookup_name(self.worker_service_name)
                    except MPI.Exception as e:
                        if e.Get_error_class() == MPI.ERR_NAME:
                            time.sleep(random.randrange(1, 5))
                        else:
                            raise
                    attempt += 1
                assert self.worker_port is not None
                self.comm.bcast(self.worker_port, root=0)
            else:
                self.worker_port = self.comm.bcast(None, root=0)
            self.comm.barrier()
            if not self.worker_port:
                raise RuntimeError(
                    f"connect_service: unable to lookup service "
                    f"{self.worker_service_name}"
                )
            self.server_worker_comm = self.comm.Connect(self.worker_port, info, root=0)
        else:
            for i in range(self.n_workers - 1):
                client_worker_comm = self.comm.Accept(self.worker_port, info, root=0)
                self.client_worker_comms.append(client_worker_comm)

    def serve(self) -> MessageTag:
        """
        Serve submitted calls until told to finish. Tasks are
        obtained via scatter and results are returned via gather,
        i.e. all collective workers spawned by a CollectiveBroker
        will participate in these collective calls.

        Returns MessageTag.SESSION_END when the broker relays the end of
        a session, keeping all communicators open so that serve() can be
        called again, or MessageTag.EXIT after closing them on exit.

        Call this function if workers need to perform initialization
        different from the controller, like this:

        >>> def workerfun(worker):
        >>>     do = whatever + initialization - is * necessary
        >>>     worker.serve()
        >>>     do = whatever + cleanup - is * necessary

        If you don't define workerfun(), serve() will be called automatically by
        run().
        """

        rank = self.comm.rank
        merged_rank = self.merged_comm.Get_rank()
        logger.info(
            f"MPI collective worker {self.worker_id}-{rank}: waiting for calls."
        )

        # wait for orders:
        while True:
            if rank == 0:
                logger.info(
                    f"MPI collective worker {self.worker_id}-{rank}: "
                    "getting next task from queue..."
                )
            # get next task from queue:
            name_to_call, args, kwargs, module, time_est, task_id = self.get_next_task()
            #            if rank == 0:
            logger.info(
                f"MPI collective worker {self.worker_id}-{rank}: "
                "received next task from queue."
            )
            # TODO: add timeout and check whether controller lives!
            if name_to_call == "session_end":
                logger.info(
                    f"MPI collective worker {self.worker_id}-{rank}: session ended."
                )
                return MessageTag.SESSION_END
            if name_to_call == "broadcast":
                self.broadcast_value = args[0]
                return MessageTag.BROADCAST
            if name_to_call == "exit":
                logger.info(
                    f"MPI collective worker {self.worker_id}-{rank}: exiting..."
                )
                if self.server_worker_comm is not None:
                    self.server_worker_comm.Disconnect()
                for client_worker_comm in self.client_worker_comms:
                    client_worker_comm.Disconnect()
                if self.service_published and (rank == 0):
                    MPI.Unpublish_name(self.worker_service_name, self.worker_port)
                    MPI.Close_port(self.worker_port)
                _leave_group(self.merged_comm, self.spawn_comm)
                return MessageTag.EXIT
            self.total_time_est[rank] += time_est
            call_time = time.time()

            # A failure is still gathered like a result, so that the
            # broker and the other workers are not left waiting.
            result, failed = call_task(
                name_to_call, module, args, kwargs, self.capture_errors
            )
            this_time = time.time() - call_time
            self.n_processed[rank] += 1
            self.stats.append(
                {
                    "id": task_id,
                    "rank": merged_rank,
                    "this_time": this_time,
                    "time_over_est": this_time / time_est,
                    "n_processed": self.n_processed[rank],
                    "total_time": time.time() - start_time,
                    "error": failed,
                }
            )
            self.gather_results(result)
            result = None

    def gather_results(self, result: Any) -> None:
        if self.collective_mode == CollectiveMode.Gather:
            req = self.merged_comm.Ibarrier()
            self.merged_comm.gather((result, self.stats[-1]), root=0)
            req.wait()
        elif self.collective_mode == CollectiveMode.SendRecv:
            req = self.merged_comm.isend(
                (result, self.stats[-1]), dest=0, tag=MessageTag.DONE.value
            )
            req.wait()
        else:
            raise RuntimeError(
                f"MPICollectiveWorker: unknown collective mode {self.collective_mode}"
            )

    def get_next_task(self) -> tuple[str, tuple[Any], dict[Any, Any], str, int, int]:
        if self.collective_mode == CollectiveMode.Gather:
            req = self.merged_comm.Ibarrier()
            (
                name_to_call,
                args,
                kwargs,
                module,
                time_est,
                task_id,
            ) = self.merged_comm.scatter(None, root=0)
            req.wait()
        elif self.collective_mode == CollectiveMode.SendRecv:
            buf = bytearray(1 << 20)  # TODO: 1 MB buffer, make it configurable
            req = self.merged_comm.irecv(buf, source=0, tag=MessageTag.TASK.value)
            (name_to_call, args, kwargs, module, time_est, task_id) = req.wait()
        else:
            raise RuntimeError(f"Unknown collective mode {self.collective_mode}")
        return (name_to_call, args, kwargs, module, time_est, task_id)

    def abort(self) -> None:
        rank = self.comm.rank
        traceback.print_exc()
        logger.info(f"MPI collective worker {self.worker_id}-{rank}: aborting...")
        self.comm.Abort(1)


class MPICollectiveBroker:
    def __init__(
        self,
        worker_id: int,
        comm: Intracomm,
        group_comm: Intracomm,
        merged_comm: Intracomm,
        nprocs_per_worker: int,
        ready_data: Any | None = None,
        is_worker: bool = False,
        collective_mode: CollectiveMode = CollectiveMode.Gather,
        capture_errors: bool = False,
        spawn_comm: Intercomm | None = None,
    ) -> None:
        logger.info(f"MPI collective broker {worker_id} starting")
        assert not spawned

        size = comm.size
        rank = comm.rank
        merged_size = merged_comm.size
        merged_rank = merged_comm.rank

        self.collective_mode = collective_mode
        self.comm = comm
        self.group_comm = group_comm
        self.merged_comm = merged_comm
        # The intercommunicator to the spawned workers, from
        # MPI_Comm_spawn; None in a split group.
        self.spawn_comm = spawn_comm
        self.worker_id = worker_id
        self.nprocs_per_worker = nprocs_per_worker

        self.start_time = start_time
        self.total_time_est = np.zeros(size) * np.nan
        self.total_time_est[rank] = 0
        self.n_processed = np.zeros(merged_size) * np.nan
        self.n_processed[merged_rank] = 0
        self.total_time = np.zeros(merged_size) * np.nan
        self.total_time[merged_rank] = 0
        self.stats = []
        self.is_worker = is_worker
        self.is_broker = True
        self.ready_data = ready_data
        # Report task exceptions, its own or its workers', to the
        # controller instead of aborting.
        self.capture_errors = capture_errors
        logger.info(f"MPI collective broker {self.worker_id}: initialized.")

    def serve(self) -> MessageTag:
        """
        Broker and serve submitted calls until told to finish. A task
        is received from the controller and sent to all collective
        workers associated with this broker via scatter.

        The end of a session and the final exit are relayed to the
        collective workers in the same way.  Returns the tag that ended
        serving: MessageTag.SESSION_END or MessageTag.EXIT.

        Call this function if workers need to perform initialization
        different from the controller, like this:

        >>> def workerfun(worker):
        >>>     do = whatever + initialization - is * necessary
        >>>     worker.serve()
        >>>     do = whatever + cleanup - is * necessary

        If you don't define workerfun(), serve() will be called automatically by
        run().
        """
        rank = self.comm.rank
        merged_rank = self.merged_comm.Get_rank()

        logger.info(f"MPI collective broker {self.worker_id}: waiting for calls.")

        # wait for orders:
        while True:
            # signal the controller this worker is ready
            _isend_bytes(
                self.comm,
                pickle.dumps(self.ready_data),
                dest=0,
                tag=MessageTag.READY.value,
            )
            logger.info(
                f"MPI collective broker {self.worker_id}: "
                "getting next task from controller..."
            )

            tag, data = self.process()

            logger.info(
                f"MPI collective broker {self.worker_id}: "
                "received message from controller..."
            )

            if tag == MessageTag.EXIT.value:
                logger.info(f"MPI collective broker {self.worker_id}: exiting...")
                self.scatter_task("exit", (), {}, "", 0, 0)
                # The collective workers leave the group when told to exit,
                # and leaving is collective, so the broker leaves with them.
                _leave_group(self.merged_comm, self.spawn_comm)
                return MessageTag.EXIT
            elif tag == MessageTag.SESSION_END.value:
                logger.info(f"MPI collective broker {self.worker_id}: session ended.")
                self.scatter_task("session_end", (), {}, "", 0, 0)
                return MessageTag.SESSION_END
            elif tag == MessageTag.BROADCAST.value:
                self.broadcast_value = data
                self.scatter_task("broadcast", (data,), {}, "", 0, 0)
                return MessageTag.BROADCAST
            elif tag == MessageTag.TASK.value:
                (name_to_call, args, kwargs, module, time_est, task_id) = data
            else:
                raise RuntimeError(f"MPI collective broker: unknown message tag {tag}")

            logger.info(
                f"MPI collective broker {self.worker_id}: "
                f"sending task {task_id} to workers..."
            )
            self.scatter_task(name_to_call, args, kwargs, module, time_est, task_id)
            logger.info(
                f"MPI collective broker {self.worker_id}: sending task complete."
            )

            self.total_time_est[rank] += time_est
            if self.is_worker:
                call_time = time.time()
                # A failure is still gathered like a result, so that the
                # collective workers are not left waiting.
                this_result, failed = call_task(
                    name_to_call, module, args, kwargs, self.capture_errors
                )
                this_time = time.time() - call_time
                self.n_processed[merged_rank] += 1
                this_stat = {
                    "id": task_id,
                    "rank": merged_rank,
                    "this_time": this_time,
                    "time_over_est": this_time / time_est,
                    "n_processed": self.n_processed[merged_rank],
                    "total_time": time.time() - start_time,
                    "error": failed,
                }
            else:
                this_result = None
                this_stat = None
                this_time = 0

            logger.info(
                f"MPI collective broker {self.worker_id}: "
                "gathering data from workers..."
            )
            results, stats = self.gather_results(this_result, this_stat)
            # The group's values as a CollectiveTaskResult, or its first
            # failure in their place, flagged in the stat sent with it.
            payload, stat = _group_outcome(results, stats)
            logger.info(
                f"MPI collective broker {self.worker_id}: "
                "sending results to controller..."
            )
            _isend_bytes(
                self.comm,
                pickle.dumps((task_id, payload, stat)),
                dest=0,
                tag=MessageTag.DONE.value,
            )
            results = payload = this_result = None

    def scatter_task(
        self,
        name_to_call: str,
        args: tuple[Any],
        kwargs: dict[Any, Any],
        module: str,
        time_est: int,
        task_id: int,
    ) -> None:
        merged_rank = self.merged_comm.Get_rank()
        merged_size = self.merged_comm.Get_size()

        if self.collective_mode == CollectiveMode.Gather:
            req = self.merged_comm.Ibarrier()
            self.merged_comm.scatter(
                [(name_to_call, args, kwargs, module, time_est, task_id)] * merged_size,
                root=merged_rank,
            )
            req.wait()
        elif self.collective_mode == CollectiveMode.SendRecv:
            msg = (name_to_call, args, kwargs, module, time_est, task_id)
            reqs = []
            status = []
            for i in range(
                self.nprocs_per_worker - 1 if self.is_worker else self.nprocs_per_worker
            ):
                dest = i + 1
                req = self.merged_comm.isend(msg, dest=dest, tag=MessageTag.TASK.value)
                reqs.append(req)
                status.append(MPI.Status())
            MPI.Request.waitall(reqs, status)
        else:
            raise RuntimeError(
                f"MPICollectiveBroker: unknown collective mode {self.collective_mode}"
            )

    def gather_results(
        self, this_result: Any, this_stat: Any
    ) -> tuple[list[Any], list[dict[str, int | float | float64]]]:
        merged_rank = self.merged_comm.Get_rank()
        if self.collective_mode == CollectiveMode.Gather:
            req = self.merged_comm.Ibarrier()
            sub_data = self.merged_comm.gather(
                (this_result, this_stat), root=merged_rank
            )
            req.wait()
            results, stats = _split_group_data(sub_data)
        elif self.collective_mode == CollectiveMode.SendRecv:
            reqs = []
            for i in range(
                self.nprocs_per_worker - 1 if self.is_worker else self.nprocs_per_worker
            ):
                buf = bytearray(1 << 20)  # TODO: 1 MB buffer, make it configurable
                source = i + 1
                req = self.merged_comm.irecv(
                    buf,
                    source=source,
                    tag=MessageTag.DONE.value,
                )
                reqs.append(req)
            status = [MPI.Status() for i in range(self.nprocs_per_worker)]
            try:
                sub_data = MPI.Request.waitall(reqs, status)
            except MPI.Exception:
                logger.error([MPI.Get_error_string(s.Get_error()) for s in status])
                raise
            del reqs
            results, stats = _split_group_data(sub_data)
            # A working broker always has a stat, even if its call
            # returned None.
            if this_stat is not None:
                results = [this_result] + results
                stats = [this_stat] + stats
        else:
            raise RuntimeError(
                f"MPICollectiveBroker: unknown collective mode {self.collective_mode}"
            )
        return results, stats

    def process(
        self,
    ) -> tuple[int, None | tuple[str, tuple[int], dict[Any, Any], str, int, int]]:
        # Blocks until the controller sends something.
        status = MPI.Status()
        self.comm.Probe(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()
        data = self.comm.recv(source=0, tag=tag)
        return tag, data

    def abort(self):
        rank = self.comm.rank
        traceback.print_exc()
        logger.info(f"MPI collective broker {rank}: aborting...")
        self.comm.Abort(1)


def do_spawn_workers(
    fun: Callable | None,
    fun_name: str | None,
    module_name: str | None,
    world_comm: Intracomm,
    group_comm: Intracomm,
    worker_id: int,
    n_workers: int,
    nprocs_per_worker: int,
    broker_is_worker: bool,
    collective_mode: str,
    enable_worker_service: bool,
    worker_service_name: str,
    spawn_args: list[Any],
    sequential_spawn: bool,
    spawn_startup_wait: int,
    spawn_executable: str | None,
    verbose: bool,
    capture_errors: bool = False,
) -> tuple[Intercomm, Intracomm, CollectiveMode]:
    worker_config = {
        "worker_id": worker_id,
        "n_workers": n_workers,
        "collective_mode": collective_mode,
        "enable_worker_service": enable_worker_service,
        "worker_service_name": worker_service_name,
        "verbose": verbose,
        "capture_errors": capture_errors,
    }
    if fun is not None:
        worker_config["init_fun_name"] = str(fun_name)
        worker_config["init_module_name"] = str(module_name)
    spawn_stmts = (
        f"import sys, runpy; sys.argv.extend"
        f"(['-', 'distwq:spawned', '{json.dumps(worker_config)}']); "
        f"runpy.run_module('distwq', run_name='__main__'); sys.exit()"
    )
    if callable(spawn_args):
        arglist = spawn_args(["-c", spawn_stmts])
    else:
        arglist = spawn_args + ["-c", spawn_stmts]
    logger.info(f"MPI broker {worker_id} : before spawn")
    worker_id = group_comm.rank + 1
    if collective_mode.lower() == "gather":
        collective_mode_arg = CollectiveMode.Gather
    elif collective_mode.lower() == "sendrecv":
        collective_mode_arg = CollectiveMode.SendRecv
    else:
        raise RuntimeError(f"Unknown collective mode {collective_mode}")

    if sequential_spawn and (worker_id > 1):
        status = MPI.Status()
        _ = group_comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
    if spawn_startup_wait is not None:
        spawn_startup_wait = max(spawn_startup_wait, 2)
        time.sleep(random.randrange(1, spawn_startup_wait))
    req = world_comm.Ibarrier()
    try:
        if spawn_executable is None:
            spawn_executable = sys.executable
        sub_comm = MPI.COMM_SELF.Spawn(
            spawn_executable,
            args=arglist,
            maxprocs=nprocs_per_worker - 1 if broker_is_worker else nprocs_per_worker,
        )
    except MPI.Exception:
        logger.error(f"MPI broker {worker_id} : spawn error")
        raise
    if sequential_spawn and (worker_id < n_workers):
        group_comm.send("spawn", dest=worker_id)
    logger.info(f"MPI broker {worker_id} : after spawn")
    req.wait()
    req = sub_comm.Ibarrier()
    merged_comm = sub_comm.Merge(False)
    req.wait()
    return sub_comm, merged_comm, collective_mode_arg


def do_split_workers(
    fun: Callable | None,
    fun_name: str | None,
    module_name: str | None,
    world_comm: Intracomm,
    group_comm: Intracomm,
    worker_id: int,
    n_workers: int,
    nprocs_per_worker: int,
    broker_is_worker: bool,
    collective_mode: str,
    is_broker: bool,
    broker_ranks: tuple[int],
    verbose: bool,
) -> tuple[Intracomm, Intracomm, CollectiveMode]:
    worker_config = {
        "worker_id": worker_id,
        "n_workers": n_workers,
        "collective_mode": collective_mode,
        "enable_worker_service": False,
        "worker_service_name": None,
        "verbose": verbose,
    }
    if fun is not None:
        worker_config["init_fun_name"] = str(fun_name)
        worker_config["init_module_name"] = str(module_name)
    if is_broker:
        logger.info(f"MPI broker {worker_id} : before split")
    else:
        logger.info(f"MPI worker {worker_id} (rank {group_comm.rank}) : before split")

    if collective_mode.lower() == "gather":
        collective_mode_arg = CollectiveMode.Gather
    elif collective_mode.lower() == "sendrecv":
        collective_mode_arg = CollectiveMode.SendRecv
    else:
        raise RuntimeError(f"Unknown collective mode {collective_mode}")

    this_broker_rank = None
    this_worker_ranks = None
    if broker_is_worker:
        this_worker_ranks = set(
            range((worker_id - 1) * nprocs_per_worker, worker_id * nprocs_per_worker)
        )
        this_broker_rank = next(iter(this_worker_ranks))
    else:
        this_worker_ranks = set(
            range(
                (worker_id - 1) * (nprocs_per_worker + 1),
                worker_id * (nprocs_per_worker + 1),
            )
        )
        this_broker_rank = (worker_id - 1) * (nprocs_per_worker + 1)
        this_worker_ranks.remove(this_broker_rank)

    group_rank = group_comm.rank
    color = (
        worker_id
        if group_rank in this_worker_ranks
        else (
            worker_id + n_workers if group_rank == this_broker_rank else MPI.UNDEFINED
        )
    )

    local_comm = group_comm.Split(color, key=group_comm.rank)
    if is_broker:
        local_leader = 0
        remote_leader = next(iter(this_worker_ranks))
    else:
        local_leader = 0
        remote_leader = this_broker_rank

    if broker_is_worker:
        sub_comm = local_comm.Dup()
    else:
        sub_comm = local_comm.Create_intercomm(
            local_leader, group_comm, remote_leader, tag=0
        )

    if broker_is_worker:
        merged_comm = sub_comm.Dup()
    else:
        req = sub_comm.Ibarrier()
        merged_comm = sub_comm.Merge(False)
        req.wait()

    if is_broker:
        logger.info(f"MPI broker {worker_id} : after split")
    else:
        logger.info(f"MPI worker {worker_id} (rank {group_comm.rank}) : after split")

    global my_config
    my_config = {}
    my_config.update(worker_config)

    return sub_comm, local_comm, merged_comm, collective_mode_arg


def get_group_comm(
    world_comm: Intracomm, size: int, rank: int, is_controller: bool, spawned: bool
) -> tuple[int, Intracomm]:
    group_id = 0
    group_comm = None
    if not spawned and (size > 1):
        if is_controller:
            group_id = 2
        elif is_worker:
            group_id = 1
        else:
            group_id = 3
        group_comm = world_comm.Split(group_id, key=rank)
    else:
        group_comm = world_comm

    return group_id, group_comm


def get_fun(fun_name: str | None, module_name: str) -> Callable | None:
    fun = None
    if fun_name is not None:
        if module_name not in sys.modules:
            importlib.import_module(module_name)
        fun = eval(fun_name, sys.modules[module_name].__dict__)
    return fun


def check_worker_grouping_method(worker_grouping_method):
    if isinstance(worker_grouping_method, str):
        if worker_grouping_method.lower() == "spawn":
            return GroupingMethod.GroupSpawn
        elif worker_grouping_method.lower() == "split":
            return GroupingMethod.GroupSplit
        else:
            raise RuntimeError(f"Unknown grouping method {worker_grouping_method}")
    else:
        return worker_grouping_method


def check_spawn_config(
    spawn_workers: bool,
    nprocs_per_worker: int,
    broker_is_worker: bool,
    enable_worker_service: bool,
    broker_fun: None,
) -> None:
    if broker_fun is not None:
        if spawn_workers is not True:
            raise RuntimeError(
                "distwq.run: cannot use broker_fun_name when "
                "spawn_workers is set to False"
            )
        if broker_is_worker:
            raise RuntimeError(
                "distwq.run: cannot use broker_fun_name when "
                "broker_is_worker is set to True"
            )
    if spawn_workers and (nprocs_per_worker == 1) and broker_is_worker:
        raise RuntimeError(
            "distwq.run: cannot spawn workers when nprocs_per_worker=1 "
            "and broker_is_worker is set to True"
        )
    if enable_worker_service and (not spawn_workers):
        raise RuntimeError(
            "distwq.run: cannot enable worker service when "
            "spawn_workers is set to False"
        )


def check_split_config(
    split_workers: bool,
    nprocs_per_worker: int,
    broker_is_worker: bool,
    broker_fun: None,
    world_size: int,
) -> None:
    """Raise RuntimeError if the split configuration cannot be used.

    world_size counts the worker processes plus one for the controller.
    """
    if broker_fun is not None:
        if split_workers is not True:
            raise RuntimeError(
                "distwq.run: cannot use broker_fun_name when "
                "split_workers is set to False"
            )
        if broker_is_worker:
            raise RuntimeError(
                "distwq.run: cannot use broker_fun_name when "
                "broker_is_worker is set to True"
            )
    if split_workers:
        if broker_is_worker:
            if (world_size - 1) % nprocs_per_worker != 0:
                raise RuntimeError(
                    "distwq.run: unable to split workers evenly "
                    f"into groups of {nprocs_per_worker} "
                    f"when there are {world_size - 1} worker processes "
                    "(the world size, or max_workers, less 1 process "
                    "for the controller)"
                )
        else:
            if (world_size - 1) % (nprocs_per_worker + 1) != 0:
                raise RuntimeError(
                    f"distwq.run: unable to split workers evenly "
                    f"into groups of {nprocs_per_worker + 1} "
                    f"when there are {world_size - 1} worker processes "
                    "(the world size, or max_workers, less 1 process "
                    "for the controller; 1 process per worker group is "
                    "needed for brokers)"
                )


def _call_init(
    fun: Callable,
    server: Any,
    args: tuple[Any, ...],
    what: str,
    abort_comm: Intracomm | None = None,
) -> None:
    """Call an init function as fun(server, *args); abort the job if it raises.

    An init function runs on one process while the others go on with
    their own part of the setup, or wait for work.  An exception left to
    propagate from it would stop only this process, and the rest of the
    job would wait for it forever.  So the exception is logged with its
    traceback, and abort_comm (by default the world communicator) is
    aborted with exit status 1.
    """
    try:
        fun(server, *args)
    except BaseException:
        logger.exception(f"distwq: {what} raised an exception; aborting the job")
        sys.stdout.flush()
        sys.stderr.flush()
        (abort_comm if abort_comm is not None else MPI.COMM_WORLD).Abort(1)
        raise


def setup_spawn_group(
    worker_id,
    spawn_workers,
    nprocs_per_worker,
    broker_is_worker,
    broker_fun,
    fun,
    fun_name,
    module_name,
    args,
    world_comm,
    group_comm,
    n_workers,
    collective_mode,
    enable_worker_service,
    worker_service_name,
    spawn_args,
    sequential_spawn,
    spawn_startup_wait,
    spawn_executable,
    verbose,
    capture_errors=False,
):
    """Spawn this broker's collective workers and return the broker.

    With capture_errors=True the broker and its workers report task
    exceptions to the controller instead of aborting.

    Performs every collective step of spawn-based grouping, including the
    optional initialization functions, but does not start serving.  Call
    serve() on the returned broker to do that.
    """
    check_spawn_config(
        spawn_workers,
        nprocs_per_worker,
        broker_is_worker,
        enable_worker_service,
        broker_fun,
    )
    spawn_comm, merged_comm, collective_mode_arg = do_spawn_workers(
        fun,
        fun_name,
        module_name,
        world_comm,
        group_comm,
        worker_id,
        n_workers,
        nprocs_per_worker,
        broker_is_worker,
        collective_mode,
        enable_worker_service,
        worker_service_name,
        spawn_args,
        sequential_spawn,
        spawn_startup_wait,
        spawn_executable,
        verbose,
        capture_errors=capture_errors,
    )
    broker = MPICollectiveBroker(
        worker_id,
        world_comm,
        group_comm,
        merged_comm,
        nprocs_per_worker,
        is_worker=broker_is_worker,
        collective_mode=collective_mode_arg,
        capture_errors=capture_errors,
        spawn_comm=spawn_comm,
    )
    if fun is not None:
        req = merged_comm.Ibarrier()
        merged_comm.bcast(args, root=0)
        req.wait()
        if broker_is_worker:
            _call_init(fun, broker, args, "worker init function")
    if broker_fun is not None:
        _call_init(broker_fun, broker, args, "broker init function")
    return broker


def run_spawn_group(
    worker_id,
    spawn_workers,
    nprocs_per_worker,
    broker_is_worker,
    broker_fun,
    fun,
    fun_name,
    module_name,
    args,
    world_comm,
    group_comm,
    n_workers,
    collective_mode,
    enable_worker_service,
    worker_service_name,
    spawn_args,
    sequential_spawn,
    spawn_startup_wait,
    spawn_executable,
    verbose,
):
    setup_spawn_group(
        worker_id,
        spawn_workers,
        nprocs_per_worker,
        broker_is_worker,
        broker_fun,
        fun,
        fun_name,
        module_name,
        args,
        world_comm,
        group_comm,
        n_workers,
        collective_mode,
        enable_worker_service,
        worker_service_name,
        spawn_args,
        sequential_spawn,
        spawn_startup_wait,
        spawn_executable,
        verbose,
    ).serve()


def setup_split_group(
    worker_id,
    n_workers,
    split_workers,
    is_broker,
    broker_ranks,
    nprocs_per_worker,
    broker_is_worker,
    broker_fun,
    fun,
    fun_name,
    module_name,
    args,
    world_comm,
    controller_worker_comm,
    group_comm,
    collective_mode,
    verbose,
    capture_errors=False,
):
    """Split this rank into its worker group and return its server.

    With capture_errors=True the group reports task exceptions to the
    controller instead of aborting.

    Returns the broker on broker ranks and the collective worker on the
    other ranks of the group.  Performs every collective step of
    split-based grouping, including the optional initialization functions,
    but does not start serving.  Call serve() on the returned object to
    do that.
    """
    broker = None
    worker = None
    # group_comm holds the worker ranks only, which max_workers can make
    # fewer than the world size implies; the size given counts one more
    # rank, for the controller.
    check_split_config(
        split_workers,
        nprocs_per_worker,
        broker_is_worker,
        broker_fun,
        group_comm.size + 1,
    )
    _sub_comm, local_comm, merged_comm, collective_mode_arg = do_split_workers(
        fun,
        fun_name,
        module_name,
        world_comm,
        group_comm,
        worker_id,
        n_workers,
        nprocs_per_worker,
        broker_is_worker,
        collective_mode,
        is_broker,
        broker_ranks,
        verbose,
    )

    if is_broker:
        local_comm.Free()
        broker = MPICollectiveBroker(
            worker_id,
            controller_worker_comm,
            group_comm,
            merged_comm,
            nprocs_per_worker,
            is_worker=broker_is_worker,
            collective_mode=collective_mode_arg,
            capture_errors=capture_errors,
        )
        req = controller_worker_comm.Ibarrier()
        req.wait()
    elif is_worker:
        worker = MPICollectiveWorker(
            local_comm,
            merged_comm,
            worker_id,
            n_workers,
            collective_mode=collective_mode_arg,
            capture_errors=capture_errors,
        )

    if fun is not None:
        req = merged_comm.Ibarrier()
        merged_comm.bcast(args, root=0)
        req.wait()
        # A broker that only relays calls computes nothing, so it runs no
        # worker init function; broker_fun is its init function.
        if is_broker and broker_is_worker:
            _call_init(fun, broker, args, "worker init function")
        elif not is_broker:
            _call_init(fun, worker, args, "worker init function")
    if is_broker:
        if broker_fun is not None:
            _call_init(broker_fun, broker, args, "broker init function")
        return broker
    return worker


def run_split_group(
    worker_id,
    n_workers,
    split_workers,
    is_broker,
    broker_ranks,
    nprocs_per_worker,
    broker_is_worker,
    broker_fun,
    fun,
    fun_name,
    module_name,
    args,
    world_comm,
    controller_worker_comm,
    group_comm,
    collective_mode,
    verbose,
):
    setup_split_group(
        worker_id,
        n_workers,
        split_workers,
        is_broker,
        broker_ranks,
        nprocs_per_worker,
        broker_is_worker,
        broker_fun,
        fun,
        fun_name,
        module_name,
        args,
        world_comm,
        controller_worker_comm,
        group_comm,
        collective_mode,
        verbose,
    ).serve()


@dataclass
class Topology:
    """The communicators and role of this rank after establish_topology().

    role is one of:

    - "controller": submits calls; controller_worker_comm connects it to
      every worker, or to every broker under worker grouping.
    - "worker": a single-process worker; server is its MPIWorker.
    - "broker": relays calls to a worker group; server is its
      MPICollectiveBroker.
    - "collective_worker": a member of a split worker group that is not
      its broker; server is its MPICollectiveWorker.
    - "idle": a rank left out by max_workers; it takes no part in the work.

    grouping_method is the method actually used, which can differ from
    the requested one (see establish_topology).
    """

    role: str
    grouping_method: GroupingMethod
    controller_worker_comm: Intracomm | None = None
    group_comm: Intracomm | None = None
    group_id: int = 0
    worker_id: int | None = None
    server: Any = None


def _worker_split_layout(group_comm, nprocs_per_worker, broker_is_worker):
    """Return (number of groups, this rank's worker id, broker ranks) for a
    split of group_comm into worker groups of nprocs_per_worker processes,
    plus one extra broker process per group when brokers do not work."""
    stride = nprocs_per_worker if broker_is_worker else nprocs_per_worker + 1
    n_groups = group_comm.size // stride
    worker_id = (group_comm.rank // stride) + 1
    broker_ranks = {x * stride for x in range(n_groups)}
    return n_groups, worker_id, broker_ranks


def establish_topology(
    worker_grouping_method: str | GroupingMethod = GroupingMethod.NoGrouping,
    nprocs_per_worker: int = 1,
    broker_is_worker: bool = False,
    collective_mode: str = "gather",
    enable_worker_service: bool = False,
    worker_service_name: str = "distwq.init",
    sequential_spawn: bool = False,
    spawn_startup_wait: int | None = None,
    spawn_executable: str | None = None,
    spawn_args: list[Any] | None = None,
    fun: Callable | None = None,
    fun_name: str | None = None,
    module_name: str = "__main__",
    broker_fun: Callable | None = None,
    args: tuple[Any] = (),
    verbose: bool = False,
    capture_errors: bool = False,
) -> Topology:
    """Set up the controller, workers and worker groups; call on every rank.

    This is the one sequence of collective MPI calls shared by run() and
    distwq.future.DistributedEnvironment, so that the controller and the
    workers always make the same calls in the same order.  It:

    1. checks the grouping configuration on every rank, before any
       collective call, so an invalid configuration fails everywhere
       instead of leaving the valid ranks waiting;
    2. splits the world into the controller, the worker ranks and any
       idle ranks (get_group_comm);
    3. splits off the controller-worker communicator, joining the
       controller with every worker, or with every broker under grouping;
    4. sets up the worker groups, if any, and passes the barrier on the
       controller-worker communicator that tells the controller every
       worker is ready.

    Under GroupSpawn or GroupSplit, fun, broker_fun and args are passed to
    setup_spawn_group() or setup_split_group(), which call them on the
    group's processes.  Without grouping they are not used here: the
    caller, run() or DistributedEnvironment, calls fun on each worker
    itself.  An init function that raises aborts the whole job (see
    _call_init()).

    NoGrouping with nprocs_per_worker > 1 is treated as GroupSplit, as
    run() has always done.

    With capture_errors=True every worker-side server reports a task's
    exception to the controller in place of its result, instead of letting
    it abort the job.  run() leaves this off.

    The returned servers have not started serving.  Call serve() on
    Topology.server to do that.
    """
    method = check_worker_grouping_method(worker_grouping_method)
    if method == GroupingMethod.NoGrouping and nprocs_per_worker > 1:
        method = GroupingMethod.GroupSplit
    if spawn_args is None:
        spawn_args = []

    if not has_mpi:
        return Topology(role="controller", grouping_method=method)

    world_size = world_comm.size
    if world_size < 2 or n_workers < 1:
        # No workers: the controller runs every call itself.
        return Topology(
            role="controller",
            grouping_method=GroupingMethod.NoGrouping,
            controller_worker_comm=world_comm,
            group_comm=world_comm,
        )

    grouped = method in (GroupingMethod.GroupSpawn, GroupingMethod.GroupSplit)
    if method == GroupingMethod.GroupSpawn:
        check_spawn_config(
            True, nprocs_per_worker, broker_is_worker, enable_worker_service, broker_fun
        )
    elif method == GroupingMethod.GroupSplit:
        # Only the worker ranks are split into groups, which max_workers can
        # make fewer than the world size implies; the size given counts one
        # more rank, for the controller.
        check_split_config(
            True, nprocs_per_worker, broker_is_worker, broker_fun, n_workers + 1
        )

    group_id, group_comm = get_group_comm(
        world_comm, world_size, rank, is_controller, spawned
    )

    if is_controller:
        # Under grouping the controller shares color 1 with the brokers;
        # otherwise color 0 with all workers.
        controller_worker_comm = world_comm.Split(1 if grouped else 0, key=0)
        controller_worker_comm.Ibarrier().wait()
        return Topology(
            role="controller",
            grouping_method=method,
            controller_worker_comm=controller_worker_comm,
            group_comm=group_comm,
            group_id=group_id,
        )

    if not is_worker:
        # Left out by max_workers: take part in the Split, but in no
        # communicator.
        world_comm.Split(MPI.UNDEFINED, key=1)
        return Topology(
            role="idle",
            grouping_method=method,
            group_comm=group_comm,
            group_id=group_id,
        )

    if method == GroupingMethod.GroupSpawn:
        worker_id = group_comm.rank + 1
        controller_worker_comm = world_comm.Split(1, key=1)
        broker = setup_spawn_group(
            worker_id,
            True,
            nprocs_per_worker,
            broker_is_worker,
            broker_fun,
            fun,
            fun_name,
            module_name,
            args,
            controller_worker_comm,
            group_comm,
            group_comm.size,
            collective_mode,
            enable_worker_service,
            worker_service_name,
            spawn_args,
            sequential_spawn,
            spawn_startup_wait,
            spawn_executable,
            verbose,
            capture_errors=capture_errors,
        )
        return Topology(
            role="broker",
            grouping_method=method,
            controller_worker_comm=controller_worker_comm,
            group_comm=group_comm,
            group_id=group_id,
            worker_id=worker_id,
            server=broker,
        )

    if method == GroupingMethod.GroupSplit:
        n_groups, worker_id, broker_ranks = _worker_split_layout(
            group_comm, nprocs_per_worker, broker_is_worker
        )
        is_broker = group_comm.rank in broker_ranks
        controller_worker_comm = world_comm.Split(1 if is_broker else 2, key=1)
        server = setup_split_group(
            worker_id,
            n_groups,
            True,
            is_broker,
            tuple(broker_ranks),
            nprocs_per_worker,
            broker_is_worker,
            broker_fun,
            fun,
            fun_name,
            module_name,
            args,
            world_comm,
            controller_worker_comm,
            group_comm,
            collective_mode,
            verbose,
            capture_errors=capture_errors,
        )
        return Topology(
            role="broker" if is_broker else "collective_worker",
            grouping_method=method,
            controller_worker_comm=controller_worker_comm if is_broker else None,
            group_comm=group_comm,
            group_id=group_id,
            worker_id=worker_id,
            server=server,
        )

    controller_worker_comm = world_comm.Split(0, key=1)
    controller_worker_comm.Ibarrier().wait()
    return Topology(
        role="worker",
        grouping_method=method,
        controller_worker_comm=controller_worker_comm,
        group_comm=group_comm,
        group_id=group_id,
        worker_id=group_comm.rank + 1,
        server=MPIWorker(
            controller_worker_comm, group_comm, capture_errors=capture_errors
        ),
    )


def release_workers_after_error(controller: MPIController) -> None:
    """
    Tell the workers to exit after the controller function has raised.

    Called from inside the handler of that exception.  The workers are
    told as in a normal exit: results still outstanding are collected
    and discarded, then every worker is sent EXIT.  If that fails as
    well, the controller can no longer reach the workers, and the whole
    job is aborted with a non-zero status.
    """
    if not controller.workers_available:
        return
    logger.error(
        "MPI controller : controller function raised an exception; "
        "telling all workers to exit..."
    )
    try:
        controller.exit()
    except BaseException:  # noqa: BLE001
        controller.abort()


def run(
    fun_name: str | None = None,
    module_name: str = "__main__",
    broker_fun_name: str | None = None,
    broker_module_name: str = "__main__",
    max_workers: int = -1,
    worker_grouping_method: str | GroupingMethod = GroupingMethod.NoGrouping,
    sequential_spawn: bool = False,
    spawn_startup_wait: int | None = None,
    spawn_executable: str | None = None,
    spawn_args: list[Any] | None = None,
    nprocs_per_worker: int = 1,
    collective_mode: str = "gather",
    broker_is_worker: bool = False,
    worker_service_name: str = "distwq.init",
    enable_worker_service: bool = False,
    time_limit: int | None = None,
    verbose: bool = False,
    args: tuple[Any] = (),
) -> None:
    """Run in controller/worker mode until fun(controller/worker) finishes.

    Must be called on all MPI nodes.

    On the controller, run() calls fun_name() and returns when fun_name() returns.

    On each worker, run() calls fun() if that is defined, or calls serve()
    otherwise, and returns when fun() returns, or when fun() returns on
    the controller, or when controller calls exit().

    If fun_name() raises an exception on the controller, run() tells the
    workers to exit, so that run() returns on every worker, and then
    raises the exception again to its caller.  If an uncaught exception
    reaches the top level of an MPI program, mpi_excepthook aborts the
    whole job with a non-zero exit status.

    :arg string module_name: module where fun_name is located
    :arg bool verbose: whether processing information should be printed.

    :arg GroupingMethod worker_grouping_method: specifies grouping method
    for workers: NoGrouping (default) GroupSpawn (spawn separate
    worker processes via MPI_Comm_spawn) GroupSplit (split separate
    worker processes via MPI_Comm_split).

    :arg bool sequential_spawn: whether to spawn processes in sequence

    :arg string spawn_executable: optional executable name for call to
    spawn (default is sys.executable)

    :arg string list spawn_args: optional arguments to prepend to list
    of arguments in call to spawn; or a callable that takes the list
    of arguments that distwq needs to pass to the python interpreter,
    and returns a new list of arguments

    :arg int nprocs_per_worker: how many processes per worker

    :arg broker_is_worker: when worker_grouping_method is GroupSpawn
    or GroupSplit and nprocs_per_worker > 1, MPI_Comm_spawn or
    MPI_Comm_split will be used to create workers, and a
    CollectiveBroker object is used to relay tasks and results between
    controller and worker.  When broker_is_worker is true, the broker
    also participates in serving tasks, otherwise it only relays
    calls.

    :arg time_limit: maximum wall clock time, in seconds

    :arg args: additional args to pass to fun

    """

    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    if spawn_args is None:
        spawn_args = []
    assert nprocs_per_worker > 0
    assert not spawned

    worker_grouping_method = check_worker_grouping_method(worker_grouping_method)

    limit_workers(max_workers)

    fun = get_fun(fun_name, module_name)
    broker_fun = get_fun(broker_fun_name, module_name)

    results = None
    if has_mpi:  # run in mpi mode
        topology = establish_topology(
            worker_grouping_method=worker_grouping_method,
            nprocs_per_worker=nprocs_per_worker,
            broker_is_worker=broker_is_worker,
            collective_mode=collective_mode,
            enable_worker_service=enable_worker_service,
            worker_service_name=worker_service_name,
            sequential_spawn=sequential_spawn,
            spawn_startup_wait=spawn_startup_wait,
            spawn_executable=spawn_executable,
            spawn_args=spawn_args,
            fun=fun,
            fun_name=fun_name,
            module_name=module_name,
            broker_fun=broker_fun,
            args=args,
            verbose=verbose,
        )
        if topology.role == "controller":
            assert fun is not None
            controller = MPIController(
                topology.controller_worker_comm, time_limit=time_limit
            )
            previous_sigint = signal.signal(
                signal.SIGINT, lambda signum, frame: controller.abort()
            )
            try:
                try:
                    results = fun(controller, *args)
                except BaseException:
                    # Release the workers, so that they return from run()
                    # too, and let the caller see the exception.
                    release_workers_after_error(controller)
                    raise
                controller.exit()
            finally:
                # The handler refers to this controller; restoring the
                # previous handler lets the controller and any results it
                # still holds be freed once run() returns.
                signal.signal(
                    signal.SIGINT,
                    previous_sigint if previous_sigint is not None else signal.SIG_DFL,
                )
        elif topology.role == "worker":
            worker = topology.server
            if fun is not None:
                _call_init(fun, worker, args, f"worker function {fun_name}")
            worker.serve()
        elif topology.role in ("broker", "collective_worker"):
            # Worker groups already ran fun and broker_fun during setup.
            topology.server.serve()
        else:
            logger.info(f"distwq.run: rank {rank} is not needed as a worker.")

    else:  # run as single processor
        assert fun is not None
        logger.info("MPI controller : not available, running as a single process.")
        controller = MPIController()
        results = fun(controller, *args)
        logger.info("MPI controller : finished.")
    return results


def worker_main():
    if is_worker:
        worker_id = int(my_config["worker_id"])
        logger.info(f"MPI collective worker {worker_id}-{rank} starting")
        collective_mode = my_config["collective_mode"]
        enable_worker_service = my_config["enable_worker_service"]
        worker_service_name = my_config["worker_service_name"]
        verbose_flag = my_config["verbose"]
        verbose = verbose_flag == 1
        if verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)
        logger.info(f"MPI collective worker {worker_id}-{rank} starting")
        logger.info(f"MPI collective worker {worker_id}-{rank} args: {my_args}")

        parent_comm = world_comm.Get_parent()
        req = parent_comm.Ibarrier()
        merged_comm = parent_comm.Merge(True)
        req.wait()

        if collective_mode.lower() == "gather":
            collective_mode_arg = CollectiveMode.Gather
        elif collective_mode.lower() == "sendrecv":
            collective_mode_arg = CollectiveMode.SendRecv
        else:
            raise RuntimeError(f"Unknown collective mode {collective_mode}")

        worker = MPICollectiveWorker(
            world_comm,
            merged_comm,
            worker_id,
            n_workers,
            collective_mode=collective_mode_arg,
            worker_service_name=worker_service_name,
            capture_errors=bool(my_config.get("capture_errors", False)),
            spawn_comm=parent_comm,
        )
        fun = None
        if "init_fun_name" in my_config:
            fun_name = my_config["init_fun_name"]
            module = my_config["init_module_name"]
            try:
                fun = get_fun(fun_name, module)
            except BaseException:
                # The broker is waiting to send the init arguments; abort
                # rather than leave it, and the controller, waiting.
                logger.exception(
                    f"MPI collective worker {worker_id}-{rank}: cannot find "
                    f"init function {fun_name} in module {module}; aborting"
                )
                merged_comm.Abort(1)
                raise
        if fun is not None:
            if enable_worker_service and (worker_id == 1):
                worker.publish_service()
            req = merged_comm.Ibarrier()
            args = merged_comm.bcast(None, root=0)
            req.wait()
            if enable_worker_service:
                worker.connect_service(n_lookup_attempts=5)
            _call_init(
                fun, worker, args, "worker init function", abort_comm=merged_comm
            )
        # A spawned worker runs no user code between sessions, so it simply
        # serves again after each session until told to exit.
        while worker.serve() != MessageTag.EXIT:
            pass
        MPI.Finalize()


if __name__ == "__main__":
    worker_main()
