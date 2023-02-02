#!/usr/bin/python
#
# Distributed work queue operations using mpi4py.
#
# Copyright (C) 2020-2023 Ivan Raikov and distwq authors.
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
import random
import signal
import sys
import time
import traceback
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from mpi4py.MPI import Intercomm, Intracomm
from numpy import float64, ndarray


class CollectiveMode(IntEnum):
    Gather = 1
    SendRecv = 2


class MessageTag(IntEnum):
    READY = 0
    DONE = 1
    TASK = 2
    EXIT = 3


class GroupingMethod(IntEnum):
    NoGrouping = 0
    GroupSpawn = 1
    GroupSplit = 2


logger = logging.getLogger(__name__)

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
my_args: Optional[List[str]] = sys.argv[my_args_start_index:] if has_args else None
my_config: Optional[Dict[Any, Any]] = None

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
start_time = time.time()


def multiple_task_arguments(N, args, kwargs, task_ids, workers):
    """
    Helper function to set default arguments, task ids, and workers for multiple tasks.
    """
    if (args is None) or (len(args) == 0):
        args = [dict() for _ in range(N)]
    if (kwargs is None) or (len(kwargs) == 0):
        kwargs = [dict() for _ in range(N)]
    if task_ids is None:
        task_ids = [None for _ in range(N)]
    if workers is None:
        workers = [None for _ in range(N)]

    return args, kwargs, task_ids, workers


class MPIController(object):
    def __init__(self, comm: Intracomm, time_limit: Any = None) -> None:
        size = comm.size

        self.comm = comm
        self.workers_available = True if size > 1 else False

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
        self.result_queue: List[int] = []
        self.task_queue: List[int] = []
        self.wait_queue: List[int] = []
        self.waiting: Dict[
            int,
            Tuple[
                str,
                Union[List[Any], Tuple[Any]],
                Dict[Any, Any],
                str,
                Optional[int],
                Optional[int],
            ],
        ] = {}
        self.ready_workers: List[int] = []
        self.ready_workers_data: Dict[int, Any] = {}
        """(list) ids of submitted calls"""
        self.assigned = {}
        """
        (dictionary)
        assigned[id] is the worker assigned to the call with that id.
        """
        self.worker_queue = [[] for i in range(0, size)]
        """
        (list of lists)
        worker_queue[i] contains the ids of calls assigned to worker i.
        """
        self.active_workers = set([])
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

    def process(self, limit: int = 1000) -> List[Union[int, Any]]:
        """
        Process incoming messages.
        """
        if not self.workers_available:
            return
        count = 0
        while self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG):
            if (limit is not None) and (limit < count):
                break

            status = MPI.Status()
            data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            worker = status.Get_source()
            tag = status.Get_tag()
            if tag == MessageTag.READY.value:
                if worker not in self.ready_workers:
                    self.ready_workers.append(worker)
                    self.ready_workers_data[worker] = data
                    self.active_workers.add(worker)
                logger.info(
                    f"MPI controller : received READY message from worker {worker}"
                )
            elif tag == MessageTag.DONE.value:
                task_id, results, stats = data
                logger.info(
                    f"MPI controller : received DONE message for task {task_id} "
                    f"from worker {worker}"
                )

                self.results[task_id] = results
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
        else:
            time.sleep(1)

        return self.submit_waiting()

    def submit_call(
        self,
        name_to_call: str,
        args: Tuple[Any] = (),
        kwargs: Dict[Any, Any] = {},
        module_name: str = "__main__",
        time_est: int = 1,
        task_id: Optional[int] = None,
        worker: Optional[int] = None,
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
            logger.info(f"MPI controller : calling {name_to_call} {args} {kwargs} ...")
            object_to_call = None
            try:
                if module_name not in sys.modules:
                    importlib.import_module(module_name)
                object_to_call = eval(name_to_call, sys.modules[module_name].__dict__)
            except NameError:
                logger.error(str(sys.modules[module_name].__dict__.keys()))
                raise
            call_time = time.time()
            self.results[task_id] = object_to_call(*args, **kwargs)
            self.result_queue.append(task_id)
            this_time = time.time() - call_time
            self.n_processed[0] += 1
            self.total_time[0] = time.time() - start_time
            self.stats.append(
                {
                    "id": task_id,
                    "rank": worker,
                    "this_time": this_time,
                    "time_over_est": this_time / time_est,
                    "n_processed": self.n_processed[0],
                    "total_time": self.total_time[0],
                }
            )

        self.total_time_est[worker] += time_est
        return task_id

    def queue_call(
        self,
        name_to_call: str,
        args: Union[List[Any], Tuple[Any]] = (),
        kwargs: Dict[Any, Any] = {},
        module_name: str = "__main__",
        time_est: int = 1,
        task_id: Optional[int] = None,
        requested_worker: Optional[int] = None,
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
            worker = 0
            logger.info(f"MPI controller : calling {name_to_call} {args} {kwargs} ...")
            object_to_call = None
            try:
                if module_name not in sys.modules:
                    importlib.import_module(module_name)
                object_to_call = eval(name_to_call, sys.modules[module_name].__dict__)
            except NameError:
                logger.error(str(sys.modules[module_name].__dict__.keys()))
                raise
            call_time = time.time()
            self.results[task_id] = object_to_call(*args, **kwargs)
            self.result_queue.append(task_id)
            this_time = time.time() - call_time
            self.n_processed[0] += 1
            self.total_time[0] = time.time() - start_time
            self.stats.append(
                {
                    "id": task_id,
                    "rank": worker,
                    "this_time": this_time,
                    "time_over_est": this_time / time_est,
                    "n_processed": self.n_processed[0],
                    "total_time": self.total_time[0],
                }
            )

        return task_id

    def submit_waiting(self) -> List[Union[int, Any]]:
        """
        Submit waiting tasks if workers are available.

        :return object: ids of calls, to be used in get_result().
        """
        task_ids = []
        if self.workers_available:
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
        args: List[List[Any]] = [],
        kwargs: Dict[Any, Any] = {},
        module_name: str = "__main__",
        time_est: int = 1,
        task_ids: Optional[int] = None,
        workers: Optional[int] = None,
    ) -> List[int]:
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
        if len(kwargs) > 0:
            assert len(args) == len(kwargs)
        submitted_task_ids = []
        N = len(args)
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
            # perform call on this rank if no workers are available:
            worker = 0
            logger.info(f"MPI controller : calling {name_to_call} {args} {kwargs} ...")
            object_to_call = None
            try:
                if module_name not in sys.modules:
                    importlib.import_module(module_name)
                object_to_call = eval(name_to_call, sys.modules[module_name].__dict__)
            except NameError:
                logger.error(str(sys.modules[module_name].__dict__.keys()))
                raise
            args, kwargs, task_ids, workers = multiple_task_arguments(
                N, args, kwargs, task_ids, workers
            )
            for this_args, this_kwargs, this_task_id, this_worker in zip(
                args, kwargs, task_ids, workers
            ):
                if this_task_id is None:
                    this_task_id = self.count
                    self.count += 1
                self.check_valid_task_id(this_task_id)
                call_time = time.time()
                self.results[this_task_id] = object_to_call(*this_args, **this_kwargs)
                self.result_queue.append(this_task_id)
                this_time = time.time() - call_time
                self.n_processed[0] += 1
                self.total_time[0] = time.time() - start_time
                self.stats.append(
                    {
                        "id": this_task_id,
                        "rank": worker,
                        "this_time": this_time,
                        "time_over_est": this_time / time_est,
                        "n_processed": self.n_processed[0],
                        "total_time": self.total_time[0],
                    }
                )
                self.total_time_est[this_worker] += time_est
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

    def get_result(
        self, task_id: int
    ) -> Union[
        Tuple[int, Tuple[ndarray, ndarray]], Tuple[int, List[Tuple[ndarray, ndarray]]]
    ]:
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
        :return: return value of call.
        """
        if task_id in self.results:
            return task_id, self.results[task_id]
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

            while not (task_id in self.results):
                self.process()

            logger.info(
                f"MPI controller : received result for call with id {task_id} "
                f"from worker {source}."
            )

        else:
            logger.info(
                f"MPI controller : returning result for call with id {task_id} " "..."
            )
        result = self.results[task_id]
        self.result_queue.remove(task_id)
        return task_id, result

    def get_next_result(
        self,
    ) -> Optional[
        Union[
            Tuple[int, Tuple[ndarray, ndarray]],
            Tuple[int, List[Tuple[ndarray, ndarray]]],
        ]
    ]:
        """
        Return result of next earlier submitted call whose result has not yet
        been obtained.

        Can only be called by the controller.

        If the call is not yet finished, waits for it to finish.

        :rtype:  object
        :return: id, return value of call, or None of there are no more calls in
                 the queue.
        """
        self.process()
        if len(self.result_queue) > 0:
            task_id = self.result_queue.pop(0)
            return task_id, self.results[task_id]
        elif len(self.task_queue) > 0:
            task_id = self.task_queue[0]
            return task_id, self.get_result(task_id)[1]
        else:
            return None

    def probe_next_result(self):
        """
        Return result of next earlier submitted call whose result has not yet
        been obtained.

        Can only be called by the controller.

        If no result is available, returns none.

        :rtype:  object
        :return: id, return value of call, or None of there are no results ready.
        """
        self.process()
        if len(self.result_queue) > 0:
            task_id = self.result_queue.pop(0)
            logger.info(
                f"MPI controller : received result for call with id {task_id} ..."
            )
            return task_id, self.results[task_id]
        else:
            return None

    def probe_all_next_results(self):
        """
        Return all available results of earlier submitted calls whose result has not yet
        been obtained.

        Can only be called by the controller.

        If no result is available, returns empty list.

        :rtype:  object
        :return: list of id, return value of call
        """
        self.process()
        ret = []
        if len(self.result_queue) > 0:
            for i in range(len(self.result_queue)):
                task_id = self.result_queue.pop(0)
                logger.info(
                    f"MPI controller : received result for call with id {task_id} ..."
                )
                ret.append((task_id, self.results[task_id]))

        return ret

    def info(self) -> None:
        """
        Print processing statistics.

        Can only be called by the controller.
        """

        if len(self.stats) == 0:
            return

        call_times = np.array([s["this_time"] for s in self.stats])
        call_quotients = np.array([s["time_over_est"] for s in self.stats])
        cvar_call_quotients = call_quotients.std() / call_quotients.mean()

        if self.workers_available:
            worker_quotients = self.total_time / self.total_time_est
            cvar_worker_quotients = worker_quotients.std() / worker_quotients.mean()
            print(
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
                f"{n_workers}\n"
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
        else:
            print(
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

    def exit(self) -> None:
        """
        Tell all workers to exit.

        Can only be called by the controller.
        """
        if self.workers_available:
            while self.get_next_result() is not None:
                pass
            # tell workers to exit:
            reqs = []
            for worker in self.active_workers:
                logger.info(f"MPI controller : telling worker {worker} " "to exit...")
                reqs.append(
                    self.comm.isend(None, dest=worker, tag=MessageTag.EXIT.value)
                )
            MPI.Request.Waitall(reqs)

    def abort(self):
        """
        Abort execution on all MPI nodes immediately.

        Can be called by controller and workers.
        """
        traceback.print_exc()
        logger.error("MPI controller : aborting...")
        self.comm.Abort()

    def check_valid_task_id(self, task_id: int) -> bool:
        """
        Given a new task id, check that it is not already assigned or waiting.
        """
        if task_id in self.assigned:
            raise RuntimeError(f"task id {task_id} already in queue!")
        if task_id in self.waiting:
            raise RuntimeError(f"task id {task_id} already in wait queue!")
        return True


class MPIWorker(object):
    def __init__(
        self, comm: Intracomm, group_comm: Intracomm, ready_data: Optional[Any] = None
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

        logger.info(f"MPI worker {self.worker_id}: initialized.")

    def serve(self) -> None:
        """
        Serve submitted calls until told to finish.

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
        ready = True
        status = MPI.Status()
        exit_flag = False
        while not exit_flag:
            # signal the controller this worker is ready
            if ready:
                req = self.comm.isend(
                    self.ready_data, dest=0, tag=MessageTag.READY.value
                )
                req.wait()

            # get next task from queue:
            if self.comm.Iprobe(source=0, tag=MPI.ANY_TAG):
                data = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                tag = status.Get_tag()

                # TODO: add timeout and check whether controller lives!
                object_to_call = None
                if tag == MessageTag.EXIT.value:
                    logger.info(f"MPI worker {self.worker_id}: exiting...")
                    exit_flag = True
                    break
                elif tag == MessageTag.TASK.value:
                    try:
                        (name_to_call, args, kwargs, module, time_est, task_id) = data
                        if module not in sys.modules:
                            importlib.import_module(module)
                        object_to_call = eval(
                            name_to_call, sys.modules[module].__dict__
                        )
                    except NameError:
                        logger.error(str(sys.modules[module].__dict__.keys()))
                        raise
                else:
                    raise RuntimeError(
                        f"MPI worker {self.worker_id}: unknown message tag {tag}"
                    )
                self.total_time_est[rank] += time_est
                call_time = time.time()
                result = object_to_call(*args, **kwargs)
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
                    }
                )
                req = self.comm.isend(
                    (task_id, result, self.stats[-1]), dest=0, tag=MessageTag.DONE.value
                )
                req.wait()
                ready = True
            else:
                ready = False
                time.sleep(1)

    def abort(self):
        traceback.print_exc()
        logger.info(f"MPI worker {self.worker_id}: aborting...")
        self.comm.Abort()


class MPICollectiveWorker(object):
    def __init__(
        self,
        comm: Intracomm,
        merged_comm: Intracomm,
        worker_id: int,
        n_workers: int,
        worker_service_name: str = "distwq.init",
        collective_mode: CollectiveMode = CollectiveMode.Gather,
    ):
        size = comm.size
        rank = comm.rank

        self.collective_mode = collective_mode
        self.n_workers = n_workers
        self.worker_id = worker_id
        self.comm = comm
        self.merged_comm = merged_comm

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
                            raise e
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

    def serve(self) -> None:
        """
        Serve submitted calls until told to finish. Tasks are
        obtained via scatter and results are returned via gather,
        i.e. all collective workers spawned by a CollectiveBroker
        will participate in these collective calls.

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
                self.merged_comm.Disconnect()
                break
            try:
                if module not in sys.modules:
                    importlib.import_module(module)
                object_to_call = eval(name_to_call, sys.modules[module].__dict__)
            except NameError:
                logger.error(str(sys.modules[module].__dict__.keys()))
                raise
            self.total_time_est[rank] += time_est
            call_time = time.time()
            result = object_to_call(*args, **kwargs)
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
                }
            )
            self.gather_results(result)

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
                "MPICollectiveWorker: unknown collective mode "
                f"{self.collective_mode}"
            )

    def get_next_task(self) -> Tuple[str, Tuple[Any], Dict[Any, Any], str, int, int]:
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
        self.comm.Abort()


class MPICollectiveBroker(object):
    def __init__(
        self,
        worker_id: int,
        comm: Intracomm,
        group_comm: Intracomm,
        merged_comm: Intracomm,
        nprocs_per_worker: int,
        ready_data: Optional[Any] = None,
        is_worker: bool = False,
        collective_mode: CollectiveMode = CollectiveMode.Gather,
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

    def serve(self) -> None:
        """
        Broker and serve submitted calls until told to finish. A task
        is received from the controller and sent to all collective
        workers associated with this broker via scatter.

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

        logger.info(f"MPI worker broker {self.worker_id}: waiting for calls.")

        # wait for orders:
        while True:
            logger.info(
                f"MPI collective broker {self.worker_id}: "
                "getting next task from controller..."
            )
            # signal the controller this worker is ready
            req = self.comm.isend(self.ready_data, dest=0, tag=MessageTag.READY.value)
            req.wait()

            while True:
                msg = self.process()
                if msg is not None:
                    tag, data = msg
                    break

            logger.info(
                f"MPI collective broker {self.worker_id}: "
                "received message from controller..."
            )

            if tag == MessageTag.EXIT.value:
                logger.info(f"MPI worker broker {self.worker_id}: exiting...")
                self.scatter_task("exit", (), {}, "", 0, 0)
                break
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
                object_to_call = None
                try:
                    if module not in sys.modules:
                        importlib.import_module(module)
                    object_to_call = eval(name_to_call, sys.modules[module].__dict__)
                except NameError:
                    logger.error(str(sys.modules[module].__dict__.keys()))
                    raise

                call_time = time.time()
                this_result = object_to_call(*args, **kwargs)
                this_time = time.time() - call_time
                self.n_processed[merged_rank] += 1
                this_stat = {
                    "id": task_id,
                    "rank": merged_rank,
                    "this_time": this_time,
                    "time_over_est": this_time / time_est,
                    "n_processed": self.n_processed[merged_rank],
                    "total_time": time.time() - start_time,
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

            stat_times = np.asarray([stat["this_time"] for stat in stats])
            max_time = 0.0
            if len(stat_times) > 0:
                max_time = np.argmax(stat_times)
                stat = stats[max_time]
            else:
                stat = None
            logger.info(
                f"MPI collective broker {self.worker_id}: "
                "sending results to controller..."
            )
            req = self.comm.isend(
                (task_id, results, stat), dest=0, tag=MessageTag.DONE.value
            )
            req.wait()

    def scatter_task(
        self,
        name_to_call: str,
        args: Tuple[Any],
        kwargs: Dict[Any, Any],
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
    ) -> Tuple[List[Any], List[Dict[str, Union[int, float, float64]]]]:
        merged_rank = self.merged_comm.Get_rank()
        if self.collective_mode == CollectiveMode.Gather:
            req = self.merged_comm.Ibarrier()
            sub_data = self.merged_comm.gather(
                (this_result, this_stat), root=merged_rank
            )
            req.wait()
            results = [result for result, stat in sub_data if result is not None]
            stats = [stat for result, stat in sub_data if stat is not None]
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
            results = [result for result, stat in sub_data if result is not None]
            stats = [stat for result, stat in sub_data if stat is not None]
            if this_result is not None:
                results = [this_result] + results
                stats = [this_stat] + stats
        else:
            raise RuntimeError(
                "MPICollectiveBroker: unknown collective mode "
                f"{self.collective_mode}"
            )
        return results, stats

    def process(
        self,
    ) -> Optional[
        Union[
            Tuple[int, None],
            Tuple[int, Tuple[str, Tuple[int], Dict[Any, Any], str, int, int]],
        ]
    ]:
        status = MPI.Status()
        if self.comm.Iprobe(source=0, tag=MPI.ANY_TAG):
            # get next task from controller queue:
            data = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()
            return tag, data
        else:
            time.sleep(1)
            return None

    def abort(self):
        rank = self.comm.rank
        traceback.print_exc()
        logger.info(f"MPI worker broker {rank}: aborting...")
        self.comm.Abort()


def do_spawn_workers(
    fun: Optional[Callable],
    fun_name: Optional[str],
    module_name: Optional[str],
    world_comm: Intracomm,
    group_comm: Intracomm,
    worker_id: int,
    n_workers: int,
    nprocs_per_worker: int,
    broker_is_worker: bool,
    collective_mode: str,
    enable_worker_service: bool,
    worker_service_name: str,
    spawn_args: List[Any],
    sequential_spawn: bool,
    spawn_startup_wait: int,
    spawn_executable: Optional[str],
    verbose: bool,
) -> Tuple[Intercomm, Intracomm, CollectiveMode]:
    worker_config = {
        "worker_id": worker_id,
        "n_workers": n_workers,
        "collective_mode": collective_mode,
        "enable_worker_service": enable_worker_service,
        "worker_service_name": worker_service_name,
        "verbose": verbose,
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
    fun: Optional[Callable],
    fun_name: Optional[str],
    module_name: Optional[str],
    world_comm: Intracomm,
    group_comm: Intracomm,
    worker_id: int,
    n_workers: int,
    nprocs_per_worker: int,
    broker_is_worker: bool,
    collective_mode: str,
    is_broker: bool,
    broker_ranks: Tuple[int],
    verbose: bool,
) -> Tuple[Intracomm, Intracomm, CollectiveMode]:
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
        this_broker_rank = tuple(this_worker_ranks)[0]
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
        remote_leader = tuple(this_worker_ranks)[0]
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
) -> Tuple[int, Intracomm]:
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


def get_fun(fun_name: Optional[str], module_name: str) -> Optional[Callable]:
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
                    f"when world size is {world_size} "
                    "(1 process is needed for controller)"
                )
        else:
            if (world_size - 1) % (nprocs_per_worker + 1) != 0:
                raise RuntimeError(
                    f"distwq.run: unable to split workers evenly "
                    f"into groups of {nprocs_per_worker+1} "
                    f"when world size is {world_size} "
                    "(1 process is needed for controller; "
                    "1 process per worker group is needed for brokers)"
                )


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
    check_spawn_config(
        spawn_workers,
        nprocs_per_worker,
        broker_is_worker,
        enable_worker_service,
        broker_fun,
    )
    sub_comm, merged_comm, collective_mode_arg = do_spawn_workers(
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
    )
    broker = MPICollectiveBroker(
        worker_id,
        world_comm,
        group_comm,
        merged_comm,
        nprocs_per_worker,
        is_worker=broker_is_worker,
        collective_mode=collective_mode_arg,
    )
    if fun is not None:
        req = merged_comm.Ibarrier()
        merged_comm.bcast(args, root=0)
        req.wait()
        if broker_is_worker:
            fun(broker, *args)
    if broker_fun is not None:
        broker_fun(broker, *args)
    broker.serve()


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
    broker = None
    worker = None
    check_split_config(
        split_workers,
        nprocs_per_worker,
        broker_is_worker,
        broker_fun,
        world_comm.size,
    )
    sub_comm, local_comm, merged_comm, collective_mode_arg = do_split_workers(
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
        )

    if fun is not None:
        req = merged_comm.Ibarrier()
        merged_comm.bcast(args, root=0)
        req.wait()
        if is_broker and broker_is_worker:
            fun(broker, *args)
        elif is_worker:
            fun(worker, *args)
    if is_broker:
        if broker_fun is not None:
            broker_fun(broker, *args)
        broker.serve()
    else:
        worker.serve()


def run(
    fun_name: Optional[str] = None,
    module_name: str = "__main__",
    broker_fun_name: Optional[str] = None,
    broker_module_name: str = "__main__",
    max_workers: int = -1,
    worker_grouping_method: Union[str, GroupingMethod] = GroupingMethod.NoGrouping,
    sequential_spawn: bool = False,
    spawn_startup_wait: Optional[int] = None,
    spawn_executable: Optional[str] = None,
    spawn_args: List[Any] = [],
    nprocs_per_worker: int = 1,
    collective_mode: str = "gather",
    broker_is_worker: bool = False,
    worker_service_name: str = "distwq.init",
    enable_worker_service: bool = False,
    time_limit: Optional[int] = None,
    verbose: bool = False,
    args: Tuple[Any] = (),
) -> None:
    """Run in controller/worker mode until fun(controller/worker) finishes.

    Must be called on all MPI nodes.

    On the controller, run() calls fun_name() and returns when fun_name() returns.

    On each worker, run() calls fun() if that is defined, or calls serve()
    otherwise, and returns when fun() returns, or when fun() returns on
    the controller, or when controller calls exit().

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

    logging.basicConfig(level=logging.INFO if verbose else logging.WARN)

    assert nprocs_per_worker > 0
    assert not spawned

    worker_grouping_method = check_worker_grouping_method(worker_grouping_method)

    global n_workers, is_worker
    if max_workers > 0:
        n_workers = min(max_workers, n_workers)
        is_worker = (not is_controller) and (rank - 1) < n_workers

    fun = get_fun(fun_name, module_name)
    broker_fun = get_fun(broker_fun_name, module_name)

    if has_mpi:  # run in mpi mode
        spawn_workers = (
            (worker_grouping_method == GroupingMethod.GroupSpawn)
            and (n_workers > 0)
            and (nprocs_per_worker > 0)
        )
        split_workers = (
            (
                (worker_grouping_method == GroupingMethod.GroupSplit)
                or (
                    (worker_grouping_method == GroupingMethod.NoGrouping)
                    and (nprocs_per_worker > 1)
                )
            )
            and (n_workers > 0)
            and (nprocs_per_worker > 0)
        )
        group_id, group_comm = get_group_comm(
            world_comm, size, rank, is_controller, spawned
        )
        if is_controller:  # I'm the controller
            assert fun is not None
            controller_worker_comm = world_comm
            if split_workers or spawn_workers:
                color = 1 if is_controller else 2
                controller_worker_comm = world_comm.Split(
                    color, key=0 if is_controller else 1
                )
            controller = MPIController(controller_worker_comm, time_limit=time_limit)
            signal.signal(signal.SIGINT, lambda signum, frame: controller.abort())
            req = controller_worker_comm.Ibarrier()
            req.wait()
            try:  # put everything in a try block to be able to exit!
                fun(controller, *args)
            except ValueError:
                controller.abort()
            controller.exit()
        elif is_worker and spawn_workers:  # I'm a broker
            worker_id = group_comm.rank + 1
            is_broker = True
            color = 1 if is_broker or is_controller else 2
            controller_worker_comm = world_comm.Split(
                color, key=0 if is_controller else 1
            )
            run_spawn_group(
                worker_id,
                spawn_workers,
                nprocs_per_worker,
                broker_is_worker,
                broker_fun,
                fun,
                fun_name,
                module_name,
                args,
                controller_worker_comm,
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
            )
        elif is_worker and split_workers:  # I'm a broker or a worker
            if broker_is_worker:
                n_workers = group_comm.size // nprocs_per_worker
                worker_id = (group_comm.rank // nprocs_per_worker) + 1
                broker_set = {(x * nprocs_per_worker) for x in range(n_workers)}
            else:
                n_workers = group_comm.size // (nprocs_per_worker + 1)
                worker_id = (group_comm.rank // (nprocs_per_worker + 1)) + 1
                broker_set = {(x * (nprocs_per_worker + 1)) for x in range(n_workers)}
            is_broker = group_comm.rank in broker_set
            color = 1 if is_broker or is_controller else 2
            controller_worker_comm = world_comm.Split(
                color, key=0 if is_controller else 1
            )
            run_split_group(
                worker_id,
                n_workers,
                split_workers,
                is_broker,
                tuple(broker_set),
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
            )

        elif is_worker:  # I'm a worker
            worker_id = rank
            req = world_comm.Ibarrier()
            req.wait()
            worker = MPIWorker(world_comm, group_comm)
            if fun is not None:
                fun(worker, *args)
            worker.serve()
            MPI.Finalize()
        else:
            raise RuntimeError("distwq.run: invalid worker configuration")

    else:  # run as single processor
        assert fun is not None
        logger.info("MPI controller : not available, running as a single process.")
        controller = MPIController()
        fun(controller, *args)
        logger.info("MPI controller : finished.")


def worker_main():
    if is_worker:
        worker_id = int(my_config["worker_id"])
        logger.info(f"MPI collective worker {worker_id}-{rank} starting")
        collective_mode = my_config["collective_mode"]
        enable_worker_service = my_config["enable_worker_service"]
        worker_service_name = my_config["worker_service_name"]
        verbose_flag = my_config["verbose"]
        verbose = True if verbose_flag == 1 else False
        if verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARN)
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
        )
        fun = None
        if "init_fun_name" in my_config:
            fun_name = my_config["init_fun_name"]
            module = my_config["init_module_name"]
            if module not in sys.modules:
                importlib.import_module(module)
            fun = eval(fun_name, sys.modules[module].__dict__)
        if fun is not None:
            if enable_worker_service and (worker_id == 1):
                worker.publish_service()
            req = merged_comm.Ibarrier()
            args = merged_comm.bcast(None, root=0)
            req.wait()
            if enable_worker_service:
                worker.connect_service(n_lookup_attempts=5)
            fun(worker, *args)
        worker.serve()
        MPI.Finalize()


if __name__ == "__main__":
    worker_main()
