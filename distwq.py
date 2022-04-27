#!/usr/bin/python
#
# Distributed work queue operations using mpi4py.
#
# Copyright (C) 2020-2022 Ivan Raikov and distwq authors.
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

import sys, time, signal, importlib, traceback, logging, random, json
from enum import Enum, IntEnum
import numpy as np

class CollectiveMode(IntEnum):
    Gather = 1
    SendRecv = 2

class MessageTag(IntEnum):
    READY = 0
    DONE = 1
    TASK = 2
    EXIT = 3

    
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

my_args = sys.argv[sys.argv.index('-')+1:] if '-' in sys.argv else None
my_config = None

# message types
tag_ctrl_to_worker = 1
tag_worker_to_ctrl = 2

# initialize:
workers_available = True
spawned = False
if has_mpi:
    spawned = (my_args[0] == 'distwq:spawned') if my_args is not None else False
    size = world_comm.size
    rank = world_comm.rank
    is_controller = (not spawned) and (rank == 0)
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
    my_config = json.loads(my_args[1])
    
n_workers = size - 1 if not spawned else int(my_config['n_workers'])
start_time = time.time()

class MPIController(object):

    def __init__(self, comm, time_limit=None):
        
        size = comm.size
        rank = comm.rank

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
        self.result_queue = []
        self.task_queue = []
        self.wait_queue = []
        self.waiting = {}
        self.ready_workers = []
        self.ready_workers_data = {}
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
        (set)
        active_workers contains the ids of workers that have communicated with the controller
        """
        self.n_processed = np.zeros(size).astype(np.int)
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

    def process(self, limit=1000):
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
                logger.info(f"MPI controller : received READY message from worker {worker}")
            elif tag == MessageTag.DONE.value:
                task_id, results, stats = data
                self.results[task_id] = results
                self.stats.append(stats)
                self.n_processed[worker] = stats["n_processed"]
                self.total_time[worker] = stats["total_time"]
                self.task_queue.remove(task_id)
                self.result_queue.append(task_id)
                self.worker_queue[worker].remove(task_id)
                self.assigned.pop(task_id)
                count += 1
                logger.info(f"MPI controller : received DONE message for task {task_id}")
            else:
                raise RuntimeError(f"MPI controller : invalid message tag {tag}")
        else:
            time.sleep(1)

        return self.submit_waiting()

            
    def submit_call(self, name_to_call, args=(), kwargs={},
                    module_name="__main__", time_est=1, task_id=None, worker=None):
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
        if task_id in self.assigned:
            raise RuntimeError(f"task id {task_id} already in queue!")
        if task_id in self.waiting:
            raise RuntimeError(f"task id {task_id} already in wait queue!")
        if self.workers_available:
            self.process()
            if len(self.ready_workers) > 0:
                if worker is None:
                    ready_total_time_est = np.asarray([self.total_time_est[worker] for worker in self.ready_workers])
                    worker = self.ready_workers[np.argmin(ready_total_time_est)]
                else:
                    if worker not in self.ready_workers:
                        raise RuntimeError(f"worker {worker} is not in ready queue!")

                # send name to call, args, time_est to worker:
                logger.info(f"MPI controller : assigning call with id {task_id} to worker "
                            f"{worker}: {name_to_call} {args} {kwargs} ...")
                req = self.comm.isend((name_to_call, args, kwargs, module_name, time_est, task_id),
                                      dest=worker, tag=MessageTag.TASK.value)
                req.wait()
                self.ready_workers.remove(worker)
                del(self.ready_workers_data[worker])
                self.task_queue.append(task_id)
                self.worker_queue[worker].append(task_id)
                self.assigned[task_id] = worker
            else:
                self.queue_call(name_to_call, args=args, kwargs=kwargs,
                                module_name=module_name, time_est=time_est,
                                task_id=task_id, requested_worker=worker)
                
        else:
            # perform call on this rank if no workers are available:
            worker = 0
            logger.info(f"MPI controller : calling {name_to_call} {args} {kwargs} ...")

            try:
                if module_name not in sys.modules:
                    importlib.import_module(module_name)
                object_to_call = eval(name_to_call,
                                      sys.modules[module_name].__dict__)
            except NameError:
                logger.error(str(sys.modules[module_name].__dict__.keys()))
                raise
            call_time = time.time()
            self.results[task_id] = object_to_call(*args, **kwargs)
            self.result_queue.append(task_id)
            this_time = time.time() - call_time
            self.n_processed[0] += 1
            self.total_time[0] = time.time() - start_time
            self.stats.append({"id":task_id, "rank": worker,
                               "this_time": this_time,
                               "time_over_est": this_time / time_est,
                               "n_processed": self.n_processed[0],
                               "total_time": self.total_time[0]})

        self.total_time_est[worker] += time_est
        return task_id
    
    def queue_call(self, name_to_call, args=(), kwargs={},
                    module_name="__main__", time_est=1, task_id=None, requested_worker=None):
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
        if task_id in self.assigned:
            raise RuntimeError(f"task id {task_id} already in queue!")
        if task_id in self.waiting:
            raise RuntimeError(f"task id {task_id} already in wait queue!")
        if self.workers_available:
            self.process()
            self.wait_queue.append(task_id)
            self.waiting[task_id] = (name_to_call, args, kwargs, module_name, time_est, requested_worker)
        else:
            # perform call on this rank if no workers are available:
            worker = 0
            logger.info(f"MPI controller : calling {name_to_call} {args} {kwargs} ...")

            try:
                if module_name not in sys.modules:
                    importlib.import_module(module_name)
                object_to_call = eval(name_to_call,
                                      sys.modules[module_name].__dict__)
            except NameError:
                logger.error(str(sys.modules[module_name].__dict__.keys()))
                raise
            call_time = time.time()
            self.results[task_id] = object_to_call(*args, **kwargs)
            self.result_queue.append(task_id)
            this_time = time.time() - call_time
            self.n_processed[0] += 1
            self.total_time[0] = time.time() - start_time
            self.stats.append({"id":task_id, "rank": worker,
                               "this_time": this_time,
                               "time_over_est": this_time / time_est,
                               "n_processed": self.n_processed[0],
                               "total_time": self.total_time[0]})

        return task_id

    def submit_waiting(self):
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
                    name_to_call, args, kwargs, module_name, time_est, requested_worker = self.waiting[task_id]
                    if (requested_worker is None) or (requested_worker not in self.ready_workers):
                        ready_total_time_est = np.asarray([self.total_time_est[worker] for worker in self.ready_workers])
                        worker = self.ready_workers[np.argmin(ready_total_time_est)]
                    else:
                        worker = requested_worker

                    # send name to call, args, time_est to worker:
                    logger.info(f"MPI controller : assigning waiting call with id {task_id} to worker "
                                f"{worker}: {name_to_call} {args} {kwargs} ...")
                    req = self.comm.isend((name_to_call, args, kwargs, module_name, time_est, task_id),
                                          dest=worker, tag=MessageTag.TASK.value)
                    reqs.append(req)
                    status.append(MPI.Status())
                    self.ready_workers.remove(worker)
                    del(self.ready_workers_data[worker])
                    self.task_queue.append(task_id)
                    self.worker_queue[worker].append(task_id)
                    self.assigned[task_id] = worker
                    del(self.waiting[task_id])
                    self.total_time_est[worker] += time_est
                    task_ids.append(task_id)
                MPI.Request.waitall(reqs, status) 
        return task_ids

    
    def submit_multiple(self, name_to_call, args=[], kwargs=[],
                        module_name="__main__", time_est=1, task_ids=None, workers=None):
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
            assert(len(args) == len(kwargs))
        submitted_task_ids = []
        if self.workers_available:
            self.process()
            N = len(args)
            if (args is None) or (len(args) == 0):
                kwargs = [dict() for _ in range(N)]
            if (kwargs is None) or (len(kwargs) == 0):
                kwargs = [dict() for _ in range(N)]
            if task_ids is None:
                task_ids = [None for _ in range(N)]
            if workers is None:
                workers = [None for _ in range(N)]
            for this_args, this_kwargs, this_task_id, this_worker in zip(args, kwargs, task_ids, workers):
                if this_task_id in self.assigned:
                    raise RuntimeError(f"task id {task_id} already in queue!")
                if this_task_id in self.waiting:
                    raise RuntimeError(f"task id {task_id} already in wait queue!")
                this_task_id = self.queue_call(name_to_call, args=this_args, kwargs=this_kwargs,
                                               module_name=module_name, time_est=time_est,
                                               task_id=this_task_id, requested_worker=this_worker)
                submitted_task_ids.append(this_task_id)
        else:
            # perform call on this rank if no workers are available:
            worker = 0
            logger.info(f"MPI controller : calling {name_to_call} {args} {kwargs} ...")

            try:
                if module_name not in sys.modules:
                    importlib.import_module(module_name)
                object_to_call = eval(name_to_call,
                                      sys.modules[module_name].__dict__)
            except NameError:
                logger.error(str(sys.modules[module_name].__dict__.keys()))
                raise
            if (args is None) or (len(args) == 0):
                kwargs = [dict() for _ in range(N)]
            if (kwargs is None) or (len(kwargs) == 0):
                kwargs = [dict() for _ in range(N)]
            if task_ids is None:
                task_ids = [None for _ in range(N)]
            if workers is None:
                workers = [None for _ in range(N)]
            for this_args, this_kwargs, this_task_id, this_worker in zip(args, kwargs, task_ids, workers):
                call_time = time.time()
                self.results[this_task_id] = object_to_call(*this_args, **this_kwargs)
                self.result_queue.append(task_id)
                this_time = time.time() - call_time
                self.n_processed[0] += 1
                self.total_time[0] = time.time() - start_time
                self.stats.append({"id":task_id, "rank": worker,
                                   "this_time": this_time,
                                   "time_over_est": this_time / time_est,
                                   "n_processed": self.n_processed[0],
                                   "total_time": self.total_time[0]})
                self.total_time_est[this_worker] += time_est
                subitted_task_ids.append(this_task_id)

        return submitted_task_ids


    
    def get_ready_worker(self):
        """
        Returns the id and data of a ready worker.
        If there are no workers, or no worker is ready, returns (None, None)
        """
        if self.workers_available:
            self.process()
            if len(self.ready_workers) > 0:
                ready_total_time_est = np.asarray([self.total_time_est[worker] for worker in self.ready_workers])
                worker = self.ready_workers[np.argmin(ready_total_time_est)]
                return worker, self.ready_workers_data[worker]
            else:
                return None, None
        else:
            return None, None
    
    def get_result(self, task_id):
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
                raise RuntimeError(f"get_result({task_id})) called before get_result("
                                   f"{worker_queue[source][0]})")
            logger.info(f"MPI controller : retrieving result for call with id {task_id} "
                        f"from worker {source} ...")
            
            while not (task_id in self.results):
                self.process()

            logger.info(f"MPI controller : received result for call with id {task_id} "
                        f"from worker {source}.")
            
        else:
            logger.info(f"MPI controller : returning result for call with id {task_id} "
                        "...")
        result = self.results[task_id]
        self.result_queue.remove(task_id)
        return task_id, result

    def get_next_result(self):
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
            return self.get_result(task_id)
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
            logger.info(f"MPI controller : received result for call with id {task_id} ...")
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
                logger.info(f"MPI controller : received result for call with id {task_id} ...")
                ret.append((task_id, self.results[task_id]))

        return ret

    def info(self):
        """
        Print processing statistics.

        Can only be called by the controller.
        """

        if len(self.stats) == 0:
            return

        call_times = np.array([s["this_time"] for s in self.stats])
        call_quotients = np.array([s["time_over_est"] for s in self.stats])
        cvar_call_quotients = call_quotients.std()/call_quotients.mean()

        if self.workers_available:
            worker_quotients = self.total_time/self.total_time_est 
            cvar_worker_quotients = worker_quotients.std()/worker_quotients.mean()
            print("\n"
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
                  f"{cvar_worker_quotients:.04f}\n")
        else:
            print("\n"
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
                  f"{cvar_call_quotients:.04f}\n")

    def exit(self):
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
                logger.info(f"MPI controller : telling worker {worker} "
                            "to exit...")
                reqs.append(self.comm.isend(None, dest=worker, tag=MessageTag.EXIT.value))
            MPI.Request.Waitall(reqs)
                
    def abort(self):
        """
        Abort execution on all MPI nodes immediately.

        Can be called by controller and workers.
        """
        traceback.print_exc()
        logger.error("MPI controller : aborting...")
        self.comm.Abort()

        
class MPIWorker(object):        

    def __init__(self, comm, group_comm, ready_data=None):
        self.comm = comm
        self.group_comm = group_comm
        self.worker_id = rank
        self.total_time_est = np.zeros(size)*np.nan
        self.total_time_est[rank] = 0
        self.n_processed = np.zeros(size)*np.nan
        self.n_processed[rank] = 0
        self.start_time = start_time
        self.total_time = np.zeros(size)*np.nan
        self.total_time[rank] = 0
        self.stats = []
        self.ready_data = None

        logger.info(f"MPI worker {self.worker_id}: initialized.")
        
    def serve(self):
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
        size = self.comm.size
        rank = self.comm.rank

        logger.info(f"MPI worker {rank}: waiting for calls.")
            
        # wait for orders:
        ready = True
        status = MPI.Status()
        exit_flag = False
        while not exit_flag:
            # signal the controller this worker is ready
            if ready:
                req = self.comm.isend(self.ready_data, dest=0, tag=MessageTag.READY.value)
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
                        object_to_call = eval(name_to_call,
                                              sys.modules[module].__dict__)
                    except NameError:
                        logger.error(str(sys.modules[module].__dict__.keys()))
                        raise
                else:
                    raise RuntimeError(f'MPI worker {self.worker_id}: unknown message tag {tag}')
                self.total_time_est[rank] += time_est
                call_time = time.time()
                result = object_to_call(*args, **kwargs)
                this_time = time.time() - call_time
                self.n_processed[rank] += 1
                self.stats.append({"id": task_id, "rank": rank,
                                   "this_time": this_time,
                                   "time_over_est": this_time / time_est,
                                   "n_processed": self.n_processed[rank],
                                   "total_time": time.time() - start_time})
                req = self.comm.isend((task_id, result, self.stats[-1]), dest=0, tag=MessageTag.DONE.value)
                req.wait()
                ready = True
            else:
                ready = False
                time.sleep(1)
    def abort(self):
        rank = self.comm.rank
        traceback.print_exc()
        logger.info(f"MPI worker {self.worker_id}: aborting...")
        comm.Abort()

class MPICollectiveWorker(object):

    def __init__(self, comm, merged_comm, worker_id, n_workers, worker_service_name="distwq.init", collective_mode=CollectiveMode.Gather):


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
        self.total_time_est = np.zeros(size)*np.nan
        self.total_time_est[rank] = 0
        self.n_processed = np.zeros(size)*np.nan
        self.n_processed[rank] = 0
        self.total_time = np.zeros(size)*np.nan
        self.total_time[rank] = 0
        self.stats = []

    def publish_service(self):
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
                
        
    def connect_service(self, n_lookup_attempts=5):
        info = MPI.INFO_NULL
        if not self.service_published:
            if self.comm.rank == 0:
                attempt = 0
                while attempt < n_lookup_attempts:
                    try:
                        self.worker_port = MPI.Lookup_name(self.worker_service_name)
                    except MPI.Exception as e:
                        if e.Get_error_class() == MPI.ERR_NAME:
                            time.sleep(random.randrange(1,5))
                        else:
                            raise e
                    attempt += 1
                assert(self.worker_port is not None)
                self.comm.bcast(self.worker_port, root=0)
            else:
                self.worker_port = self.comm.bcast(None, root=0)
            self.comm.barrier()
            if not self.worker_port:
                raise RuntimeError(f"connect_service: unable to lookup service {self.worker_service_name}")
            self.server_worker_comm = self.comm.Connect(self.worker_port, info, root=0)
        else:
            for i in range(self.n_workers-1):
                client_worker_comm = self.comm.Accept(self.worker_port, info, root=0)
                self.client_worker_comms.append(client_worker_comm)
        

        
    def serve(self):
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
        merged_size = self.merged_comm.Get_size()
        logger.info(f"MPI collective worker {self.worker_id}-{rank}: waiting for calls.")

        # wait for orders:
        while True:
            if rank == 0:
                logger.info(f"MPI collective worker {self.worker_id}-{rank}: getting next task from queue...")
            # get next task from queue:
            if self.collective_mode == CollectiveMode.Gather:
                req = self.merged_comm.Ibarrier()
                (name_to_call, args, kwargs, module, time_est, task_id) = \
                    self.merged_comm.scatter(None, root=0)
                req.wait()
            elif self.collective_mode == CollectiveMode.SendRecv:
                buf = bytearray(1<<20) # TODO: 1 MB buffer, make it configurable
                req = self.merged_comm.irecv(buf, source=0, tag=MessageTag.TASK.value)
                (name_to_call, args, kwargs, module, time_est, task_id) = req.wait()
            else:
                raise RuntimeError(f"Unknown collective mode {collective_mode}")
            if rank == 0:
                logger.info(f"MPI collective worker {self.worker_id}-{rank}: received next task from queue.")
            # TODO: add timeout and check whether controller lives!
            if name_to_call == "exit":
                logger.info(f"MPI collective worker {self.worker_id}-{rank}: exiting...")
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
                object_to_call = eval(name_to_call,
                                      sys.modules[module].__dict__)
            except NameError:
                logger.error(str(sys.modules[module].__dict__.keys()))
                raise
            self.total_time_est[rank] += time_est
            call_time = time.time()
            result = object_to_call(*args, **kwargs)
            this_time = time.time() - call_time
            self.n_processed[rank] += 1
            self.stats.append({"id": task_id,
                               "rank": merged_rank,
                               "this_time": this_time,
                               "time_over_est": this_time / time_est,
                               "n_processed": self.n_processed[rank],
                               "total_time": time.time() - start_time})
            if self.collective_mode == CollectiveMode.Gather:
                req = self.merged_comm.Ibarrier()
                self.merged_comm.gather((result, self.stats[-1]), root=0)
                req.wait()
            elif self.collective_mode == CollectiveMode.SendRecv:
                req = self.merged_comm.isend((result, self.stats[-1]), dest=0, tag=MessageTag.DONE.value)
                req.wait()
            else:
                raise RuntimeError(f"MPICollectiveWorker: unknown collective mode {self.collective_mode}")
                

    def abort(self):
        rank = self.comm.rank
        traceback.print_exc()
        logger.info(f"MPI collective worker {self.worker_id}-{rank}: aborting...")
        comm.Abort()

        
class MPICollectiveBroker(object):        

    def __init__(self, comm, group_comm, merged_comm, nprocs_per_worker, ready_data=None, is_worker=False, 
                 collective_mode=CollectiveMode.Gather):
        logger.info(f'MPI collective broker {rank} starting')
        assert(not spawned)
        self.collective_mode=collective_mode
        self.comm = comm
        self.group_comm = group_comm
        self.merged_comm = merged_comm
        self.worker_id = rank
        self.nprocs_per_worker = nprocs_per_worker

        self.start_time = start_time
        self.total_time_est = np.zeros(size)*np.nan
        self.total_time_est[rank] = 0
        self.n_processed = np.zeros(size)*np.nan
        self.n_processed[rank] = 0
        self.total_time = np.zeros(size)*np.nan
        self.total_time[rank] = 0
        self.stats = []
        self.is_worker = is_worker
        self.is_broker = True
        self.ready_data = ready_data
        
    def serve(self):
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
        size = self.comm.size
        rank = self.comm.rank
        merged_rank = self.merged_comm.Get_rank()
        merged_size = self.merged_comm.Get_size()
        
        logger.info(f"MPI worker broker {self.worker_id}: waiting for calls.")
            
        # wait for orders:
        while True:
            logger.info(f"MPI collective broker {self.worker_id}: getting next task from controller...")
            # signal the controller this worker is ready
            req = self.comm.isend(self.ready_data, dest=0, tag=MessageTag.READY.value)
            req.wait()

            while True:
                msg = self.process()
                if msg is not None:
                    tag, data = msg
                    break

            logger.info(f"MPI collective broker {self.worker_id}: received message from controller...")

            if tag == MessageTag.EXIT.value:
                logger.info(f"MPI worker broker {self.worker_id}: exiting...")
                msg = ("exit", (), {}, "", 0, 0)
                if self.collective_mode == CollectiveMode.Gather:
                    req = self.merged_comm.Ibarrier()
                    self.merged_comm.scatter([msg]*merged_size, root=0)
                    req.wait()
                elif self.collective_mode == CollectiveMode.SendRecv:
                    reqs = []
                    status = []
                    for i in range(self.nprocs_per_worker):
                        req = self.merged_comm.isend(msg, dest=i if self.is_worker else i+1, tag=MessageTag.TASK.value)                    
                        reqs.append(req)
                        status.append(MPI.Status())
                    MPI.Request.waitall(reqs, status)
                else:
                    raise RuntimeError(f'MPICollectiveBroker: unknown collective mode {self.collective_mode}')
                
                break
            elif tag == MessageTag.TASK.value:
                (name_to_call, args, kwargs, module, time_est, task_id) = data
            else:
                raise RuntimeError(f'MPI collective broker: unknown message tag {tag}')
                 
            logger.info(f"MPI collective broker {self.worker_id}: sending task {task_id} to workers...")
            if self.collective_mode == CollectiveMode.Gather:
                req = self.merged_comm.Ibarrier()
                self.merged_comm.scatter([(name_to_call, args, kwargs, module, time_est, task_id)]*merged_size,
                                         root=merged_rank)
                req.wait()
            elif self.collective_mode == CollectiveMode.SendRecv:
                msg = (name_to_call, args, kwargs, module, time_est, task_id)
                reqs = []
                status = [] 
                for i in range(self.nprocs_per_worker):
                    req = self.merged_comm.isend(msg, dest=i if self.is_worker else i+1, tag=MessageTag.TASK.value)                    
                    reqs.append(req)
                    status.append(MPI.Status())
                MPI.Request.waitall(reqs, status)
            else:
                raise RuntimeError(f'MPICollectiveBroker: unknown collective mode {self.collective_mode}')
                
            logger.info(f"MPI collective broker {self.worker_id}: sending task complete.")
            
            self.total_time_est[rank] += time_est
            if self.is_worker:

                try:
                    if module not in sys.modules:
                        importlib.import_module(module)
                        object_to_call = eval(name_to_call,
                                              sys.modules[module].__dict__)
                except NameError:
                    logger.error(str(sys.modules[module].__dict__.keys()))
                    raise

                call_time = time.time()
                this_result = object_to_call(*args, **kwargs)
                this_time = time.time() - call_time
                self.n_processed[rank] += 1
                this_stat = {"id": task_id,
                             "rank": merged_rank,
                             "this_time": this_time,
                             "time_over_est": this_time / time_est,
                             "n_processed": self.n_processed[rank],
                             "total_time": time.time() - start_time}
            else:
                this_result = None
                this_stat = None
                this_time = 0
                

            logger.info(f"MPI collective broker {self.worker_id}: gathering data from workers...")
            if self.collective_mode == CollectiveMode.Gather:
                req = self.merged_comm.Ibarrier()
                sub_data = self.merged_comm.gather((this_result, this_stat), root=merged_rank)
                req.wait()
                results = [result for result, stat in sub_data if result is not None]
                stats = [stat for result, stat in sub_data if stat is not None]
            elif self.collective_mode == CollectiveMode.SendRecv:
                reqs = []
                for i in range(self.nprocs_per_worker):
                    buf = bytearray(1<<20) # TODO: 1 MB buffer, make it configurable
                    req = self.merged_comm.irecv(buf, source=i if self.is_worker else i+1, tag=MessageTag.DONE.value)
                    reqs.append(req)
                status = [MPI.Status() for i in range(self.nprocs_per_worker)]
                try:
                    sub_data = MPI.Request.waitall(reqs, status)
                except:
                    print([MPI.Get_error_string(s.Get_error()) for s in status])
                    raise
                del(reqs)
                results = [result for result, stat in sub_data if result is not None]
                stats = [stat for result, stat in sub_data if stat is not None]
                if this_result is not None:
                    results = [this_result] + results
                    stats = [this_stat] + stats
            else:
                raise RuntimeError(f'MPICollectiveBroker: unknown collective mode {self.collective_mode}')
            logger.info(f"MPI collective broker {self.worker_id}: gathered {len(results)} results from workers...")

            stat_times = np.asarray([stat["this_time"] for stat in stats])
            max_time = 0.
            if len(stat_times) > 0:
                max_time = np.argmax(stat_times)
                stat = stats[max_time]
            else:
                stat = None
            ready = True
            logger.info(f"MPI collective broker {self.worker_id}: sending results to controller...")
            req = self.comm.isend((task_id, results, stat), dest=0, tag=MessageTag.DONE.value)
            req.wait()

    def process(self):
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
        comm.Abort()



def run(fun_name=None, module_name='__main__',
        broker_fun_name=None, broker_module_name='__main__', max_workers=-1,
        spawn_workers=False, sequential_spawn=False, spawn_startup_wait=None,
        spawn_executable=None, spawn_args=[],
        nprocs_per_worker=1, collective_mode="gather",
        broker_is_worker=False, worker_service_name="distwq.init", enable_worker_service=False,
        time_limit=None, verbose=False, args=()):
    """
    Run in controller/worker mode until fun(controller/worker) finishes.

    Must be called on all MPI nodes.

    On the controller, run() calls fun_name() and returns when fun_name() returns.

    On each worker, run() calls fun() if that is defined, or calls serve()
    otherwise, and returns when fun() returns, or when fun() returns on
    the controller, or when controller calls exit().

    :arg string module_name: module where fun_name is located
    :arg bool verbose: whether processing information should be printed.
    :arg bool spawn_workers: whether to spawn separate worker processes via MPI_Comm_spawn
    :arg bool sequential_spawn: whether to spawn processes in sequence
    :arg string spawn_executable: optional executable name for call to spawn (default is sys.executable)
    :arg string list spawn_args: optional arguments to prepend to list of arguments in call to spawn; or a callable that takes the list of arguments that distwq needs to pass to the python interpreter, and returns a new list of arguments
    :arg int nprocs_per_worker: how many processes per worker
    :arg broker_is_worker: when spawn_worker is True or nprocs_per_worker > 1, MPI_Spawn will be used to create workers, 
    and a CollectiveBroker object is used to relay tasks and results between controller and worker.
    When broker_is_worker is true, the broker also participates in serving tasks, otherwise it only 
    relays calls.
    :arg time_limit: maximum wall clock time, in seconds
    :arg args: additional args to pass to fun
 

    """

    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARN)
        
    assert nprocs_per_worker > 0
    assert not spawned

    global n_workers, is_worker
    if max_workers > 0:
        n_workers = min(max_workers, n_workers)
        if rank > n_workers:
            is_worker = False

    if not spawned and (size > 1):
        group_id = 0
        if is_controller:
            group_id = 2
        elif is_worker:
            group_id = 1
        else:
            group_id = 3
        group_comm = world_comm.Split(group_id, rank)
    else:
        group_comm = world_comm

    fun = None
    if fun_name is not None:
        if module_name not in sys.modules:
            importlib.import_module(module_name)
        fun = eval(fun_name, sys.modules[module_name].__dict__)

    broker_fun = None
    if broker_fun_name is not None:
        if broker_module_name not in sys.modules:
            importlib.import_module(broker_module_name)
        broker_fun = eval(broker_fun_name, sys.modules[broker_module_name].__dict__)

    if has_mpi:  # run in mpi mode
        if (n_workers > 0) and (nprocs_per_worker > 1):
            spawn_workers = True
        if is_controller:  # I'm the controller
            assert(fun is not None)
            controller = MPIController(world_comm, time_limit=time_limit)
            signal.signal(signal.SIGINT, lambda signum, frame: controller.abort())
            req = world_comm.Ibarrier()
            req.wait()
            try:  # put everything in a try block to be able to exit!
                fun(controller, *args)
            except ValueError:
                controller.abort()
            controller.exit()
        elif is_worker:  # I'm a worker or a broker
            worker_id = rank
            if broker_fun is not None:
                if spawn_workers is not True:
                    raise RuntimeError("distwq.run: cannot use broker_fun_name when spawn_workers is set to False")
                if broker_is_worker: 
                    raise RuntimeError("distwq.run: cannot use broker_fun_name when broker_is_worker is set to True")
            if spawn_workers and (nprocs_per_worker==1) and broker_is_worker:
                raise RuntimeError("distwq.run: cannot spawn workers when nprocs_per_worker=1 and broker_is_worker is set to True")
            if enable_worker_service and (not spawn_workers):
                raise RuntimeError("distwq.run: cannot enable worker service when spawn_workers is set to False")
            if spawn_workers:
                worker_config = { 'worker_id': worker_id,
                                  'n_workers': n_workers,
                                  'collective_mode': collective_mode,
                                  'enable_worker_service': enable_worker_service,
                                  'worker_service_name': worker_service_name,
                                  'verbose': verbose }
                if fun is not None:
                    worker_config['init_fun_name'] = str(fun_name)
                    worker_config['init_module_name'] = str(module_name)
                spawn_stmts = f"import sys, runpy; sys.argv.extend(['-', 'distwq:spawned', '{json.dumps(worker_config)}']); runpy.run_module('distwq', run_name='__main__'); sys.exit()"
                if callable(spawn_args):
                    arglist = spawn_args(['-c', spawn_stmts])
                else:
                    arglist = spawn_args + ['-c', spawn_stmts]
                logger.info(f"MPI broker {worker_id} : before spawn")
                worker_id = rank
                if collective_mode.lower() == "gather":
                    collective_mode_arg = CollectiveMode.Gather
                elif collective_mode.lower() == "sendrecv":
                    collective_mode_arg = CollectiveMode.SendRecv
                else:
                    raise RuntimeError(f"Unknown collective mode {collective_mode}")
                    
                if sequential_spawn and (worker_id > 1):
                    status = MPI.Status()
                    data = group_comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                if spawn_startup_wait is not None:
                    spawn_startup_wait = max(spawn_startup_wait, 2)
                    time.sleep(random.randrange(1, spawn_startup_wait))
                req = world_comm.Ibarrier()
                try:
                    if spawn_executable is None:
                        spawn_executable = sys.executable
                    sub_comm = MPI.COMM_SELF.Spawn(spawn_executable, args=arglist,
                                                   maxprocs=nprocs_per_worker-1 
                                                   if broker_is_worker else nprocs_per_worker)
                except:
                    logger.error(f"MPI broker {worker_id} : spawn error")
                    raise
                if sequential_spawn and (worker_id < n_workers):
                    group_comm.send("spawn", dest=worker_id)
                logger.info(f"MPI broker {worker_id} : after spawn")
                req.wait()
                req = sub_comm.Ibarrier()
                merged_comm = sub_comm.Merge(False)
                req.wait()
                broker=MPICollectiveBroker(world_comm, group_comm, merged_comm, 
                                           nprocs_per_worker,
                                           is_worker=broker_is_worker,
                                           collective_mode=collective_mode_arg)
                if fun is not None:
                    req = merged_comm.Ibarrier()
                    merged_comm.bcast(args, root=0)
                    req.wait()
                if broker_is_worker and (fun is not None):
                    fun(broker, *args)
                elif broker_fun is not None:
                    broker_fun(broker, *args)
                broker.serve()
            else:
                req = world_comm.Ibarrier()
                req.wait()
                worker = MPIWorker(world_comm, group_comm)
                if fun is not None:
                    fun(worker, *args)
                worker.serve()
    else:  # run as single processor
        assert(fun is not None)
        logger.info("MPI controller : not available, running as a single process.")
        controller = MPIController()
        fun(controller, *args)
        logger.info("MPI controller : finished.")

if __name__ == '__main__':
    if is_worker:
        worker_id = int(my_config['worker_id'])
        logger.info(f'MPI collective worker {worker_id}-{rank} starting')
        collective_mode = my_config['collective_mode']
        enable_worker_service = my_config['enable_worker_service']
        worker_service_name = my_config['worker_service_name']
        verbose_flag = my_config['verbose']
        verbose = True if verbose_flag == 1 else False
        if verbose:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARN)
        logger.info(f'MPI collective worker {worker_id}-{rank} starting')
        logger.info(f'MPI collective worker {worker_id}-{rank} args: {my_args}')

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
        
        worker = MPICollectiveWorker(world_comm, merged_comm, worker_id, n_workers,  
                                     collective_mode=collective_mode_arg,
                                     worker_service_name=worker_service_name)
        fun = None
        if 'init_fun_name' in my_config:
            fun_name = my_config['init_fun_name']
            module = my_config['init_module_name']
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
            ret_val = fun(worker, *args)
        worker.serve()
        MPI.Finalize()
