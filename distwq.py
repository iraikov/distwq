#!/usr/bin/python
#
# Distributed work queue operations using mpi4py.
#
# Copyright (C) 2020 Ivan Raikov and distwq authors.
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

import sys, importlib, time, traceback, logging, uuid
import numpy as np

logger = logging.getLogger(__name__)

# try to get the communicator object to see whether mpi is available:
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
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
    sys.stderr.flush()
    MPI.COMM_WORLD.Abort(1)

if has_mpi:
    sys_excepthook = sys.excepthook
    sys.excepthook = mpi_excepthook


# initialize:
workers_available = True
spawned = False
if has_mpi:
    spawned = (__name__ == '__main__')
    size = comm.size
    rank = comm.rank
    is_controller = (not spawned) and (rank == 0)
    if size < 2:
        workers_available = False
else:
    size = 1
    rank = 0
    is_controller = True

    
is_worker = not is_controller
n_workers = size - 1
start_time = time.time()

class MPIController(object):

    def __init__(self, comm):
        
        size = comm.size
        rank = comm.rank

        self.comm = comm
        self.workers_available = True if size > 1 else False

        self.total_time_est = np.zeros(size)
        """
        (numpy array of ints)
        total_time_est[i] is the current estimate of the total time
        MPI worker i will work on already submitted calls.
        On worker i, only total_time_est[i] is available.
        """
        self.total_time_est[0] = np.inf
        self.queue = []
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

    def submit_call(self, name_to_call, args=(), kwargs={},
                    module_name="__main__", time_est=1, id=None, worker=None):
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
        if id is None:
            id = 'distwq_%s' % str(uuid.uuid4())
        if id in self.assigned:
            raise RuntimeError("id ", str(id), " already in queue!")
        if worker is not None and is_worker:
            raise RuntimeError(
                "only the controller can use worker= in submit_call()")
        logger.info("MPI controller: total_time_est is {self.total_time_est}")
        if worker is None or worker < 1 or worker >= size:
            # find worker with least estimated total time:
            worker = np.argmin(self.total_time_est)
        if self.workers_available:
            # send name to call, args, time_est to worker:
            logger.info(f"MPI controller : assigning call with id {id} to worker "
                        f"{worker}: {name_to_call} {args} {kwargs} ...")
            self.comm.send((name_to_call, args, kwargs, module_name, time_est, id),
                           dest=worker)
        else:
            # perform call on this rank if no workers are available:
            worker = 0
            logger.info(f"MPI controller : calling {name_to_call} {args} {kwargs} "
                        "...")
            try:
                object_to_call = eval(name_to_call,
                                      sys.modules[module_name].__dict__)
            except NameError:
                logger.error(str(sys.modules[module_name].__dict__.keys()))
                raise
            call_time = time.time()
            self.results[id] = object_to_call(*args, **kwargs)
            this_time = time.time() - call_time
            self.n_processed[0] += 1
            self.total_time[0] = time.time() - start_time
            self.stats.append({"id": id, "rank": 0,
                               "this_time": this_time,
                               "time_over_est": this_time / time_est,
                               "n_processed": self.n_processed[0],
                               "total_time": self.total_time[0]})

        self.total_time_est[worker] += time_est
        self.queue.append(id)
        self.worker_queue[worker].append(id)
        self.assigned[id] = worker
        return id

    def get_result(self, id):
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
        source = self.assigned[id]
        if self.workers_available:
            if self.worker_queue[source][0] != id:
                raise RuntimeError("get_result(" + str(id)
                                   + ") called before get_result("
                                   + str(worker_queue[source][0]) + ")")
            logger.info(f"MPI controller : retrieving result for call with id {id} "
                        f"from worker {source} ...")
            data = self.comm.recv(source=source)
            
            (result, this_stats) = data
            self.stats.append(this_stats)
            self.n_processed[source] = this_stats["n_processed"]
            self.total_time[source] = this_stats["total_time"]
        else:
            logger.info(f"MPI controller : returning result for call with id {id} "
                        "...")
            result = self.results[id]
        self.queue.remove(id)
        self.worker_queue[source].remove(id)
        self.assigned.pop(id)
        return result

    def get_next_result(self):
        """
        Return result of next earlier submitted call whose result has not yet
        been got.

        Can only be called by the controller.

        If the call is not yet finished, waits for it to finish.

        :rtype:  object
        :return: return value of call, or None of there are no more calls in
                 the queue.
        """
        if len(self.queue) > 0:
            id = self.queue[0]
            return self.get_result(id)
        else:
            return None

    def info(self):
        """
        Print processing statistics.

        Can only be called by the controller.
        """

        call_times = np.array([s["this_time"] for s in self.stats])
        call_quotients = np.array([s["time_over_est"] for s in self.stats])

        if self.workers_available:
            worker_quotients = self.total_time/self.total_time_est
            print("\n"
                  "MPI run statistics\n"
                  "     =====================\n"
                  "     results collected:         "
                  f"{self.n_processed[1:].sum()}\n"
                  "     results not yet collected: "
                  f"{len(self.queue)}\n"
                  "     total reported time:       "
                  f"{call_times.sum()}\n"
                  "     mean time per call:        "
                  f"{call_times.mean()}\n"
                  "     std.dev. of time per call: "
                  f"{call_times.std()}\n"
                  "     coeff. of var. of actual over estd. time per call: "
                  f"{call_quotients.std()/call_quotients.mean()}\n"
                  "     workers:                      "
                  f"{n_workers}\n"
                  "     mean calls per worker:        "
                  f"{self.n_processed[1:].mean()}\n"
                  "     std.dev. of calls per worker: "
                  f"{self.n_processed[1:].std()}\n"
                  "     min calls per worker:         "
                  f"{self.n_processed[1:].min()}\n"
                  "     max calls per worker:         "
                  f"{self.n_processed[1:].max()}\n"
                  "     mean time per worker:        "
                  f"{self.total_time.mean()}\n"
                  "     std.dev. of time per worker: "
                  f"{self.total_time.std()}\n"
                  "     coeff. of var. of actual over estd. time per worker: "
                  f"{worker_quotients.std()/worker_quotients.mean()}\n")
        else:
            print("\n"
                  "MPI run statistics\n"
                  "     =====================\n"
                  "     results collected:         "
                  f"{self.n_processed[0]}\n"
                  "     results not yet collected: "
                  f"{len(self.queue)}\n"
                  "     total reported time:       "
                  f"{call_times.sum()}\n"
                  "     mean time per call:        "
                  f"{call_times.mean()}\n"
                  "     std.dev. of time per call: "
                  f"{call_times.std()}\n"
                  "     coeff. of var. of actual over estd. time per call: "
                  f"{call_quotients.std()/call_quotients.mean()}\n")

    def terminate(self):
        """
        Tell all workers to terminate.

        Can only be called by the controller.
        """
        if self.workers_available:
            # tell workers to terminate:
            for worker in range(1, size):
                logger.info(f"MPI controller : telling worker {worker} "
                            "to terminate...")
                comm.send(("terminate", (), {}, "", 0, 0), dest=worker)
            self.workers_available = False

    def abort(self):
        """
        Abort execution on all MPI nodes immediately.

        Can be called by controller and workers.
        """
        traceback.print_exc()
        logger.error("MPI controller : aborting...")
        self.comm.Abort()

        
class MPIWorker(object):        

    def __init__(self, comm):
        self.comm = comm
        self.total_time_est = np.zeros(size)*np.nan
        self.total_time_est[rank] = 0
        self.n_processed = np.zeros(size)*np.nan
        self.n_processed[rank] = 0
        self.total_time = np.zeros(size)*np.nan
        self.total_time[rank] = 0
        self.stats = []
        logger.info("MPI worker %d: initialized." % self.comm.rank)
        
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

        logger.info("MPI worker %d: waiting for calls." % rank)
            
        # wait for orders:
        while True:
            # get next task from queue:
            (name_to_call, args, kwargs, module, time_est, call_id) = \
                self.comm.recv(source=0)
            # TODO: add timeout and check whether controller lives!
            if name_to_call == "terminate":
                logger.info("MPI worker %d: terminating..." % rank)
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
            self.stats.append({"id": call_id, "rank": rank,
                               "this_time": this_time,
                               "time_over_est": this_time / time_est,
                               "n_processed": self.n_processed[rank],
                               "total_time": time.time() - start_time})
            comm.send((result, self.stats[-1]), dest=0)

    def abort(self):
        rank = self.comm.rank
        traceback.print_exc()
        logger.info("MPI worker %d: aborting..." % rank)
        comm.Abort()

class MPICollectiveWorker(object):        

    def __init__(self, comm, worker_id):
        self.worker_id = worker_id
        self.comm = comm
        self.parent_comm = self.comm.Get_parent()
        assert self.parent_comm != MPI.COMM_NULL
        self.merged_comm = self.parent_comm.Merge(True)

        self.total_time_est = np.zeros(size)*np.nan
        self.total_time_est[rank] = 0
        self.n_processed = np.zeros(size)*np.nan
        self.n_processed[rank] = 0
        self.total_time = np.zeros(size)*np.nan
        self.total_time[rank] = 0
        self.stats = []
        
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

        size = self.parent_comm.size
        rank = self.parent_comm.rank
        merged_rank = self.merged_comm.Get_rank()
        merged_size = self.merged_comm.Get_size()
        logger.info("MPI collective worker %d-%d: waiting for calls." % (self.worker_id, rank))

        # wait for orders:
        while True:
            logger.info("MPI collective worker %d-%d: getting next task from queue..." % (self.worker_id, rank))
            # get next task from queue:
            (name_to_call, args, kwargs, module, time_est, call_id) = \
                self.merged_comm.scatter(None, root=0)
            # TODO: add timeout and check whether controller lives!
            if name_to_call == "terminate":
                logger.info("MPI collective worker %d-%d: terminating..." % (self.worker_id, rank))
                self.parent_comm.Disconnect()
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
            self.stats.append({"id": call_id,
                               "rank": merged_rank,
                               "this_time": this_time,
                               "time_over_est": this_time / time_est,
                               "n_processed": self.n_processed[rank],
                               "total_time": time.time() - start_time})
            self.merged_comm.gather((result, self.stats[-1]), root=0)

    def abort(self):
        rank = self.comm.rank
        traceback.print_exc()
        logger.info("MPI collective worker %d-%d: aborting..." % (self.worker_id, rank))
        comm.Abort()

        
class MPICollectiveBroker(object):        

    def __init__(self, comm, sub_comm, is_worker=False):
        self.comm = comm
        self.sub_comm = sub_comm
        self.merged_comm = sub_comm.Merge(False)
        self.total_time_est = np.zeros(size)*np.nan
        self.total_time_est[rank] = 0
        self.n_processed = np.zeros(size)*np.nan
        self.n_processed[rank] = 0
        self.total_time = np.zeros(size)*np.nan
        self.total_time[rank] = 0
        self.stats = []
        self.is_worker = is_worker
        
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

        logger.info("MPI worker broker %d: waiting for calls." % (rank-1))
            
        # wait for orders:
        while True:
            # get next task from controller queue:
            (name_to_call, args, kwargs, module, time_est, call_id) = \
                self.comm.recv(source=0)
            # TODO: add timeout and check whether controller lives!
            if name_to_call == "terminate":
                logger.info("MPI worker broker %d: terminating..." % (rank-1))
                self.merged_comm.scatter([("terminate", (), {}, "", 0, 0)]*merged_size, root=merged_rank)
                self.sub_comm.Disconnect()
                self.merged_comm.Disconnect()
                break
                
            logger.info("MPI collective broker %d: sending task to workers..." % (rank-1))
            self.merged_comm.scatter([(name_to_call, args, kwargs, module, time_est, call_id)]*merged_size,
                                     root=merged_rank)

            self.total_time_est[rank] += time_est
            if self.is_worker:
                call_time = time.time()
                result = object_to_call(*args, **kwargs)
                this_time = time.time() - call_time
                self.n_processed[rank] += 1
                this_stat = {"id": call_id,
                             "rank": merged_rank,
                             "this_time": this_time,
                             "time_over_est": this_time / time_est,
                             "n_processed": self.n_processed[rank],
                             "total_time": time.time() - start_time}
            else:
                result = None
                this_stat = None
                this_time = 0
                
            if this_stat is not None:
                self.stats.append(this_stat)
            
            sub_results_and_stats = self.merged_comm.gather((result, this_stat), root=merged_rank)
            results = [result for result, stat in sub_results_and_stats if result is not None]
            logger.info("MPI collective broker %d: gathered %s results from workers..." % (rank-1, len(results)))
            stats = [stat for result, stat in sub_results_and_stats if result is not None]
            stat_times = np.asarray([stat["this_time"] for stat in stats])
            max_time = np.argmax(stat_times)
            stat = stats[max_time]
            logger.info("MPI collective broker %d: sending results to controller..." % (rank-1))
            self.comm.send((results, stat), dest=0)

    def abort(self):
        rank = self.comm.rank
        traceback.print_exc()
        logger.info("MPI worker broker %d: aborting..." % rank)
        comm.Abort()



def run(fun_name=None, module_name='__main__', verbose=False, nprocs_per_worker=1, broker_is_worker=False, args=()):
    """
    Run in controller/worker mode until fun(controller/worker) finishes.

    Must be called on all MPI nodes.

    On the controller, run() calls fun_name() and returns when fun_name() returns.

    On each worker, run() calls fun() if that is defined, or calls serve()
    otherwise, and returns when fun() returns, or when fun() returns on
    the controller, or when controller calls terminate().

    :arg string module_name: module where fun_name is located
    :arg bool verbose: whether processing information should be printed.
    :arg int nprocs_per_worker: how many processes per worker
    :arg broker_is_worker: when nprocs_per_worker, MPI Spawn will be used to create workers, 
    and a CollectiveBroker object is used to relay tasks and results between controller and worker.
    When broker_is_worker is true, the broker also participates in serving tasks, otherwise it only 
    relays calls.
    :arg args: additional args to pass to fun
 

    """

    if verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARN)

    assert nprocs_per_worker > 0
    fun = None
    if fun_name is not None:
        fun = eval(fun_name, sys.modules[module_name].__dict__)
        
    if has_mpi:  # run in mpi mode
        if is_controller:  # I'm the controller
            assert(fun is not None)
            controller = MPIController(comm)
            try:  # put everything in a try block to be able to terminate!
                fun(controller, *args)
            except ValueError:
                controller.abort()
            controller.terminate()
        else:  # I'm a worker or a broker
            if nprocs_per_worker > 1:
                arglist = ['-m', 'distwq', '-', '%d' % (rank-1)]
                if fun is not None:
                    arglist += [str(fun_name), str(module_name)]
                sub_comm = MPI.COMM_SELF.Spawn(sys.executable,
                                                args=arglist,
                                                maxprocs=nprocs_per_worker-1
                                                   if broker_is_worker else nprocs_per_worker)
                broker=MPICollectiveBroker(comm, sub_comm, is_worker=broker_is_worker)
                if fun is not None:
                    fun(broker, *args)
                else:
                    broker.serve()
            else:
                worker = MPIWorker(comm)
                if fun is not None:
                    fun(worker, *args)
                else:
                    worker.serve()
    else:  # run as single processor
        assert(fun is not None)
        logger.info("MPI controller : not available, running as a single process.")
        controller = MPIController()
        fun(controller, *args)
        logger.info("MPI controller : finished.")

        
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if is_worker:
        worker_id = int(sys.argv[2])
        logger.info('MPI collective worker %d-%d starting' % (worker_id, rank))
        worker = MPICollectiveWorker(comm, worker_id)
        fun = None
        if len(sys.argv) > 3:
            fun_name = sys.argv[4]
            module = sys.argv[5]
            fun = eval(fun_name, sys.modules[module].__dict__)
        if fun is not None:
            fun(worker, sys.argv[6:])
        else:
            worker.serve()
    
