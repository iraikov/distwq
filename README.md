# distwq

Distributed work queue operations using mpi4py.

Allows for easy parallelization in controller/worker mode with one
controller submitting function or method calls to workers. Supports
multiple ranks per worker (collective workers). Uses mpi4py if
available, otherwise processes calls sequentially in one process.

Based on mpi.py from the pyunicorn project.

## Global Variables

- `is_controller` (bool): True if current process is the controller
- `is_worker` (bool): True if current process is a worker
- `spawned` (bool): True if current process was spawned via MPI_Comm_spawn
- `workers_available` (bool): True if workers are available
- `size` (int): Total number of MPI processes
- `rank` (int): Rank of current MPI process
- `n_workers` (int): Number of worker processes

## Enums

### CollectiveMode
- `Gather = 1`: Use MPI gather/scatter for collective operations
- `SendRecv = 2`: Use MPI send/receive for collective operations

### MessageTag
- `READY = 0`: Worker ready message
- `DONE = 1`: Task completion message
- `TASK = 2`: Task assignment message
- `EXIT = 3`: Worker exit message

### GroupingMethod
- `NoGrouping = 0`: No worker grouping
- `GroupSpawn = 1`: Group workers via MPI_Comm_spawn
- `GroupSplit = 2`: Group workers via MPI_Comm_split

## EXAMPLE

```python
# Example of using distributed work queue distwq
# PYTHONPATH must include the directories in which distwq and this file are located.

import distwq
import numpy as np  
import scipy
from scipy import signal

def do_work(freq):
    fs = 10e3
    N = 1e5
    amp = 2*np.sqrt(2)
    freq = float(freq)
    noise_power = 0.001 * fs / 2
    time = np.arange(N) / fs
    x = amp*np.sin(2*np.pi*freq*time)
    x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
    f, pdens = signal.periodogram(x, fs)
    return f, pdens

def main(controller):
    n = 150
    for i in range(0, n):
        controller.submit_call("do_work", (i+1,), module_name="example_distwq")
    s = []
    for i in range(0, n):
        s.append(controller.get_next_result())
    print("results length : %d" % len(s))
    print(s)
    controller.info()

if __name__ == '__main__':
    if distwq.is_controller:
        distwq.run(fun_name="main", verbose=True, nprocs_per_worker=3)
    else:
        distwq.run(fun_name=None, verbose=True, nprocs_per_worker=3)
```

## API

### MPIController

#### submit_call

```python
submit_call(name_to_call, args=(), kwargs={}, module_name="__main__", time_est=1, task_id=None, worker=None)
```

Submit a call for parallel execution.

If called by the controller and workers are available, the call is submitted
to a worker for asynchronous execution.

If called by a worker or if no workers are available, the call is instead
executed synchronously on this MPI node.

**Examples:**

1. Provide ids and time estimate explicitly:

```python
for n in range(0,10):
    controller.submit_call("doit", (n,A[n]), task_id=n, time_est=n**2)

for n in range(0,10):
    result[n] = controller.get_result(n)
```

2. Use generated ids stored in a list:

```python
for n in range(0,10):
    ids.append(controller.submit_call("doit", (n,A[n])))

for n in range(0,10):
    results.append(controller.get_result(ids.pop()))
```

3. Ignore ids altogether:

```python
for n in range(0,10):
    controller.submit_call("doit", (n,A[n]))

for n in range(0,10):
    results.append(controller.get_next_result())
```

4. Call a module function and use keyword arguments:

```python
controller.submit_call("solve", (), {"a":a, "b":b}, module_name="numpy.linalg")
```

**Parameters:**
- `name_to_call` (str): name of callable object (usually a function or static method of a class) as contained in the namespace specified by module.
- `args` (tuple): the positional arguments to provide to the callable object. Tuples of length 1 must be written (arg,). Default: ()
- `kwargs` (dict): the keyword arguments to provide to the callable object. Default: {}
- `module_name` (str): optional name of the imported module or submodule in whose namespace the callable object is contained. For objects defined on the script level, this is "__main__", for objects defined in an imported package, this is the package name. Must be a key of the dictionary sys.modules (check there after import if in doubt). Default: "__main__"
- `time_est` (int): estimated relative completion time for this call; used to find a suitable worker. Default: 1
- `task_id` (int or None): unique id for this call. Must be a possible dictionary key. If None, a random id is assigned and returned. Can be re-used after get_result() for this call. Default: None
- `worker` (int > 0 and < comm.size, or None): optional no. of worker to assign the call to. If None, the call is assigned to the worker with the smallest current total time estimate. Default: None

**Returns:**
id of call, to be used in get_result().

#### queue_call

```python
queue_call(name_to_call, args=(), kwargs={}, module_name="__main__", time_est=1, task_id=None, requested_worker=None)
```

Submit a call for later execution.

If called by the controller and workers are available, the call is put on the wait queue and submitted to a worker when it is available. Method process() checks the wait queue and submits calls on the wait queue.

**Parameters:**
Same as submit_call(), except:
- `requested_worker` (int > 0 and < comm.size, or None): optional no. of worker to assign the call to. If None, or the worker is not available, the call is assigned to the worker with the smallest current total time estimate. Default: None

**Returns:**
id of call, to be used in get_result().

#### submit_multiple

```python
submit_multiple(name_to_call, args=[], kwargs=[], module_name="__main__", time_est=1, task_ids=None, workers=None)
```

Submit multiple calls for parallel execution.

Analogous to submit_call, but accepts lists of arguments and submits to multiple workers for asynchronous execution.

**Parameters:**
- `name_to_call` (str): name of callable object
- `args` (list): the positional arguments to provide to the callable object for each task, as a list of tuples. Default: []
- `kwargs` (list): the keyword arguments to provide to the callable object for each task, as a list of dictionaries. Default: []
- `module_name` (str): optional name of the imported module or submodule. Default: "__main__"
- `time_est` (int): estimated relative completion time for this call. Default: 1
- `task_ids` (list or None): unique ids for each call. If None, random ids are assigned. Default: None
- `workers` (list of int > 0 and < comm.size, or None): optional worker ids to assign the tasks to. If None, the tasks are assigned in order to the workers with the smallest current total time estimate. Default: None

**Returns:**
List of task ids for the submitted calls.

#### get_result

```python
get_result(task_id)
```

Return result of earlier submitted call.

Can only be called by the controller.

If the call is not yet finished, waits for it to finish.
Results should be collected in the same order as calls were submitted.
For each worker, the results of calls assigned to that worker must be
collected in the same order as those calls were submitted.
Can only be called once per call.

**Parameters:**
- `task_id` (int): id of an earlier submitted call, as provided to or returned by submit_call().

**Returns:**
Tuple of (task_id, return value of call).

#### get_next_result

```python
get_next_result()
```

Return result of next earlier submitted call whose result has not yet
been obtained.

Can only be called by the controller.

If the call is not yet finished, waits for it to finish.

**Returns:**
Tuple of (id, return value of call), or None if there are no more calls in the queue.

#### probe_next_result

```python
probe_next_result()
```

Return result of next earlier submitted call whose result has not yet
been obtained.

Can only be called by the controller.

If no result is available, returns None.

**Returns:**
Tuple of (id, return value of call), or None if there are no results ready.

#### probe_all_next_results

```python
probe_all_next_results()
```

Return all available results of earlier submitted calls whose result has not yet
been obtained.

Can only be called by the controller.

If no result is available, returns empty list.

**Returns:**
List of tuples (id, return value of call).

#### get_ready_worker

```python
get_ready_worker()
```

Returns the id and data of a ready worker.

**Returns:**
Tuple of (worker_id, worker_data) if a worker is available, or (None, None) if no workers are available or ready.

#### info

```python
info()
```

Print processing statistics.

Can only be called by the controller.

#### exit

```python
exit()
```

Tell all workers to exit.

Can only be called by the controller.

#### abort

```python
abort()
```

Abort execution on all MPI nodes immediately.

Can be called by controller and workers.

### MPIWorker

#### serve

```python
serve()
```

Serve submitted calls until told to finish.

Call this function if workers need to perform initialization
different from the controller, like this:

```python
def workerfun(worker):
    do = whatever + initialization - is * necessary
    worker.serve()
    do = whatever + cleanup - is * necessary
```

If workerfun() is not defined, serve() will be called automatically by run().

#### abort

```python
abort()
```

Abort execution on all MPI nodes immediately.

### MPICollectiveWorker

Used for collective operations where multiple processes work together on tasks.

#### serve

```python
serve()
```

Serve submitted calls until told to finish. Tasks are obtained via scatter and results are returned via gather, i.e. all collective workers spawned by a CollectiveBroker will participate in these collective calls.

#### publish_service

```python
publish_service()
```

Publish the worker service for discovery by other processes.

#### connect_service

```python
connect_service(n_lookup_attempts=5)
```

Connect to the worker service.

**Parameters:**
- `n_lookup_attempts` (int): Number of attempts to lookup the service. Default: 5

#### abort

```python
abort()
```

Abort execution on all MPI nodes immediately.

### MPICollectiveBroker

#### serve

```python
serve()
```

Broker and serve submitted calls until told to finish. A task
is received from the controller and sent to all collective
workers associated with this broker via scatter.

Call this function if workers need to perform initialization
different from the controller, like this:

```python
def workerfun(worker):
    do = whatever + initialization - is * necessary
    worker.serve()
    do = whatever + cleanup - is * necessary
```

If workerfun() is not defined, serve() will be called automatically by run().

#### abort

```python
abort()
```

Abort execution on all MPI nodes immediately.

## Procedures

### run

```python
run(fun_name=None, module_name="__main__", broker_fun_name=None, broker_module_name="__main__", 
    max_workers=-1, worker_grouping_method=GroupingMethod.NoGrouping, sequential_spawn=False, 
    spawn_startup_wait=None, spawn_executable=None, spawn_args=[], nprocs_per_worker=1, 
    collective_mode="gather", broker_is_worker=False, worker_service_name="distwq.init", 
    enable_worker_service=False, time_limit=None, verbose=False, args=())
```

Run in controller/worker mode until fun(controller/worker) finishes.

Must be called on all MPI nodes.

On the controller, run() calls fun_name() and returns when fun_name() returns.

On each worker, run() calls fun() if that is defined, or calls serve()
otherwise, and returns when fun() returns, or when fun() returns on
the controller, or when controller calls exit().

**Parameters:**
- `fun_name` (str or None): name of function to call on controller. Default: None
- `module_name` (str): module where fun_name is located. Default: "__main__"
- `broker_fun_name` (str or None): name of function to call on brokers. Default: None
- `broker_module_name` (str): module where broker_fun_name is located. Default: "__main__"
- `max_workers` (int): maximum number of workers to use. -1 means use all available. Default: -1
- `worker_grouping_method` (str or GroupingMethod): specifies grouping method for workers: "spawn", "split", or GroupingMethod enum values. Default: GroupingMethod.NoGrouping
- `sequential_spawn` (bool): whether to spawn processes in sequence. Default: False
- `spawn_startup_wait` (int or None): optional startup wait time for spawned processes. Default: None
- `spawn_executable` (str or None): optional executable name for call to spawn (default is sys.executable). Default: None
- `spawn_args` (list): optional arguments to prepend to list of arguments in call to spawn; or a callable that takes the list of arguments that distwq needs to pass to the python interpreter, and returns a new list of arguments. Default: []
- `nprocs_per_worker` (int): how many processes per worker. Default: 1
- `collective_mode` (str): collective communication mode ("gather" or "sendrecv"). Default: "gather"
- `broker_is_worker` (bool): when worker_grouping_method is GroupSpawn or GroupSplit and nprocs_per_worker > 1, MPI_Comm_spawn or MPI_Comm_split will be used to create workers, and a CollectiveBroker object is used to relay tasks and results between controller and worker. When broker_is_worker is true, the broker also participates in serving tasks, otherwise it only relays calls. Default: False
- `worker_service_name` (str): name for worker service discovery. Default: "distwq.init"
- `enable_worker_service` (bool): whether to enable worker service for discovery. Default: False
- `time_limit` (int or None): maximum wall clock time, in seconds. Default: None
- `verbose` (bool): whether processing information should be printed. Default: False
- `args` (tuple): additional args to pass to fun. Default: ()

**Returns:**
Return value of the controller function, or None for workers.
