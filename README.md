<a name=".distwq"></a>
# distwq
Distributed work queue operations using mpi4py.


Allows for easy parallelization in controller/worker mode with one
controller submitting function or method calls to workers.  Supports
multiple ranks per worker (collective workers). Uses mpi4py if
available, otherwise processes calls sequentially in one process.

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


<a name=".distwq.MPIController.submit_call"></a>
#### submit\_call

```python
 | submit_call(name_to_call, args=(), kwargs={}, module_name="__main__", time_est=1, task_id=None)
```

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

<a name=".distwq.MPIController.get_result"></a>
#### get\_result

```python
 | get_result(task_id)
```

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

**Returns**:

return value of call.

<a name=".distwq.MPIController.get_next_result"></a>
#### get\_next\_result

```python
 | get_next_result()
```

Return result of next earlier submitted call whose result has not yet
been obtained.

Can only be called by the controller.

If the call is not yet finished, waits for it to finish.

:rtype:  object

**Returns**:

id, return value of call, or None of there are no more calls in
the queue.

<a name=".distwq.MPIController.info"></a>
#### info

```python
 | info()
```

Print processing statistics.

Can only be called by the controller.

<a name=".distwq.MPIController.exit"></a>
#### exit

```python
 | exit()
```

Tell all workers to exit.

Can only be called by the controller.

<a name=".distwq.MPIController.abort"></a>
#### abort

```python
 | abort()
```

Abort execution on all MPI nodes immediately.

Can be called by controller and workers.

<a name=".distwq.MPIWorker.serve"></a>
#### serve

```python
 | serve()
```

Serve submitted calls until told to finish.

Call this function if workers need to perform initialization
different from the controller, like this:

>>> def workerfun(worker):
>>>     do = whatever + initialization - is * necessary
>>>     worker.serve()
>>>     do = whatever + cleanup - is * necessary

If you don't define workerfun(), serve() will be called automatically by
run().

<a name=".distwq.MPICollectiveWorker.serve"></a>
#### serve

```python
 | serve()
```

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

<a name=".distwq.MPICollectiveBroker.serve"></a>
#### serve

```python
 | serve()
```

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

<a name=".distwq.run"></a>
#### run

```python
run(fun_name=None, module_name='__main__', verbose=False, spawn_workers=False, nprocs_per_worker=1, broker_is_worker=False, args=())
```

Run in controller/worker mode until fun(controller/worker) finishes.

Must be called on all MPI nodes.

On the controller, run() calls fun_name() and returns when fun_name() returns.

On each worker, run() calls fun() if that is defined, or calls serve()
otherwise, and returns when fun() returns, or when fun() returns on
the controller, or when controller calls exit().

:arg string module_name: module where fun_name is located
:arg bool verbose: whether processing information should be printed.
:arg bool spawn_workers: whether to spawn separate worker processes via MPI_Spawn
:arg int nprocs_per_worker: how many processes per worker
:arg broker_is_worker: when spawn_worker is True or nprocs_per_worker > 1, MPI_Spawn will be used to create workers, 
and a CollectiveBroker object is used to relay tasks and results between controller and worker.
When broker_is_worker is true, the broker also participates in serving tasks, otherwise it only 
relays calls.
:arg args: additional args to pass to fun

