# Example of using distributed work queue distwq
# PYTHONPATH must include the directories in which distwq and this file are located.

import pprint
import distwq
import numpy as np  
import scipy
from scipy import signal
from mpi4py import MPI

def do_work(freq):
    rng = np.random.RandomState()
    fs = 10e3
    N = 1e5
    amp = 2*np.sqrt(2)
    freq = float(freq)
    noise_power = 0.001 * fs / 2
    time = np.arange(N) / fs
    x = amp*np.sin(2*np.pi*freq*time)
    x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    f, pdens = signal.periodogram(x, fs)
    return f, pdens

def init(worker):
    if worker.worker_id == 1:
        if worker.comm.rank == 0:
            root=MPI.ROOT
        else:
            root=MPI.PROC_NULL
    else:
        root = 0
    data = worker.worker_comm.alltoall(['inter alltoall']*3)
    assert (data == ['inter alltoall']*3)
    
    
def main(controller):
    n = 5
    for i in range(0, n):
        controller.submit_call("do_work", (i+1,), module_name="example_distwq")
    s = []
    for i in range(0, n):
        s.append(controller.get_next_result())
    controller.info()
    pprint.pprint(s)

if __name__ == '__main__':
    if distwq.is_controller:
        distwq.run(fun_name="main", verbose=True, nprocs_per_worker=3)
    else:
        distwq.run(fun_name="init", module_name="example_distwq", verbose=True, nprocs_per_worker=3)
