# Example of using distributed work queue distwq
# PYTHONPATH must include the directories in which distwq and this file are located.

import distwq
import numpy as np  
import scipy
from scipy import signal
from mpi4py import MPI
from neuron import h
h.load_file('stdrun.hoc')

def do_work(i):
    pc = h.ParallelContext()
    rank = int(pc.id())
    nhost = int(pc.nhost())
    print ("worker %d of %d: %i"  %  (rank, nhost, i))
    soma = h.Section(name='soma')
    soma.L = 20
    soma.diam = 20
    soma.insert('hh')
    iclamp = h.IClamp(soma(0.5))
    iclamp.delay = 2
    iclamp.dur = 0.1
    iclamp.amp = 0.9
    rec_v = h.Vector()
    rec_t = h.Vector()
    rec_v.record(soma(0.5)._ref_v) # Membrane potential vector
    rec_t.record(h._ref_t)
    h.finitialize(-65)
    h.fadvance()
    h.continuerun(250)
    return rec_v.max()


def main(controller):
    n = 10
    for i in range(0, n):
        controller.submit_call("do_work", (i+1,), module_name="example_nrn")
    s = []
    for i in range(0, n):
        s.append(controller.get_next_result())
    print(s)
    controller.info()

if __name__ == '__main__':
    if distwq.is_controller:
        distwq.run(fun_name="main", verbose=True, nprocs_per_worker=3)
    else:
        distwq.run(fun_name=None, verbose=True, nprocs_per_worker=3)
