# Example of using distributed work queue distwq
# PYTHONPATH must include the directories in which distwq and this file are located.

import itertools
import time

import numpy as np
from mpi4py import MPI
from neuron import coreneuron, gui, h

import distwq

h.load_file("stdrun.hoc")


cells = []
nclist = []
vrecs = []
stims = []


class MyCell:
    _ids = itertools.count(0)

    def __repr__(self):
        return "MyCell[%d]" % self.id

    def __init__(self):
        self.id = next(self._ids)
        # create the morphology and connect it
        self.soma = h.Section(name="soma", cell=self)
        self.dend = h.Section(name="dend", cell=self)
        self.dend.connect(self.soma(0.5))
        self.soma.insert("pas")
        self.dend.insert("pas")
        self.dend(0.5).pas.e = -65
        self.soma(0.5).pas.e = -65
        self.synlist = []


## Creates half-gap junction mechanism
def mkgap(pc, sec, gid, secpos, sgid, dgid, w, gjlist):
    myrank = int(pc.id())

    seg = sec(secpos)
    gj = h.ggap(seg)
    gj.g = w

    pc.source_var(seg._ref_v, sgid, sec=sec)
    pc.target_var(gj, gj._ref_vgap, dgid)

    if myrank == 0:
        print(
            "mkgap: gid %i: sec=%s sgid=%i dgid=%i w=%f"
            % (gid, str(sec), sgid, dgid, w)
        )

    gjlist.append(gj)

    return gj


def mkcells(pc, ngids):
    nranks = int(pc.nhost())
    myrank = int(pc.id())

    assert nranks <= ngids

    for gid in range(ngids):

        if gid % nranks == myrank:

            cell = MyCell()
            nc = h.NetCon(cell.soma(0.5)._ref_v, None, sec=cell.soma)
            pc.set_gid2node(gid, myrank)
            pc.cell(gid, nc, 1)
            cells.append(cell)

            # Current injection into section
            stim = h.IClamp(cell.soma(0.5))
            if gid % 2 == 0:
                stim.delay = 10
            else:
                stim.delay = 20
            stim.dur = 20
            stim.amp = 10
            stims.append(stim)

            # Record membrane potential
            v = h.Vector()
            v.record(cell.dend(0.5)._ref_v)
            vrecs.append(v)

            if myrank == 0:
                print(
                    "Rank %i: created gid %i; stim delay = %.02f"
                    % (myrank, gid, stim.delay)
                )


## Creates connections:
def connectcells(pc, ngids):
    nranks = int(pc.nhost())
    myrank = int(pc.id())

    for gid in range(0, ngids, 2):

        # source gid: all even gids
        src = gid
        # destination gid: all odd gids
        dst = gid + 1

        if pc.gid_exists(dst) > 0:
            cell = pc.gid2cell(dst)
            sec = cell.dend
            syn = h.Exp2Syn(sec(0.5))
            nc = pc.gid_connect(src, syn)
            nc.delay = 0.5
            nclist.append(nc)
            cell.synlist.append(syn)


def do_work(i):

    pc = h.ParallelContext()
    pc.gid_clear()
    myrank = int(pc.id())

    ngids = 5 * int(pc.nhost())
    mkcells(pc, ngids)
    connectcells(pc, ngids)

    h.cvode.use_fast_imem(1)
    h.cvode.cache_efficient(1)

    if hasattr(h, "nrn_sparse_partrans"):
        h.nrn_sparse_partrans = 1

    rec_t = h.Vector()
    rec_t.record(h._ref_t)

    wt = time.time()

    h.dt = 0.025
    pc.set_maxstep(10)

    coreneuron.enable = True
    coreneuron.verbose = 1

    h.finitialize(-65)
    pc.psolve(500)

    total_wt = time.time() - wt

    print("rank %d: total compute time: %.02f" % (myrank, total_wt))

    output = itertools.chain(
        [np.asarray(rec_t.to_python())],
        [np.asarray(vrec.to_python()) for vrec in vrecs],
    )
    return output


def main(controller):
    n = 10
    for i in range(0, n):
        controller.submit_call("do_work", (i + 1,), module_name="example_nrn_direct")
    s = []
    for i in range(0, n):
        s.append(controller.get_next_result())
    print(s)
    controller.info()


if __name__ == "__main__":
    if distwq.is_controller:
        distwq.run(fun_name="main", verbose=True, nprocs_per_worker=3)
    else:
        distwq.run(fun_name=None, verbose=True, nprocs_per_worker=3)
