# Example of using distributed work queue distwq
# PYTHONPATH must include the directories in which distwq and this file are located.

import pprint
import sys

import numpy as np
from scipy import signal

import distwq

nprocs_per_worker = 3


def do_work(freq):
    rng = np.random.RandomState()
    fs = 10e3
    N = 1e5
    amp = 2 * np.sqrt(2)
    freq = float(freq)
    noise_power = 0.001 * fs / 2
    time = np.arange(N) / fs
    x = amp * np.sin(2 * np.pi * freq * time)
    x += rng.normal(scale=np.sqrt(noise_power), size=time.shape)
    f, pdens = signal.periodogram(x, fs)
    return f, pdens


def init(worker):
    if worker.server_worker_comm is not None:
        data = worker.server_worker_comm.alltoall(
            ["inter alltoall"] * nprocs_per_worker
        )
        assert data == ["inter alltoall"] * nprocs_per_worker
        worker.server_worker_comm.barrier()
    else:
        for client_worker_comm in worker.client_worker_comms:
            data = client_worker_comm.alltoall(["inter alltoall"] * nprocs_per_worker)
            assert data == ["inter alltoall"] * nprocs_per_worker
            client_worker_comm.barrier()
    print(f"worker init: data = {data}")
    sys.stdout.flush()


def main(controller):
    n = 5
    for i in range(0, n):
        controller.submit_call(
            "do_work", (i + 1,), module_name="example_distwq_service"
        )
    s = []
    for i in range(0, n):
        s.append(controller.get_next_result())
    controller.info()
    pprint.pprint(s)


if __name__ == "__main__":
    if distwq.is_controller:
        distwq.run(
            fun_name="main",
            verbose=True,
            worker_grouping_method="spawn",
            nprocs_per_worker=nprocs_per_worker,
        )
    else:
        distwq.run(
            fun_name="init",
            module_name="example_distwq_service",
            enable_worker_service=True,
            worker_grouping_method="spawn",
            nprocs_per_worker=nprocs_per_worker,
            verbose=True,
        )
