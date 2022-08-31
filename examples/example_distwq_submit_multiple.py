# Example of using distributed work queue distwq
# PYTHONPATH must include the directories in which distwq and this file are located.

import pprint

import numpy as np
import scipy
from mpi4py import MPI
from scipy import signal

import distwq


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
    pass


def main(controller):

    n = 10
    args = list(([i + 1] for i in range(n)))
    controller.submit_multiple(
        "do_work", args, module_name="example_distwq_submit_multiple"
    )
    s = []
    for i in range(0, n):
        s.append(controller.get_next_result())
    controller.info()
    pprint.pprint(s)


if __name__ == "__main__":
    if distwq.is_controller:
        distwq.run(fun_name="main", verbose=True, spawn_workers=False)
    else:
        distwq.run(
            fun_name="init",
            module_name="example_distwq_submit_multiple",
            spawn_workers=False,
            verbose=True,
        )
