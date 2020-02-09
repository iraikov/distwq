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
    n = 10
    for i in range(0, n):
        controller.submit_call("do_work", (i+1,), module="example_distwq")
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
