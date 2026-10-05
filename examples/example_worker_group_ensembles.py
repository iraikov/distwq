"""
Run stochastic simulation ensembles on worker groups with distwq.

Run with a world size that splits into groups of GROUP_SIZE processes
plus the controller, for example with GROUP_SIZE = 2:

    mpirun -n 5 python examples/example_worker_group_ensembles.py

Under split grouping every task sent to a group runs on each process of
the group, and the task's result is a CollectiveTaskResult, whose values
are the processes' results.
Here each process uses that to run one independent replicate of a
stochastic model: the leaky integrate-and-fire neuron below is driven by
an Ornstein-Uhlenbeck noise current, seeded from the task's seed and the
process's world rank.  A task with one parameter set therefore returns a
small ensemble, whose mean and spread the controller computes.

Across the parameter sets, the controller compares the mean firing rate
with the rate of the same neuron without noise, to show how much the
noise raises the rate near threshold.  broadcast() then gives the summary
to every process, including the members of the worker groups.
"""

import logging

import numpy as np
from mpi4py import MPI

from distwq.functional import map_distributed
from distwq.future import create_split_environment

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("distwq").setLevel(logging.WARNING)
logger = logging.getLogger("example_worker_group_ensembles")

GROUP_SIZE = 2

# Leaky integrate-and-fire neuron, in ms, mV, MOhm and nA.
TAU_M = 20.0
V_REST = -70.0
V_THRESHOLD = -50.0
V_RESET = -65.0
R_M = 100.0
REFRACTORY = 2.0
DT = 0.1
DURATION = 2000.0
# Ornstein-Uhlenbeck noise current: correlation time (ms).
TAU_NOISE = 5.0


def lif_rate(drive, noise_std, seed):
    """Simulate the neuron for DURATION ms with a constant drive (nA) plus
    an Ornstein-Uhlenbeck current of standard deviation noise_std (nA).

    Returns the firing rate in Hz.  The random stream depends on seed and
    on the world rank of the process, so each member of a worker group
    runs a different replicate of the same task.
    """
    rng = np.random.default_rng([seed, MPI.COMM_WORLD.Get_rank()])
    n_steps = int(DURATION / DT)
    decay = np.exp(-DT / TAU_NOISE)
    kicks = rng.normal(scale=noise_std * np.sqrt(1.0 - decay**2), size=n_steps)
    v = V_REST
    noise = 0.0
    refractory_left = 0.0
    n_spikes = 0
    for kick in kicks:
        noise = noise * decay + kick
        if refractory_left > 0.0:
            refractory_left -= DT
            continue
        v += DT / TAU_M * (V_REST - v + R_M * (drive + noise))
        if v >= V_THRESHOLD:
            v = V_RESET
            refractory_left = REFRACTORY
            n_spikes += 1
    return 1000.0 * n_spikes / DURATION


def ensemble(params):
    """Return this process's replicate for one parameter set."""
    return lif_rate(params["drive"], params["noise_std"], params["seed"])


def deterministic_rate(drive):
    """Return the rate of the noiseless neuron, from its closed form."""
    v_inf = V_REST + R_M * drive
    if v_inf <= V_THRESHOLD:
        return 0.0
    time_to_spike = TAU_M * np.log((v_inf - V_RESET) / (v_inf - V_THRESHOLD))
    return 1000.0 / (REFRACTORY + time_to_spike)


def main():
    drives = [0.15, 0.19, 0.20, 0.21, 0.25, 0.30]
    tasks = [
        {"drive": drive, "noise_std": 0.05, "seed": seed}
        for seed, drive in enumerate(drives)
    ]

    with create_split_environment(
        nprocs_per_worker=GROUP_SIZE, broker_is_worker=True
    ) as env:
        summary = None
        for ctl in env.controller_session():
            replicates = map_distributed(ctl, ensemble, tasks)
            summary = [
                {
                    "drive": task["drive"],
                    "mean": float(np.mean(rates)),
                    "std": float(np.std(rates)),
                    "n": len(rates),
                    "noiseless": deterministic_rate(task["drive"]),
                }
                for task, rates in zip(tasks, (r.values for r in replicates))
            ]
            logger.info(
                f"{env.worker_count} worker groups of {GROUP_SIZE} processes "
                f"ran {len(tasks)} ensembles"
            )
        summary = env.broadcast(summary)

    if env.role == "controller":
        logger.info("drive (nA)  noisy rate (Hz)       noiseless rate (Hz)")
        for row in summary:
            logger.info(
                f"  {row['drive']:.2f}      {row['mean']:6.2f} +/- {row['std']:5.2f}"
                f" (n={row['n']})   {row['noiseless']:6.2f}"
            )
    # Every process, including group members, now holds the same summary.
    assert len(summary) == len(drives)


if __name__ == "__main__":
    main()
