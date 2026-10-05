"""
Estimate pi by Monte Carlo sampling with the task interface of distwq.

Run on two or more ranks, with the distwq repository on PYTHONPATH:

    mpirun -n 4 python examples/example_task_interface.py

Each task draws a batch of random points in the unit square and counts
those inside the quarter circle.  The controller keeps a fixed number of
batches running and, as each one finishes, decides whether the estimate
is precise enough or another batch is needed.  The decision is made on
the controller only, while the workers simply serve whatever they are
sent; this is the kind of result-dependent control flow that the session
design allows.

The example also shows:

- a task that fails on a worker, and the exception raised from its
  future on the controller;
- two sessions in one environment;
- broadcast(), which returns the controller's final estimate on every
  rank.
"""

import logging
import math

import numpy as np

from distwq.future import FIRST_COMPLETED, DistributedEnvironment, wait

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("distwq").setLevel(logging.WARNING)
logger = logging.getLogger("example_task_interface")

BATCH_SIZE = 200_000
TARGET_STDERR = 2e-4
MAX_BATCHES = 400


def count_inside(seed, n_samples):
    """Return how many of n_samples random points fall inside the quarter
    circle of radius 1.  Each batch has its own seed, so the run can be
    repeated exactly."""
    rng = np.random.default_rng(seed)
    points = rng.random((n_samples, 2))
    return int(np.count_nonzero((points**2).sum(axis=1) <= 1.0))


def pi_estimate(inside, total):
    """Return the estimate of pi and its standard error."""
    p = inside / total
    return 4.0 * p, 4.0 * math.sqrt(p * (1.0 - p) / total)


def sample_until_precise(ctl, in_flight):
    """Keep in_flight batches running until the standard error of the
    estimate falls below TARGET_STDERR, or MAX_BATCHES have been used."""
    next_seed = 0
    pending = set()
    while next_seed < in_flight:
        pending.add(ctl.submit(count_inside, next_seed, BATCH_SIZE))
        next_seed += 1

    inside = total = 0
    estimate, stderr = float("nan"), float("inf")
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            inside += future.result()
            total += BATCH_SIZE
        estimate, stderr = pi_estimate(inside, total)
        if stderr > TARGET_STDERR and next_seed < MAX_BATCHES:
            # Not precise enough yet: replace each finished batch.
            for _ in done:
                pending.add(ctl.submit(count_inside, next_seed, BATCH_SIZE))
                next_seed += 1
        elif pending:
            # Precise enough: the batches still running are collected when
            # the session ends, but not used.
            break
    return estimate, stderr, total


def main():
    with DistributedEnvironment() as env:
        result = None

        # Session 1: a task that fails.  rng.random() rejects a negative
        # size, and the ValueError raised on the worker comes back here.
        for ctl in env.controller_session():
            try:
                ctl.submit(count_inside, 0, -1).result()
            except ValueError as exc:
                logger.info(f"A failing task raised on the controller: {exc!r}")
                logger.info(
                    f"  worker traceback ends with: {exc.__cause__.tb.splitlines()[-1]}"
                )

        # Session 2: adaptive sampling.
        for ctl in env.controller_session():
            in_flight = 2 * max(env.worker_count, 1)
            estimate, stderr, total = sample_until_precise(ctl, in_flight)
            result = (estimate, stderr, total)

        estimate, stderr, total = env.broadcast(result)
        if env.role == "controller":
            logger.info(
                f"pi = {estimate:.5f} +/- {stderr:.5f} from {total:,} samples "
                f"(error {abs(estimate - math.pi):.5f})"
            )


if __name__ == "__main__":
    main()
