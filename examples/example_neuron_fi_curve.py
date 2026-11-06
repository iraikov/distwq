"""
Measure the f-I curve and rheobase of a Hodgkin-Huxley soma in NEURON,
using the workflow functions of distwq.

Run on two or more ranks, with distwq and NEURON's Python package on
PYTHONPATH:

    mpirun -n 4 python examples/example_neuron_fi_curve.py

Each task builds a single-compartment soma with Hodgkin-Huxley channels,
injects a current step, and returns the firing rate.  Every task builds
its own model, so tasks are independent and can run on any worker.

Three stages run in one session, each depending on the previous one's
results:

1. parameter_sweep: a coarse f-I curve over current amplitude, at two
   temperatures.
2. adaptive_parameter_exploration: bisection for the rheobase, the
   smallest current that makes the cell fire repetitively, at each
   temperature.  The interval to bisect comes from the coarse sweep, and
   each round's interval comes from the previous round's results.  The
   refinement runs on the controller.
3. adaptive_map: two step durations are tried on a sample of amplitudes,
   and the one whose estimates agree best with the longest step is used
   for a fine f-I curve.
"""

import logging

import numpy as np
from neuron import h

from distwq.functional import adaptive_map, map_distributed
from distwq.future import DistributedEnvironment
from distwq.workflow import adaptive_parameter_exploration, parameter_sweep

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("distwq").setLevel(logging.WARNING)
logger = logging.getLogger("example_neuron_fi_curve")

h.load_file("stdrun.hoc")

STIM_DELAY = 50.0  # ms
# A cell "fires repetitively" if it spikes at least this often during the
# step; a single onset spike does not count.
MIN_RATE = 5.0  # Hz


def firing_rate(params):
    """Simulate a current step on a Hodgkin-Huxley soma; return the rate.

    params holds "amp" (nA), "celsius" (degrees C) and optionally
    "duration" (ms, default 500).  Returns params with "rate" added, in
    spikes per second during the step.
    """
    duration = params.get("duration", 500.0)
    h.celsius = params["celsius"]
    soma = h.Section(name="soma")
    soma.L = soma.diam = 20.0  # um
    soma.insert("hh")
    stim = h.IClamp(soma(0.5))
    stim.delay = STIM_DELAY
    stim.dur = duration
    stim.amp = params["amp"]
    detector = h.NetCon(soma(0.5)._ref_v, None, sec=soma)
    detector.threshold = 0.0  # mV
    spike_times = h.Vector()
    detector.record(spike_times)

    h.finitialize(-65.0)
    h.continuerun(STIM_DELAY + duration)

    times = np.array(spike_times.to_python())
    in_step = np.count_nonzero(times >= STIM_DELAY)
    return {**params, "rate": 1000.0 * in_step / duration}


def bisect_rheobase(celsius, low, high, tolerance):
    """Return a refinement function for adaptive_parameter_exploration that
    narrows [low, high] around the rheobase until it is narrower than
    tolerance (nA)."""
    bounds = {"low": low, "high": high}

    def refine(results):
        result = results[0]
        if result["rate"] >= MIN_RATE:
            bounds["high"] = result["amp"]
        else:
            bounds["low"] = result["amp"]
        if bounds["high"] - bounds["low"] < tolerance:
            return []
        return [{"amp": 0.5 * (bounds["low"] + bounds["high"]), "celsius": celsius}]

    return refine, bounds


def rate_with_short_step(params):
    return firing_rate({**params, "duration": 200.0})


def rate_with_medium_step(params):
    return firing_rate({**params, "duration": 1000.0})


def rate_with_reference_step(params):
    return firing_rate({**params, "duration": 2000.0})


def main():
    temperatures = [6.3, 16.3]
    coarse_amps = [round(float(a), 3) for a in np.arange(0.0, 0.41, 0.05)]

    with DistributedEnvironment() as env:
        report = None
        for ctl in env.controller_session():
            # 1. Coarse f-I curve.
            sweep = parameter_sweep(
                ctl, firing_rate, {"celsius": temperatures, "amp": coarse_amps}
            )
            rates = {(p["celsius"], p["amp"]): r["rate"] for p, r in sweep}

            # 2. Rheobase by bisection, starting from the bracket the sweep
            # found at each temperature.
            rheobase = {}
            for celsius in temperatures:
                firing = [a for a in coarse_amps if rates[(celsius, a)] >= MIN_RATE]
                if not firing:
                    rheobase[celsius] = None
                    continue
                high = firing[0]
                low = max(a for a in coarse_amps if a < high) if high > 0 else 0.0
                refine, bounds = bisect_rheobase(celsius, low, high, tolerance=1e-3)
                start = [{"amp": 0.5 * (low + high), "celsius": celsius}]
                adaptive_parameter_exploration(
                    ctl, firing_rate, start, refine, max_iterations=20
                )
                rheobase[celsius] = (bounds["low"], bounds["high"])

            # 3. Fine f-I curve at 6.3 C with the step duration that agrees
            # best with a long reference step on a sample of amplitudes.
            fine_amps = [
                {"amp": float(a), "celsius": 6.3} for a in np.linspace(0.05, 0.4, 15)
            ]
            # adaptive_map samples the first items, so put a spread of
            # amplitudes first.
            sample = fine_amps[::5]
            ordered = sample + [p for p in fine_amps if p not in sample]
            reference = [
                r["rate"]
                for r in map_distributed(ctl, rate_with_reference_step, sample)
            ]

            def disagreement(results, _elapsed, reference=reference):
                return sum(abs(r["rate"] - ref) for r, ref in zip(results, reference))

            chosen, fine = adaptive_map(
                ctl,
                [rate_with_short_step, rate_with_medium_step],
                ordered,
                sample_size=len(sample),
                cost=disagreement,
            )
            report = {
                "sweep": sorted((c, a, r) for (c, a), r in rates.items()),
                "rheobase": rheobase,
                "chosen": chosen.__name__,
                "fine": sorted((r["amp"], r["rate"]) for r in fine),
            }
        report = env.broadcast(report)

    if env.role == "controller":
        logger.info("Coarse f-I curve (rate in Hz):")
        for celsius in temperatures:
            row = "  ".join(f"{r:5.1f}" for c, a, r in report["sweep"] if c == celsius)
            logger.info(f"  {celsius:4.1f} C: {row}")
        logger.info(f"  amplitudes (nA): {coarse_amps}")
        for celsius, bracket in report["rheobase"].items():
            if bracket is None:
                logger.info(f"Rheobase at {celsius} C: above {coarse_amps[-1]} nA")
            else:
                logger.info(
                    f"Rheobase at {celsius} C: between {bracket[0]:.4f} and "
                    f"{bracket[1]:.4f} nA"
                )
        logger.info(f"Step duration chosen for the fine curve: {report['chosen']}")
        for amp, rate in report["fine"]:
            logger.info(f"  {amp:.3f} nA: {rate:6.1f} Hz")


if __name__ == "__main__":
    main()
