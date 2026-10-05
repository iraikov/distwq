"""
Analyse a set of noisy recordings with the functional interface of distwq.

Run on two or more ranks, with the distwq repository on PYTHONPATH:

    mpirun -n 4 python examples/example_functional_pipeline.py

Each recording is a sum of sine waves in Gaussian noise.  The analysis is
written as functional steps: functions sent to the workers are defined at
module level, and the steps that decide what to keep (filters,
conditions, reductions) run on the controller and may be lambdas.

The example uses, in one session:

- starmap_distributed, to synthesise recordings from (seed, frequency,
  noise) triples;
- a FunctionalPipeline with a map step (spectrum), a filter step that
  keeps recordings with a clear peak, and a flatmap step that lists each
  recording's spectral peaks;
- map_reduce, to sum the power spectra into a mean spectrum;
- parallel_map, to apply several summary measures to one recording;
- conditional_map, to choose an analysis on the controller.

At the end, quick_map runs one more map in a fresh environment and
returns its result on every rank.
"""

import logging
import operator

import numpy as np

from distwq.functional import (
    conditional_map,
    map_reduce,
    parallel_map,
    pipeline,
    quick_map,
    starmap_distributed,
)
from distwq.future import DistributedEnvironment

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("distwq").setLevel(logging.WARNING)
logger = logging.getLogger("example_functional_pipeline")

SAMPLING_RATE = 1000.0  # Hz
DURATION = 4.0  # s
# Peak-to-median power ratio above which a recording counts as clear.  At
# noise_std 8 the ratio is about 10 to 25; at noise_std 2 or less it is
# above 200.
PEAK_RATIO = 100.0


def synthesise(seed, frequency, noise_std):
    """Return a recording: a sine wave at frequency (Hz) and its second
    harmonic at half the amplitude, in Gaussian noise of noise_std."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(SAMPLING_RATE * DURATION)) / SAMPLING_RATE
    signal = np.sin(2 * np.pi * frequency * t) + 0.5 * np.sin(4 * np.pi * frequency * t)
    return {
        "seed": seed,
        "frequency": frequency,
        "samples": signal + rng.normal(scale=noise_std, size=t.shape),
    }


def power_spectrum(recording):
    """Return the one-sided power spectrum and its peak-to-median ratio."""
    samples = recording["samples"]
    power = np.abs(np.fft.rfft(samples * np.hanning(samples.size))) ** 2
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / SAMPLING_RATE)
    return {
        "seed": recording["seed"],
        "frequency": recording["frequency"],
        "freqs": freqs,
        "power": power,
        "peak_ratio": float(power.max() / np.median(power)),
    }


def spectral_peaks(spectrum, n_peaks=2):
    """Return the n_peaks strongest local maxima as (seed, Hz, power)."""
    power = spectrum["power"]
    is_peak = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
    candidates = np.flatnonzero(is_peak) + 1
    strongest = candidates[np.argsort(power[candidates])[::-1][:n_peaks]]
    return [
        (spectrum["seed"], float(spectrum["freqs"][i]), float(power[i]))
        for i in sorted(strongest)
    ]


def spectrum_power(recording):
    """Return the power spectrum of a recording as an array."""
    return power_spectrum(recording)["power"]


def rms(recording):
    return float(np.sqrt(np.mean(recording["samples"] ** 2)))


def zero_crossing_rate(recording):
    """Return the number of sign changes per second."""
    signs = np.signbit(recording["samples"])
    return float(np.count_nonzero(signs[1:] != signs[:-1]) / DURATION)


def dominant_frequency(recording):
    spectrum = power_spectrum(recording)
    return float(spectrum["freqs"][np.argmax(spectrum["power"])])


def quick_mean(recording):
    return float(np.mean(recording["samples"]))


def square(x):
    return x * x


def analyse(ctl):
    """Run the analysis in the session of ctl; return a summary."""
    rng = np.random.default_rng(2026)
    specs = [
        (seed, float(rng.uniform(5.0, 60.0)), float(noise))
        for seed, noise in enumerate([0.5, 1.0, 2.0, 8.0] * 3)
    ]
    recordings = starmap_distributed(ctl, synthesise, specs)

    # A recording is kept only if its spectral peak stands well clear of
    # the noise: at least PEAK_RATIO times the median power.  The
    # threshold is a controller-side decision.
    peaks = (
        pipeline(power_spectrum)
        .filter(lambda spectrum: spectrum["peak_ratio"] > PEAK_RATIO)
        .flatmap(spectral_peaks)
        .run(ctl, recordings)
    )

    # Summing spectra is associative and commutative, so the arrival-order
    # reduction gives the same mean spectrum as the ordered one.
    total_power = map_reduce(
        ctl, spectrum_power, operator.add, recordings, ordered=False
    )
    mean_power = total_power / len(recordings)

    measures = parallel_map(
        ctl, [rms, zero_crossing_rate, dominant_frequency], recordings[0]
    )

    # Use the full spectral estimate only if there are few recordings.
    frequencies = conditional_map(
        ctl, recordings, lambda: len(recordings) <= 20, dominant_frequency, quick_mean
    )
    clear = {seed for seed, _, _ in peaks}
    return {
        "n_recordings": len(recordings),
        "n_clear": len(clear),
        "peaks": peaks,
        "mean_power_peak_hz": float(
            np.fft.rfftfreq(4000, 1.0 / SAMPLING_RATE)[np.argmax(mean_power)]
        ),
        "first_recording": dict(
            zip(["rms", "zero_crossings_per_s", "dominant_hz"], measures)
        ),
        "true_vs_found": [
            (spec[1], found)
            for spec, found in zip(specs, frequencies)
            if spec[0] in clear
        ],
    }


def main():
    with DistributedEnvironment() as env:
        summary = None
        for ctl in env.controller_session():
            summary = analyse(ctl)
        summary = env.broadcast(summary)

    squares = quick_map(square, range(10))

    if env.role == "controller":
        logger.info(
            f"{summary['n_clear']} of {summary['n_recordings']} recordings have a clear peak"
        )
        for seed, hz, power in summary["peaks"]:
            logger.info(
                f"  recording {seed:2d}: peak at {hz:6.2f} Hz, power {power:.3g}"
            )
        logger.info(f"Mean spectrum peaks at {summary['mean_power_peak_hz']:.2f} Hz")
        logger.info(f"First recording: {summary['first_recording']}")
        worst = max(abs(true - found) for true, found in summary["true_vs_found"])
        logger.info(
            "Largest error of the dominant-frequency estimate on clear "
            f"recordings: {worst:.2f} Hz (resolution {1.0 / DURATION:.2f} Hz)"
        )
        logger.info(f"quick_map result on every rank: {squares}")


if __name__ == "__main__":
    main()
