"""Numbers in docs/design/views/envelope-peaks.md, from NumPy and SciPy alone.

Shares no code with sonore. Prints each claim's number with its tag.

  T1  levels drawn from an exponential distribution in dB, with scale
      20 / (alpha ln 10), give amplitudes with a Pareto tail of exponent alpha
  T2  the Hill estimator on the largest k of n Pareto samples: its spread over
      trials against the stated uncertainty alpha / sqrt(k), for the k that
      1, 2, 5 and 7 s of 5 ms bins give at the top 5%
  T3  sample kurtosis of Pareto samples as the sample grows: it keeps growing
      for alpha below 4 and settles above
  T4  the whole measurement (high-pass, envelope, maximum in each 5 ms bin,
      Hill on the top 5%) on 7 s of shot noise with Pareto amplitudes over a
      faint noise floor: how close it gets to the alpha that made it
  T5  the same measurement on Gaussian noise
"""

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

FS = 44100
BIN = 0.005  # seconds
rng = np.random.default_rng(0)


def hill(values, fraction=0.05, min_count=10):
    """Hill estimate of the tail exponent from the largest ``fraction``."""
    ordered = np.sort(values)[::-1]
    count = max(min_count, int(fraction * len(ordered)))
    return count / np.sum(np.log(ordered[:count] / ordered[count]))


def bin_maxima(x, cutoff):
    """Maximum in each 5 ms bin of the 500 Hz smoothed Hilbert envelope of
    ``x`` high-passed at ``cutoff`` Hz."""
    high = sosfiltfilt(butter(4, cutoff, "highpass", fs=FS, output="sos"), x)
    envelope = sosfiltfilt(butter(2, 500, "lowpass", fs=FS, output="sos"), np.abs(hilbert(high)))
    bin_length = int(BIN * FS)
    n_bins = len(envelope) // bin_length
    return envelope[: n_bins * bin_length].reshape(n_bins, bin_length).max(axis=1)


def pareto_levels_db(alpha, size):
    return rng.exponential(20 / (alpha * np.log(10)), size)


print("T1  exponential dB levels give Pareto amplitudes [check]")
for alpha in (1.0, 1.5, 2.5):
    amplitudes = 10 ** (pareto_levels_db(alpha, 200_000) / 20)
    # survival P(A > a) = a**-alpha for a >= 1: fit the slope between a = 2 and a = 20
    survival = [np.mean(amplitudes > a) for a in (2.0, 20.0)]
    slope = -np.log(survival[1] / survival[0]) / np.log(10)
    print(f"    alpha {alpha}: survival slope {slope:.3f}")

print("T2  Hill on the top 5% of n Pareto samples, 400 trials: spread vs alpha / sqrt(k) [check]")
for seconds in (1, 2, 5, 7):
    n = int(seconds / BIN)
    k = int(0.05 * n)
    row = []
    for alpha in (1.5, 4.0):
        estimates = np.array([hill(10 ** (pareto_levels_db(alpha, n) / 20), min_count=1) for _ in range(400)])
        spread = 100 * np.std(estimates) / alpha
        row.append(f"alpha {alpha}: median {np.median(estimates):.2f}, spread {spread:3.0f}%")
    print(f"    {seconds} s (n={n}, k={k}, 1/sqrt(k) = {100 / np.sqrt(k):3.0f}%): " + "; ".join(row))

print("T3  median sample kurtosis of Pareto samples by sample size, 200 trials [check]")
for alpha in (1.5, 3.0, 6.0):
    row = []
    for size in (1_000, 10_000, 100_000):
        kurtosis = []
        for _ in range(200):
            a = 10 ** (pareto_levels_db(alpha, size) / 20)
            kurtosis.append(np.mean((a - a.mean()) ** 4) / np.var(a) ** 2)
        row.append(f"n={size}: {np.median(kurtosis):9.1f}")
    print(f"    alpha {alpha}: " + "  ".join(row))

print("T4  whole measurement on 7 s of shot noise, 8 seeds [check]")
n = 7 * FS
burst_length = int(0.004 * FS)
burst_shape = np.exp(-np.arange(burst_length) / (0.0008 * FS))
for alpha in (1.0, 1.5, 2.5):
    estimates = []
    for _ in range(8):
        onsets = np.sort(rng.choice(n - burst_length, int(150 * 7), replace=False))  # 150 events per second
        amplitudes = 10 ** (np.minimum(pareto_levels_db(alpha, len(onsets)), 40) / 20)  # capped 40 dB up
        x = 1e-3 * rng.standard_normal(n)
        for onset, amplitude in zip(onsets, amplitudes, strict=True):
            x[onset : onset + burst_length] += amplitude * burst_shape * rng.standard_normal(burst_length)
        estimates.append(hill(bin_maxima(x, 2000)))
    low, high = min(estimates), max(estimates)
    print(f"    alpha {alpha}: median estimate {np.median(estimates):.2f} (range {low:.2f}-{high:.2f})")

print("T5  whole measurement on 7 s of Gaussian noise, 8 seeds [check]")
estimates = [hill(bin_maxima(rng.standard_normal(n), 2000)) for _ in range(8)]
print(f"    median {np.median(estimates):.1f} (range {min(estimates):.1f}-{max(estimates):.1f})")
