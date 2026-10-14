"""Numbers in docs/design/texture/pop-tails.md, from NumPy and SciPy alone.

Shares no code with sonore. Prints each claim's number with its tag.

  T1  levels drawn from an exponential distribution in dB, with scale
      20 / (alpha ln 10), give amplitudes with a Pareto tail of exponent alpha
  T2  the Hill estimator on the top 5% of 1000 Pareto samples (about the
      number of envelope peaks in 7 s of a texture): its spread over trials
  T3  sample kurtosis of Pareto samples as the sample grows: it keeps growing
      for alpha below 4 and settles above
  T4  the whole measurement (high-pass, envelope, peaks at least 5 ms apart,
      Hill on the top 5%) on 7 s of shot noise with Pareto amplitudes over a
      faint noise floor: how close it gets to the alpha that made it
  T5  the same measurement on Gaussian noise
"""

import numpy as np
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

FS = 44100
rng = np.random.default_rng(0)


def hill(values, fraction=0.05, min_count=10):
    """Hill estimate of the tail exponent from the largest ``fraction``."""
    ordered = np.sort(values)[::-1]
    count = max(min_count, int(fraction * len(ordered)))
    return count / np.sum(np.log(ordered[:count] / ordered[count]))


def envelope_peaks(x, cutoff):
    """Peaks, at least 5 ms apart, of the 500 Hz smoothed Hilbert envelope
    of ``x`` high-passed at ``cutoff`` Hz."""
    high = sosfiltfilt(butter(4, cutoff, "highpass", fs=FS, output="sos"), x)
    envelope = sosfiltfilt(butter(2, 500, "lowpass", fs=FS, output="sos"), np.abs(hilbert(high)))
    index, _ = find_peaks(envelope, distance=int(0.005 * FS))
    return envelope[index]


def pareto_levels_db(alpha, size):
    return rng.exponential(20 / (alpha * np.log(10)), size)


print("T1  exponential dB levels give Pareto amplitudes [check]")
for alpha in (1.0, 1.5, 2.5):
    amplitudes = 10 ** (pareto_levels_db(alpha, 200_000) / 20)
    # survival P(A > a) = a**-alpha for a >= 1: fit the slope between a = 2 and a = 20
    survival = [np.mean(amplitudes > a) for a in (2.0, 20.0)]
    slope = -np.log(survival[1] / survival[0]) / np.log(10)
    print(f"    alpha {alpha}: survival slope {slope:.3f}")

print("T2  Hill on the top 5% of 1000 samples, 400 trials [check]")
for alpha in (1.0, 1.5, 2.5, 4.0):
    estimates = np.array([hill(10 ** (pareto_levels_db(alpha, 1000) / 20)) for _ in range(400)])
    spread = np.std(estimates) / alpha
    print(f"    alpha {alpha}: median {np.median(estimates):.2f}, relative spread {100 * spread:.0f}%")

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
        estimates.append(hill(envelope_peaks(x, 2000)))
    low, high = min(estimates), max(estimates)
    print(f"    alpha {alpha}: median estimate {np.median(estimates):.2f} (range {low:.2f}-{high:.2f})")

print("T5  whole measurement on 7 s of Gaussian noise, 8 seeds [check]")
estimates = [hill(envelope_peaks(rng.standard_normal(n), 2000)) for _ in range(8)]
print(f"    median {np.median(estimates):.1f} (range {min(estimates):.1f}-{max(estimates):.1f})")
