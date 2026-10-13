"""Numbers in docs/design/sources/event-texture.md, from NumPy and SciPy alone.

Shares no code with sonore. Prints each claim's number with its tag.

The event is one short chirp, h: a rising exponential sweep from 2 to 6 kHz
lasting 30 ms under a Hann window, at 44.1 kHz. The process places copies of
h at Poisson onset times with rate `rate` per second, each copy scaled by
1 / sqrt(rate) so that the expected power does not depend on the rate.

  E1  expected power spectrum: the same at every rate (Campbell's theorem
      gives rate * |H(f)|^2 before the scaling)
  E2  excess kurtosis of the waveform: integral(h^4) / (rate * integral(h^2)^2),
      falling as 1 / rate toward Gaussian noise's 0
  E3  how often events overlap: an event of length T overlaps another with
      probability 1 - exp(-2 * rate * T)
  E4  one band's Hilbert envelope (2.5-5 kHz): its variance over its squared
      mean, falling toward the Rayleigh value 4 / pi - 1 of Gaussian noise
  E5  a regular train at the same rate instead: its spectrum is lines spaced
      by the rate, unlike the Poisson process's continuous spectrum
"""

import numpy as np
from scipy.signal import butter, chirp, fftconvolve, hilbert, sosfiltfilt, welch

FS = 44100
EVENT_DUR = 0.030
DURATION = 60.0
RATES = [1, 3, 10, 30, 100, 300, 1000]
SEEDS = range(4)

t = np.arange(int(EVENT_DUR * FS)) / FS
event = chirp(t, 2000, EVENT_DUR, 6000, method="logarithmic") * np.hanning(len(t))


def poisson_train(rate, rng, n_samples):
    """Unit impulses at Poisson onset times, rounded to the nearest sample."""
    n_events = rng.poisson(rate * n_samples / FS)
    impulses = np.zeros(n_samples)
    np.add.at(impulses, rng.integers(0, n_samples, n_events), 1.0)
    return impulses


def chorus(rate, rng):
    n_samples = int(DURATION * FS)
    return fftconvolve(poisson_train(rate, rng, n_samples), event)[:n_samples] / np.sqrt(rate)


def excess_kurtosis(x):
    centered = x - x.mean()
    return np.mean(centered**4) / np.mean(centered**2) ** 2 - 3


band = butter(4, [2500, 5000], btype="bandpass", fs=FS, output="sos")
event_energy = np.sum(event**2) / FS  # integral of h^2
event_fourth = np.sum(event**4) / FS  # integral of h^4

# E1: Welch spectra of each rate, compared with the expected |H(f)|^2 (per second of sound).
H = np.fft.rfft(event, 8192)
f_h = np.fft.rfftfreq(8192, 1 / FS)
# The continuous transform is H / fs, and Welch's density is one-sided, hence 2 |H|^2 / fs^2.
expected = 2 * np.abs(H) ** 2 / FS**2
print("E1 Welch spectrum over 2.5-5 kHz minus the expected 2|H(f)|^2 / fs^2, mean and max |dB|:")
results = {}
for rate in RATES:
    sounds = [chorus(rate, np.random.default_rng(seed)) for seed in SEEDS]
    results[rate] = sounds
    f, psd = welch(np.concatenate(sounds), FS, nperseg=8192)
    model = np.interp(f, f_h, expected)
    keep = (f >= 2500) & (f <= 5000)
    diff_db = 10 * np.log10(psd[keep] / model[keep])
    print(f"   rate {rate:5d}/s: mean {diff_db.mean():+.2f} dB, max |{np.abs(diff_db).max():.2f}| dB")

print("E2 waveform excess kurtosis, measured (mean over seeds) and predicted:")
for rate in RATES:
    measured = np.mean([excess_kurtosis(x) for x in results[rate]])
    predicted = event_fourth / (rate * event_energy**2)
    print(f"   rate {rate:5d}/s: measured {measured:8.2f}, predicted {predicted:8.2f}")

print(f"E3 chance an event of {EVENT_DUR * 1e3:.0f} ms overlaps another; mean events sounding at once:")
for rate in RATES:
    print(f"   rate {rate:5d}/s: {1 - np.exp(-2 * rate * EVENT_DUR):.3f}, {rate * EVENT_DUR:.2f}")

print(f"E4 envelope variance / mean^2 in 2.5-5 kHz (Rayleigh limit {4 / np.pi - 1:.3f}):")
for rate in RATES:
    ratios = []
    for x in results[rate]:
        envelope = np.abs(hilbert(sosfiltfilt(band, x)))
        ratios.append(envelope.var() / envelope.mean() ** 2)
    print(f"   rate {rate:5d}/s: {np.mean(ratios):.3f}")
gaussian = np.random.default_rng(99).standard_normal(int(DURATION * FS))
envelope = np.abs(hilbert(sosfiltfilt(band, gaussian)))
print(f"   Gaussian noise: {envelope.var() / envelope.mean() ** 2:.3f}")

# E5: a regular train at 100/s. Its spectrum's peak-to-median ratio over 2.5-5 kHz, in dB,
# against the Poisson chorus at the same rate.
rate = 100
n_samples = int(DURATION * FS)
regular = np.zeros(n_samples)
regular[:: FS // rate] = 1.0
regular = fftconvolve(regular, event)[:n_samples]
for name, x in [("regular", regular), ("Poisson", results[rate][0])]:
    f, psd = welch(x, FS, nperseg=2**16)
    keep = (f >= 2500) & (f <= 5000)
    ratio = 10 * np.log10(psd[keep].max() / np.median(psd[keep]))
    print(f"E5 {name} train at {rate}/s: spectrum peak over median in 2.5-5 kHz {ratio:.1f} dB")
