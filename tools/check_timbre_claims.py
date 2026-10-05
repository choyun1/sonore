"""Numerical checks for the timbre claims in docs/design/music/timbre-page.md (T-C1 to T-C5).

Like the other design checkers, this shares no code with sonore: NumPy and SciPy, every
formula written out from Peeters et al. (2011), not from the Timbre Toolbox source (whose
licence forbids redistribution). Each line prints the claim number and the number that
supports it.

    python tools/check_timbre_claims.py
"""

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, lfilter

FS = 44100
F0 = 311.13  # E-flat 4, the pitch of Grey (1977) and McAdams et al. (1995) [Hz]


def report(claim, text, value):
    print(f"{claim:6s} {text:<80s} {value:10.4f}")


def tone(amplitudes, attack, duration=1.0, f0=F0, release=0.05):
    """Harmonic complex with a linear attack of ``attack`` seconds, a flat sustain and a
    linear release, all partials starting together."""
    t = np.arange(int(duration * FS)) / FS
    harmonics = np.arange(1, len(amplitudes) + 1)
    signal = (np.asarray(amplitudes)[:, None] * np.sin(2 * np.pi * f0 * harmonics[:, None] * t)).sum(0)
    envelope = np.minimum(1, t / attack) * np.clip((duration - t) / release, 0, 1)
    return signal * envelope


def energy_envelope(signal, cutoff, two_pass):
    """Peeters et al. (2011) II B 1: amplitude of the analytic signal, low-passed by a third-order
    Butterworth filter, once (the paper's descriptor setting, 5 Hz) or forward and backward (its
    onset-detection setting, 20 Hz)."""
    b, a = butter(3, cutoff / (FS / 2))
    amplitude = np.abs(hilbert(signal))
    return filtfilt(b, a, amplitude) if two_pass else lfilter(b, a, amplitude)


def attack_weakest_effort(envelope, alpha=3.0):
    """Start and end of the attack [s] by the weakest-effort method as the paper's text gives it
    (III A 2 a): thresholds 0.1, ..., 1 of the maximum, efforts between successive crossing times,
    start at the first threshold whose effort is below alpha times the mean effort, end at the
    last one; then the minimum and maximum of the envelope within those two efforts."""
    peak_index = int(np.argmax(envelope))
    normalized = envelope[: peak_index + 1] / envelope[peak_index]
    thresholds = np.arange(1, 11) / 10
    crossings = np.array([np.argmax(normalized >= level) for level in thresholds])
    efforts = np.diff(crossings)
    weak = np.flatnonzero(efforts < alpha * efforts.mean())
    first, last = weak[0], weak[-1]
    start_span = slice(crossings[first], crossings[first + 1] + 1)
    end_span = slice(crossings[last], crossings[last + 1] + 1)
    start = crossings[first] + np.argmin(envelope[start_span])
    end = crossings[last] + np.argmax(envelope[end_span])
    return start / FS, end / FS


def attack_fixed(envelope, low=0.1, high=0.9):
    """Attack from the first crossing of ``low`` to the first of ``high`` times the maximum."""
    normalized = envelope / envelope.max()
    return np.argmax(normalized >= low) / FS, np.argmax(normalized >= high) / FS


SAWTOOTH_LIKE = 1 / np.arange(1, 21)

# T-C1: the envelope filter decides what short attacks measure as. A 5 Hz single-pass filter
# turns every attack shorter than about 100 ms into roughly the filter's own rise time.
for attack in (0.005, 0.02, 0.08, 0.3):
    signal = tone(SAWTOOTH_LIKE, attack)
    report("T-C1", f"true attack {1000 * attack:5.0f} ms: log10 of the true attack", np.log10(attack))
    for name, cutoff, two_pass in (("5 Hz one pass", 5, False), ("20 Hz two pass", 20, True)):
        envelope = energy_envelope(signal, cutoff, two_pass)
        start, end = attack_weakest_effort(envelope)
        report("T-C1", f"  {name}, weakest effort: LAT", np.log10(end - start))
    amplitude = np.abs(hilbert(signal))
    b, a = butter(3, 50 / (FS / 2))
    start, end = attack_fixed(filtfilt(b, a, amplitude))
    report("T-C1", "  50 Hz two pass, 10% to 90% of maximum: LAT", np.log10(end - start))

# T-C2: spectral centroid. For a steady harmonic complex the centroid of the magnitude spectrum
# is sum(n f0 a_n) / sum(a_n); an STFT (Hamming 23.2 ms, hop 5.8 ms, as in the paper) recovers it.
signal = tone(SAWTOOTH_LIKE, 0.02)
n_window, n_hop = round(0.0232 * FS), round(0.0058 * FS)
window = np.hamming(n_window)
frames = np.lib.stride_tricks.sliding_window_view(signal, n_window)[::n_hop]
magnitude = np.abs(np.fft.rfft(frames * window, axis=1))
freqs = np.fft.rfftfreq(n_window, 1 / FS)
centroids = (magnitude * freqs).sum(1) / magnitude.sum(1)
analytic = (np.arange(1, 21) * F0 * SAWTOOTH_LIKE).sum() / SAWTOOTH_LIKE.sum()
middle = slice(len(centroids) // 4, 3 * len(centroids) // 4)
report("T-C2", "1/n amplitudes, 20 harmonics: analytic magnitude centroid [Hz]", analytic)
report("T-C2", "STFT median centroid over the sustain [Hz]", np.median(centroids[middle]))
power_analytic = (np.arange(1, 21) * F0 * SAWTOOTH_LIKE**2).sum() / (SAWTOOTH_LIKE**2).sum()
report("T-C2", "same tone, power-spectrum centroid instead [Hz]", power_analytic)

# T-C3: brightness steps for the page. Amplitudes n^(-slope) over 20 harmonics; the magnitude
# centroid in multiples of F0 for a few slopes.
for slope in (0.5, 1.0, 1.5, 2.0, 3.0):
    amplitudes = np.arange(1, 21) ** -slope
    centroid = (np.arange(1, 21) * amplitudes).sum() / amplitudes.sum()
    report("T-C3", f"amplitudes n^-{slope}: magnitude centroid / F0", centroid)


# T-C4: spectral variation (flux), 1 minus the normalized correlation of successive magnitude
# spectra. A steady tone gives about zero; a tone whose spectral slope glides from 2 to 0.5 over
# its second does not.
def variation(spectra):
    """1 minus the normalized correlation of each magnitude spectrum with the one before."""
    products = (spectra[1:] * spectra[:-1]).sum(1)
    return 1 - products / np.sqrt((spectra[1:] ** 2).sum(1) * (spectra[:-1] ** 2).sum(1))


report(
    "T-C4",
    "steady tone: median variation over the sustain [x 1e-6]",
    1e6 * np.median(variation(magnitude)[middle]),
)
t = np.arange(FS) / FS
slope_track = 2.0 - 1.5 * t
harmonics = np.arange(1, 21)
partials = np.sin(2 * np.pi * F0 * harmonics[:, None] * t)
gliding = (harmonics[:, None] ** -slope_track[None, :] * partials).sum(0)
frames = np.lib.stride_tricks.sliding_window_view(gliding, n_window)[::n_hop]
moving = np.abs(np.fft.rfft(frames * window, axis=1))
report(
    "T-C4",
    "slope gliding 2 -> 0.5 over 1 s: median variation [x 1e-6]",
    1e6 * np.median(variation(moving)[middle]),
)
# the same measure between spectra 100 ms apart rather than one hop (5.8 ms) apart
step = round(0.1 / 0.0058)
apart = 1 - (moving[step:] * moving[:-step]).sum(1) / np.sqrt(
    (moving[step:] ** 2).sum(1) * (moving[:-step] ** 2).sum(1)
)
report("T-C4", "  same tone, spectra 100 ms apart: median variation [x 1e-6]", 1e6 * np.median(apart))
steady_apart = 1 - (magnitude[step:] * magnitude[:-step]).sum(1) / np.sqrt(
    (magnitude[step:] ** 2).sum(1) * (magnitude[:-step] ** 2).sum(1)
)
report(
    "T-C4",
    "  steady tone, spectra 100 ms apart: median variation [x 1e-6]",
    1e6 * np.median(steady_apart[middle]),
)
gliding_centroid = (moving * freqs).sum(1) / moving.sum(1)
report("T-C4", "  its centroid at the start / F0", gliding_centroid[2] / F0)
report("T-C4", "  its centroid at the end / F0", gliding_centroid[-3] / F0)

# T-C5: the studies the page cites. Grey (1977): 16 tones near E-flat 4, 3-D INDSCAL solution,
# goodness of fit 0.68 (2-D), 0.75 (3-D), 0.78 (4-D). Printed from the paper, not computed.
for dims, fit in ((2, 0.68), (3, 0.75), (4, 0.78)):
    report("T-C5", f"Grey (1977) INDSCAL fit, {dims} dimensions (from the paper)", fit)
