"""Cross-check sonore's gammatone filters against a Hohmann (2002) style
complex gammatone filterbank.

The reference is the IIR filterbank from Cho's earlier notebook
(gammatone_testing.ipynb): each band is a cascade of ``order`` identical
complex one-pole filters with pole ``lambda * exp(i beta)``,
``lambda = exp(-2 pi b / fs)``, ``beta = 2 pi cf / fs``, and
``b = ERB_aud(cf) / a_gamma``, with ``ERB_aud = 24.7 + cf / 9.265`` and
``a_gamma = pi (2n - 2)! 2**-(2n - 2) / ((n - 1)!)**2``. The same constants,
the gain ``2 (1 - |pole|)**order`` and the cascade appear in pyfilterbank's
port of Hohmann (2002) (``pyfilterbank/gammatone.py``, which cites Eq. 13 for
``ERB_aud``). The real part of the complex output is the filtered signal and
its magnitude is the envelope.

The notebook's stages have numerator ``[g, g]`` then ``[1, 1]`` (a zero at
Nyquist in every stage); pyfilterbank's have ``[g]`` then ``[1]``. Both are checked here: "notebook" is the
notebook exactly, "hohmann" is the same cascade with numerator ``[g]``.

sonore's side is ``so.gammatone_filterbank(..., edges=False)``: the exact
Fourier transform of ``t**(n-1) exp(-2 pi b t) cos(2 pi cf t)`` on the DFT
grid, normalized to unit gain at cf, applied by FFT.

What it found (2026-10-06): with sonore's ``bandwidth_factor = 1/a_gamma``
(1.0186; sonore's default 1.019 is that number rounded) the two banks agree:
the same measured ERB, unit gain at cf, magnitudes within 0.06 dB down to
-20 dB for cf up to 8 kHz at 44.1 kHz, and impulse responses aligned to a
twentieth of a sample. The notebook's ``[1, 1]`` numerators add a gain of
about 16 (24 dB) at low cf that falls toward Nyquist, a two-sample delay,
and a high-frequency tilt; they are not in Hohmann's filter.

    python tools/crosscheck_gammatone_hohmann.py
"""

from math import factorial

import numpy as np
from scipy.signal import hilbert, lfilter

import sonore as so
from sonore.core.utils import erb_bandwidth

FS = 44100.0
N_SAMPLES = 44100  # 1 s
ORDER = 4
CFS = np.array([100.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 12000.0, 16000.0])


def report(text, value):
    print(f"{text:<78s} {value:.4g}")


def table(title, rows, header):
    print(f"\n{title}")
    print("  cf [Hz] " + "".join(f"{h:>13s}" for h in header))
    for cf, values in zip(CFS, rows, strict=True):
        print(f"  {cf:7.0f} " + "".join(f"{v:13.4g}" for v in values))


# ------------------------------------------------------------- reference
A_GAMMA = np.pi * factorial(2 * ORDER - 2) * 2.0 ** -(2 * ORDER - 2) / factorial(ORDER - 1) ** 2


def erb_aud(cf):
    return 24.7 + cf / 9.265


def reference_coefficients(cf, bandwidth_factor=1.0):
    """The notebook's gf_coef: gain and complex pole of one band."""
    b = erb_aud(cf) * bandwidth_factor / A_GAMMA
    pole = np.exp(-2 * np.pi * b / FS) * np.exp(1j * 2 * np.pi * cf / FS)
    return 2 * (1 - abs(pole)) ** ORDER, pole


def reference_filter(x, cf, zero_at_nyquist):
    """The notebook's gf_process for one band (zero_at_nyquist=True) or the
    pyfilterbank/Hohmann cascade (False). Returns the complex output."""
    gain, pole = reference_coefficients(cf)
    y = np.asarray(x, complex)
    numerator = [gain, gain] if zero_at_nyquist else [gain]
    for _ in range(ORDER):
        y = lfilter(numerator, [1.0, -pole], y)
        numerator = [1.0, 1.0] if zero_at_nyquist else [1.0]
    return y


def reference_response(freqs, cf, zero_at_nyquist):
    """Frequency response of the real part of the output, 0.5 (H(f) + conj H(-f))."""
    gain, pole = reference_coefficients(cf)

    def complex_response(f):
        z_inv = np.exp(-2j * np.pi * f / FS)
        numerator = gain * (1 + z_inv) ** ORDER if zero_at_nyquist else gain
        return numerator / (1 - pole * z_inv) ** ORDER

    return 0.5 * (complex_response(freqs) + np.conj(complex_response(-freqs)))


# ------------------------------------------------------------- sonore
def sonore_bank(bandwidth_factor):
    centers = np.concatenate([[CFS[0] / 2], CFS, [min(CFS[-1] * 1.2, FS / 2 * 0.999)]])
    return so.gammatone_filterbank(centers=centers, edges=False, bandwidth_factor=bandwidth_factor)


def sonore_b(cf, bandwidth_factor):
    return bandwidth_factor * erb_bandwidth(cf)


def measured_bandwidths(freqs, magnitude):
    """-3 dB width and equivalent rectangular bandwidth of |H| normalized to
    its peak, on a fine linear grid."""
    power = (magnitude / magnitude.max()) ** 2
    above = freqs[power >= 0.5]
    step = freqs[1] - freqs[0]
    return above.max() - above.min(), np.sum(power) * step


# ------------------------------------------------------------- bandwidth
print("Bandwidth parameter b (the gammatone's decay rate, exp(-2 pi b t))")
report("1/a_gamma for order 4 (b / ERB of the filter)", 1 / A_GAMMA)
ratio = (erb_aud(CFS) / A_GAMMA) / sonore_b(CFS, 1.019)
report("reference b / sonore b at its default 1.019, smallest over cfs", ratio.min())
report("reference b / sonore b at its default 1.019, largest over cfs", ratio.max())
erb_ratio = erb_aud(CFS) / erb_bandwidth(CFS)
report(
    "ERB formulas: (24.7 + cf/9.265) / 24.7(4.37 cf/1000 + 1), max |ratio - 1|", np.abs(erb_ratio - 1).max()
)

# Matched: sonore with bandwidth_factor = 1/a_gamma, so b differs only by the ERB formulas.
MATCHED = 1 / A_GAMMA
bank_default = sonore_bank(1.019)
bank_matched = sonore_bank(MATCHED)

fine = np.linspace(0, FS / 2, 2_000_001)
rows = []
for k, cf in enumerate(CFS):
    sonore_mag = np.abs(bank_matched.response(fine)[:, k])
    hohmann_mag = np.abs(reference_response(fine, cf, zero_at_nyquist=False))
    width_s, erb_s = measured_bandwidths(fine, sonore_mag)
    width_h, erb_h = measured_bandwidths(fine, hohmann_mag)
    rows.append([erb_s / sonore_b(cf, MATCHED), erb_h / (erb_aud(cf) / A_GAMMA), erb_s, erb_h, erb_aud(cf)])
table(
    f"Measured ERB of each filter (matched b); the two ratios should be a_gamma = {A_GAMMA:.4f}",
    rows,
    ["ERB/b sonore", "ERB/b hohm.", "ERB sonore", "ERB hohmann", "ERB_aud(cf)"],
)


# ------------------------------------------------------------- gain and shape
def shape_db(reference, sonore, mask, at_cf):
    """Largest difference [dB] between the two magnitudes, each normalized at cf, where mask holds."""
    ref_db = 20 * np.log10(np.abs(reference[mask]) / np.abs(reference[at_cf]))
    son_db = 20 * np.log10(np.abs(sonore[mask]) / np.abs(sonore[at_cf]))
    return np.abs(ref_db - son_db).max()


rfft_freqs = np.fft.rfftfreq(N_SAMPLES, 1 / FS)
rows = []
for k, cf in enumerate(CFS):
    sonore = bank_matched.response(rfft_freqs)[:, k]
    notebook = reference_response(rfft_freqs, cf, zero_at_nyquist=True)
    hohmann = reference_response(rfft_freqs, cf, zero_at_nyquist=False)
    at_cf = np.argmin(np.abs(rfft_freqs - cf))
    band = 20 * np.log10(np.abs(sonore) / np.abs(sonore).max()) > -20
    deep = 20 * np.log10(np.abs(sonore) / np.abs(sonore).max()) > -60

    rows.append(
        [
            np.abs(sonore[at_cf]),
            np.abs(hohmann[at_cf]),
            np.abs(notebook[at_cf]),
            shape_db(hohmann, sonore, band, at_cf),
            shape_db(hohmann, sonore, deep, at_cf),
            shape_db(notebook, sonore, band, at_cf),
        ]
    )
table(
    "Gain at cf (real part) and largest shape difference [dB] after normalizing at cf (matched b)",
    rows,
    ["gain sonore", "gain hohm.", "gain notebk", "hohm. >-20dB", "hohm. >-60dB", "nb >-20dB"],
)

# ------------------------------------------------------------- impulse responses
impulse = np.zeros(N_SAMPLES)
impulse[0] = 1.0
sonore_ir = bank_matched.analyze(so.Sound(impulse, FS), pad=0).data[:, :, 0]
t = np.arange(N_SAMPLES) / FS


def best_lag(a, b):
    """Lag (samples, refined by a parabola) that best aligns b to a, and the
    normalized correlation there."""
    spectrum = np.fft.rfft(a) * np.conj(np.fft.rfft(b))
    xcorr = np.fft.irfft(spectrum, n=len(a))
    peak = int(np.argmax(xcorr))
    left, mid, right = xcorr[peak - 1], xcorr[peak], xcorr[(peak + 1) % len(a)]
    shift = 0.5 * (left - right) / (left - 2 * mid + right)
    lag = peak + shift
    if lag > len(a) / 2:
        lag -= len(a)
    return lag, mid / np.sqrt(np.sum(a**2) * np.sum(b**2))


rows = []
for k, cf in enumerate(CFS):
    son = sonore_ir[:, k]
    ref = reference_filter(impulse, cf, zero_at_nyquist=False).real
    nb = reference_filter(impulse, cf, zero_at_nyquist=True).real
    lag_h, corr_h = best_lag(son, ref)
    lag_n, corr_n = best_lag(son, nb)
    son_unit, ref_unit = son / np.linalg.norm(son), ref / np.linalg.norm(ref)
    residual = np.linalg.norm(son_unit - ref_unit)
    b = sonore_b(cf, MATCHED)
    env_s = np.abs(hilbert(son))
    env_h = np.abs(reference_filter(impulse, cf, zero_at_nyquist=False))
    rows.append(
        [
            lag_h,
            1 - corr_h,
            residual,
            lag_n,
            1000 * t[np.argmax(env_s)],
            1000 * t[np.argmax(env_h)],
            1000 * (ORDER - 1) / (2 * np.pi * b),
        ]
    )
table(
    "Impulse responses (real part) with matched b. lag > 0: sonore later than the reference [samples]",
    rows,
    [
        "lag hohm.",
        "1-corr hohm.",
        "resid. no lag",
        "lag notebk",
        "env pk son ms",
        "env pk hoh ms",
        "(n-1)/2pib ms",
    ],
)

# ------------------------------------------------------------- envelopes of noise
rng = np.random.default_rng(0)
noise = rng.standard_normal(N_SAMPLES)
sonore_bands = bank_matched.analyze(so.Sound(noise, FS), pad=0)
sonore_env = sonore_bands.envelopes().data[:, :, 0]
sonore_sub = sonore_bands.data[:, :, 0]
rows = []
for k, cf in enumerate(CFS):
    ref = reference_filter(noise, cf, zero_at_nyquist=False)
    keep = slice(N_SAMPLES // 4, N_SAMPLES)  # skip the IIR filter's start-up
    lag, _ = best_lag(sonore_sub[:, k], ref.real)
    shift = int(round(lag))
    son_env = np.roll(sonore_env[:, k], shift)[keep]
    ref_env = np.abs(ref)[keep]
    son_sub = np.roll(sonore_sub[:, k], shift)[keep]
    rows.append(
        [
            np.corrcoef(son_sub, ref.real[keep])[0, 1],
            np.corrcoef(son_env, ref_env)[0, 1],
            np.sqrt(np.mean(ref_env**2) / np.mean(son_env**2)),
            np.linalg.norm(son_env - ref_env * np.linalg.norm(son_env) / np.linalg.norm(ref_env))
            / np.linalg.norm(son_env),
        ]
    )
table(
    "White noise through both banks (matched b, Hohmann numerator), aligned by the IR lag",
    rows,
    ["corr subband", "corr envelope", "rms env h/s", "rel env diff"],
)

# ------------------------------------------------------------- default bank
rows = []
for k, cf in enumerate(CFS):
    son = np.abs(bank_default.response(fine)[:, k])
    _, erb_s = measured_bandwidths(fine, son)
    hoh = np.abs(reference_response(fine, cf, zero_at_nyquist=False))
    _, erb_h = measured_bandwidths(fine, hoh)
    rows.append([erb_s, erb_h, erb_s / erb_h])
table(
    "sonore at its default 1.019 vs the reference at factor 1: measured ERB [Hz]",
    rows,
    ["sonore", "hohmann", "ratio"],
)
