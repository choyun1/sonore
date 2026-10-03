"""Klatt's (1980) digital resonator against the textbook damped oscillator.

The driven, damped harmonic oscillator x'' + 2 sigma x' + w0^2 x = w0^2 u has
the transfer function H(s) = w0^2 / (s^2 + 2 sigma s + w0^2): unit gain at
0 Hz and a pole pair at s = -sigma +- j wd, with wd^2 = w0^2 - sigma^2. Writing
sigma = pi bw and wd = 2 pi f, this checks how ``so.resonator(sound, f, bw)``
compares with it. Like the other checkers it uses only NumPy and SciPy, with
Klatt's coefficients written out from the paper. Each line prints the claim
and the number that supports it.

    python tools/check_resonator_physics.py

It runs in well under a second.
"""

import numpy as np
from scipy.signal import freqz

FS = 16000.0
CASES = [(500, 60), (1500, 100), (3500, 200), (300, 300), (7000, 300)]


def report(claim, text, value):
    print(f"{claim:4s} {text:<84s} {value:.4g}")


def klatt_coefs(f, bw, fs=FS):
    """y[n] = A x[n] + B y[n-1] + C y[n-2] (Klatt, 1980)."""
    c = -np.exp(-2 * np.pi * bw / fs)
    b = 2 * np.exp(-np.pi * bw / fs) * np.cos(2 * np.pi * f / fs)
    return 1 - b - c, b, c


def klatt_gain(f, bw, freqs, fs=FS):
    a, b, c = klatt_coefs(f, bw, fs)
    _, h = freqz([a], [1, -b, -c], worN=freqs, fs=fs)
    return np.abs(h)


def oscillator_gain(f, bw, freqs):
    sigma, damped = np.pi * bw, 2 * np.pi * f
    w0_squared = damped**2 + sigma**2
    s = 2j * np.pi * freqs
    return np.abs(w0_squared / (s**2 + 2 * sigma * s + w0_squared))


def peak_and_width(freqs, gain):
    above = freqs[gain >= gain.max() / np.sqrt(2)]
    return freqs[np.argmax(gain)], above.max() - above.min()


freqs = np.arange(0, FS / 2, 0.1)

# R1: the poles are the oscillator's poles mapped by z = exp(s / fs)
# (impulse invariance), so the ringing frequency and decay rate are exact.
pole_error = 0.0
for f, bw in CASES:
    a, b, c = klatt_coefs(f, bw)
    analog = np.exp((-np.pi * bw + 2j * np.pi * f) / FS)
    pole_error = max(pole_error, np.min(np.abs(np.roots([1, -b, -c]) - analog)))
report("R1", "largest distance of Klatt's poles from exp((-pi bw + j 2 pi f)/fs)", pole_error)

# R2: unit gain at 0 Hz, as the oscillator's static response.
dc_error = max(abs(klatt_gain(f, bw, [0.0])[0] - 1) for f, bw in CASES)
report("R2", "largest |gain at 0 Hz - 1| over the cases", dc_error)

# R3: f is the ringing frequency and bw the decay rate (bw = sigma / pi).
# They are the peak and the -3 dB width only when bw is much smaller than f;
# that is true of the oscillator itself, not an error of the digital filter.
for f, bw in CASES:
    peak, width = peak_and_width(freqs, klatt_gain(f, bw, freqs))
    peak_osc, width_osc = peak_and_width(freqs, oscillator_gain(f, bw, freqs))
    print(
        f"R3   f={f:4d} bw={bw:3d}: peak {peak:7.1f} Hz (oscillator {peak_osc:7.1f}),"
        f" -3 dB width {width:5.1f} Hz (oscillator {width_osc:5.1f})"
    )

# R4: around the resonance the digital magnitude follows the oscillator's
# while f is well below fs/2. Above it the digital filter falls off more
# slowly, because its response repeats every fs and has no zero at fs/2.
for f, bw in CASES:
    diff = 20 * np.log10(klatt_gain(f, bw, freqs) / oscillator_gain(f, bw, freqs))
    in_band = np.max(np.abs(diff[np.abs(freqs - f) <= bw]))
    at_4k, near_nyquist = diff[np.searchsorted(freqs, 4000)], diff[np.searchsorted(freqs, 7900)]
    print(
        f"R4   f={f:4d} bw={bw:3d}: dB from the oscillator: largest within f +- bw {in_band:5.2f},"
        f" at 4 kHz {at_4k:5.2f}, at 7.9 kHz {near_nyquist:5.2f}"
    )

# R5: the antiresonator's coefficients invert the resonator exactly.
a, b, c = klatt_coefs(1000, 80)
x = np.random.default_rng(0).standard_normal(4000)
y = np.zeros_like(x)
for n in range(len(x)):
    y[n] = a * x[n] + b * (y[n - 1] if n > 0 else 0) + c * (y[n - 2] if n > 1 else 0)
back = (y - b * np.r_[0, y[:-1]] - c * np.r_[0, 0, y[:-2]]) / a
report("R5", "largest |antiresonator(resonator(x)) - x| for unit-variance noise", np.max(np.abs(back - x)))
